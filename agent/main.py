"""FrugalRouter orchestrator.

Contract: read /input/tasks.json -> write /output/results.json, exit 0.
Never crash, never time out, never leave invalid JSON."""

import json
import os
import signal
import sys
import threading

from . import config, fireworks, llm, pipelines
from .classify import classify
from .util import atomic_write_results, elapsed, log

RESULTS = {}          # task_id -> answer string
CONF = {}             # task_id -> confidence
_FALLBACK = "I could not determine the answer."


def flush():
    atomic_write_results(config.OUTPUT_PATH, RESULTS)


def emit_calibration(tid, cat, res):
    """Local-eval calibration producer — no-op unless CALIBRATION_LOG_PATH is
    set (the grading harness never sets it). Emits the agent-side half of the
    eval/calibrate.py record; join judge.py pass/fail as "judge_pass" before
    running calibration. Signals come from llm.LAST_SIGNALS (the most recent
    local completion for this task)."""
    if not config.CALIBRATION_LOG_PATH:
        return
    try:
        rec = {"task_id": tid, "category": cat,
               "confidence": res.confidence,
               "signals": dict(llm.LAST_SIGNALS)}
        with open(config.CALIBRATION_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:  # noqa: BLE001 — logging must never break the run
        log(f"calibration log error: {e}")


def watchdog():
    import time
    flushed = False
    while True:
        time.sleep(2)
        t = elapsed()
        if t >= config.FLUSH_S and not flushed:
            log("WATCHDOG: forced flush")
            flush()
            flushed = True
        if t >= config.HARD_EXIT_S:
            log("WATCHDOG: hard exit")
            flush()
            os._exit(0)


def load_tasks():
    with open(config.INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("tasks", [])
    tasks = []
    for item in data:
        tid = str(item.get("task_id", f"t{len(tasks) + 1}"))
        tasks.append((tid, str(item.get("prompt", ""))))
    return tasks


def pick_mode(tps: float, tasks_left: int) -> str:
    base = "full" if tps >= config.TPS_FULL else (
        "lean" if tps >= config.TPS_LEAN else "panic")
    remaining = config.SOFT_NEW_WORK_S - elapsed()
    est = {"full": 26.0, "lean": 13.0, "panic": 6.0}
    mode = base
    order = ["full", "lean", "panic"]
    i = order.index(mode)
    while i < 2 and tasks_left * est[order[i]] > max(remaining, 1):
        i += 1
    return order[i]


def _esc_threshold(cat, override):
    """Per-category escalation threshold, unless an explicit override is given
    (second pass / emergency reuse the single caller-supplied value)."""
    if override is not None:
        return override
    thr = getattr(config, "CATEGORY_THRESHOLDS", None) or {}
    return thr.get(cat, config.ESCALATE_CONF_THRESHOLD)


def escalate_candidates(tasks_meta, threshold=None):
    cands = [(tid, cat, prompt, res) for tid, cat, prompt, res in tasks_meta
             if CONF.get(tid, 0) < _esc_threshold(cat, threshold)]
    if not cands or config.ESCALATION_BUDGET_TOKENS <= 0:
        if cands:
            log(f"{len(cands)} weak tasks but escalation budget is 0 — staying local")
        return
    # group by category: same-category weak questions ride one batched call
    # (byte-stable prefix -> cache-friendly). Cheapest category first, and
    # cheapest member first within it, so the shared budget fixes the most.
    groups = {}
    for cand in cands:
        groups.setdefault(cand[1], []).append(cand)
    for members in groups.values():
        members.sort(key=lambda c: fireworks.est_tokens(c[2]) + c[3].esc_max_tokens)
    ordered = sorted(groups.values(),
                     key=lambda ms: sum(fireworks.est_tokens(c[2]) + c[3].esc_max_tokens
                                        for c in ms) / len(ms))
    batch_on = getattr(config, "BATCH_ESCALATION", True)
    log(f"escalating {len(cands)} weak tasks in {len(ordered)} category groups, "
        f"cheapest first (budget {fireworks.BUDGET.spent}/{fireworks.BUDGET.total})")
    for members in ordered:
        if elapsed() > config.FLUSH_S - 15:
            break
        cat = members[0][1]
        if batch_on and len(members) >= 2:
            # keep each member's category-specific format hint (esc_suffix) —
            # exact-match judging depends on it, same as the single-item path
            answers = fireworks.batch_chat([(tid, prompt + res.esc_suffix)
                                            for tid, _c, prompt, res in members],
                                           cat)
            for tid, _c, _p, _r in members:
                ans = answers.get(tid)
                if ans and ans.strip():
                    RESULTS[tid] = ans.strip()
                    CONF[tid] = 0.88
            flush()
        else:
            for tid, _c, prompt, res in members:
                if elapsed() > config.FLUSH_S - 15:
                    break
                ans, _spent = fireworks.chat(prompt + res.esc_suffix, cat,
                                             max_tokens=res.esc_max_tokens)
                if ans and ans.strip():
                    RESULTS[tid] = ans.strip()
                    CONF[tid] = 0.88
                    flush()


def main() -> int:
    log("FrugalRouter starting")
    llm.CAPTURE_SIGNALS = bool(config.CALIBRATION_LOG_PATH)  # local eval only
    threading.Thread(target=watchdog, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: (flush(), os._exit(0)))

    try:
        tasks = load_tasks()
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: cannot read tasks: {e}")
        atomic_write_results(config.OUTPUT_PATH, {})
        return 0
    for tid, _ in tasks:
        RESULTS[tid] = _FALLBACK
        CONF[tid] = 0.0
    flush()
    log(f"{len(tasks)} tasks loaded")

    servers_ok = False
    try:
        servers_ok = llm.start_all()
    except Exception as e:  # noqa: BLE001
        log(f"server start crashed: {e}")
    tps = llm.probe_tps() if servers_ok else 0.0

    # classify + sort by category so llama.cpp reuses cached prompt prefixes
    tasks_c = sorted(((tid, classify(p), p) for tid, p in tasks),
                     key=lambda t: t[1])
    log("categories: " + ", ".join(f"{tid}={cat}" for tid, cat, _ in tasks_c))

    local_usable = servers_ok and tps >= config.TPS_DEAD
    if config.TEST_FORCE_EMERGENCY:
        # local-eval test hook (eval.yml force_emergency): exercise the
        # emergency escalate-all path exactly as a dead local would
        log("TEST_FORCE_EMERGENCY=1 — forcing emergency escalate-all path")
        local_usable = False
    if not local_usable:
        # gate survival outranks token rank: answer everything via Fireworks
        log(f"LOCAL PATH UNAVAILABLE (servers_ok={servers_ok}, tps={tps:.1f}) "
            f"— emergency escalation of all tasks")
        fireworks.raise_budget(config.EMERGENCY_BUDGET_TOKENS)

    tasks_meta = []
    if local_usable:
        done = 0
        for i, (tid, cat, prompt) in enumerate(tasks_c):
            if elapsed() > config.SOFT_NEW_WORK_S - 25:
                log(f"soft deadline — {len(tasks_c) - done} tasks go straight "
                    f"to escalation")
                for tid2, cat2, prompt2 in tasks_c[i:]:
                    tasks_meta.append((tid2, cat2, prompt2,
                                       pipelines.Result(_FALLBACK, 0.0,
                                                        esc_max_tokens=300)))
                needed = fireworks.BUDGET.spent + (len(tasks_c) - done) * 260
                fireworks.raise_budget(min(config.EMERGENCY_BUDGET_TOKENS, needed))
                break
            mode = pick_mode(tps, len(tasks_c) - done)
            res = pipelines.run_task(cat, prompt, mode)
            RESULTS[tid] = res.answer or _FALLBACK
            CONF[tid] = res.confidence
            tasks_meta.append((tid, cat, prompt, res))
            emit_calibration(tid, cat, res)
            flush()
            done += 1
            log(f"[{done}/{len(tasks_c)}] {tid} ({cat}, {mode}) conf={res.confidence:.2f}")
            # machine-readable record: joined with judge verdicts by eval
            # tooling to fit CATEGORY_THRESHOLDS (eval/calibrate.py)
            log("CALIB " + json.dumps({"task_id": tid, "category": cat,
                                       "confidence": round(res.confidence, 3),
                                       "mode": mode}))
    else:
        for tid, cat, prompt in tasks_c:
            tasks_meta.append((tid, cat, prompt,
                               pipelines.Result(_FALLBACK, 0.0,
                                                esc_max_tokens=300)))

    escalate_candidates(tasks_meta)

    # anytime loop: spend leftover wall-clock strengthening the weakest local answers
    if local_usable:
        weak = sorted((t for t in tasks_meta
                       if 0.05 < CONF.get(t[0], 0) < 0.70),
                      key=lambda t: CONF.get(t[0], 0))
        for tid, cat, prompt, _res in weak:
            if elapsed() > config.SOFT_NEW_WORK_S - 30:
                break
            log(f"anytime: re-running {tid} ({cat})")
            try:
                res2 = pipelines.run_task(cat, prompt, "full")
                conf2 = res2.confidence
                if cat == "factual":
                    # agreement can't verify facts — never let a factual
                    # answer look solid enough to dodge the second escalation
                    conf2 = min(conf2, 0.75)
                if conf2 > CONF.get(tid, 0) and res2.answer.strip():
                    RESULTS[tid] = res2.answer
                    CONF[tid] = conf2
                    flush()
                    log(f"anytime: improved {tid} to conf={conf2:.2f}")
            except Exception as e:  # noqa: BLE001
                log(f"anytime error on {tid}: {e}")

    # second escalation pass: leftover budget goes to anything still shaky
    escalate_candidates(tasks_meta, threshold=0.60)

    # last-ditch sweep: nothing may ship as fallback text while budget remains
    for tid, cat, prompt, res in tasks_meta:
        if CONF.get(tid, 0) >= 0.1 or elapsed() > config.FLUSH_S - 12:
            continue
        _r, per_item = fireworks._esc_cap(cat)
        ans, _sp = fireworks.chat(prompt + res.esc_suffix, cat, max_tokens=per_item)
        if ans and ans.strip():
            RESULTS[tid] = ans.strip()
            CONF[tid] = 0.85
            flush()
            log(f"last-ditch escalation rescued {tid}")

    flush()
    try:
        llm.stop_all()
    except Exception:  # noqa: BLE001
        pass
    log(f"done: {len(tasks)} answers, fireworks tokens spent={fireworks.BUDGET.spent}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — absolute last resort
        log(f"UNCAUGHT: {e}")
        try:
            flush()
        except Exception:  # noqa: BLE001
            pass
        sys.exit(0)
