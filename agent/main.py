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


def escalate_candidates(tasks_meta, threshold=None):
    threshold = threshold if threshold is not None else config.ESCALATE_CONF_THRESHOLD
    cands = [(tid, cat, prompt, res) for tid, cat, prompt, res in tasks_meta
             if CONF.get(tid, 0) < threshold]
    if not cands or config.ESCALATION_BUDGET_TOKENS <= 0:
        if cands:
            log(f"{len(cands)} weak tasks but escalation budget is 0 — staying local")
        return
    # cheapest-estimated-first, sequentially: actual spend (not max_tokens
    # reservations) gates later calls, maximizing tasks fixed per budget
    cands.sort(key=lambda c: fireworks.est_tokens(c[2]) + c[3].esc_max_tokens)
    log(f"escalating {len(cands)} weak tasks, cheapest first "
        f"(budget {fireworks.BUDGET.spent}/{fireworks.BUDGET.total})")
    for tid, cat, prompt, res in cands:
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

    tasks_meta = []
    if servers_ok and tps > 1.0:
        done = 0
        for tid, cat, prompt in tasks_c:
            mode = pick_mode(tps, len(tasks_c) - done)
            res = pipelines.run_task(cat, prompt, mode)
            RESULTS[tid] = res.answer or _FALLBACK
            CONF[tid] = res.confidence
            tasks_meta.append((tid, cat, prompt, res))
            flush()
            done += 1
            log(f"[{done}/{len(tasks_c)}] {tid} ({cat}, {mode}) conf={res.confidence:.2f}")
    else:
        log("LOCAL PATH UNAVAILABLE — escalating everything within budget")
        for tid, cat, prompt in tasks_c:
            tasks_meta.append((tid, cat, prompt,
                               pipelines.Result(_FALLBACK, 0.0, esc_max_tokens=300)))

    escalate_candidates(tasks_meta)

    # anytime loop: spend leftover wall-clock strengthening the weakest local answers
    if servers_ok and tps > 1.0:
        weak = sorted((t for t in tasks_meta
                       if 0.05 < CONF.get(t[0], 0) < 0.78),
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
