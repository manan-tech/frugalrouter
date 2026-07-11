#!/usr/bin/env python3
"""Fit per-category escalation thresholds from eval run records.

Input: JSONL, one record per task:
  {"task_id": "m1", "category": "math", "confidence": 0.62,
   "signals": {...}, "judge_pass": true}

Model: a task is escalated iff confidence < threshold[category]; an escalated
task passes with ESC_PASS_RATE probability, a local task passes iff its
judge_pass. Starting from escalate-everything, each category's threshold is
lowered to the smallest (most token-frugal) candidate such that the projected
accuracy across ALL categories keeps a one-sided 95% binomial (Wilson) lower
bound >= TARGET (0.88, safety margin over the 80% gate). Categories with the
largest potential token savings are relaxed first.

Emits a CATEGORY_THRESHOLDS dict on stdout, ready to paste into
agent/config.py. Diagnostics go to stderr. Pure stdlib.

Usage: python3 eval/calibrate.py records.jsonl [--target 0.88]
       [--esc-pass-rate 0.95]
"""

import json
import math
import sys

TARGET = 0.88
ESC_PASS_RATE = 0.95
Z95 = 1.6448536269514722  # one-sided 95%

# Mirrors ESC_CAPS max_tokens in agent/config.py — used only to weight token
# savings when ordering categories; keep loosely in sync.
ESC_TOKEN_COST = {
    "factual": 300, "sentiment": 200, "ner": 250, "summary": 500,
    "math": 400, "logic": 400, "code_gen": 900, "code_debug": 900,
}
DEFAULT_ESC_COST = 400
ESCALATE_ALL = 1.01  # every confidence < this -> escalate everything

# Full category set from agent/config.py (ESC_CAPS / classify.CATEGORIES) and
# its global default threshold (ESCALATE_CONF_THRESHOLD). Categories absent
# from the input JSONL are emitted at the default so the stdout dict can be
# pasted verbatim into agent/config.py without dropping keys.
CONFIG_CATEGORIES = tuple(sorted(ESC_TOKEN_COST))
DEFAULT_THRESHOLD = 0.55


def wilson_lower(passes: float, n: int, z: float = Z95) -> float:
    """One-sided Wilson score lower bound for a binomial proportion.
    Accepts fractional passes (expected passes under ESC_PASS_RATE)."""
    if n <= 0:
        return 0.0
    p = passes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    rad = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (center - rad) / denom)


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip line {ln}: bad JSON ({e})", file=sys.stderr)
                continue
            try:
                recs.append({
                    "task_id": str(r["task_id"]),
                    "category": str(r["category"]),
                    "confidence": float(r["confidence"]),
                    "signals": r.get("signals") or {},
                    "judge_pass": bool(r["judge_pass"]),
                })
            except (KeyError, TypeError, ValueError) as e:
                print(f"skip line {ln}: missing/bad field ({e})", file=sys.stderr)
    return recs


def esc_cost(rec, esc_pass_rate=None):
    """Estimated remote tokens if this task escalates. Prefers a per-record
    signal, falls back to the category cap."""
    sig = rec["signals"]
    for key in ("esc_tokens", "escalation_tokens"):
        v = sig.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return float(ESC_TOKEN_COST.get(rec["category"], DEFAULT_ESC_COST))


def projected(recs, thresholds, esc_pass_rate):
    """(expected_passes, n, escalated_count, escalated_tokens) under the
    escalate-if-conf-below-threshold policy."""
    passes = 0.0
    esc_n = 0
    esc_tok = 0.0
    for r in recs:
        if r["confidence"] < thresholds[r["category"]]:
            passes += esc_pass_rate
            esc_n += 1
            esc_tok += esc_cost(r)
        else:
            passes += 1.0 if r["judge_pass"] else 0.0
    return passes, len(recs), esc_n, esc_tok


def candidate_thresholds(cat_recs):
    """Sorted candidates: 0.0 (never escalate), each observed confidence
    (keeps that task local, escalates strictly-lower ones), escalate-all."""
    cands = {0.0, ESCALATE_ALL}
    for r in cat_recs:
        cands.add(round(r["confidence"], 4))
    return sorted(cands)


def fit(recs, target, esc_pass_rate):
    by_cat = {}
    for r in recs:
        by_cat.setdefault(r["category"], []).append(r)

    # start safest: escalate everything
    thresholds = {cat: ESCALATE_ALL for cat in by_cat}
    base_passes, n, _, _ = projected(recs, thresholds, esc_pass_rate)
    if wilson_lower(base_passes, n) < target:
        print(f"WARNING: even escalate-everything only projects a "
              f"{wilson_lower(base_passes, n):.3f} lower bound (< {target}); "
              f"emitting escalate-all thresholds", file=sys.stderr)
        return thresholds

    # relax categories with the biggest potential token savings first
    def savings(cat):
        return sum(esc_cost(r) for r in by_cat[cat])

    for cat in sorted(by_cat, key=savings, reverse=True):
        for cand in candidate_thresholds(by_cat[cat]):  # frugal-first (low->high)
            trial = dict(thresholds)
            trial[cat] = cand
            passes, n, _, _ = projected(recs, trial, esc_pass_rate)
            if wilson_lower(passes, n) >= target:
                thresholds[cat] = cand
                break
        # no candidate below ESCALATE_ALL was safe -> stays escalate-all
    return thresholds


def main():
    args = [a for a in sys.argv[1:]]
    target, esc_pass_rate = TARGET, ESC_PASS_RATE
    path = None
    i = 0
    while i < len(args):
        if args[i] == "--target":
            target = float(args[i + 1]); i += 2
        elif args[i] == "--esc-pass-rate":
            esc_pass_rate = float(args[i + 1]); i += 2
        else:
            path = args[i]; i += 1
    if not path:
        sys.exit("usage: calibrate.py records.jsonl [--target 0.88] "
                 "[--esc-pass-rate 0.95]")

    recs = load_records(path)
    if not recs:
        sys.exit("no usable records")

    thresholds = fit(recs, target, esc_pass_rate)
    passes, n, esc_n, esc_tok = projected(recs, thresholds, esc_pass_rate)
    lb = wilson_lower(passes, n)
    print(f"records={n} escalated={esc_n} est_esc_tokens={esc_tok:.0f} "
          f"projected_acc={passes / n:.3f} wilson95_lb={lb:.3f} "
          f"(target {target})", file=sys.stderr)
    for cat in sorted(thresholds):
        cat_recs = [r for r in recs if r["category"] == cat]
        esc = sum(1 for r in cat_recs if r["confidence"] < thresholds[cat])
        print(f"  {cat:12} thr={thresholds[cat]:.4f}  "
              f"escalates {esc}/{len(cat_recs)}", file=sys.stderr)

    # paste-ready output (stdout only): seed every config category so a
    # verbatim paste is complete; unfitted ones keep the global default.
    full = {cat: DEFAULT_THRESHOLD for cat in CONFIG_CATEGORIES}
    full.update(thresholds)
    print("# generated by eval/calibrate.py — paste into agent/config.py")
    print("CATEGORY_THRESHOLDS = {")
    for cat in sorted(full):
        note = "" if cat in thresholds else "  # not in input; global default"
        print(f'    "{cat}": {full[cat]:.4f},{note}')
    print("}")


if __name__ == "__main__":
    main()
