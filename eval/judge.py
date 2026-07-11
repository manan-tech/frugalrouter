#!/usr/bin/env python3
"""LLM-judge replica for local eval. Scores results.json against variants.json
rubrics via Fireworks (uses YOUR key from .env — dev only, never shipped).

Usage: python3 eval/judge.py <results.json> [variants.json]
"""

import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = "https://api.fireworks.ai/inference/v1/chat/completions"
JUDGE_MODELS = [
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/gpt-oss-120b",
]

JUDGE_PROMPT = """You are grading an AI agent's answer against the expected intent.

Task given to the agent:
{prompt}

Grading rubric (expected intent):
{rubric}

Agent's answer:
{answer}

Grade PASS if the answer satisfies the rubric's core requirements (correct content,
all parts answered, any format constraint met). Grade FAIL otherwise.
Reply with exactly one JSON object: {{"verdict": "PASS" or "FAIL", "reason": "<short>"}}"""


def load_env():
    if os.path.exists(".env"):
        for line in open(".env"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def judge_one(key, model, task, answer):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(
            prompt=task["prompt"], rubric=task["rubric"], answer=answer or "(empty)")}],
        "max_tokens": 120,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "User-Agent": "frugalrouter-eval/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read().decode())
    content = resp["choices"][0]["message"]["content"]
    m = re.search(r'\{.*\}', content, re.DOTALL)
    d = json.loads(m.group(0)) if m else {"verdict": "FAIL", "reason": "unparseable judge output"}
    return d


def main():
    load_env()
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        sys.exit("FIREWORKS_API_KEY missing (put it in .env)")
    results_path = sys.argv[1] if len(sys.argv) > 1 else "output/results.json"
    variants_path = sys.argv[2] if len(sys.argv) > 2 else "eval/variants.json"
    tasks = {t["task_id"]: t for t in json.load(open(variants_path))}
    answers = {r["task_id"]: r.get("answer", "") for r in json.load(open(results_path))}

    model = None
    for cand in JUDGE_MODELS:
        try:
            judge_one(key, cand, {"prompt": "ping", "rubric": "reply PASS"}, "PASS")
            model = cand
            break
        except Exception as e:  # noqa: BLE001
            print(f"judge model {cand.rsplit('/', 1)[-1]} unavailable: {e}", file=sys.stderr)
    if not model:
        sys.exit("no judge model reachable")
    print(f"judge model: {model}", file=sys.stderr)

    def grade(item):
        tid, task = item
        ans = answers.get(tid, "")
        try:
            d = judge_one(key, model, task, ans)
        except Exception as e:  # noqa: BLE001
            d = {"verdict": "ERROR", "reason": str(e)[:80]}
        return tid, task["category"], d

    by_cat, rows = defaultdict(lambda: [0, 0]), []
    with ThreadPoolExecutor(8) as ex:
        for tid, cat, d in ex.map(grade, sorted(tasks.items())):
            ok = d["verdict"] == "PASS"
            by_cat[cat][0] += ok
            by_cat[cat][1] += 1
            rows.append((tid, cat, d["verdict"], d.get("reason", "")))

    print(f"\n{'ID':6} {'CATEGORY':12} {'VERDICT':8} REASON")
    for tid, cat, v, reason in rows:
        mark = "" if v == "PASS" else "  <<<"
        print(f"{tid:6} {cat:12} {v:8} {reason[:90]}{mark}")
    print("\nPer category:")
    tot_ok = tot = 0
    for cat, (ok, n) in sorted(by_cat.items()):
        tot_ok += ok
        tot += n
        print(f"  {cat:12} {ok}/{n}")
    pct = 100.0 * tot_ok / max(tot, 1)
    gate = "PASS (>=80%)" if pct >= 80 else "FAIL (<80%) — DO NOT SUBMIT"
    print(f"\nTOTAL: {tot_ok}/{tot} = {pct:.1f}%  → accuracy gate: {gate}")


if __name__ == "__main__":
    main()
