"""One-shot inference for the fine-tuned model (v23+).

The SFT model was trained on exactly one distribution: THIS system prompt,
the raw task prompt as the user turn, thinking disabled, and the answer in
the category's exact output format. Prompting it any other way (the legacy
per-category system prompts, few-shot blocks, grammar constraints) moves it
off-distribution and throws away what the fine-tune bought.

Math is the one category with a post-step by construction: the trained
answer IS a fenced Python program, and the shipped answer is that program's
stdout (finetune/sft.jsonl math rows). A 1.7B can't do compound interest in
its head, but it reliably writes the four-line program that can.

Confidence semantics: 0.95 = trained-path answer (above the 0.70 anytime-loop
bar, below EARLY_CONF's reserved 1.0). Escalation is structurally off in this
build (all budgets are hard zero), so confidence only steers the anytime loop.
"""

import re

from . import config
from .llm import GENERAL
from .sandbox import run_python
from .util import elapsed, extract_code, log

# BYTE-IDENTICAL to finetune/train.py SYSTEM_PROMPT — the single most
# load-bearing string in this build. Do not reflow, "fix" punctuation, or
# deduplicate spaces: the model saw these exact bytes 2,708 times.
SYSTEM_PROMPT = (
    "You answer the user's task directly and in the exact format requested. "
    "No preamble, no restating the question, no explanation of your process. "
    "Answer once, correctly."
)

# Generation caps: generous vs the trained answer lengths (p50=127 tokens
# across the SFT set) so a correct answer is never truncated, bounded so a
# rare runaway can't eat the wall-clock. The trained model stops at
# <|im_end|> well before these.
_CAPS = {
    "math": 300,
    "code_gen": 400,
    "code_debug": 400,
    "logic": 220,
    "factual": 200,
    "summary": 200,
    "ner": 200,
    "sentiment": 120,
}

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _chat(prompt: str, cap: int, temperature: float = 0.0) -> str:
    return GENERAL.chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        max_tokens=cap, temperature=temperature, thinking=False,
        timeout_s=120)


def _exec_math(raw: str):
    """(shipped_answer, ok). The trained math answer is a fenced program;
    the answer the judge sees is what it prints."""
    m = _FENCE.search(raw)
    code = m.group(1) if m else raw
    ok, out, err = run_python(code)
    if ok and out.strip():
        return out.strip(), True
    log(f"oneshot math program failed ({(err or 'no output')[:120]})")
    return raw, False


# --------------------------------------------------------------------------
# code_gen behavioral majority vote
#
# Why not the legacy test-gen/verify/repair machinery: the SFT model no
# longer follows out-of-distribution instructions — asked for asserts, it
# emits an implementation (measured: 0/12 testgen attempts produced a single
# assert). Model-generated tests and inputs are structurally dead here.
#
# What measurably works instead: the model's errors on code_gen are
# knife-edge sampling accidents, not beliefs — rehearsal g2 produced the
# correct behavior in 3 of 5 samples (and temp-0 itself flips between
# correct/wrong with the runner's thread count, the same float-noise class
# as the router's f3 coin flip). So: 3 samples, fingerprint each on a
# DETERMINISTIC input battery (no model in the loop), ship temp-0's answer
# unless it sits outside a >=2-sample behavior cluster.
#
# code_debug deliberately gets NO vote: its misses are systematic (d2: 5/5
# samples share the same wrong behavior — resampling can't help), and fix
# samples are anchored to the given buggy code anyway.
# --------------------------------------------------------------------------

_STR_BATTERY = ["", "a", "A", "Aa", "ab", "racecar", "RaceCar",
                "A man a plan a canal Panama", "hello", "Hello World",
                "12321", "AEIOUaeiou xyz"]
_INT_BATTERY = [0, 1, 2, 5, 7, 10, -3, 100]
_LIST_BATTERY = [[], [1], [3, 1, 2], [1, 1, 2, 2, 3], [5, 4, 3, 2, 1],
                 [2, 2, 2], [10, -1, 7, 7, 0, 3]]
# batteries tried per positional-arg count; first one where >= half the
# calls return without raising is the fingerprint domain
_BATTERIES = {
    1: [[(s,) for s in _STR_BATTERY],
        [(x,) for x in _LIST_BATTERY],
        [(n,) for n in _INT_BATTERY]],
    2: [[(a, b) for a, b in [([], []), ([1], []), ([1, 3, 5], [2, 4, 6]),
                             ([1, 2], [1, 2]), ([7], [3, 9]),
                             ([-1, 0], [0, 1])]],
        [(a, b) for a, b in [("abc", "cba"), ("", ""), ("listen", "silent"),
                             ("Hello", "world")]],
        [(a, b) for a, b in [(3, 4), (0, 0), (-2, 7), (10, 5)]]],
}


def _fingerprint(code: str):
    """repr of the first function's outputs over the first usable battery,
    or None. Purely deterministic — no model involvement."""
    m = re.search(r"^def\s+(\w+)\s*\(([^)]*)\)", code, re.MULTILINE)
    if not m:
        return None
    name = m.group(1)
    arity = len([a for a in m.group(2).split(",")
                 if a.strip() and "=" not in a and not a.strip().startswith("*")])
    parts = []
    for battery in _BATTERIES.get(arity, []):
        prog = (f"{code}\n\n_ARGS = {battery!r}\n_out = []\n"
                "for _a in _ARGS:\n"
                "    try:\n"
                f"        _out.append(repr({name}(*_a)))\n"
                "    except Exception as _e:\n"
                "        _out.append('ERR:' + type(_e).__name__)\n"
                "print('|'.join(_out))\n")
        ok, stdout, _err = run_python(prog)
        if not ok or not stdout.strip():
            continue
        fp = stdout.strip()
        errs = fp.count("ERR:")
        if errs * 2 <= len(battery):     # at least half the calls ran
            # keep EVERY usable battery, not just the first: a list function
            # also "runs" on strings (both iterate), and two impls can agree
            # on strings while differing on the lists that actually matter
            parts.append(fp)
    return f"{arity}:" + "||".join(parts) if parts else None


def _majority_code(prompt: str, raw0: str):
    """Two resamples + behavioral vote. Returns (answer_to_ship, confident):
    ship temp-0's answer unless a >=2 cluster it isn't part of overrules it;
    `confident` is True iff some behavior recurred across >=2 samples. When
    every sample scatters to a different behavior (no consensus), the problem
    was too hard for the 1.7B this run — confident=False routes it to
    escalation instead of shipping a coin-flip."""
    code0 = extract_code(raw0)
    if not code0 or "def " not in code0:
        return raw0, False                # unparseable — let escalation decide
    if elapsed() > config.SOFT_NEW_WORK_S * 0.6:
        return raw0, True                 # protect the deadline; trust temp-0
    fp0 = _fingerprint(code0)
    if fp0 is None:
        return raw0, True                 # no usable battery — can't vote, keep it
    votes = [(fp0, raw0)]
    for temp in (0.5, 0.9):
        try:
            raw = _chat(prompt, _CAPS["code_gen"], temperature=temp)
        except Exception as e:  # noqa: BLE001 — voting is a bonus, never a risk
            log(f"oneshot[code_gen] resample failed (non-fatal): {e}")
            continue
        code = extract_code(raw)
        if not code or "def " not in code:
            continue
        fp = _fingerprint(code)
        if fp is not None:
            votes.append((fp, raw))
    counts = {}
    for fp, _raw in votes:
        counts[fp] = counts.get(fp, 0) + 1
    top_fp = max(counts, key=lambda k: counts[k])
    confident = counts[top_fp] >= 2
    if confident and fp0 != top_fp:
        winner = next(r for f, r in votes if f == top_fp)
        log(f"oneshot[code_gen] majority overruled temp-0 "
            f"({counts[top_fp]}/{len(votes)} agree)")
        return winner, True
    if not confident:
        log(f"oneshot[code_gen] no behavioral consensus across {len(votes)} "
            f"samples — routing to escalation")
    return raw0, confident


def answer(category: str, prompt: str):
    """One trained-distribution generation; math executes its program.

    Never raises: any local-inference failure surfaces as a low-confidence
    Result so main.py's dead-streak detector still sees it.
    """
    from .pipelines import Result  # local import — pipelines imports us too

    cap = _CAPS.get(category, 300)
    try:
        raw = _chat(prompt, cap)
    except Exception as e:  # noqa: BLE001 — a task must never take down the run
        log(f"oneshot[{category}] generation failed: {e}")
        return Result("Unable to determine.", 0.0)

    if not raw.strip():
        # rare with temp 0, but a blank answer is an automatic judge FAIL —
        # one nudged retry is cheap insurance
        try:
            raw = _chat(prompt, cap, temperature=0.3)
        except Exception as e:  # noqa: BLE001
            log(f"oneshot[{category}] retry failed: {e}")
        if not raw.strip():
            return Result("Unable to determine.", 0.0)

    if category == "code_gen":
        shipped, confident = _majority_code(prompt, raw)
        # confident (a >=2 behavioral cluster) -> trust it, zero tokens.
        # no consensus -> conf below the 0.55 escalation threshold so
        # escalate_candidates buys a remote answer; the local pick stands as
        # the fallback if the budget/proxy can't.
        return Result(shipped.strip(), 0.95 if confident else 0.45,
                      esc_max_tokens=420)

    if category == "math":
        shipped, ok = _exec_math(raw)
        if not ok:
            # a second sample usually writes a working program; keep the raw
            # text (which at least shows the working) if it doesn't
            try:
                raw2 = _chat(prompt, cap, temperature=0.3)
                shipped2, ok2 = _exec_math(raw2)
                if ok2:
                    return Result(shipped2, 0.95)
            except Exception as e:  # noqa: BLE001
                log(f"oneshot[math] retry failed: {e}")
            return Result(shipped, 0.30)
        return Result(shipped, 0.95)

    if category == "factual":
        # A 1.7B is confidently wrong on facts too often to trust locally, and
        # nothing here can verify a fact (grader box: "Venus is the Red
        # Planet"). Ship LOW conf with the remote format hint: main.early_escalate
        # answers factual in a batch up front (primary path), and if that proxy
        # call fails, this sub-threshold Result lets escalate_candidates retry it
        # individually. The local answer stands only if both remote paths fail.
        from .pipelines import FACTUAL_ESC
        return Result(raw.strip(), 0.45, esc_suffix=FACTUAL_ESC,
                      esc_max_tokens=160)

    return Result(raw.strip(), 0.95)
