"""Per-category pipelines. Each converts raw model ability into *verified*
answers: math runs generated Python, code runs generated tests, logic
brute-forces extracted constraints, everything else votes across samples.

Handlers return Result(answer, confidence, esc_suffix, esc_max_tokens).
Confidence semantics: >=0.8 verified/agreed, ~0.5 plausible single-shot,
<=0.4 shaky (escalation candidate)."""

import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass

from . import config
from .llm import CODER, GENERAL
from .sandbox import run_python, run_with_tests
from .util import (extract_code, extract_last_number, fmt_number, log,
                   similarity, split_sentences)


@dataclass
class Result:
    answer: str
    confidence: float
    esc_suffix: str = ""
    esc_max_tokens: int = 120


def _samples_for(mode: str, full: int, lean: int) -> int:
    return {"full": full, "lean": lean}.get(mode, 1)


# --------------------------------------------------------------------------
# factual
# --------------------------------------------------------------------------
FACTUAL_SYS = ("You are a precise assistant. Answer the question directly and "
               "completely in 1-2 short sentences. If the question has multiple "
               "parts, answer every part.")


def factual(prompt: str, mode: str) -> Result:
    n = _samples_for(mode, 3, 2)
    outs = []
    for i in range(n):
        outs.append(GENERAL.chat(
            [{"role": "system", "content": FACTUAL_SYS},
             {"role": "user", "content": prompt}],
            max_tokens=90, temperature=0.3 if i == 0 else config.GEN_TEMP))
    best, agree = _majority_by_similarity(outs)
    conf = {3: 0.85, 2: 0.7, 1: 0.35 if n > 1 else 0.5}.get(agree, 0.5)
    return Result(best, conf, esc_suffix=" Answer briefly.", esc_max_tokens=160)


def _majority_by_similarity(outs):
    """Cluster short answers by content-word overlap; return (best, cluster_size)."""
    outs = [o for o in outs if o.strip()]
    if not outs:
        return "Unable to determine.", 0
    best_i, best_size = 0, 1
    for i, a in enumerate(outs):
        size = sum(1 for b in outs if similarity(a, b) >= 0.45)
        if size > best_size:
            best_i, best_size = i, size
    return outs[best_i], best_size


# --------------------------------------------------------------------------
# math
# --------------------------------------------------------------------------
MATH_SYS = ("Convert the word problem into a short Python program that prints "
            "only the final numeric answer. Output only a python code block.")
MATH_FEWSHOT_U = ("A shop has 100 apples and sells 20% of them. How many are left?")
MATH_FEWSHOT_A = "```python\ntotal = 100\nsold = total * 20 / 100\nprint(total - sold)\n```"


def _math_via_code(prompt: str, temp: float):
    out = CODER.chat(
        [{"role": "system", "content": MATH_SYS},
         {"role": "user", "content": MATH_FEWSHOT_U},
         {"role": "assistant", "content": MATH_FEWSHOT_A},
         {"role": "user", "content": prompt}],
        max_tokens=220, temperature=temp)
    code = extract_code(out)
    if not code:
        return None
    ok, stdout, _ = run_python(code)
    if not ok:
        return None
    return extract_last_number(stdout)


def math(prompt: str, mode: str) -> Result:
    temps = [0.2, 0.7, 1.0] if mode != "panic" else [0.2]
    vals = []
    for t in temps[:2]:
        v = _math_via_code(prompt, t)
        if v is not None:
            vals.append(v)
        if len(vals) == 2 and abs(vals[0] - vals[1]) < 1e-9:
            return Result(_math_answer(vals[0]), 0.92,
                          esc_suffix=" Give only the final number.", esc_max_tokens=160)
    if mode == "panic" and vals:
        return Result(_math_answer(vals[0]), 0.55,
                      esc_suffix=" Give only the final number.", esc_max_tokens=160)
    # tie-break: third code sample + a direct general-model answer
    v3 = _math_via_code(prompt, temps[-1]) if mode != "panic" else None
    if v3 is not None:
        vals.append(v3)
    direct = GENERAL.chat(
        [{"role": "system", "content": "Solve step by step, then give the final number."},
         {"role": "user", "content": prompt}],
        max_tokens=280, thinking=(mode == "full"))
    dv = extract_last_number(direct)
    if dv is not None:
        vals.append(dv)
    if not vals:
        return Result(direct.strip() or "Unable to determine.", 0.25,
                      esc_suffix=" Give only the final number.", esc_max_tokens=160)
    top, cnt = Counter(vals).most_common(1)[0]
    conf = 0.8 if cnt >= 3 else (0.6 if cnt == 2 else 0.3)
    return Result(_math_answer(top), conf,
                  esc_suffix=" Give only the final number.", esc_max_tokens=160)


def _math_answer(v) -> str:
    return f"The answer is {fmt_number(v)}."


# --------------------------------------------------------------------------
# sentiment
# --------------------------------------------------------------------------
SENTIMENT_SYS = ("Classify the sentiment of the given text as Positive, Negative, "
                 "Neutral, or Mixed, and briefly justify.")
SENTIMENT_GRAMMAR = r'''root ::= label " - " [^\n]{12,150}
label ::= "Positive" | "Negative" | "Neutral" | "Mixed"'''
_LABEL_RE = re.compile(r"^(Positive|Negative|Neutral|Mixed)\b", re.IGNORECASE)


def sentiment(prompt: str, mode: str) -> Result:
    n = _samples_for(mode, 2, 2)
    outs, labels = [], []
    for i in range(n):
        o = GENERAL.chat(
            [{"role": "system", "content": SENTIMENT_SYS},
             {"role": "user", "content": prompt}],
            max_tokens=70, temperature=0.2 if i == 0 else config.GEN_TEMP,
            grammar=SENTIMENT_GRAMMAR)
        outs.append(o)
        m = _LABEL_RE.match(o.strip())
        labels.append(m.group(1).title() if m else "?")
    if len(set(labels)) == 1 and labels[0] != "?":
        return Result(outs[0], 0.85, esc_max_tokens=160)
    if n > 1:  # tie-break vote
        o3 = GENERAL.chat(
            [{"role": "system", "content": SENTIMENT_SYS},
             {"role": "user", "content": prompt}],
            max_tokens=70, temperature=0.9, grammar=SENTIMENT_GRAMMAR)
        outs.append(o3)
        m = _LABEL_RE.match(o3.strip())
        labels.append(m.group(1).title() if m else "?")
        top, cnt = Counter(l for l in labels if l != "?").most_common(1)[0]
        best = next(o for o, l in zip(outs, labels) if l == top)
        return Result(best, 0.7 if cnt >= 2 else 0.45, esc_max_tokens=160)
    return Result(outs[0], 0.5, esc_max_tokens=160)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
SUMMARY_SYS = ("Summarize the given passage. Obey the stated length/format "
               "constraint exactly. Output only the summary, nothing else.")


def _summary_constraint(prompt: str):
    p = prompt.lower()
    m = re.search(r"exactly (one|two|three|four|five|\d+) sentences?", p)
    if m:
        w = m.group(1)
        n = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}.get(w) or int(w)
        return ("sentences_exact", n)
    m = re.search(r"in (one|a single|two|three|\d+) sentences?", p)
    if m:
        w = m.group(1)
        n = {"one": 1, "a single": 1, "two": 2, "three": 3}.get(w) or int(w)
        return ("sentences_exact", n)
    m = re.search(r"(?:at most|under|no more than|maximum of|within) (\d+) words", p)
    if m:
        return ("words_max", int(m.group(1)))
    return (None, None)


def summary(prompt: str, mode: str) -> Result:
    kind, n = _summary_constraint(prompt)
    out = GENERAL.chat(
        [{"role": "system", "content": SUMMARY_SYS},
         {"role": "user", "content": prompt}],
        max_tokens=220, temperature=0.5)
    for attempt in range(config.MAX_LOCAL_RETRIES):
        ok, detail = _summary_ok(out, kind, n)
        if ok:
            return Result(out.strip(), 0.85 if attempt == 0 else 0.75,
                          esc_max_tokens=220)
        out = GENERAL.chat(
            [{"role": "system", "content": SUMMARY_SYS},
             {"role": "user", "content": prompt},
             {"role": "assistant", "content": out},
             {"role": "user", "content":
              f"Your summary violates the constraint ({detail}). "
              f"Rewrite it — shorter and complete — so it satisfies the "
              f"constraint exactly. Output only the corrected summary."}],
            max_tokens=220, temperature=0.4)
    # programmatic last resort
    if kind == "sentences_exact" and n:
        sents = split_sentences(out)
        out = " ".join(sents[:n]) if len(sents) >= n else out
    elif kind == "words_max" and n:
        out = " ".join(out.split()[:n]).rstrip(",;") + "."
    return Result(out.strip(), 0.55, esc_max_tokens=220)


def _summary_ok(out: str, kind, n):
    if not out.strip():
        return False, "empty"
    if not re.search(r'[.!?]["\')\]]*\s*$', out.strip()):
        return False, "it is cut off mid-sentence"
    if kind == "sentences_exact":
        k = len(split_sentences(out))
        return k == n, f"has {k} sentences, needs exactly {n}"
    if kind == "words_max":
        k = len(out.split())
        return k <= n, f"has {k} words, max {n}"
    return True, ""


# --------------------------------------------------------------------------
# NER
# --------------------------------------------------------------------------
NER_SYS = ('Extract all named entities from the given text. Output one entity '
           'per line, formatted exactly as "Entity | Type". Allowed types: '
           'Person, Organization, Location, Date, Event, Product. '
           'If there are none, output exactly: None')
_NER_LINE = re.compile(r"^(.{1,60}?)\s*\|\s*(Person|Organization|Location|Date|Event|Product)\s*$",
                       re.IGNORECASE | re.MULTILINE)


def ner(prompt: str, mode: str) -> Result:
    n = _samples_for(mode, 2, 2)
    parsed = []
    for i in range(n):
        o = GENERAL.chat(
            [{"role": "system", "content": NER_SYS},
             {"role": "user", "content": prompt}],
            max_tokens=110, temperature=0.15 if i == 0 else 0.7)
        pairs = [(m.group(1).strip(), m.group(2).title())
                 for m in _NER_LINE.finditer(o)]
        parsed.append(pairs)
    if not any(parsed):
        return Result("None", 0.4, esc_max_tokens=200)
    base = parsed[0] or parsed[-1]
    if len(parsed) > 1 and parsed[0] and parsed[1]:
        k0 = {e.lower() for e, _ in parsed[0]}
        k1 = {e.lower() for e, _ in parsed[1]}
        # keep entities either agreed on, or literally present in the task text
        base = [(e, t) for e, t in parsed[0]
                if e.lower() in k1 or e in prompt]
        extras = [(e, t) for e, t in parsed[1]
                  if e.lower() not in k0 and e in prompt]
        base += extras
        conf = 0.85 if k0 == k1 else 0.7
    else:
        base = [(e, t) for e, t in base if e in prompt or len(parsed) == 1]
        conf = 0.55
    if not base:
        return Result("None", 0.4, esc_max_tokens=200)
    seen, lines = set(), []
    for e, t in base:
        if e.lower() not in seen:
            seen.add(e.lower())
            lines.append(f"{e} | {t}")
    return Result("\n".join(lines), conf, esc_max_tokens=200)


# --------------------------------------------------------------------------
# code generation
# --------------------------------------------------------------------------
CODEGEN_SYS = ("Write clean, correct Python. Output only a single python code "
               "block containing the complete function. No explanations.")
TESTGEN_SYS = ("Write exactly 3 Python assert statements that test a function "
               "against the given specification. Cover normal and edge cases. "
               "Output only a python code block containing the asserts, "
               "no function definition.")


def _gen_tests(spec: str, signature: str, temp: float = 0.3) -> str:
    out = CODER.chat(
        [{"role": "system", "content": TESTGEN_SYS},
         {"role": "user", "content":
          f"Specification: {spec}\nFunction signature: {signature}"}],
        max_tokens=160, temperature=temp)
    code = extract_code(out)
    return "\n".join(ln for ln in code.splitlines()
                     if ln.strip().startswith("assert"))


def _signature_of(code: str) -> str:
    m = re.search(r"^def .+?:", code, re.MULTILINE)
    return m.group(0) if m else "unknown"


def code_gen(prompt: str, mode: str) -> Result:
    out = CODER.chat(
        [{"role": "system", "content": CODEGEN_SYS},
         {"role": "user", "content": prompt}],
        max_tokens=380, temperature=config.CODE_TEMP)
    code = extract_code(out)
    if not code or "def " not in code:
        out = CODER.chat(
            [{"role": "system", "content": CODEGEN_SYS},
             {"role": "user", "content": prompt + "\nOutput only the code block."}],
            max_tokens=380, temperature=0.6)
        code = extract_code(out)
    if not code:
        return Result(out.strip() or "Unable to produce code.", 0.2, esc_max_tokens=400)
    return _verify_and_repair(prompt, code, mode, esc_max=400)


def _verify_and_repair(spec: str, code: str, mode: str, esc_max: int,
                       preamble: str = "") -> Result:
    tests = _gen_tests(spec, _signature_of(code))
    if not tests:
        return Result(_code_answer(preamble, code), 0.5, esc_max_tokens=esc_max)
    ok, err = run_with_tests(code, tests)
    attempts = 0
    while not ok and attempts < config.MAX_LOCAL_RETRIES and mode != "panic":
        attempts += 1
        fix = CODER.chat(
            [{"role": "system", "content": CODEGEN_SYS},
             {"role": "user", "content":
              f"Task: {spec}\n\nThis implementation fails its tests.\n"
              f"```python\n{code}\n```\nFailure:\n{err}\n\nTests:\n{tests}\n"
              f"Output the corrected complete function only."}],
            max_tokens=380, temperature=0.4)
        new_code = extract_code(fix)
        if new_code and "def " in new_code:
            code = new_code
        ok, err = run_with_tests(code, tests)
    if not ok and mode != "panic":
        # maybe the tests are wrong: regenerate once and re-check original code
        tests2 = _gen_tests(spec, _signature_of(code), temp=0.8)
        if tests2 and tests2 != tests:
            ok2, _ = run_with_tests(code, tests2)
            if ok2:
                return Result(_code_answer(preamble, code), 0.65, esc_max_tokens=esc_max)
    conf = 0.9 if ok else 0.35
    return Result(_code_answer(preamble, code), conf, esc_max_tokens=esc_max)


def _code_answer(preamble: str, code: str) -> str:
    block = f"```python\n{code}\n```"
    return f"{preamble}\n\n{block}".strip() if preamble else block


# --------------------------------------------------------------------------
# code debugging
# --------------------------------------------------------------------------
DEBUG_BUG_SYS = ("State in ONE short sentence what the bug is. Start your "
                 "answer with 'Bug:'. Do not write any code.")
DEBUG_FIX_SYS = ("You are given code with a bug and its intended behavior. "
                 "Output only the corrected complete code in a python code "
                 "block. No explanations.")


def code_debug(prompt: str, mode: str) -> Result:
    # small models are unreliable at "sentence then code" combos — split calls
    bug = CODER.chat(
        [{"role": "system", "content": DEBUG_BUG_SYS},
         {"role": "user", "content": prompt}],
        max_tokens=60, temperature=0.2)
    bug_line = bug.strip().splitlines()[0] if bug.strip() else ""
    if bug_line and not bug_line.startswith("Bug:"):
        bug_line = "Bug: " + bug_line
    code = ""
    for temp in (config.CODE_TEMP, 0.6):
        out = CODER.chat(
            [{"role": "system", "content": DEBUG_FIX_SYS},
             {"role": "user", "content": prompt}],
            max_tokens=340, temperature=temp)
        code = extract_code(out)
        if code and "def " in code:
            break
    if not code or "def " not in code:
        return Result((bug_line + "\n" + out).strip() or "Unable to determine.",
                      0.3, esc_max_tokens=400)
    return _verify_and_repair(prompt, code, mode, esc_max=400,
                              preamble=bug_line or "Bug: see corrected code below.")


# --------------------------------------------------------------------------
# logic
# --------------------------------------------------------------------------
LOGIC_EXTRACT_SYS = (
    "Convert the puzzle into JSON with keys: entities (list of names), "
    "attributes (list of things assigned to entities), constraints (list of "
    "Python boolean expressions over dict A mapping entity to attribute), "
    "question_attribute (the attribute the question asks about, or null). "
    "Output only the JSON.")
LOGIC_FEWSHOT_U = ("Two kids, Ann and Max, each like a different fruit: apple "
                   "or pear. Ann does not like the pear. Who likes the pear?")
LOGIC_FEWSHOT_A = ('{"entities": ["Ann", "Max"], "attributes": ["apple", "pear"], '
                   '"constraints": ["A[\'Ann\'] != \'pear\'"], '
                   '"question_attribute": "pear"}')
LOGIC_DIRECT_SYS = ("Solve the puzzle. State the answer in one short sentence, "
                    "then give one brief supporting reason.")


def logic(prompt: str, mode: str) -> Result:
    if mode != "panic":
        solved = _logic_solver(prompt)
        if solved:
            return solved
    outs = []
    n = _samples_for(mode, 2, 1)
    for i in range(n):
        outs.append(GENERAL.chat(
            [{"role": "system", "content": LOGIC_DIRECT_SYS},
             {"role": "user", "content": prompt}],
            max_tokens=340 if mode == "full" else 160,
            thinking=(mode == "full"), temperature=0.6))
    best, agree = _majority_by_similarity(outs)
    conf = 0.7 if (n > 1 and agree >= 2) else (0.45 if best else 0.2)
    return Result(best, conf, esc_suffix=" State the answer in one sentence.",
                  esc_max_tokens=200)


def _logic_solver(prompt: str):
    try:
        raw = GENERAL.chat(
            [{"role": "system", "content": LOGIC_EXTRACT_SYS},
             {"role": "user", "content": LOGIC_FEWSHOT_U},
             {"role": "assistant", "content": LOGIC_FEWSHOT_A},
             {"role": "user", "content": prompt}],
            max_tokens=240, temperature=0.2)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        spec = json.loads(raw)
        entities = [str(e) for e in spec.get("entities", [])]
        attrs = [str(a) for a in spec.get("attributes", [])]
        cons = [str(c) for c in spec.get("constraints", [])]
        q_attr = spec.get("question_attribute")
        if not entities or not attrs or len(entities) != len(attrs) or len(entities) > 6:
            return None
        sols = []
        for perm in itertools.permutations(attrs):
            A = dict(zip(entities, perm))
            try:
                if all(eval(c, {"__builtins__": {}}, {"A": A}) for c in cons):
                    sols.append(A)
            except Exception:  # noqa: BLE001 — malformed constraint kills solver path
                return None
        if len(sols) != 1:
            return None
        A = sols[0]
        if q_attr and q_attr in attrs:
            who = next((e for e, a in A.items() if a == q_attr), None)
            if who:
                verb = _logic_verb(prompt)
                reason = "; ".join(f"{e} has the {a}" for e, a in A.items())
                return Result(f"{who} {verb} the {q_attr}. ({reason}.)", 0.92,
                              esc_suffix=" State the answer in one sentence.",
                              esc_max_tokens=200)
        assign = ", ".join(f"{e}: {a}" for e, a in A.items())
        return Result(f"The unique solution is {assign}.", 0.85,
                      esc_suffix=" State the answer in one sentence.",
                      esc_max_tokens=200)
    except Exception as e:  # noqa: BLE001
        log(f"logic solver path failed: {e}")
        return None


def _logic_verb(prompt: str) -> str:
    m = re.search(r"who (\w+)", prompt.lower())
    return m.group(1) if m else "has"


HANDLERS = {
    "factual": factual,
    "math": math,
    "sentiment": sentiment,
    "summary": summary,
    "ner": ner,
    "code_gen": code_gen,
    "code_debug": code_debug,
    "logic": logic,
}


def run_task(category: str, prompt: str, mode: str) -> Result:
    handler = HANDLERS.get(category, factual)
    try:
        return handler(prompt, mode)
    except Exception as e:  # noqa: BLE001 — a task must never take down the run
        log(f"pipeline[{category}] crashed: {e}; falling back to direct answer")
        try:
            out = GENERAL.chat(
                [{"role": "system", "content": FACTUAL_SYS},
                 {"role": "user", "content": prompt}],
                max_tokens=150, temperature=0.5)
            return Result(out or "Unable to determine.", 0.3)
        except Exception as e2:  # noqa: BLE001
            log(f"fallback also failed: {e2}")
            return Result("Unable to determine.", 0.0)
