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


# Static escalation format hint for factual — shared with main.early_escalate,
# which fires factual batches before any pipeline has produced a Result.
FACTUAL_ESC = (" Lead with the direct answer covering every "
               "part asked; one short sentence per part; "
               "no hedging.")


def factual(prompt: str, mode: str) -> Result:
    n = _samples_for(mode, 3, 2)
    outs = []
    for i in range(n):
        outs.append(GENERAL.chat(
            [{"role": "system", "content": FACTUAL_SYS},
             {"role": "user", "content": prompt}],
            max_tokens=90,
            temperature=0.3 if i == 0 else config.SC_TEMP,
            min_p=None if i == 0 else config.SC_MIN_P))
    best, agree = _majority_by_similarity(outs)
    # sample agreement cannot verify facts — a small model is confidently
    # wrong too often. Cap below the escalation threshold: factual always
    # escalates when budget allows (cheapest category to fix remotely),
    # and the local majority answer stands as the fallback.
    conf = {3: 0.50, 2: 0.45, 1: 0.30}.get(agree, 0.40)
    return Result(best, conf, esc_suffix=FACTUAL_ESC, esc_max_tokens=160)


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
MATH_SYS = ("Convert the word problem into a short Python program. Never write "
            "digit separators in numbers (write 2400, not 2,400). If the "
            "problem asks for multiple values, print each one on its own line "
            "as 'label: value' using unit words (cups:, cost:, total:) — never "
            "item names copied from examples. If it asks for one value, print just that "
            "number. Output only a python code block.")
MATH_FEWSHOT_U = ("A depot starts with 1,500 boxes. It ships 24% of its stock, "
                  "then receives 300 boxes, then ships 200 more. How many remain?")
MATH_FEWSHOT_A = ("```python\nstock = 1500\nstock -= stock * 24 / 100\n"
                  "stock += 300\nstock -= 200\nprint(stock)\n```")
MATH_FEWSHOT2_U = ("A recipe uses 2/3 cup of flour for 8 buns. How much flour is "
                   "needed for 20 buns, and what does that flour cost at $1.80 "
                   "per cup?")
MATH_FEWSHOT2_A = ("```python\ncups = (2 / 3) * 20 / 8\ncost = cups * 1.80\n"
                   "print(f\"cups: {cups}\")\nprint(f\"cost: ${cost:.2f}\")\n```")

_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _math_via_code(prompt: str, temp: float, min_p=None):
    """Returns (values_tuple, stdout) — a problem may ask for several values
    (e.g. an amount AND its cost); the judge requires all of them."""
    clean = _THOUSANDS_RE.sub("", prompt)  # '2,400' would be a Python tuple
    out = CODER.chat(
        [{"role": "system", "content": MATH_SYS},
         {"role": "user", "content": MATH_FEWSHOT_U},
         {"role": "assistant", "content": MATH_FEWSHOT_A},
         {"role": "user", "content": MATH_FEWSHOT2_U},
         {"role": "assistant", "content": MATH_FEWSHOT2_A},
         {"role": "user", "content": clean}],
        max_tokens=240, temperature=temp, min_p=min_p)
    code = extract_code(out)
    if not code:
        return None
    ok, stdout, _ = run_python(code)
    if not ok or not stdout.strip():
        return None
    from .util import extract_last_number as _last
    nums = tuple(round(float(v), 6) for v in
                 (_last(line) for line in stdout.strip().splitlines())
                 if v is not None)[:6]
    if not nums:
        return None
    return nums, stdout.strip()


_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


_MULTI_ASK_RE = re.compile(r"\?.*\?|,\s*(?:and\s+)?(?:what|how much|how many)\b",
                           re.IGNORECASE | re.DOTALL)


def _math_direct_value(prompt: str, mode: str):
    """Independent answer from the OTHER model — heterogeneous check."""
    direct = GENERAL.chat(
        [{"role": "system", "content":
          "Solve step by step, then give the final answer(s). "
          "Put your final answer within \\boxed{}."},
         {"role": "user", "content": prompt}],
        max_tokens=560 if mode == "full" else 280, thinking=(mode == "full"))
    boxed = _BOXED_RE.search(direct)
    dv = extract_last_number(boxed.group(1)) if boxed else None
    if dv is None:
        dv = extract_last_number(direct)
    return dv, direct


def math(prompt: str, mode: str) -> Result:
    """The coder model miscodes some problem shapes REPRODUCIBLY, so two
    coder samples agreeing proves nothing (measured: both said 832.32 on a
    task whose answer is 1672). Confidence now requires the coder's executed
    program and the general model's independent answer to agree."""
    v1 = _math_via_code(prompt, 0.2)
    if mode == "panic":
        if v1:
            return Result(_math_answer(v1), 0.55,
                          esc_suffix=_MATH_ESC, esc_max_tokens=200)
        return Result("Unable to determine.", 0.2,
                      esc_suffix=_MATH_ESC, esc_max_tokens=200)

    # completeness: a two-part question must yield two printed values
    if v1 and len(v1[0]) < 2 and _MULTI_ASK_RE.search(prompt):
        retry = _math_via_code(
            prompt + "\nPrint EVERY value the question asks for, one per line.",
            0.4)
        if retry and len(retry[0]) > len(v1[0]):
            v1 = retry

    dv, direct = _math_direct_value(prompt, mode)

    if v1 and dv is not None and any(abs(x - dv) < 0.01 for x in v1[0]):
        # heterogeneous agreement: executed program ∧ independent reasoning
        return Result(_math_answer(v1), 0.92,
                      esc_suffix=_MATH_ESC, esc_max_tokens=200)
    if v1 and dv is None:
        v2 = _math_via_code(prompt, config.SC_TEMP, min_p=config.SC_MIN_P)
        if v2 and v2[0] == v1[0]:
            return Result(_math_answer(v1), 0.60,  # same-model only: capped
                          esc_suffix=_MATH_ESC, esc_max_tokens=200)
    # disagreement or missing halves — never trust, always escalate
    fallback = (_math_answer(v1) if v1 else
                (f"The answer is {fmt_number(dv)}." if dv is not None
                 else direct.strip() or "Unable to determine."))
    return Result(fallback, 0.35, esc_suffix=_MATH_ESC, esc_max_tokens=200)


_MATH_ESC = (" Give the final answer(s) with brief working; if multiple values "
             "are asked for, state each one clearly.")


def _math_answer(v) -> str:
    values, stdout = v
    if len(values) == 1:
        return f"The answer is {fmt_number(values[0])}."
    # multi-value problems: the program's labeled output IS the answer
    return stdout


# --------------------------------------------------------------------------
# sentiment
# --------------------------------------------------------------------------
SENTIMENT_SYS = ("Classify the sentiment of the given text as Positive, Negative, "
                 "Neutral, or Mixed, then give a one-sentence reason. If the text "
                 "contains BOTH good and bad aspects, never label it Negative — "
                 "use Mixed (or Positive if the outcome is good) and the reason "
                 "must explicitly mention both the negative and positive aspects.")
SENTIMENT_GRAMMAR = r'''root ::= label " - " [^\n]{12,220}
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
            max_tokens=70, temperature=config.SC_TEMP,
            min_p=config.SC_MIN_P, grammar=SENTIMENT_GRAMMAR)
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
               "constraint exactly. If bullet points are requested, output "
               "each as a line starting with '- ', packing in every distinct "
               "item the passage lists for that point within any word cap. "
               "Output only the summary, nothing else.")


_WORDNUM = {"one": 1, "a single": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _summary_constraint(prompt: str):
    p = prompt.lower()
    # bullet formats: "exactly three bullet points, each no longer than 15 words"
    m = re.search(r"(?:exactly |in )?(one|two|three|four|five|\d+) bullet"
                  r"(?: point)?s?", p)
    if m:
        n = _WORDNUM.get(m.group(1)) or int(m.group(1))
        mw = re.search(r"each (?:no longer than|no more than|under|at most|"
                       r"with(?:in)? (?:a maximum of )?)\s*(\d+) words", p)
        return ("bullets", (n, int(mw.group(1)) if mw else None))
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
    if kind == "bullets":
        want, word_cap = n
        lines = _bullet_lines(out) or split_sentences(out)
        lines = (lines + [""] * want)[:want]
        cap = word_cap or 15
        out = "\n".join("- " + " ".join(ln.split()[:cap]).rstrip(".,;")
                         for ln in lines if ln)
    elif kind == "sentences_exact" and n:
        sents = split_sentences(out)
        out = " ".join(sents[:n]) if len(sents) >= n else out
    elif kind == "words_max" and n:
        out = " ".join(out.split()[:n]).rstrip(",;") + "."
    return Result(out.strip(), 0.55, esc_max_tokens=220)


def _bullet_lines(out: str):
    return [ln.strip().lstrip("-*•").strip() for ln in out.splitlines()
            if ln.strip().lstrip("-*•").strip()]


def _summary_ok(out: str, kind, n):
    if not out.strip():
        return False, "empty"
    if kind == "bullets":
        want, word_cap = n
        lines = _bullet_lines(out)
        if len(lines) != want:
            return False, f"has {len(lines)} bullet points, needs exactly {want}"
        if word_cap:
            for ln in lines:
                if len(ln.split()) > word_cap:
                    return False, (f"a bullet has {len(ln.split())} words, "
                                   f"max {word_cap} per bullet")
        return True, ""
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
           'per line, formatted exactly as "Entity | TYPE". Allowed types: '
           'PERSON, ORGANIZATION, LOCATION, DATE, EVENT, PRODUCT. '
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
            max_tokens=110,
            temperature=0.15 if i == 0 else config.SC_TEMP,
            min_p=None if i == 0 else config.SC_MIN_P)
        pairs = [(m.group(1).strip(), m.group(2).upper())
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
        e = _expand_span(e, prompt)
        if not _proper_noun_like(e):
            continue  # generic noun phrases are judge-penalized extras
        if e.lower() not in seen:
            seen.add(e.lower())
            lines.append(f"{e} | {t}")
    return Result("\n".join(lines), conf, esc_max_tokens=200)


_QUALIFIERS = r"(?:last|this|next|early|late|mid|summer|winter|spring|fall|autumn|dr\.?|mr\.?|mrs\.?|ms\.?|president|professor|prof\.?)"


def _expand_span(entity: str, text: str) -> str:
    """Small models clip entity spans ('2024' for 'summer 2024', 'March' for
    'March 15 2023'). Re-attach preceding qualifiers and trailing day/year."""
    for _ in range(2):
        m = re.search(r"\b(" + _QUALIFIERS + r")\s+" + re.escape(entity),
                      text, re.IGNORECASE)
        if not m:
            break
        entity = text[m.start(1):m.start(1) + len(m.group(1))] + " " + entity
    m = re.search(re.escape(entity) + r"((?:,?\s+\d{1,2})?(?:,?\s+\d{4})?)", text)
    if m and m.group(1):
        entity = entity + m.group(1)
    return entity


_ALLOWED_LOWER = {"of", "the", "de", "for", "and", "&"}


def _proper_noun_like(entity: str) -> bool:
    """Generic noun phrases ('large language model') are judge-penalized
    extras — require every content word capitalized or numeric."""
    words = entity.replace(",", " ").split()
    return all(w[0].isupper() or w[0].isdigit() or w.lower() in _ALLOWED_LOWER
               for w in words if w)


# --------------------------------------------------------------------------
# code generation
# --------------------------------------------------------------------------
CODEGEN_SYS = ("Write clean, correct Python. Output only a single python code "
               "block containing the complete function. No explanations. "
               "If the task says to handle duplicates, operate on distinct "
               "values unless it states otherwise.")
TESTGEN_SYS = ("Write exactly 3 Python assert statements that test a function "
               "against the given specification. Cover normal and edge cases. "
               "Output only a python code block containing the asserts, "
               "no function definition.")
INPUTGEN_SYS = ("Given a function specification and signature, write a Python "
                "list named INPUTS of 4 argument tuples to test it with, "
                "covering normal and edge cases, e.g. "
                "INPUTS = [([3, 1, 2],), ([],), ([5, 5],), ([1],)]. "
                "Only the argument values — no expected results, no function "
                "code. Output only the INPUTS assignment.")


def _func_name(code: str):
    m = re.search(r"^def\s+(\w+)", code, re.MULTILINE)
    return m.group(1) if m else None


def _gen_inputs(spec: str, signature: str) -> str:
    out = CODER.chat(
        [{"role": "system", "content": INPUTGEN_SYS},
         {"role": "user", "content":
          f"Specification: {spec}\nFunction signature: {signature}"}],
        max_tokens=140, temperature=0.3)
    code = extract_code(out)
    m = re.search(r"INPUTS\s*=\s*\[.*", code, re.DOTALL)
    return m.group(0) if m else ""


def _fingerprint(code: str, inputs_code: str):
    """Run code's first function over INPUTS; return list of repr(result)."""
    name = _func_name(code)
    if not name or not inputs_code:
        return None
    prog = (
        f"{code}\n\n{inputs_code}\n"
        "_out = []\n"
        "for _args in INPUTS:\n"
        "    if not isinstance(_args, tuple):\n"
        "        _args = (_args,)\n"
        "    try:\n"
        f"        _r = {name}(*_args)\n"
        "        _out.append(repr(sorted(_r.items())) if isinstance(_r, dict) else repr(_r))\n"
        "    except Exception as _e:\n"
        "        _out.append('ERR:' + type(_e).__name__)\n"
        "print('\\n'.join(_out))\n")
    ok, stdout, _err = run_python(prog)
    if not ok:
        return None
    fp = stdout.splitlines()
    if not fp or all(x.startswith("ERR:") for x in fp):
        return None
    return fp


def _behavior_verify(spec: str, impls: list, inputs_code: str):
    """Cross-check independent implementations by observed behavior.
    Returns (best_code, confidence) or None if machinery unavailable."""
    fps = [(c, _fingerprint(c, inputs_code)) for c in impls if c]
    fps = [(c, fp) for c, fp in fps if fp]
    if len(fps) < 2:
        return None
    counts = Counter(tuple(fp) for _c, fp in fps)
    top_fp, top_n = counts.most_common(1)[0]
    if top_n >= 2:
        best = next(c for c, fp in fps if tuple(fp) == top_fp)
        return best, (0.9 if top_n >= 2 and len(fps) == 2 else
                      0.9 if top_n >= 3 else 0.72)
    return None


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


def _gen_impl(prompt: str, temp: float, min_p=None) -> str:
    out = CODER.chat(
        [{"role": "system", "content": CODEGEN_SYS},
         {"role": "user", "content": prompt}],
        max_tokens=380, temperature=temp, min_p=min_p)
    code = extract_code(out)
    return code if code and "def " in code else ""


def code_gen(prompt: str, mode: str) -> Result:
    impl_a = _gen_impl(prompt, config.CODE_TEMP)
    if not impl_a:
        impl_a = _gen_impl(prompt, 0.6)
    if not impl_a:
        return Result("Unable to produce code.", 0.2, esc_max_tokens=400)
    if mode == "panic":
        return Result(_code_answer("", impl_a), 0.5, esc_max_tokens=400)

    # behavioral cross-verification: two independent impls must agree on
    # observed outputs (asserts with model-guessed expected values are the
    # thing that fails — inputs alone are easy to generate correctly)
    impl_b = _gen_impl(prompt, config.SC_TEMP, min_p=config.SC_MIN_P)
    inputs_code = _gen_inputs(prompt, _signature_of(impl_a))
    verified = _behavior_verify(prompt, [impl_a, impl_b], inputs_code)
    if verified:
        return Result(_code_answer("", verified[0]), verified[1], esc_max_tokens=400)
    if impl_b and inputs_code:  # disagreement — majority vote with a third impl
        impl_c = _gen_impl(prompt, 0.45)
        verified = _behavior_verify(prompt, [impl_a, impl_b, impl_c], inputs_code)
        if verified:
            return Result(_code_answer("", verified[0]), 0.72, esc_max_tokens=400)
    # machinery unavailable — fall back to assert-based verify/repair
    return _verify_and_repair(prompt, impl_a, mode, esc_max=400)


_ASSERT_EQ_RE = re.compile(r"^assert\s+(.+?)\s*==\s*(.+)$")


def _failure_detail(code: str, tests: str) -> str:
    """Find the first failing assert, then run the candidate on that call to
    capture the actual traceback/observed output. Returns an observed-vs-
    expected line for the repair prompt, or '' if nothing parseable failed."""
    try:
        for line in tests.splitlines():
            line = line.strip()
            if not line.startswith("assert"):
                continue
            ok, _out, _err = run_python(f"{code}\n{line}\n")
            if ok:
                continue
            m = _ASSERT_EQ_RE.match(line)
            if not m:
                return ""
            call_expr, expected = m.group(1).strip(), m.group(2).strip()
            ok2, out, err = run_python(f"{code}\nprint(repr({call_expr}))\n")
            if ok2:
                observed = out.strip() or "no output"
            else:
                tb = err.strip().splitlines()
                observed = tb[-1] if tb else "crashed with no traceback"
            im = re.match(r"\w+\((.*)\)$", call_expr, re.DOTALL)
            failing_input = im.group(1) if im else call_expr
            return (f"Failing input: {failing_input}; "
                    f"expected (from tests): {expected}; observed: {observed}")
    except Exception as e:  # noqa: BLE001 — diagnostics must never break repair
        log(f"failure-detail probe failed: {e}")
    return ""


def _verify_and_repair(spec: str, code: str, mode: str, esc_max: int,
                       preamble: str = "") -> Result:
    tests = _gen_tests(spec, _signature_of(code))
    if not tests:
        return Result(_code_answer(preamble, code), 0.5, esc_max_tokens=esc_max)
    ok, err = run_with_tests(code, tests)
    attempts = 0
    while not ok and attempts < config.MAX_LOCAL_RETRIES and mode != "panic":
        attempts += 1
        detail = _failure_detail(code, tests)
        if detail:
            err = f"{err}\n{detail}" if err else detail
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


DEBUG_DIFF_SYS = ("You are shown a task with buggy code, and the corrected "
                  "code. State in ONE short sentence the functional defect "
                  "that was fixed. Start with 'Bug:'. No code, no speculation "
                  "beyond the visible difference.")


def _bug_line_from_diff(prompt: str, fixed_code: str) -> str:
    """Describe the bug AFTER fixing — grounded by the actual before/after
    diff. Uses the general model: the coder model hallucinates descriptions."""
    try:
        bug = GENERAL.chat(
            [{"role": "system", "content": DEBUG_DIFF_SYS},
             {"role": "user", "content":
              f"{prompt}\n\nCorrected code:\n```python\n{fixed_code}\n```"}],
            max_tokens=70, temperature=0.2)
        line = bug.strip().splitlines()[0] if bug.strip() else ""
        if line and not line.startswith("Bug:"):
            line = "Bug: " + line
        return line
    except Exception:  # noqa: BLE001
        return ""


def _normalize_code(code: str) -> str:
    lines = []
    for ln in code.splitlines():
        ln = re.sub(r"#.*", "", ln).strip()
        if ln:
            lines.append(re.sub(r"\s+", " ", ln))
    return "\n".join(lines)


def _is_echo_of_buggy(prompt: str, fixed_code: str) -> bool:
    """The coder model sometimes returns the buggy code unchanged."""
    m = re.search(r"def .*?(?=\.\s|$)", prompt, re.DOTALL)
    if not m:
        return False
    return _normalize_code(m.group(0)) == _normalize_code(fixed_code)


def code_debug(prompt: str, mode: str) -> Result:
    code = ""
    out = ""
    for temp in (config.CODE_TEMP, 0.6, 0.8):
        out = CODER.chat(
            [{"role": "system", "content": DEBUG_FIX_SYS},
             {"role": "user", "content": prompt if temp != 0.8 else
              prompt + "\nThe corrected code must differ from the original."}],
            max_tokens=340, temperature=temp)
        code = extract_code(out)
        if code and "def " in code and not _is_echo_of_buggy(prompt, code):
            break
        code = code if code and "def " in code else ""
    if not code or "def " not in code:
        return Result(out.strip() or "Unable to determine.", 0.3, esc_max_tokens=400)
    if mode == "panic":
        bug = _bug_line_from_diff(prompt, code) or "Bug: see corrected code below."
        return Result(_code_answer(bug, code), 0.5, esc_max_tokens=400)

    # cross-check the fix against an independent from-scratch implementation
    # of the described intent (same name/signature), by observed behavior
    ref = _gen_impl(
        f"{prompt}\n\nWrite the correct implementation from scratch. "
        f"Keep the same function name and signature as the original.", 0.7)
    inputs_code = _gen_inputs(prompt, _signature_of(code))
    if ref and _func_name(ref) != _func_name(code):
        ref = ""  # different name — behavior comparison would be meaningless
    verified = _behavior_verify(prompt, [code, ref], inputs_code) if ref else None
    if verified:
        best_code, conf = verified
        bug = _bug_line_from_diff(prompt, best_code) or "Bug: see corrected code below."
        return Result(_code_answer(bug, best_code), conf, esc_max_tokens=400)
    # fix and reference disagree behaviorally — let generated asserts arbitrate
    if ref:
        tests = _gen_tests(prompt, _signature_of(code))
        if tests:
            fix_ok, _ = run_with_tests(code, tests)
            ref_ok, _ = run_with_tests(ref, tests)
            # an echoed buggy function can pass weak asserts — never keep the
            # echo when a working from-scratch reference exists
            if ref_ok and (not fix_ok or _is_echo_of_buggy(prompt, code)):
                code = ref
            if fix_ok or ref_ok:
                bug = _bug_line_from_diff(prompt, code) or "Bug: see corrected code below."
                return Result(_code_answer(bug, code), 0.7, esc_max_tokens=400)
    res = _verify_and_repair(prompt, code, mode, esc_max=400)
    final_code = extract_code(res.answer) or code
    if _is_echo_of_buggy(prompt, final_code) and ref:
        final_code = ref  # never ship the original buggy code back
    bug = _bug_line_from_diff(prompt, final_code) or "Bug: see corrected code below."
    return Result(_code_answer(bug, final_code),
                  min(res.confidence, 0.5) if _is_echo_of_buggy(prompt, final_code)
                  else res.confidence, esc_max_tokens=400)


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
    n = _samples_for(mode, 2, 2)
    for i in range(n):
        # thinking eats max_tokens from the inside: 640 leaves room for the
        # answer after the think block (340 starved it to empty content)
        o = GENERAL.chat(
            [{"role": "system", "content": LOGIC_DIRECT_SYS},
             {"role": "user", "content": prompt}],
            max_tokens=640 if mode == "full" else 200,
            thinking=(mode == "full"), temperature=0.6)
        if not o.strip():  # think-block starvation — retry without thinking
            o = GENERAL.chat(
                [{"role": "system", "content": LOGIC_DIRECT_SYS},
                 {"role": "user", "content": prompt}],
                max_tokens=200, thinking=False, temperature=0.6)
        outs.append(o)
    best, agree = _majority_by_similarity(outs)
    conf = 0.7 if (n > 1 and agree >= 2) else (0.45 if best else 0.2)
    return Result(best, conf, esc_suffix=" State the answer in one sentence.",
                  esc_max_tokens=200)


def _logic_solver(prompt: str):
    res, status = _logic_solve_once(prompt, 0.2)
    if res is None and status == "bad_count":
        # 0 or >1 solutions usually means a missed or mangled constraint —
        # one hotter re-extraction often recovers it
        res, _ = _logic_solve_once(prompt, 0.7)
    return res


def _logic_solve_once(prompt: str, temp: float):
    """One extraction + brute-force pass. Returns (Result|None, status);
    status == 'bad_count' means the solver ran but found 0 or >1 solutions
    (retryable extraction failure), '' means anything else."""
    try:
        raw = GENERAL.chat(
            [{"role": "system", "content": LOGIC_EXTRACT_SYS},
             {"role": "user", "content": LOGIC_FEWSHOT_U},
             {"role": "assistant", "content": LOGIC_FEWSHOT_A},
             {"role": "user", "content": prompt}],
            max_tokens=240, temperature=temp)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        # models emit A["Sam"] != "bird" inside JSON strings — heal the quotes.
        # the value's closing quote may butt directly against the constraint
        # string's own closing quote, so `"` must be in the lookahead
        raw = re.sub(r'A\[\s*"([^"\]]+)"\s*\]', r"A['\1']", raw)
        raw = re.sub(r'(==|!=)\s*"([^",\]}]+)"(?=[\s,\]"}])', r"\1 '\2'", raw)
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError:
            log(f"solver raw after heal: {raw[:220]!r}")
            raise
        entities = [str(e) for e in spec.get("entities", [])]
        attrs = [str(a) for a in spec.get("attributes", [])]
        cons = [str(c) for c in spec.get("constraints", [])]
        q_attr = spec.get("question_attribute")
        if not entities or not attrs or len(entities) > 6 or len(attrs) > 6:
            return None, ""
        # partial-assignment puzzles name fewer people than attributes
        # ("three houses, two named residents") — pad with placeholders
        while len(entities) < len(attrs):
            entities.append(f"_unnamed{len(entities)}")
        if len(entities) != len(attrs):
            return None, ""
        sols = []
        for perm in itertools.permutations(attrs):
            A = dict(zip(entities, perm))
            try:
                if all(eval(c, {"__builtins__": {}}, {"A": A}) for c in cons):
                    sols.append(A)
            except Exception:  # noqa: BLE001 — malformed constraint kills solver path
                return None, ""
        if len(sols) != 1:
            return None, "bad_count"
        A = sols[0]
        named = {e: a for e, a in A.items() if not e.startswith("_unnamed")}
        reason = "; ".join(f"{e} has the {a}" for e, a in named.items())
        # "who has X?" questions
        if q_attr and q_attr in attrs:
            who = next((e for e, a in named.items() if a == q_attr), None)
            if who:
                verb = _logic_verb(prompt)
                return Result(f"{who} {verb} the {q_attr}. ({reason}.)", 0.92,
                              esc_suffix=" State the answer in one sentence.",
                              esc_max_tokens=200), ""
        # "which X does E have?" questions — entity named in the question
        qsents = re.findall(r"[^.?!]*\?", prompt)
        qsent = qsents[-1] if qsents else prompt
        ent_in_q = next((e for e in named if re.search(
            rf"\b{re.escape(e)}\b", qsent)), None)
        if ent_in_q:
            return Result(f"{ent_in_q} has the {A[ent_in_q]}. ({reason}.)", 0.9,
                          esc_suffix=" State the answer in one sentence.",
                          esc_max_tokens=200), ""
        assign = ", ".join(f"{e}: {a}" for e, a in named.items())
        return Result(f"The unique solution is {assign}.", 0.85,
                      esc_suffix=" State the answer in one sentence.",
                      esc_max_tokens=200), ""
    except Exception as e:  # noqa: BLE001
        log(f"logic solver path failed: {e}")
        return None, ""


def _logic_verb(prompt: str) -> str:
    m = re.search(r"who (\w+)", prompt.lower())
    return m.group(1) if m else "has"


# cues that a task carries explicit format/structure requirements our
# specialized parsers may not recognize (the hidden set can invent new ones)
_CONSTRAINT_CUES = re.compile(
    r"exactly|at most|no more than|fewer than|no longer than|bullet|numbered|"
    r"\bjson\b|\btable\b|uppercase|lowercase|one paragraph|single word|"
    r"comma[- ]separated|\bin \d+ words\b|maximum of|begin with|end with|"
    r"format(?:ted)? as", re.IGNORECASE)
# formats our sentiment/NER templates would actively violate if requested
_FORMAT_OVERRIDE_CUES = re.compile(
    r"\bjson\b|\btable\b|comma[- ]separated|single word|numbered list|"
    r"uppercase|lowercase", re.IGNORECASE)

COMPLIANCE_SYS = ("You check format compliance only — not factual accuracy. "
                  "Reply exactly 'PASS' if the answer satisfies every explicit "
                  "format, length, and structural requirement stated in the "
                  "task, else 'FAIL: <the requirement violated>'.")


def _compliance_check(prompt: str, answer: str):
    """Zero-token generic verifier for requirement shapes we don't parse."""
    verdict = GENERAL.chat(
        [{"role": "system", "content": COMPLIANCE_SYS},
         {"role": "user", "content": f"Task:\n{prompt}\n\nAnswer:\n{answer}"}],
        max_tokens=48, temperature=0.1)
    v = verdict.strip()
    return v.upper().startswith("PASS"), v


def _enforce_generic_format(category: str, prompt: str, res: Result) -> Result:
    """When a task states requirements our parsers didn't handle, verify
    compliance with the local model; one guided retry, then escalate-worthy."""
    if category not in ("summary", "ner", "sentiment", "factual"):
        return res
    if not _CONSTRAINT_CUES.search(prompt):
        return res
    if category == "summary" and _summary_constraint(prompt)[0] is not None:
        return res  # the specialized parser already enforced this shape
    if category in ("ner", "sentiment") and not _FORMAT_OVERRIDE_CUES.search(prompt):
        return res  # our pinned template is compatible with the request
    try:
        ok, verdict = _compliance_check(prompt, res.answer)
        if ok:
            return res
        log(f"generic-format check failed ({category}): {verdict[:80]}")
        redo = GENERAL.chat(
            [{"role": "system", "content":
              "Answer the task, following its stated format requirements "
              "EXACTLY. Output only the answer."},
             {"role": "user", "content":
              f"{prompt}\n\n(Your previous answer violated: {verdict[:120]})"}],
            max_tokens=max(res.esc_max_tokens, 220), temperature=0.4)
        if redo.strip():
            ok2, _v2 = _compliance_check(prompt, redo)
            conf = res.confidence if ok2 else min(res.confidence, 0.45)
            return Result(redo.strip(), conf, res.esc_suffix, res.esc_max_tokens)
        return Result(res.answer, min(res.confidence, 0.45),
                      res.esc_suffix, res.esc_max_tokens)
    except Exception as e:  # noqa: BLE001
        log(f"generic-format check error: {e}")
        return res


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
        res = handler(prompt, mode)
        if mode != "panic":
            res = _enforce_generic_format(category, prompt, res)
        return res
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
