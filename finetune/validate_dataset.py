#!/usr/bin/env python3
"""Validate + CLEAN the generated SFT set before it burns a GPU hour.

Garbage in = a fine-tune that is confidently wrong in the wrong format, and we
only get one shot at the GPU budget. So this does not *assert* quality, it
*checks* it:

  math        every equation the answer states is re-evaluated with a sandboxed
              ast evaluator; a false equation drops the row. Also: must end in a
              labelled value, must not be a derivation dump.
  code_gen    the fenced Python is compile()d. Placeholder bodies (`...`, TODO,
              NotImplementedError) are rejected — that is exactly the junk a
              code-named model emits.
  code_debug  same, plus: the answer must open with ONE sentence naming the bug,
              and the fixed code is ast-compared against the buggy code lifted
              out of the PROMPT — an echo of the bug is a drop.
  ner         every line must be "Entity | TYPE" with an allowed TYPE, and the
              entity must actually OCCUR in the prompt (hallucination check).
  sentiment   "^(Positive|Negative|Neutral|Mixed) - <reason>"; a Mixed label must
              name both sides (contrast word or two clauses).
  summary     the constraint is parsed out of the PROMPT ("exactly two
              sentences", "three bullet points ... 15 words", "max N words") and
              the answer is COUNTED against it.
  factual     no preamble, no hedging, non-empty.
  logic       one-sentence answer + a parenthesised reason.
  ALL         no markdown headers, no "As an AI", no restatement of the question,
              no chat-template leakage, and a length guard: anything over
              train.py's --maxlen (2048 == GENERAL_CTX) gets dropped there anyway,
              and would not even fit the server's context at inference. Catch it
              here, where it is free, instead of on the GPU clock.

CLEAN, not just reject: near-misses are canonicalised into the target format
(strip <think>, strip prose around a code fence, normalise "Entity|TYPE" spacing,
normalise en-dashes in "Label – reason", normalise bullet markers) so a good
example is not lost to a typo. Cleaning never invents content.

Usage:
    python finetune/validate_dataset.py                     # raw -> sft.jsonl
    python finetune/validate_dataset.py --show-drops 20     # eyeball the losses
    python finetune/validate_dataset.py --allow-code-prose  # loosen if yield dies

Exit status is nonzero if a category ends up empty or under --min-per-cat, so a
driver script halts BEFORE paying for the GPU.

Pure stdlib.
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, OrderedDict

# --------------------------------------------------------------------------
# split_sentences: use the agent's own splitter when the repo is importable so
# the training data is counted the same way the runtime counts it. Fall back to
# an identical vendored copy (the GPU box has torch, not onnxruntime).
# --------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    from agent.util import split_sentences  # noqa: E402  (pure stdlib module)
    _SPLITTER = "agent.util"
except Exception:  # pragma: no cover - standalone use
    _SPLITTER = "vendored"
    _ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|St|No|Fig)\.$")

    def split_sentences(text):
        parts, buf = [], ""
        for chunk in re.split(r"(?<=[.!?])\s+", text.strip()):
            buf = (buf + " " + chunk).strip() if buf else chunk
            if _ABBREV.search(buf.rstrip('"').rstrip(")")):
                continue
            if buf:
                parts.append(buf)
                buf = ""
        if buf:
            parts.append(buf)
        return [p for p in parts if p]


CATEGORIES = ("ner", "sentiment", "summary", "math", "logic",
              "code_gen", "code_debug", "factual")

NER_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "DATE", "EVENT", "PRODUCT"}

SENTIMENT_LABELS = ("Positive", "Negative", "Neutral", "Mixed")

# --------------------------------------------------------------------------
# regexes
# --------------------------------------------------------------------------
FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
MD_HEADER_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S", re.MULTILINE)
TEMPLATE_LEAK_RE = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|^\s*(assistant|user|system)\s*:?\s*$",
    re.MULTILINE)
AI_RE = re.compile(
    r"\b(as an ai|as a language model|i am an ai|i'm an ai|"
    r"as an? (?:large )?language model)\b", re.I)
PREAMBLE_RE = re.compile(
    r"^\s*(sure|certainly|of course|absolutely|great question|"
    r"here(?:'s| is| are)\b|below (?:is|are)\b|i'd be happy|i would be happy|"
    r"let me\b|okay|ok[,!.]|alright|as requested|no problem)", re.I)
HEDGE_RE = re.compile(
    r"\b(i (?:do not|don't) have (?:access|real[- ]time)|"
    r"as of my (?:last|knowledge)|i cannot (?:provide|answer|be sure)|"
    r"i'm not (?:sure|certain)|it (?:is )?depends? entirely|"
    r"my training data|i am unable to)\b", re.I)
RESTATE_RE = re.compile(
    r"^\s*(the (?:question|task|prompt) (?:asks|is|requires)|you (?:asked|want)|"
    r"you'd like me to|we are asked|the user (?:asks|wants))", re.I)
PLACEHOLDER_RE = re.compile(
    r"(your code here|todo|fixme|implement (?:this|me)|"
    r"raise NotImplementedError|pass\s*#\s*(?:todo|stub))", re.I)
CONTRAST_RE = re.compile(
    r"\b(but|however|although|though|yet|while|whereas|despite|"
    r"in spite of|on the other hand|even (?:if|so)|nonetheless|"
    r"still|offset|mixed|both)\b", re.I)
STEP_MARKER_RE = re.compile(
    r"^\s*(step\s*\d|first[,:]|second[,:]|next[,:]|then[,:]|finally[,:]|"
    r"let'?s\b|we (?:start|begin|first)|\d\.\s)", re.I | re.MULTILINE)
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_WORDNUM = {"one": 1, "a single": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7}
_STOP = {"the", "a", "an", "is", "are", "was", "of", "in", "on", "at", "to",
         "and", "or", "it", "its", "near", "by", "for", "with", "as", "that",
         "this", "following", "each", "all", "from"}


def norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def norm_text(s):
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def content_words(s):
    return {w for w in norm_text(s).split() if w not in _STOP and len(w) > 1}


def jaccard(a, b):
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def est_tokens(s):
    """Cheap token estimate (no tokenizer on the GPU box's critical path)."""
    return len(s) // 4 + len(s.split()) // 3


# --------------------------------------------------------------------------
# code helpers
# --------------------------------------------------------------------------
def fences(text):
    """[(lang, code)] for every fenced block."""
    return [(m.group(1) or "", m.group(2)) for m in FENCE_RE.finditer(text)]


def outside_fences(text):
    return FENCE_RE.sub("\n", text)


def parses(code):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def ast_key(code):
    """Structural fingerprint: whitespace/comment/format insensitive."""
    try:
        return ast.dump(ast.parse(code), annotate_fields=True)
    except Exception:
        return None


def is_trivial_code(code):
    """A body that promises code instead of being code."""
    if PLACEHOLDER_RE.search(code):
        return True
    try:
        tree = ast.parse(code)
    except Exception:
        return True
    if not tree.body:
        return True

    # every function body is just `...` / `pass` / a docstring
    def stub(node):
        body = [n for n in node.body
                if not (isinstance(n, ast.Expr)
                        and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, str))]
        if not body:
            return True
        return all(isinstance(n, ast.Pass)
                   or (isinstance(n, ast.Expr)
                       and isinstance(n.value, ast.Constant)
                       and n.value.value is Ellipsis)
                   for n in body)

    funcs = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if funcs and all(stub(f) for f in funcs):
        return True
    # no def/class and fewer than 2 real statements: not a "complete program"
    if not funcs and not [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        real = [n for n in tree.body
                if not isinstance(n, (ast.Import, ast.ImportFrom, ast.Pass))]
        if len(real) < 2:
            return True
    return False


# Prose glued onto the end of an inline snippet. The trap: "return nums[1]. Fix"
# PARSES — `.Fix` is an attribute access — so "does it compile?" is NOT enough to
# tell code from code+prose. Real Python never writes a dot followed by a space
# and a name; prose after a sentence-ending dot always does.
_DOT_PROSE_RE = re.compile(r"\.\s+[A-Za-z_]")


def code_from_prompt(text):
    """Lift the buggy snippet out of a code_debug prompt.

    Prompts come both fenced and inline ("...has a bug: def get_sum(nums):
    return nums[0] + nums[1]. Find and fix it."), so: prefer a fence; otherwise
    cut the trailing instruction off and keep the longest slice that is *clean*
    code — parses AND carries no prose.
    """
    fs = fences(text)
    if fs:
        return max((c for _, c in fs), key=len).strip()
    m = re.search(r"(?m)^\s*(?:def |class |import |from )", text)
    if not m:
        m = re.search(r"\b(?:def|class)\s+\w+\s*[(:]", text)
        if not m:
            return ""
    body = text[m.start():]

    cands = []
    # 1. drop trailing sentences ("Find and fix it."), longest slice first
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z])", body)
    for k in range(len(chunks), 0, -1):
        cands.append(" ".join(chunks[:k]))
    # 2. fall back to shaving one token at a time
    toks = body.split(" ")
    for cut in range(0, min(len(toks), 80)):
        cands.append(" ".join(toks[:len(toks) - cut]) if cut else body)

    for cand in cands:
        for c in (cand.strip(), cand.strip().rstrip(".").rstrip()):
            if len(c) < 12 or _DOT_PROSE_RE.search(c):
                continue
            if parses(c):
                return c.strip()
    return ""      # no clean extraction: the caller falls back to a text compare


# --------------------------------------------------------------------------
# sandboxed arithmetic — verify the equations a math answer states
# --------------------------------------------------------------------------
_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
           ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
           ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
           ast.Pow: lambda a, b: a ** b}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-numeric")
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 64 or abs(a) > 1e6):
            raise ValueError("pow too large")   # no 9**9**9 bombs
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and b == 0:
            raise ValueError("div by zero")
        return _BINOPS[type(node.op)](a, b)
    raise ValueError("disallowed node")


def safe_arith(expr):
    """Evaluate a pure-numeric expression, or None."""
    expr = expr.strip()
    if len(expr) > 200 or not re.search(r"[-+*/%]", expr):
        return None
    try:
        return _eval(ast.parse(expr, mode="eval"))
    except Exception:
        return None


def _arith_normalise(text):
    t = (text.replace("×", "*").replace("·", "*").replace("÷", "/")
             .replace("−", "-").replace("–", "-").replace("—", "-"))
    # "1.875 x 2.40" — models write multiplication as an ASCII x constantly, and
    # without this the equation is skipped rather than checked. Digit-on-both-
    # sides only, so the variable in "2x = 10" is untouched.
    t = re.sub(r"(?<=\d)\s*[xX]\s*(?=[\d(])", "*", t)
    t = re.sub(r"(\d),(\d{3})\b", r"\1\2", t)               # 1,672 -> 1672
    t = re.sub(r"(\d),(\d{3})\b", r"\1\2", t)               # 1,234,567
    t = t.replace("$", "").replace("€", "").replace("£", "")
    # "37% of 2400" -> "(37/100*2400)"
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
               r"(\1/100*\2)", t, flags=re.I)
    t = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", t)      # bare 37%
    return t


_EQ_RE = re.compile(
    r"(?<![\w.])((?:\(?-?\d[\d.]*\)?)(?:\s*[-+*/%]\s*\(?-?\d[\d.]*\)?)+)"
    r"\s*=\s*(-?\d[\d.]*)(?![\w.])")

# Tolerance for a stated equation. Loose enough for the rounding the rubrics
# themselves bless ("1.87 or 1.88 cups" for 1.875; $4.488 shown as $4.50), tight
# enough to catch a real miscalculation. 2% was NOT: it silently accepted
# "37% of 2400 = 900" (off by 12, i.e. 1.3%). Do not loosen without a case.
ARITH_ABS_TOL = 0.05
ARITH_REL_TOL = 0.005


def bad_equations(answer):
    """Every stated `a op b = c` whose arithmetic is FALSE."""
    t = _arith_normalise(answer)
    bad = []
    for m in _EQ_RE.finditer(t):
        lhs, rhs = m.group(1), m.group(2)
        got = safe_arith(lhs)
        if got is None:
            continue
        try:
            want = float(rhs)
        except ValueError:
            continue
        if abs(got - want) > max(ARITH_ABS_TOL, ARITH_REL_TOL * abs(want)):
            bad.append(f"{lhs.strip()} = {rhs} (actually {got:g})")
    return bad


def numbers_in(text):
    out = []
    for tok in NUM_RE.findall(text.replace(",", "")):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


# --------------------------------------------------------------------------
# summary constraint — parsed out of the PROMPT (superset of agent/pipelines.py)
# --------------------------------------------------------------------------
def summary_constraint(prompt):
    p = prompt.lower()
    m = re.search(r"(?:exactly |in )?(one|two|three|four|five|six|seven|\d+)"
                  r"\s+bullet(?:\s*point)?s?", p)
    if m:
        n = _WORDNUM.get(m.group(1)) or int(m.group(1))
        mw = re.search(r"each\s+(?:no longer than|no more than|under|at most|"
                       r"fewer than|less than|within|with(?:in)?\s+a maximum of|"
                       r"max(?:imum)?(?:\s+of)?)\s*(\d+)\s*words", p)
        return ("bullets", (n, int(mw.group(1)) if mw else None))
    m = re.search(r"exactly\s+(one|two|three|four|five|six|seven|\d+)\s+sentences?", p)
    if not m:
        m = re.search(r"\bin\s+(one|a single|two|three|four|five|\d+)\s+sentences?", p)
    if m:
        w = m.group(1)
        return ("sentences_exact", _WORDNUM.get(w) or int(w))
    m = re.search(r"(?:at most|under|no more than|maximum of|within|max)\s*(\d+)\s*words", p)
    if m:
        return ("words_max", int(m.group(1)))
    return (None, None)


def bullet_lines(out):
    lines = []
    for ln in out.splitlines():
        s = ln.strip()
        if not s:
            continue
        s = re.sub(r"^(?:[-*•‣▪]|\d+[.)])\s*", "", s).strip()
        if s:
            lines.append(s)
    return lines


# --------------------------------------------------------------------------
# CLEANERS — canonicalise a near-miss into the target format. Never invents.
# --------------------------------------------------------------------------
def clean_common(ans):
    a = THINK_RE.sub("", ans)
    if "<think>" in a:
        a = a.split("<think>")[-1]
    a = a.replace("\r\n", "\n").strip()
    a = re.sub(r"^\s*(?:answer|response|output)\s*:\s*", "", a, flags=re.I)
    a = re.sub(r"\n{3,}", "\n\n", a)
    return a.strip()


def clean_code(ans, want_prose):
    """Canonical: [one bug sentence]\n\n```python\n<code>\n``` — nothing else."""
    fs = fences(ans)
    if not fs:
        return ans
    code = max((c for _, c in fs), key=len).strip()
    head = ans[:ans.index("```")].strip() if "```" in ans else ""
    head = re.sub(r"^\s*(?:the\s+)?(?:bug|issue|problem|fix)\s*:\s*", "",
                  head, flags=re.I).strip()
    block = "```python\n" + code + "\n```"
    if not want_prose:
        return block                              # code_gen: fence only, no prose
    if not head:
        return block                              # caller's checks will drop it
    sents = split_sentences(head)
    if len(sents) > 1:
        buggy = re.compile(
            r"\b(bug|error|off[- ]by[- ]one|index|missing|wrong|incorrect|typo|"
            r"fails?|should|instead|only|never|always|condition|loop|return|"
            r"variable|division|mutat|overwrit|shadow|initiali[sz])\b", re.I)
        pick = next((s for s in sents if buggy.search(s)), sents[0])
        head = pick.strip()
    else:
        head = sents[0].strip() if sents else head
    return head + "\n\n" + block


def clean_ner(ans):
    if norm_ws(ans).lower() in ("none", "none."):
        return "None"
    out = []
    for ln in ans.splitlines():
        s = ln.strip()
        if not s:
            continue
        s = re.sub(r"^(?:[-*•‣]|\d+[.)])\s*", "", s).strip()   # drop bullets
        s = re.sub(r"[│｜]", "|", s)                  # unicode bars
        if "|" not in s and re.match(r"^[^:]{1,60}:\s*[A-Za-z_]+$", s):
            s = s.replace(":", " | ", 1)                       # "Google: ORG"
        if "|" not in s:
            out.append(s)
            continue
        ent, _, typ = s.partition("|")
        typ = re.sub(r"[^A-Za-z_ ]", "", typ).strip().upper().replace(" ", "")
        alias = {"ORG": "ORGANIZATION", "ORGANISATION": "ORGANIZATION",
                 "COMPANY": "ORGANIZATION", "PER": "PERSON", "PEOPLE": "PERSON",
                 "LOC": "LOCATION", "GPE": "LOCATION", "PLACE": "LOCATION",
                 "TIME": "DATE", "PROD": "PRODUCT"}
        typ = alias.get(typ, typ)
        ent = ent.strip().strip('*`"')          # 3.9-safe: no backslash inside the f-string expr
        out.append(f"{ent} | {typ}")
    # de-dupe lines, keep order
    return "\n".join(OrderedDict.fromkeys(out))


def clean_sentiment(ans):
    a = norm_ws(ans)
    a = re.sub(r"^\**\s*sentiment\s*:?\s*\**\s*", "", a, flags=re.I)
    a = re.sub(r"^\**([A-Za-z]+)\**\s*[–—:-]\s*", r"\1 - ", a, count=1)
    m = re.match(r"^([A-Za-z]+)\b[\s,]*(?:[-–—:]|because|since)?\s*(.*)$", a)
    if m:
        lab = m.group(1).capitalize()
        rest = m.group(2).lstrip("-–—: ").strip()
        if lab in SENTIMENT_LABELS and rest:
            return f"{lab} - {rest}"
    return a


def clean_summary(ans, kind):
    a = clean_common(ans)
    if kind == "bullets":
        lines = [ln for ln in (l.strip() for l in a.splitlines()) if ln]
        return "\n".join("- " + re.sub(r"^(?:[-*•‣]|\d+[.)])\s*", "", ln).strip()
                         for ln in lines)
    return norm_ws(a)


# --------------------------------------------------------------------------
# CHECKS — return a list of drop reasons ([] == pass)
# --------------------------------------------------------------------------
def check_universal(prompt, ans, cat, args):
    bad = []
    if not ans.strip():
        return ["empty_answer"]
    prose = outside_fences(ans)          # never lint '#' inside Python!
    if MD_HEADER_RE.search(prose):
        bad.append("markdown_header")
    if AI_RE.search(prose):
        bad.append("as_an_ai")
    if TEMPLATE_LEAK_RE.search(ans):
        bad.append("chat_template_leak")
    if cat != "code_debug" and PREAMBLE_RE.match(prose.lstrip()):
        bad.append("preamble")
    if RESTATE_RE.match(prose.lstrip()):
        bad.append("restates_question")
    else:
        psents = split_sentences(prompt)
        asents = split_sentences(prose.strip())
        if psents and asents:
            thr = 0.90 if cat in ("summary", "ner", "sentiment") else 0.75
            if len(content_words(psents[0])) >= 6 and \
                    jaccard(asents[0], psents[0]) >= thr:
                bad.append("restates_question")
        if cat in ("factual", "logic", "math"):
            head = norm_text(prose)[:60]
            if len(head) == 60 and head in norm_text(prompt):
                bad.append("restates_question")
    n = est_tokens(prompt) + est_tokens(ans)
    if n > args.max_tokens:
        bad.append("too_long_would_truncate")
    return bad


def check_ner(prompt, ans, args):
    if ans.strip() == "None":
        return []
    lines = [l for l in (x.strip() for x in ans.splitlines()) if l]
    if not lines:
        return ["ner_empty"]
    bad, pnorm = [], norm_ws(prompt).lower()
    for ln in lines:
        m = re.match(r"^(.+?)\s\|\s([A-Z]+)$", ln)
        if not m:
            bad.append("ner_bad_line_format")
            break
        ent, typ = m.group(1).strip(), m.group(2)
        if typ not in NER_TYPES:
            bad.append("ner_bad_type")
            break
        if len(ent) < 2:
            bad.append("ner_bad_line_format")
            break
        if norm_ws(ent).lower() not in pnorm:
            bad.append("ner_entity_not_in_prompt")   # hallucinated span
            break
    if len(lines) != len(set(lines)):
        bad.append("ner_duplicate_lines")
    return bad


def check_sentiment(prompt, ans, args):
    if "\n" in ans.strip():
        return ["sentiment_multiline"]
    m = re.match(r"^(Positive|Negative|Neutral|Mixed)\s-\s(.{10,})$", ans.strip())
    if not m:
        return ["sentiment_bad_format"]
    label, reason = m.group(1), m.group(2)
    if len(split_sentences(reason)) > 1:
        return ["sentiment_not_one_sentence"]
    if label == "Mixed":
        two_clauses = ("," in reason and re.search(r"\b(and|but)\b", reason, re.I))
        if not (CONTRAST_RE.search(reason) or two_clauses):
            return ["sentiment_mixed_one_sided"]
    return []


def check_summary(prompt, ans, args):
    kind, n = summary_constraint(prompt)
    if kind is None:
        return []                       # nothing stated; nothing to enforce
    if kind == "bullets":
        want, cap = n
        lines = bullet_lines(ans)
        if len(lines) != want:
            return [f"summary_bullets_{len(lines)}_want_{want}"]
        if cap:
            for ln in lines:
                if len(ln.split()) > cap:
                    return ["summary_bullet_over_word_cap"]
        return []
    if not re.search(r'[.!?]["\')\]]*\s*$', ans.strip()):
        return ["summary_cut_off"]
    if kind == "sentences_exact":
        k = len(split_sentences(ans))
        if k != n:
            return [f"summary_sentences_{k}_want_{n}"]
        return []
    if kind == "words_max":
        if len(ans.split()) > n:
            return ["summary_over_word_cap"]
    return []


def check_math(prompt, ans, args):
    bad = bad_equations(ans)
    if bad:
        return ["math_false_arithmetic"]
    nums = numbers_in(ans)
    if not nums:
        return ["math_no_number"]
    words = len(ans.split())
    if words > args.math_max_words:
        return ["math_derivation_dump"]
    if STEP_MARKER_RE.search(ans) and words > 30:
        return ["math_derivation_dump"]
    tail = split_sentences(ans)[-1] if split_sentences(ans) else ans
    if not numbers_in(tail):
        return ["math_no_final_value"]           # ends on prose, not a value
    if not re.search(r"[A-Za-z]", ans):
        return ["math_no_label"]                 # bare number, no unit/label
    if prompt.count("?") >= 2 and len(set(nums)) < 2:
        return ["math_missing_multipart_value"]
    return []


def check_logic(prompt, ans, args):
    a = ans.strip()
    if len(a.split()) > args.logic_max_words:
        return ["logic_derivation_dump"]
    m = re.search(r"\(([^)]{8,})\)", a)
    if not m:
        return ["logic_no_paren_reason"]
    head = a[:a.index("(")].strip()
    if not head:
        return ["logic_no_answer_sentence"]
    if len(split_sentences(head)) > 1:
        return ["logic_not_one_sentence"]
    return []


def check_code(prompt, ans, args, debug):
    fs = fences(ans)
    if not fs:
        return ["code_no_fence"]
    code = max((c for _, c in fs), key=len).strip()
    if not code:
        return ["code_empty_fence"]
    if not parses(code):
        return ["code_does_not_compile"]
    if is_trivial_code(code):
        return ["code_placeholder_or_stub"]
    prose = outside_fences(ans).strip()
    if not debug:
        if prose and not args.allow_code_prose:
            return ["code_gen_prose_outside_fence"]
        return []
    # code_debug: one sentence naming the bug, BEFORE the code
    head = ans[:ans.index("```")].strip() if "```" in ans else ""
    if not head:
        return ["debug_no_bug_sentence"]
    sents = split_sentences(head)
    if len(sents) != 1:
        return ["debug_bug_desc_not_one_sentence"]
    if len(head) < 15:
        return ["debug_bug_sentence_too_short"]
    if PREAMBLE_RE.match(head):
        return ["debug_preamble_not_bug"]
    buggy = code_from_prompt(prompt)
    if buggy:
        ka, kb = ast_key(code), ast_key(buggy)
        if (ka and kb and ka == kb) or norm_ws(code) == norm_ws(buggy):
            return ["debug_echoes_buggy_code"]   # "fixed" nothing
    return []


def check_factual(prompt, ans, args):
    if HEDGE_RE.search(ans):
        return ["factual_hedging"]
    if len(ans.split()) > args.factual_max_words:
        return ["factual_too_long"]
    if len(ans.split()) < 3:
        return ["factual_too_short"]
    return []


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def validate_row(row, args):
    """-> (clean_row | None, [reasons], cleaned_bool)"""
    cat = (row.get("category") or "").strip()
    prompt = row.get("prompt") or ""
    answer = row.get("answer") or ""
    if cat not in CATEGORIES:
        return None, ["bad_category"], False
    if not prompt.strip():
        return None, ["empty_prompt"], False
    if not answer.strip():
        return None, ["empty_answer"], False

    orig = answer
    answer = clean_common(answer)
    if cat == "ner":
        answer = clean_ner(answer)
    elif cat == "sentiment":
        answer = clean_sentiment(answer)
    elif cat == "summary":
        answer = clean_summary(answer, summary_constraint(prompt)[0])
    elif cat in ("code_gen", "code_debug"):
        answer = clean_code(answer, want_prose=(cat == "code_debug"))
    cleaned = (answer.strip() != orig.strip())

    reasons = check_universal(prompt, answer, cat, args)
    if not reasons:
        fn = {"ner": check_ner, "sentiment": check_sentiment,
              "summary": check_summary, "math": check_math,
              "logic": check_logic, "factual": check_factual}.get(cat)
        if fn:
            reasons = fn(prompt, answer, args)
        else:
            reasons = check_code(prompt, answer, args, debug=(cat == "code_debug"))
    if reasons:
        return None, reasons, cleaned
    return {"prompt": prompt, "answer": answer, "category": cat}, [], cleaned


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", default="finetune/sft_raw.jsonl")
    ap.add_argument("--out", dest="out", default="finetune/sft.jsonl")
    ap.add_argument("--drops", default="finetune/sft_dropped.jsonl",
                    help="every rejected row + its reasons, for eyeballing")
    ap.add_argument("--max-tokens", type=int, default=1900,
                    help="prompt+answer budget (rough estimate, runs ~40%% high). "
                         "train.py drops rows over --maxlen=2048; so does the "
                         "2048-token serving context. Keep headroom for both.")
    ap.add_argument("--math-max-words", type=int, default=80)
    ap.add_argument("--logic-max-words", type=int, default=70)
    ap.add_argument("--factual-max-words", type=int, default=180)
    ap.add_argument("--min-per-cat", type=int, default=30,
                    help="exit nonzero if a category ends up below this")
    ap.add_argument("--allow-code-prose", action="store_true",
                    help="keep code_gen answers with prose outside the fence")
    ap.add_argument("--show-drops", type=int, default=0,
                    help="print N dropped examples per category")
    a = ap.parse_args()

    # resolve relative to the repo, not the cwd
    for k in ("inp", "out", "drops"):
        v = getattr(a, k)
        if not os.path.isabs(v):
            setattr(a, k, os.path.join(_REPO, v))

    if not os.path.exists(a.inp):
        print(f"FATAL: no input at {a.inp}", file=sys.stderr)
        return 2

    kept, dropped = [], []
    seen, n_lines, n_bad_json, n_dupe, n_cleaned = {}, 0, 0, 0, 0
    per_cat = {c: Counter() for c in CATEGORIES}
    reasons_all = Counter()
    reasons_by_cat = {c: Counter() for c in CATEGORIES}
    examples = {c: [] for c in CATEGORIES}

    with open(a.inp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                n_bad_json += 1
                reasons_all["bad_json"] += 1
                continue
            if not isinstance(row, dict):
                n_bad_json += 1
                reasons_all["bad_json"] += 1
                continue

            cat = (row.get("category") or "").strip()
            key = norm_text(row.get("prompt") or "")
            if key and key in seen:
                n_dupe += 1
                reasons_all["duplicate_prompt"] += 1
                if cat in per_cat:
                    per_cat[cat]["dropped"] += 1
                    reasons_by_cat[cat]["duplicate_prompt"] += 1
                dropped.append({"category": cat, "prompt": row.get("prompt"),
                                "answer": row.get("answer"),
                                "reasons": ["duplicate_prompt"]})
                continue
            if key:
                seen[key] = True

            good, why, was_cleaned = validate_row(row, a)
            if cat in per_cat:
                if was_cleaned:
                    per_cat[cat]["cleaned"] += 1
                    n_cleaned += was_cleaned
            if good:
                kept.append(good)
                per_cat[cat]["kept"] += 1
            else:
                dropped.append({"category": cat, "prompt": row.get("prompt"),
                                "answer": row.get("answer"), "reasons": why})
                for r in why:
                    reasons_all[r] += 1
                    if cat in reasons_by_cat:
                        reasons_by_cat[cat][r] += 1
                if cat in per_cat:
                    per_cat[cat]["dropped"] += 1
                    if len(examples[cat]) < max(a.show_drops, 0):
                        examples[cat].append((why, row.get("answer") or ""))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.drops, "w", encoding="utf-8") as f:
        for r in dropped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- report ----------------
    W = 72
    print("=" * W)
    print(f"SFT DATASET VALIDATION   ({_SPLITTER} sentence splitter)")
    print(f"  in : {a.inp}")
    print(f"  out: {a.out}")
    print("=" * W)
    print(f"{'category':<12}{'kept':>7}{'dropped':>9}{'cleaned':>9}{'keep%':>8}"
          f"   top drop reasons")
    print("-" * W)
    for c in CATEGORIES:
        k = per_cat[c]["kept"]
        d = per_cat[c]["dropped"]
        cl = per_cat[c]["cleaned"]
        tot = k + d
        pct = (100.0 * k / tot) if tot else 0.0
        top = ", ".join(f"{r}:{n}" for r, n in reasons_by_cat[c].most_common(2))
        flag = "  <-- LOW" if k < a.min_per_cat else ""
        print(f"{c:<12}{k:>7}{d:>9}{cl:>9}{pct:>7.0f}%   {top}{flag}")
    print("-" * W)
    tot_in = n_lines
    print(f"{'TOTAL':<12}{len(kept):>7}{len(dropped):>9}{n_cleaned:>9}"
          f"{(100.0 * len(kept) / tot_in if tot_in else 0):>7.0f}%")
    print()
    print(f"read {tot_in} lines  |  bad json {n_bad_json}  |  dupe prompts {n_dupe}")
    print()
    print("top drop reasons (all categories)")
    for r, n in reasons_all.most_common(12):
        print(f"  {n:>5}  {r}")

    if kept:
        lens = sorted(est_tokens(r["prompt"]) + est_tokens(r["answer"])
                      for r in kept)
        p50 = lens[len(lens) // 2]
        p95 = lens[int(len(lens) * 0.95) - 1] if len(lens) > 1 else lens[0]
        print()
        print(f"est. tokens/example  p50={p50}  p95={p95}  max={lens[-1]}"
              f"   (train.py --maxlen=2048)")

    if a.show_drops:
        for c in CATEGORIES:
            if not examples[c]:
                continue
            print()
            print(f"--- dropped: {c} " + "-" * (W - 14 - len(c)))
            for why, ansr in examples[c]:
                snip = norm_ws(ansr)[:110]
                print(f"  [{','.join(why)}] {snip}")

    print()
    print(f"dropped rows written to {a.drops}")

    empty = [c for c in CATEGORIES if per_cat[c]["kept"] == 0]
    low = [c for c in CATEGORIES if 0 < per_cat[c]["kept"] < a.min_per_cat]
    if empty:
        print(f"\nFAIL: no surviving examples for: {', '.join(empty)}")
        print("      do NOT train — the model will have no format to copy there.")
        return 1
    if low:
        print(f"\nFAIL: below --min-per-cat={a.min_per_cat}: {', '.join(low)}")
        print("      regenerate those categories before spending the GPU hour.")
        return 1
    print(f"\nOK -> {len(kept)} examples ready for "
          f"`python finetune/train.py --data {a.out}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
