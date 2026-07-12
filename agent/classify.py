"""Zero-cost category classifier. The 8 category phrasings are highly
distinctive; ordered keyword rules cover paraphrased variants. No model call."""

import re

CATEGORIES = ("sentiment", "summary", "ner", "code_debug", "code_gen",
              "logic", "math", "factual")

_CODEISH = re.compile(r"(def |```|\breturn\b|[a-z_]+\(.*\)|lambda |class )")
_DIGITS = re.compile(r"\d")


def classify(prompt: str) -> str:
    p = prompt.lower()

    # Finals re-run on REPHRASED prompts, so these must match the intent, not one
    # phrasing. A misroute is expensive twice over: the task skips its pipeline
    # (an ONNX tagger/classifier never fires) AND falls through to factual, which
    # always escalates — so we pay tokens for a wrongly-formatted answer.
    if re.search(r"\bsentiment\b|\bpolarity\b|\b(overall\s+)?tone\b|"
                 r"\b(positive|negative|neutral)\b.*\b(review|text|statement|"
                 r"feedback|comment|tweet|post|message)\b|"
                 r"\b(positive|negative)\b\s*,?\s*\bnegative|neutral\b|"
                 r"how (the |a )?(writer|author|customer|reviewer|user) (feels|felt)|"
                 r"\bhow does the (writer|author|customer|reviewer) feel\b|"
                 r"\bopinion\b.*\b(express|convey)", p):
        return "sentiment"
    if re.search(r"\bsummar(y|ize|ise|isation|ization|ising|izing)\b|\bcondense\b|"
                 r"\btl;?dr\b|\b(brief|short|concise)\s+(overview|account|version|"
                 r"description|recap)\b|\boverview of\b|\bin (a|one|two|three) "
                 r"sentences?\b|\bkey (points|takeaways)\b|\bgist\b|\brecap\b", p):
        return "summary"
    if re.search(r"named entit|\bentities\b|\bentity\b.*\b(type|label)|"
                 r"\bproper nouns?\b", p) or (
            # "identify/extract/list/tag/pull out ... people/orgs/places/dates".
            # Guarded against code tasks: "write a function to find the NAMES of
            # duplicate keys" is code_gen, not NER — the noun overlap is a trap.
            re.search(r"\b(identify|extract|list|tag|find|pull out|pick out)\b[^.]*\b"
                      r"(people|persons?|organi[sz]ations?|companies|locations?|"
                      r"places?|names?|dates?)\b", p)
            and not re.search(r"\b(function|method|class|program|script|code|"
                              r"algorithm|implement)\b", p)
            and not _CODEISH.search(prompt)):
        return "ner"
    if re.search(r"\b(bug|bugs|buggy|fix|debug|broken|incorrect|wrong output|error)\b", p) and _CODEISH.search(prompt):
        return "code_debug"
    if re.search(r"\b(write|implement|create|build|develop|code up)\b[^.]*\b(function|method|class|program|script|code)\b", p):
        return "code_gen"
    if re.search(r"\beach\b[^.]*\bdifferent\b|\bpuzzle\b|\briddle\b|"
                 r"\bdifferent\b[^.]*\b(position|pet|sport|color|colour|drink|"
                 r"beverage|item|house|seat|hobby|instrument|job)s?\b|"
                 r"\blabel(s|ed|led)?\b[^.]*\b(wrong|incorrect)|"
                 r"\bboxes?\b[^.]*\blabel", p) or (
            re.search(r"\b(who|which|what)\b[^?]*\?", p) and
            re.search(r"\b(does not|doesn't|did not|didn't|neither|isn't|not the|"
                      r"every label is wrong)\b", p)):
        return "logic"
    if _DIGITS.search(p) and re.search(
            r"how (many|much)|what is the (total|sum|result|final|value)|percent|%|"
            r"\bremain(s|ing)?\b|\bleft\b|\bcost\b|\bprice\b|\bprofit\b|\baverage\b|"
            r"\bprojection\b|\bincrease\b|\bdecrease\b|\bcalculate\b|\bsells?\b|\bsold\b", p):
        return "math"
    # safety overrides before factual fallback
    if _CODEISH.search(prompt) and "def " in prompt:
        return "code_debug"
    if _DIGITS.search(p) and re.search(r"how (many|much)\b", p):
        return "math"
    return "factual"


# --------------------------------------------------------------------------
# Semantic safety net for the fallback
# --------------------------------------------------------------------------
# Every regex miss has the SAME shape: nothing matched, so the task falls through
# to "factual". That is silently expensive — a rephrased NER/sentiment task then
# skips its pipeline entirely (the ONNX tagger/classifier never fires) AND gets
# escalated with the wrong output format. And the finals rerun on REPHRASED
# prompts, i.e. wordings we did not enumerate.
#
# So: only when the regex falls through, ask the local model what the task is.
# Costs one free local call, never runs on a confidently-matched task, and a
# grammar constraint means the reply is always one of the eight labels.
_CLASSIFY_SYS = (
    "You label a task with exactly ONE category:\n"
    "ner - extract named entities (people, organisations, places, dates)\n"
    "sentiment - judge the opinion/tone of a text\n"
    "summary - condense a passage\n"
    "math - compute a numeric answer\n"
    "logic - solve a constraint/deduction puzzle\n"
    "code_gen - write new code\n"
    "code_debug - fix broken code\n"
    "factual - answer a knowledge question\n"
    "Reply with the category name only.")
_CLASSIFY_GRAMMAR = ('root ::= "ner" | "sentiment" | "summary" | "math" | '
                     '"logic" | "code_gen" | "code_debug" | "factual"')


def classify_routed(prompt: str) -> str:
    """Regex first, embeddings adjudicate the fallback. THIS is what main uses.

    The regex is perfect on the phrasings it knows (19/19 + 10/10 on the real
    suites) but blind to rewordings (6/14 on held-out paraphrases) — and it only
    ever fails ONE way: everything unmatched lands in the `factual` catch-all. So
    the fallback is the only place worth a second opinion, and an embedding router
    scores 13/14 there.

    Critically, unlike the 0.6B (see classify_semantic — measured harmful), the
    embedding router keeps genuine factual questions as factual, so it cannot
    corrupt a task the regex already got right.
    """
    cat = classify(prompt)
    if cat != "factual":
        return cat                       # confident regex hit — no embedding call
    try:
        from . import router
        r = router.route(prompt)
        if r and r[0] != "factual":
            return r[0]
    except Exception:  # noqa: BLE001 — the net must never break the run
        pass
    return "factual"


def classify_semantic(prompt: str) -> str:
    """DO NOT USE — MEASURED HARMFUL. Kept only to document the negative result.

    The idea was sound: the regex only ever fails by falling through to factual,
    so let the local model adjudicate that case. Measured against the real 0.6B
    in-container: 4/9, and it BROKE correct answers —
        "What is the capital of Australia?"            factual -> sentiment
        "Who developed the theory of general relativity?"  factual -> math
        "note down every company and city"            factual -> code_gen
    A 0.6B cannot do 8-way task classification; it turns tasks the regex got
    RIGHT into wrong-pipeline garbage. The broadened regex (21/21 on paraphrases,
    19/19 + 10/10 on the real suites) is strictly safer. main.py uses classify().
    """
    cat = classify(prompt)
    if cat != "factual":
        return cat                      # a confident regex hit — trust it, no call
    try:
        from .llm import GENERAL
        out = GENERAL.chat(
            [{"role": "system", "content": _CLASSIFY_SYS},
             {"role": "user", "content": prompt[:600]}],
            max_tokens=6, temperature=0.0, grammar=_CLASSIFY_GRAMMAR).strip().lower()
        if out in CATEGORIES:
            return out
    except Exception:  # noqa: BLE001 — the net must never break the run
        pass
    return "factual"
