"""Zero-cost category classifier. The 8 category phrasings are highly
distinctive; ordered keyword rules cover paraphrased variants. No model call."""

import re

CATEGORIES = ("sentiment", "summary", "ner", "code_debug", "code_gen",
              "logic", "math", "factual")

_CODEISH = re.compile(r"(def |```|\breturn\b|[a-z_]+\(.*\)|lambda |class )")
_DIGITS = re.compile(r"\d")


def classify(prompt: str) -> str:
    p = prompt.lower()

    if re.search(r"\bsentiment\b|\b(positive|negative|neutral)\b.*\b(review|text|statement)\b", p):
        return "sentiment"
    if re.search(r"\bsummar(y|ize|ise|isation|ization)\b|\bcondense\b|\btl;?dr\b", p):
        return "summary"
    if re.search(r"named entit|\bentities\b|\bentity\b.*\b(type|label)", p):
        return "ner"
    if re.search(r"\b(bug|bugs|buggy|fix|debug|broken|incorrect|wrong output|error)\b", p) and _CODEISH.search(prompt):
        return "code_debug"
    if re.search(r"\b(write|implement|create|build|develop|code up)\b[^.]*\b(function|method|class|program|script|code)\b", p):
        return "code_gen"
    if re.search(r"\beach\b[^.]*\bdifferent\b|\bpuzzle\b|\briddle\b|"
                 r"\bdifferent\b[^.]*\b(position|pet|sport|color|colour|drink|"
                 r"beverage|item|house|seat|hobby|instrument|job)s?\b", p) or (
            re.search(r"\b(who|which)\b[^?]*\?", p) and
            re.search(r"\b(does not|doesn't|did not|didn't|neither|isn't|not the)\b", p)):
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
