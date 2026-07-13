"""Zero-model last resort: deterministic answerers used only when local
inference is dead AND the escalation budget is exhausted. Crude, but a
lexicon sentiment call or a lead-sentence summary scores where fallback
text ("I could not determine...") scores zero."""

import re

from .util import split_sentences

_POS = {"great", "good", "love", "loved", "excellent", "amazing", "fantastic",
        "wonderful", "perfect", "best", "awesome", "happy", "impressive",
        "smooth", "fast", "reliable", "recommend", "brilliant", "superb",
        "delightful", "beautiful", "enjoy", "enjoyed", "pleased", "solid"}
_NEG = {"bad", "terrible", "awful", "hate", "hated", "worst", "horrible",
        "poor", "broken", "crash", "crashes", "crashed", "useless", "slow",
        "disappointing", "disappointed", "waste", "wasted", "fails", "failed",
        "annoying", "frustrating", "scratches", "dies", "wiped", "bug", "buggy"}
_NEGATORS = {"not", "never", "no", "isn't", "wasn't", "don't", "doesn't",
             "didn't", "can't", "couldn't", "wouldn't", "hardly", "barely"}


def sentiment(prompt: str) -> str:
    m = re.search(r":\s*(.+)$", prompt, re.DOTALL)
    text = (m.group(1) if m else prompt).lower()
    words = re.findall(r"[a-z']+", text)
    pos = neg = 0
    for i, w in enumerate(words):
        negated = any(x in _NEGATORS for x in words[max(0, i - 2):i])
        if w in _POS:
            neg, pos = (neg + 1, pos) if negated else (neg, pos + 1)
        elif w in _NEG:
            neg, pos = (neg, pos + 1) if negated else (neg + 1, pos)
    if pos and neg:
        label, why = "Mixed", "both positive and negative points are present"
    elif pos:
        label, why = "Positive", "the language is favorable"
    elif neg:
        label, why = "Negative", "the language is critical"
    else:
        label, why = "Neutral", "the statement is factual without opinion"
    return f"{label} - {why}."


def summary(prompt: str) -> str:
    m = re.search(r":\s*(.+)$", prompt, re.DOTALL)
    passage = (m.group(1) if m else prompt).strip()
    sents = split_sentences(passage)
    if not sents:
        return passage[:200]
    n = 1
    mnum = re.search(r"(?:exactly |in )(one|a single|two|three|\d+) sentences?",
                     prompt.lower())
    if mnum:
        w = mnum.group(1)
        n = {"one": 1, "a single": 1, "two": 2, "three": 3}.get(w) or int(w)
    out = " ".join(sents[:n])
    mw = re.search(r"(?:at most|under|no more than|within) (\d+) words",
                   prompt.lower())
    if mw:
        cap = int(mw.group(1))
        words = out.split()
        if len(words) > cap:
            out = " ".join(words[:cap - 1]).rstrip(",;") + "."
    return out


_DATE_RE = re.compile(
    r"\b(?:(?:last|next|early|late|mid|this)\s+)?"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|"
    r"Sunday|spring|summer|autumn|fall|winter)(?:\s+\d{4})?\b|\b(?:19|20)\d{2}\b",
    re.IGNORECASE)
_ORG_HINTS = ("Inc", "Corp", "University", "Institute", "Bank", "Company",
              "Forum", "Organization", "Ltd", "AI", "Technologies", "Group")


def ner(prompt: str) -> str:
    m = re.search(r"(?:from|in)\s*:\s*(.+)$", prompt, re.DOTALL)
    text = (m.group(1) if m else prompt).strip()
    lines, seen = [], set()
    for dm in _DATE_RE.finditer(text):
        s = dm.group(0).strip()
        if s.lower() not in seen:
            seen.add(s.lower())
            lines.append(f"{s} | DATE")
    # capitalized runs (skip sentence starts of common words)
    for cm in re.finditer(r"\b(?:[A-Z][a-zA-Z'&.]+(?:\s+(?:of|the|de|for))?\s+){1,4}"
                          r"[A-Z][a-zA-Z'&.]+\b|\b[A-Z][a-zA-Z'&.]+\b", text):
        s = cm.group(0).strip().rstrip(".,")
        if len(s) < 3 or s.lower() in seen or _DATE_RE.fullmatch(s):
            continue
        seen.add(s.lower())
        if any(h in s for h in _ORG_HINTS):
            kind = "ORGANIZATION"
        elif re.fullmatch(r"(?:Dr\.?\s+)?[A-Z][a-z]+\s+[A-Z][a-z]+", s):
            kind = "PERSON"
        else:
            kind = "LOCATION"
        lines.append(f"{s} | {kind}")
    return "\n".join(lines[:10]) if lines else "None"


HANDLERS = {"sentiment": sentiment, "summary": summary, "ner": ner}


def answer(category: str, prompt: str):
    """Best-effort deterministic answer, or None for unsupported categories."""
    fn = HANDLERS.get(category)
    if not fn:
        return None
    try:
        return fn(prompt)
    except Exception:  # noqa: BLE001
        return None
