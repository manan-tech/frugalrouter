"""Small shared helpers: logging, timing, atomic output, text normalization."""

import json
import os
import re
import sys
import threading
import time

_T0 = time.monotonic()


def elapsed() -> float:
    return time.monotonic() - _T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", file=sys.stderr, flush=True)


_write_lock = threading.Lock()


def atomic_write_results(path: str, answers: dict) -> None:
    """Write [{task_id, answer}] atomically. Safe to call from any thread."""
    payload = [{"task_id": tid, "answer": str(ans)} for tid, ans in answers.items()]
    tmp = path + ".tmp"
    with _write_lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)


THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (and a dangling open tag) from output."""
    text = THINK_RE.sub("", text)
    if "<think>" in text:  # unterminated think block: keep only what follows it
        text = text.split("<think>")[-1]
    return text.strip()


_ABBREV = re.compile(r"\b(e\.g|i\.e|etc|vs|Dr|Mr|Mrs|Ms|St|No|Fig)\.$")


def split_sentences(text: str) -> list:
    """Conservative sentence splitter good enough for constraint checking."""
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


NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_last_number(text: str):
    """Last numeric literal in text, as float, or None."""
    m = NUM_RE.findall(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def fmt_number(x: float) -> str:
    """144.0 -> '144'; keep decimals only when needed."""
    if x is None:
        return ""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def content_words(s: str) -> set:
    stop = {"the", "a", "an", "is", "are", "was", "of", "in", "on", "at", "to",
            "and", "or", "it", "its", "near", "by", "for", "with", "as", "that"}
    return {w for w in norm_text(s).split() if w not in stop and len(w) > 1}


def similarity(a: str, b: str) -> float:
    """Jaccard on content words — cheap agreement check for short answers."""
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 1.0 if norm_text(a) == norm_text(b) else 0.0
    return len(wa & wb) / len(wa | wb)


CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Best-effort python code extraction from a model reply."""
    m = CODE_FENCE_RE.findall(text)
    if m:
        return max(m, key=len).strip()
    # no fence: keep lines that look like code
    lines = text.splitlines()
    keep, started = [], False
    for ln in lines:
        if re.match(r"\s*(def |class |import |from |print\(|#|@|\w+\s*=)", ln):
            started = True
        if started:
            keep.append(ln)
    return "\n".join(keep).strip() or text.strip()
