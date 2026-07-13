#!/usr/bin/env python3
"""Build the FrugalRouter RAG corpus: Simple English Wikipedia intros -> chunks.jsonl.

WHAT THIS DOES
    Downloads Simple English Wikipedia (HuggingFace `wikimedia/wikipedia`,
    config `20231101.simple`, 241,787 articles), extracts and cleans the LEAD
    (intro) of each article, drops non-answer pages (redirects / lists / indexes
    / disambiguation / namespace / stubs), caps each intro to <=150 words at a
    sentence boundary, and writes ONE JSON object per surviving article to
    chunks.jsonl in the exact RAG-bundle format.

WHY SIMPLE ENGLISH + INTRO-ONLY
    N lands in the target envelope (~220k-235k chunks after filtering) so the
    int8 embedding matrix stays ~85-90 MB. Wikimedia's parquet `text` field is
    ALREADY cleaned plain prose (no wikitext/HTML, redirects dropped), so no
    XML / mwparserfromhell parsing is needed -- we just slice the lead out of
    ready prose. Intro-only maximises answer density per token for a 1.7B reader.

THE BUNDLE CONTRACT (this script owns line/id assignment; embed_corpus.py and
agent/rag.py must agree with it byte-for-byte):
    chunks.jsonl : one JSON object per line, exactly the keys
        {"id": <int == 0-based line index>, "title": <str>, "text": <str, <=150 words>}
    Row i of emb_int8.npy (written later by embed_corpus.py) is the embedding of
    THIS file's line-i `text`. So the ordering here is load-bearing: we iterate
    the source ONCE in fixed parquet row order and assign id = kept-counter, which
    is identical to the output line index.

PROPERTIES
    * Deterministic  -- no randomness, no seeds; parquet row order is stable and
      identical whether loaded via `datasets` or the raw parquet shard.
    * Streaming/flat -- never holds the whole corpus in RAM (Arrow memory-map or
      parquet row-group batches); safe on the 4 GB envelope and on a dev box.
    * Safe to re-run  -- writes to `<out>.tmp` then atomically os.replace()s it
      over `<out>`; a crashed run never leaves a half-written chunks.jsonl.
    * No `datasets` hard dep -- prefers the HF `datasets` library when present,
      else falls back to a direct urllib download of the single parquet shard +
      pyarrow (both yield the SAME row order, so ids are source-independent).

RUN
    python3 finetune/rag/build_corpus.py                 # -> /models/rag/chunks.jsonl
    python3 finetune/rag/build_corpus.py --out /tmp/chunks.jsonl --limit 5000  # smoke
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# Source identity (verified in the corpus plan)
# ---------------------------------------------------------------------------
HF_DATASET = "wikimedia/wikipedia"
DEFAULT_CONFIG = "20231101.simple"
# Single-shard configs only (simple is one shard). For multi-shard configs
# (e.g. 20231101.en) use --source datasets, which resolves all shards itself.
PARQUET_URL_TMPL = (
    "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
    "{config}/train-00000-of-00001.parquet"
)
DEFAULT_OUT = "/models/rag/chunks.jsonl"
DEFAULT_CACHE = os.path.expanduser("~/.cache/frugalrouter_rag")

# ---------------------------------------------------------------------------
# Extraction / filtering knobs
# ---------------------------------------------------------------------------
TARGET_WORDS = 120     # greedily accumulate lead blocks until >= this many words
MAX_WORDS = 150        # hard cap on an intro (contract: text <= ~150 words)
MAX_BLOCKS = 3         # ...or after this many paragraph blocks, whichever first
MIN_WORDS = 6          # drop intros shorter than this (near-empty stubs)
MIN_CHARS = 40         # ...or shorter than this many characters
DISAMBIG_MIN_PROSE = 12  # a lead this short that ends in ':' reads as a disambig header

# Titles that are indexes/lists, not answer-bearing articles.
LIST_TITLE_PREFIXES = (
    "List of ", "Lists of ", "Index of ", "Glossary of ",
    "Timeline of ", "Comparison of ", "Outline of ",
)
# Non-main namespaces. wikimedia parquet is main-namespace only; this is defensive.
NAMESPACE_PREFIXES = (
    "Wikipedia:", "Template:", "Category:", "Help:", "Portal:",
    "Module:", "File:", "Draft:", "MediaWiki:", "TimedText:",
)
# Lead phrases that mark a disambiguation page.
DISAMBIG_PHRASES = (
    " may refer to", " may mean", " can refer to", " may stand for",
    " can mean", " may also refer to",
)

# ---------------------------------------------------------------------------
# Cleanup regexes (input is already clean prose; these are cheap insurance)
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")            # residual HTML tags
_BRACKET_RE = re.compile(r"\[[^\]]*\]")     # [1] [edit] [note 2] reference/edit markers
_EMPTY_PAREN_RE = re.compile(r"\(\s*[,;]?\s*\)")  # () left behind after removals
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_OPEN_PAREN_SP_RE = re.compile(r"\(\s+")
_CLOSE_PAREN_SP_RE = re.compile(r"\s+\)")
_WS_RE = re.compile(r"\s+")
_SENTENCE_ENDERS = ".!?"


def log(msg: str) -> None:
    """Progress/diagnostics go to stderr so stdout stays clean."""
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Text cleaning + intro extraction
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Defensive cleanup of an intro. Mostly no-ops on wikimedia's cleaned prose.

    Deliberately does NOT strip parenthetical *content* -- an acronym gloss such
    as "(HTTP)" is exactly the fact a factual question wants, so we only remove
    genuinely EMPTY parens and tighten spacing that our own removals may leave.
    """
    text = html.unescape(text)
    text = _TAG_RE.sub("", text)
    text = _BRACKET_RE.sub("", text)          # drop [1]/[edit]/[note 2] markers
    text = _EMPTY_PAREN_RE.sub("", text)      # drop parens emptied by the above
    text = _OPEN_PAREN_SP_RE.sub("(", text)   # "( x" -> "(x"
    text = _CLOSE_PAREN_SP_RE.sub(")", text)  # "x )" -> "x)"
    text = _EMPTY_PAREN_RE.sub("", text)      # re-check now that parens are tight
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)  # " ,"/" ." -> ","/"."
    text = _WS_RE.sub(" ", text)              # collapse all whitespace to single space
    return text.strip()


def split_blocks(text: str) -> list[str]:
    """Split cleaned article text into non-empty paragraph blocks (top to bottom)."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            out.append(line)
    return out


def extract_intro(blocks: list[str]) -> str:
    """Greedily accumulate lead blocks until >= TARGET_WORDS or MAX_BLOCKS taken."""
    acc: list[str] = []
    words = 0
    for block in blocks:
        acc.append(block)
        words += len(block.split())
        if words >= TARGET_WORDS or len(acc) >= MAX_BLOCKS:
            break
    return " ".join(acc)


def cap_words(text: str, max_words: int = MAX_WORDS) -> str:
    """Cap to <= max_words, backing off to the last sentence end so we never cut
    mid-sentence. If no sentence end sits in the back portion, hard-cut instead."""
    words = text.split()
    if len(words) <= max_words:
        return text
    capped = " ".join(words[:max_words])
    last = max((capped.rfind(c) for c in _SENTENCE_ENDERS), default=-1)
    # Only honour the backoff if it keeps most of the window; otherwise a single
    # very long sentence would collapse to a fragment -- prefer the hard cut then.
    if last >= int(len(capped) * 0.6):
        return capped[: last + 1].strip()
    return capped.strip()


# ---------------------------------------------------------------------------
# Page-level filters
# ---------------------------------------------------------------------------
def dropped_reason(title: str, raw_text: str, first_block: str, intro: str):
    """Return a short reason string if this page should be dropped, else None."""
    if not raw_text or not raw_text.strip():
        return "redirect_or_empty"
    if any(title.startswith(p) for p in NAMESPACE_PREFIXES):
        return "namespace"
    if any(title.startswith(p) for p in LIST_TITLE_PREFIXES):
        return "list_or_index"

    low_first = first_block.lower()
    if title.endswith("(disambiguation)"):
        return "disambiguation"
    if any(p in low_first for p in DISAMBIG_PHRASES):
        return "disambiguation"
    # A very short lead that ends in ':' is a disambiguation/list header.
    if (len(first_block.split()) < DISAMBIG_MIN_PROSE
            and first_block.rstrip().endswith(":")):
        return "disambiguation"

    if len(intro.split()) < MIN_WORDS or len(intro) < MIN_CHARS:
        return "stub"
    return None


# ---------------------------------------------------------------------------
# Source iterators -- both yield (title, text) in identical parquet row order
# ---------------------------------------------------------------------------
def _iter_via_datasets(config: str, streaming: bool):
    from datasets import load_dataset  # imported lazily so the parquet path has no dep
    ds = load_dataset(HF_DATASET, config, split="train", streaming=streaming)
    for row in ds:
        yield (row.get("title") or "", row.get("text") or "")


def _download_shard(url: str, cache_dir: str) -> str:
    """Download the parquet shard to cache_dir if not already present. Re-run safe."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = url.rstrip("/").split("/")[-1]
    dest = os.path.join(cache_dir, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        log(f"[parquet] using cached shard {dest} ({os.path.getsize(dest) / 1e6:.0f} MB)")
        return dest
    tmp = dest + ".part"
    headers = {"User-Agent": "frugalrouter-build-corpus/1.0"}
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    log(f"[parquet] downloading {url}")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if total:
                log(f"[parquet] {done / 1e6:6.0f} / {total / 1e6:.0f} MB")
    os.replace(tmp, dest)
    return dest


def _iter_via_parquet(config: str, parquet_url: str, cache_dir: str, batch_size: int):
    import pyarrow.parquet as pq  # lazy; only the parquet path needs it
    url = parquet_url or PARQUET_URL_TMPL.format(config=config)
    path = _download_shard(url, cache_dir)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["title", "text"]):
        cols = batch.to_pydict()
        for title, text in zip(cols["title"], cols["text"]):
            yield (title or "", text or "")


def iter_articles(args):
    """Yield (title, text) from the chosen source, with auto-fallback."""
    if args.source == "datasets":
        log(f"[source] HuggingFace datasets: {HF_DATASET} {args.config}")
        yield from _iter_via_datasets(args.config, args.streaming)
        return
    if args.source == "parquet":
        log(f"[source] direct parquet shard: {args.config}")
        yield from _iter_via_parquet(args.config, args.parquet_url, args.cache_dir, args.batch_size)
        return

    # auto: prefer datasets, fall back to raw parquet on import- or load-failure.
    try:
        import datasets  # noqa: F401
    except ImportError:
        log("[source] `datasets` not installed -> direct parquet shard")
        yield from _iter_via_parquet(args.config, args.parquet_url, args.cache_dir, args.batch_size)
        return
    try:
        gen = _iter_via_datasets(args.config, args.streaming)
        first = next(gen)  # forces load_dataset() to execute; catches load failures up front
    except StopIteration:
        return
    except Exception as exc:  # noqa: BLE001 -- any load failure -> parquet fallback
        log(f"[source] datasets path failed ({exc!r}) -> direct parquet shard")
        yield from _iter_via_parquet(args.config, args.parquet_url, args.cache_dir, args.batch_size)
        return
    log(f"[source] HuggingFace datasets: {HF_DATASET} {args.config}")
    yield first
    yield from gen


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build(args) -> int:
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".tmp"

    scanned = 0
    kept = 0
    drops: dict[str, int] = {}
    t0 = time.time()

    with open(tmp, "w", encoding="utf-8") as fh:
        for title, raw_text in iter_articles(args):
            scanned += 1

            blocks = split_blocks(raw_text)
            first_block = blocks[0] if blocks else ""
            intro = cap_words(clean_text(extract_intro(blocks))) if blocks else ""

            reason = dropped_reason(title, raw_text, first_block, intro)
            if reason is not None:
                drops[reason] = drops.get(reason, 0) + 1
            else:
                # id == kept == 0-based output line index (contract-critical)
                rec = {"id": kept, "title": title, "text": intro}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1

            if scanned % args.progress_every == 0:
                rate = scanned / max(time.time() - t0, 1e-6)
                log(f"[build] scanned={scanned:,} kept={kept:,} "
                    f"dropped={scanned - kept:,} ({rate:,.0f}/s)")

            if args.limit and scanned >= args.limit:
                log(f"[build] --limit {args.limit} reached; stopping")
                break

        fh.flush()
        os.fsync(fh.fileno())

    os.replace(tmp, out)  # atomic publish; re-run never leaves a partial file

    log("-" * 60)
    log(f"[done] scanned={scanned:,}  kept={kept:,}  dropped={scanned - kept:,}")
    for reason in sorted(drops):
        log(f"       dropped[{reason}] = {drops[reason]:,}")
    log(f"[done] wrote {kept:,} chunks -> {out} "
        f"({os.path.getsize(out) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")
    if kept == 0:
        log("[done] WARNING: zero chunks written -- check the source/config.")
        return 1
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build Simple English Wikipedia intro chunks (RAG bundle chunks.jsonl).",
    )
    p.add_argument("--out", default=DEFAULT_OUT,
                   help=f"output chunks.jsonl path (default: {DEFAULT_OUT})")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"wikimedia/wikipedia config (default: {DEFAULT_CONFIG})")
    p.add_argument("--source", choices=("auto", "datasets", "parquet"), default="auto",
                   help="load path: auto (datasets, else parquet), datasets, or parquet")
    p.add_argument("--parquet-url", default="",
                   help="override the parquet shard URL (single-shard configs only)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE,
                   help=f"where the parquet fallback caches its shard (default: {DEFAULT_CACHE})")
    p.add_argument("--streaming", action="store_true",
                   help="stream via datasets instead of downloading+caching the full split")
    p.add_argument("--batch-size", type=int, default=4096,
                   help="parquet row-group batch size for the fallback reader")
    p.add_argument("--progress-every", type=int, default=20000,
                   help="print a progress line every N scanned articles")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after scanning N articles (0 = all; for smoke tests)")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
