#!/usr/bin/env python3
"""Offline corpus embedder for the RAG bundle — the WRITE side of the contract.

WHY THIS EXISTS
    agent/rag.py (the runtime READER) embeds a query with router._embed — the ONE
    all-MiniLM-L6-v2 int8 session already resident from /models/router — and scores
    it against a precomputed matrix of chunk vectors. This script is what PRODUCES
    that matrix, offline, once, on a fat box. Query and chunk vectors must live in
    the SAME space, so the embedding math here has to match router._embed BYTE FOR
    BYTE — same tokenizer, same 256-token cap, same feed built by iterating the
    session's declared inputs, same mean-pool over the REAL attention_mask, same L2
    normalisation. The only difference is speed: we PAD WITHIN A BATCH and run many
    chunks per session call. MiniLM is a symmetric encoder, so a passage embedded
    here == the same string embedded by router._embed at query time.

THE attention_mask BUG (the whole reason batching is delicate)
    router.py has the lesson baked in: masking with ones() averages ~120 [PAD]
    vectors into every sentence and collapses every cosine to ~0.94. Batching
    RE-INTRODUCES padding on purpose (short rows padded up to the batch's longest
    row), so the real per-row attention_mask is non-negotiable here: 1 for real
    tokens, 0 for [PAD]. We feed it to the model (so padded positions are masked in
    self-attention and real-token outputs are identical to single-row inference) AND
    we mean-pool with it (so padding contributes exactly zero). Never ones().

BATCH SIZE = 1 BY DEFAULT — measured, not assumed
    The task asked for BATCHED embedding "for speed". Measured against the SHIPPING
    int8 model.onnx + this tokenizer, two facts overturn that premise:
      1. Batching does NOT speed anything up. /models/router's tokenizer.json has
         Fixed(128) padding baked in, so EVERY row is already a full 128-wide
         forward pass — batch=1 vs batch=64 both run ~130 chunks/s (1 thread) /
         ~197 (2 threads). There is no short-sequence padding to amortise and the
         per-call overhead is noise next to a 128-token pass. Threads, not batch
         size, are the throughput lever.
      2. Batching CHANGES the vectors. onnxruntime's dynamically-quantized int8
         GEMM kernels are batch-size-dependent: a row embedded at batch=64 lands
         cos~0.9946 from the same row at batch=1, a drift LARGER than int8x127
         quantisation's own ~0.999 noise floor. The query is embedded single-row by
         router._embed at runtime, so a batched corpus would sit in a subtly
         DIFFERENT numeric space than the queries it must be compared with —
         exactly what "same space" forbids. At batch=1, _embed_batch reproduces
         router._embed BYTE-FOR-BYTE (cos 1.0, identical int8 rows).
    So the default is 1: exact AND no slower. --batch-size >1 stays available (it
    helps models whose tokenizer pads to `longest`, not `fixed`) but WARNS, because
    with this bundle it only trades accuracy for nothing.

BUNDLE FORMAT (must match agent/rag.py's reader exactly)
    Reads   /models/rag/chunks.jsonl : one JSON/line {"id": <0-based line idx>,
            "title": <str>, "text": <str, <=~150 words>}. Only `text` is embedded.
    Writes  /models/rag/emb_int8.npy : int8, shape (N, 384), C-contiguous. Row i is
            quantize(L2normalize(embed(text_i))), quantize(v)=clip(round(v*127),
            -127, 127). Row order == chunks.jsonl line order (row i <-> id i).
    Writes  /models/rag/meta.json    : {"n": N, "dim": 384,
            "model": "all-MiniLM-L6-v2", "quant": "int8x127"}
    At query time: cosine = dot(L2normalize(embed(q)), emb_int8[i].astype(f32)/127).

    N is ~100k-300k, so emb_int8.npy is ~40-115 MB. We write it as a real .npy via
    np.lib.format.open_memmap and fill it row-block by row-block, so peak RAM is one
    batch of activations, not the whole matrix — and the runtime reader memmaps it.

RESUMABLE
    Skips if emb_int8.npy AND meta.json both exist and are newer than chunks.jsonl,
    unless --force. The matrix is written to a .tmp and atomically renamed, and
    meta.json is written LAST, so an interrupted run never leaves a file that would
    be mistaken for a finished bundle.

DEPS: python stdlib + numpy + onnxruntime + tokenizers (same set the image already
    has). No agent/ import — this runs on the build/prep box, not in the container.
"""

import argparse
import json
import os
import sys
import time

DIM = 384
MAX_TOKENS = 256            # identical cap to router._embed (ids[:256])
MODEL_NAME = "all-MiniLM-L6-v2"
QUANT = "int8x127"
PROGRESS_EVERY = 2000       # log a progress line roughly every this many chunks


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _up_to_date(emb_path: str, meta_path: str, chunks_path: str) -> bool:
    """True iff a finished bundle already exists and is newer than the input."""
    if not (os.path.exists(emb_path) and os.path.exists(meta_path)):
        return False
    src = os.path.getmtime(chunks_path)
    return os.path.getmtime(emb_path) >= src and os.path.getmtime(meta_path) >= src


def _count_lines(path: str) -> int:
    """Number of non-blank JSONL records — one embedding row each."""
    n = 0
    with open(path, "rb") as f:
        for raw in f:
            if raw.strip():
                n += 1
    return n


def _iter_records(path: str):
    """Yield (line_index, text) for each non-blank line, in file order.

    line_index is the 0-based row this record maps to (== chunks.jsonl `id` by
    contract). We key rows off line ORDER, not the `id` field, and only warn on a
    mismatch — the format spec says they agree, but order is the load-bearing rule.
    """
    idx = 0
    warned = False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            if not warned and obj.get("id") != idx:
                _log(f"  WARN: chunks.jsonl id={obj.get('id')!r} != line index "
                     f"{idx} — using LINE ORDER for row assignment (warned once)")
                warned = True
            yield idx, str(obj.get("text", ""))
            idx += 1


def _load_session(model_dir: str, intra: int, inter: int):
    """all-MiniLM-L6-v2 int8 ONNX + its tokenizer — same files router.py loads."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    model = os.path.join(model_dir, "model.onnx")
    tok = os.path.join(model_dir, "tokenizer.json")
    if not (os.path.exists(model) and os.path.exists(tok)):
        raise FileNotFoundError(
            f"embedder files missing under {model_dir} "
            f"(need model.onnx + tokenizer.json)")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = intra
    opts.inter_op_num_threads = inter
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])
    return session, Tokenizer.from_file(tok)


def _embed_batch(session, input_names, tokenizer, texts):
    """Batched, padded, real-mask mean-pooled, L2-normalised — (B, 384) float32.

    Identical math to router._embed, one row at a time -> a whole batch: each text
    tokenised and capped at 256, rows right-padded to the batch's longest row with
    [PAD]=0, the REAL attention_mask (1=token / 0=pad) fed to the model AND used for
    the mean-pool. Padded positions therefore never touch a real-token embedding.
    """
    import numpy as np

    encs = tokenizer.encode_batch(list(texts))
    rows = [(e.ids[:MAX_TOKENS], e.attention_mask[:MAX_TOKENS]) for e in encs]
    width = max((len(ids) for ids, _ in rows), default=1) or 1
    b = len(rows)

    ids_arr = np.zeros((b, width), dtype=np.int64)          # [PAD] id == 0
    mask_arr = np.zeros((b, width), dtype=np.int64)         # 0 for [PAD]
    for r, (ids, att) in enumerate(rows):
        k = len(ids)
        ids_arr[r, :k] = ids
        mask_arr[r, :k] = att                               # real mask, never ones()

    feed = {}
    for name in input_names:
        if name == "input_ids":
            feed[name] = ids_arr
        elif name == "attention_mask":
            feed[name] = mask_arr
        elif name == "token_type_ids":                      # omitted by some exports
            feed[name] = np.zeros((b, width), dtype=np.int64)

    out = session.run(None, feed)[0]                        # (B, seq, 384)
    m = mask_arr[..., None].astype(np.float32)
    vec = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)   # mean-pool
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return (vec / np.clip(norm, 1e-9, None)).astype(np.float32)


def _quantize(vec):
    """int8x127: clip(round(v*127), -127, 127). v is unit-norm, so v*127 in range."""
    import numpy as np
    return np.clip(np.rint(vec * 127.0), -127, 127).astype(np.int8)


def embed_corpus(chunks_path, out_dir, model_dir, batch_size,
                 intra_threads, inter_threads, force):
    import numpy as np

    emb_path = os.path.join(out_dir, "emb_int8.npy")
    meta_path = os.path.join(out_dir, "meta.json")

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")
    if not force and _up_to_date(emb_path, meta_path, chunks_path):
        _log(f"up to date — {emb_path} + meta.json newer than input; "
             f"pass --force to rebuild")
        return emb_path

    os.makedirs(out_dir, exist_ok=True)

    if batch_size > 1:
        _log(f"WARN: --batch-size {batch_size} > 1. onnxruntime's int8 GEMM is "
             f"batch-size-dependent, so rows drift ~cos0.995 from the single-row "
             f"space router._embed produces at query time — and with this fixed-pad "
             f"tokenizer batching buys NO speedup. Use 1 unless you know why.")

    _log(f"counting records in {chunks_path} ...")
    n = _count_lines(chunks_path)
    if n == 0:
        raise ValueError(f"{chunks_path} has no records")
    _log(f"  {n} chunks -> emb_int8.npy shape ({n}, {DIM}) int8 "
         f"(~{n * DIM / 1e6:.0f} MB)")

    _log(f"loading embedder from {model_dir} "
         f"(intra={intra_threads}, inter={inter_threads}) ...")
    session, tokenizer = _load_session(model_dir, intra_threads, inter_threads)
    input_names = [i.name for i in session.get_inputs()]
    _log(f"  session inputs: {input_names}")

    # Write into a real .npy (correct header) via a memmap, so peak RAM is one
    # batch, not the whole matrix. Fill a .tmp then atomically rename.
    tmp_path = emb_path + ".tmp"
    mm = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.int8, shape=(n, DIM))

    t0 = time.monotonic()
    written = 0
    next_log = PROGRESS_EVERY
    batch_idx, batch_txt = [], []

    def _flush():
        nonlocal written
        if not batch_txt:
            return
        vecs = _embed_batch(session, input_names, tokenizer, batch_txt)
        q = _quantize(vecs)
        for row, out_row in zip(batch_idx, q):
            mm[row] = out_row               # scatter by line index: order-independent
        written += len(batch_txt)
        batch_idx.clear()
        batch_txt.clear()

    for idx, text in _iter_records(chunks_path):
        batch_idx.append(idx)
        batch_txt.append(text)
        if len(batch_txt) >= batch_size:
            _flush()
            if written >= next_log:
                rate = written / max(time.monotonic() - t0, 1e-9)
                eta = (n - written) / max(rate, 1e-9)
                _log(f"  {written}/{n} ({100 * written / n:5.1f}%)  "
                     f"{rate:6.0f} chunks/s  ETA {eta / 60:5.1f} min")
                next_log += PROGRESS_EVERY
    _flush()

    if written != n:
        mm.flush()
        del mm
        os.remove(tmp_path)
        raise RuntimeError(
            f"embedded {written} rows but counted {n} — refusing to write a "
            f"misaligned bundle")

    mm.flush()
    del mm                                      # release the memmap before rename
    os.replace(tmp_path, emb_path)              # atomic: partial never at final path

    meta = {"n": n, "dim": DIM, "model": MODEL_NAME, "quant": QUANT}
    meta_tmp = meta_path + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(meta_tmp, meta_path)             # written LAST -> signals completion

    dt = time.monotonic() - t0
    _log(f"done: {n} chunks in {dt / 60:.1f} min "
         f"({n / max(dt, 1e-9):.0f} chunks/s)")
    _log(f"  {emb_path}  ({os.path.getsize(emb_path) / 1e6:.1f} MB)")
    _log(f"  {meta_path}  {json.dumps(meta)}")
    return emb_path


def main():
    ap = argparse.ArgumentParser(
        description="Embed a RAG corpus into the int8x127 bundle "
                    "(agent/rag.py reads it).")
    ap.add_argument("--chunks", default="/models/rag/chunks.jsonl",
                    help="input JSONL: {id,title,text} per line (default: %(default)s)")
    ap.add_argument("--out-dir", default="/models/rag",
                    help="write emb_int8.npy + meta.json here (default: %(default)s)")
    ap.add_argument("--model-dir", default="/models/router",
                    help="all-MiniLM-L6-v2 int8 ONNX dir (default: %(default)s)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="chunks per session call. KEEP AT 1: int8 GEMM is "
                         "batch-dependent (>1 drifts from the runtime query space) "
                         "and this fixed-pad tokenizer makes batching no faster "
                         "(default: %(default)s)")
    ap.add_argument("--intra-threads", type=int, default=2,
                    help="onnxruntime intra_op threads (default: %(default)s)")
    ap.add_argument("--inter-threads", type=int, default=2,
                    help="onnxruntime inter_op threads (default: %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if outputs look up to date")
    args = ap.parse_args()

    path = embed_corpus(
        chunks_path=args.chunks,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
        force=args.force,
    )
    print(path)


if __name__ == "__main__":
    main()
