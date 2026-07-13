"""Retrieval over a baked int8 corpus — reuses the router's MiniLM ONNX embedder.

WHY THIS EXISTS. Some tasks are answerable from a small local knowledge corpus
that we bake into the image at build time; retrieving the right passage lets the
agent ground its answer locally instead of paying Fireworks to recall the fact.

NO ONNX STATE OF ITS OWN. Query embedding delegates to `router._embed`, which runs
on the ONE all-MiniLM-L6-v2 session already loaded from /models/router. That costs
zero extra RAM (the router is core to the main flow, so its session is resident)
and — critically — guarantees the real-attention_mask pooling. A duplicated pooler
is exactly how the ones() [PAD]-averaging bug (every cosine ~0.94) gets silently
re-introduced. One embedder, one session.

THE BAKED BUNDLE (offline writer and this reader must agree byte-for-byte):
  /models/rag/chunks.jsonl : one JSON per line {"id": <0-based line idx>,
                             "title": str, "text": str}
  /models/rag/emb_int8.npy : int8, shape (N, 384), C-contiguous. Row i is
                             quantize(L2normalize(embed(chunk_i.text))), where
                             quantize(v) = clip(round(v*127), -127, 127).
  /models/rag/meta.json    : {"n": N, "dim": 384, "model": "all-MiniLM-L6-v2",
                             "quant": "int8x127"}
Cosine(query, chunk_i) = dot(L2normalize(embed(query)), emb_int8[i]/127.0).
N is ~100k-300k, so emb_int8.npy is ~40-115 MB — it is loaded as a numpy MEMMAP,
never fully into RAM, and scored with a chunked matmul so peak RAM stays flat.

FAILURE-CONTAINED (mirrors ner_onnx.py / router.py exactly): _init() never raises
and sets state on success only; available() is False if onnxruntime/tokenizers or
any bundle file is missing or anything raises; retrieve() returns None and the
caller falls back. An optional accelerator must never break the run.
"""

import os

from agent_core import BASE_DIR, router
from agent_core.util import log

MODEL_DIR = os.environ.get("RAG_DIR",
                           os.path.join(BASE_DIR, "models", "rag"))

_EMB = None            # np.memmap (N, 384) int8 — the baked corpus, never in RAM
_CHUNKS = None         # list[dict]: line i (== id i) -> {"title", "text", ...}
_META = None           # {"n", "dim", "model", "quant"}
_INIT_TRIED = False

_MATMUL_ROWS = 8192    # rows dequantised per chunked-matmul block (~12 MB float32)


def _init():
    """Lazy one-shot load. Never raises — sets _EMB/_CHUNKS/_META on success only."""
    global _EMB, _CHUNKS, _META, _INIT_TRIED
    if _INIT_TRIED:
        return
    _INIT_TRIED = True
    try:
        import json

        import numpy as np

        # The query embedder is the router's session; no embedder => no retrieval.
        if not router.available():
            log("rag: query embedder unavailable — retrieval off")
            return

        emb_path = os.path.join(MODEL_DIR, "emb_int8.npy")
        chunks_path = os.path.join(MODEL_DIR, "chunks.jsonl")
        meta_path = os.path.join(MODEL_DIR, "meta.json")
        if not all(os.path.exists(p) for p in (emb_path, chunks_path, meta_path)):
            log("rag: bundle files absent — retrieval off")
            return

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        chunks = []
        bad = 0
        with open(chunks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:                     # blank lines get no emb row on
                    continue                     # the writer side either — skip
                try:
                    chunks.append(json.loads(line))
                except Exception:                # noqa: BLE001
                    # One malformed line must not discard the whole 240k-chunk
                    # corpus. Keep a placeholder so row<->line alignment (and thus
                    # every LATER chunk's id) is preserved; the dead chunk just
                    # retrieves empty text.
                    chunks.append({"title": "", "text": ""})
                    bad += 1
        if bad:
            log(f"rag: {bad} malformed chunk line(s) kept as empty placeholders")

        # mmap_mode='r' keeps the (N, 384) int8 matrix on disk — never resident.
        emb = np.load(emb_path, mmap_mode="r")

        # Row order must align 1:1 with chunks.jsonl line order (row i <-> id i);
        # a mismatch means the bundle is corrupt and every score would be wrong.
        # The dim check catches a right-count/wrong-width bundle at load (with a
        # clear log) instead of failing silently on every retrieve()'s matmul.
        exp_dim = int(meta.get("dim", 384))
        if (emb.ndim != 2 or emb.shape[0] != len(chunks)
                or emb.shape[1] != exp_dim):
            log(f"rag: shape mismatch (emb {emb.shape} vs {len(chunks)} chunks, "
                f"dim {exp_dim}) — off")
            return

        _EMB, _CHUNKS, _META = emb, chunks, meta
        log(f"rag: loaded ({len(chunks)} chunks, dim {emb.shape[1]}, int8 memmap)")
    except Exception as e:  # noqa: BLE001 — an optional accelerator must never break the run
        log(f"rag: unavailable ({e}) — retrieval off")


def available() -> bool:
    _init()
    return _EMB is not None and _CHUNKS is not None


def embed_query(text: str):
    """Embed one query/passage into a unit-norm float32 (384,) vector, or None.

    Reuses router._embed (the ONE MiniLM session) so the pooling uses the real
    attention_mask. Returns None when the embedder is unavailable or errors — the
    vector is L2-normalised, so a plain dot product with a chunk row IS cosine.
    """
    if not router.available():
        return None
    try:
        import numpy as np

        vec = router._embed(text)                              # (1, 384) f32, unit-norm
        return np.ascontiguousarray(vec[0], dtype=np.float32)  # -> (384,)
    except Exception as e:  # noqa: BLE001
        log(f"rag: embed failed ({e}) — caller falls back")
        return None


# MiniLM is a symmetric encoder, so passages embed on the identical path.
embed_passage = embed_query


def select_mmr(cand_embs, sim_q, k: int, lam: float):
    """Maximal Marginal Relevance selection for a RAG diversity re-rank.

    Greedily selects the k chunks that are relevant to the query yet mutually
    distinct:  MMR(c) = lam*sim_q[c] - (1-lam)*max_{s in selected} sim(c, s).
    cand_embs are (approximately) L2-normalised, so chunk-chunk cosine == dot.
    Returns candidate indices (into 0..M-1) in selection order: index 0 is the
    most-relevant anchor, each later index the most distinct given those chosen.
    """
    import numpy as np

    cand_embs = np.asarray(cand_embs, dtype=np.float64)
    sim_q = np.asarray(sim_q, dtype=np.float64).ravel()
    M = cand_embs.shape[0]

    if sim_q.shape[0] != M:
        raise ValueError(f"sim_q has {sim_q.shape[0]} entries but cand_embs has {M} rows")

    if k <= 0 or M == 0:                     # nothing to select / to select from
        return []
    k = min(k, M)                            # can't keep more chunks than exist
    lam = float(np.clip(lam, 0.0, 1.0))      # tolerate an out-of-range lambda

    selected = []
    # Running max similarity of every candidate to the already-selected set,
    # floored at the cosine lower bound (-1.0), NOT 0.0. Pick #1 is unaffected:
    # with S empty every candidate shares the same constant redundancy (-1), so
    # the first choice still reduces to argmax(sim_q). But 0.0 would clamp the
    # redundancy of a candidate that is ANTI-correlated with the chosen set up to
    # 0 — robbing the most-diverse chunks of the diversity credit they've earned,
    # the exact opposite of what this re-rank is for. -1.0 lets the true
    # (possibly negative) redundancy through from pick #2 onward.
    max_sim_to_sel = np.full(M, -1.0, dtype=np.float64)
    remaining = np.ones(M, dtype=bool)

    for _ in range(k):
        mmr = lam * sim_q - (1.0 - lam) * max_sim_to_sel
        mmr[~remaining] = -np.inf            # never re-pick a selected chunk
        c = int(np.argmax(mmr))              # argmax breaks ties -> lowest index
        selected.append(c)
        remaining[c] = False
        # Incremental redundancy update (O(k*M*d), no M*M matrix). Self-sim = 1 is
        # harmless: c is already masked out of `remaining`.
        sim_to_c = cand_embs @ cand_embs[c]
        max_sim_to_sel = np.maximum(max_sim_to_sel, sim_to_c)

    return selected


def retrieve(query: str, k: int = 3, pool: int = 20, lam: float = 0.6):
    """Retrieve k diverse, relevant chunks for `query`, or None if unavailable.

    Embeds the query, scores it against ALL chunk rows (int8 dequantised /127.0 in
    a chunked matmul so peak RAM stays small), takes the top `pool` by cosine, then
    MMR-re-ranks those down to k DISTINCT chunks. Returns a list of
    {"title", "text", "score"} ordered by MMR selection, where `score` is the
    chunk's RAW query-cosine (not the MMR score) — the caller uses the max score as
    a retrieval-confidence gate.
    """
    if not available():
        return None
    try:
        import numpy as np

        v = embed_query(query)               # (384,) f32 unit-norm, or None
        if v is None:
            return None

        N = _EMB.shape[0]
        if N == 0 or k <= 0:
            return []

        # Cosine vs every chunk, one memmap block at a time (never dequantise the
        # whole (N, 384) matrix — that would defeat the memmap).
        sims = np.empty(N, dtype=np.float32)
        for i in range(0, N, _MATMUL_ROWS):
            block = np.asarray(_EMB[i:i + _MATMUL_ROWS], dtype=np.float32) / 127.0
            sims[i:i + block.shape[0]] = block @ v

        # Top `pool` candidates by cosine, sorted descending (argpartition then
        # sort only the pool — O(N) select, not an O(N log N) full sort).
        p = min(pool, N)
        idx = np.argpartition(-sims, p - 1)[:p]
        idx = idx[np.argsort(-sims[idx])]

        cand_embs = np.asarray(_EMB[idx], dtype=np.float32) / 127.0   # (p, 384)
        sim_q = sims[idx].astype(np.float64)                          # (p,)

        order = select_mmr(cand_embs, sim_q, k, lam)

        out = []
        for j in order:
            ch = _CHUNKS[int(idx[j])]
            out.append({
                "title": ch.get("title", ""),
                "text": ch.get("text", ""),
                # code-exemplar bundles carry the implementation alongside the
                # embedded description; absent (e.g. a text corpus) it's "".
                "code": ch.get("code", ""),
                "score": float(sim_q[j]),      # raw query-cosine, NOT the mmr score
            })
        return out
    except Exception as e:  # noqa: BLE001
        log(f"rag: retrieve failed ({e}) — caller falls back")
        return None
