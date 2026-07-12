"""Tiny ONNX NER — a purpose-trained BERT beats a 0.6B LLM at entity tagging.

dslim/bert-base-NER (Xenova ONNX int8, ~108 MB) tags PER/ORG/LOC/MISC. It does
NOT tag DATE (CoNLL-2003 label set), so dates come from a regex pass and the two
are merged. Runs on CPU in ~30ms — no wall-clock cost, no Fireworks tokens.

Everything here is failure-contained: if onnxruntime/tokenizers/model files are
missing or anything raises, available() is False and pipelines.ner() falls back
to the LLM path exactly as before.
"""

import os
import re

from .util import log

MODEL_DIR = os.environ.get("NER_ONNX_DIR", "/models/ner")
_SESSION = None
_TOKENIZER = None
_ID2LABEL = {}
_INIT_TRIED = False

# CoNLL types -> the label vocabulary our judge/answer format expects.
_TYPE_MAP = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION"}
# MISC is deliberately dropped: it is noisy (nationalities, adjectives) and our
# format has no matching bucket — a wrong label costs more than a missing one.

# DATE is not in the model's label set, so recognise it lexically. Ordered
# longest-first; overlapping matches are suppressed by span containment.
_MONTH = (r"January|February|March|April|May|June|July|August|September|"
          r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec")
_DATE_RES = [
    re.compile(rf"\b(?:{_MONTH})\s+\d{{1,2}},?\s+\d{{4}}\b"),   # March 5, 2022
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTH})\s+\d{{4}}\b"),      # 5 March 2022
    re.compile(rf"\b(?:{_MONTH})\s+\d{{4}}\b"),                  # March 2022
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                        # 2022-03-05
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                  # 03/05/2022
    re.compile(rf"\b(?:{_MONTH})\s+\d{{1,2}}\b"),                # March 5
    re.compile(r"\b(?:last|next|this)\s+(?:year|month|week|"
               rf"{_MONTH})\b", re.IGNORECASE),                  # last April
    re.compile(r"\b(?:19|20)\d{2}\b"),                           # bare year
]


def _init():
    """Lazy one-shot load. Never raises — sets _SESSION on success only."""
    global _SESSION, _TOKENIZER, _ID2LABEL, _INIT_TRIED
    if _INIT_TRIED:
        return
    _INIT_TRIED = True
    try:
        import json

        import onnxruntime as ort
        from tokenizers import Tokenizer

        model = os.path.join(MODEL_DIR, "model.onnx")
        tok = os.path.join(MODEL_DIR, "tokenizer.json")
        cfg = os.path.join(MODEL_DIR, "config.json")
        if not all(os.path.exists(p) for p in (model, tok, cfg)):
            log("ner_onnx: model files absent — falling back to the LLM path")
            return

        opts = ort.SessionOptions()
        # 2 vCPUs total and llama-server wants them — never oversubscribe.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model, opts,
                                       providers=["CPUExecutionProvider"])
        with open(cfg, encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

        _SESSION, _TOKENIZER, _ID2LABEL = session, Tokenizer.from_file(tok), id2label
        log(f"ner_onnx: loaded ({len(id2label)} labels, 1 thread)")
    except Exception as e:  # noqa: BLE001 — an optional accelerator must never break the run
        log(f"ner_onnx: unavailable ({e}) — falling back to the LLM path")


def available() -> bool:
    _init()
    return _SESSION is not None


def _model_spans(text: str):
    """BIO-decode the model's logits into (surface, TYPE) using CHAR OFFSETS —
    slicing the original text avoids all '##' subword-merge bugs."""
    import numpy as np

    enc = _TOKENIZER.encode(text)
    ids = enc.ids[:512]                      # BERT hard limit
    offsets = enc.offsets[:512]
    n = len(ids)
    feed = {}
    for inp in _SESSION.get_inputs():
        if inp.name == "input_ids":
            feed[inp.name] = np.array([ids], dtype=np.int64)
        elif inp.name == "attention_mask":
            feed[inp.name] = np.ones((1, n), dtype=np.int64)
        elif inp.name == "token_type_ids":
            feed[inp.name] = np.zeros((1, n), dtype=np.int64)
    logits = _SESSION.run(None, feed)[0][0]  # (seq, num_labels)
    pred = logits.argmax(-1)

    spans, cur, start, end = [], None, 0, 0
    for i in range(n):
        s, e = offsets[i]
        if s == e:                            # special token ([CLS]/[SEP])
            continue
        tag = _ID2LABEL.get(int(pred[i]), "O")
        if tag == "O":
            if cur:
                spans.append((cur, start, end))
                cur = None
            continue
        pos, _, typ = tag.partition("-")
        typ = _TYPE_MAP.get(typ)
        if typ is None:                       # MISC — ignore
            if cur:
                spans.append((cur, start, end))
                cur = None
            continue
        if pos == "B" or cur != typ:
            if cur:
                spans.append((cur, start, end))
            cur, start, end = typ, s, e
        else:                                 # I- continuing the same type
            end = e
    if cur:
        spans.append((cur, start, end))
    return [(text[s:e].strip(), t) for t, s, e in spans if text[s:e].strip()]


def _date_spans(text: str):
    out, taken = [], []
    for rx in _DATE_RES:
        for m in rx.finditer(text):
            s, e = m.span()
            if any(s >= ts and e <= te for ts, te in taken):
                continue                      # inside a longer date already found
            taken.append((s, e))
            out.append((m.group(0).strip(), "DATE"))
    return out


def extract(text: str):
    """[(surface, TYPE), …] or None when the ONNX path is unavailable."""
    if not available():
        return None
    try:
        ents = _model_spans(text) + _date_spans(text)
        seen, out = set(), []
        for surface, typ in ents:
            key = (surface.lower(), typ)
            if surface and key not in seen:
                seen.add(key)
                out.append((surface, typ))
        return out
    except Exception as e:  # noqa: BLE001
        log(f"ner_onnx: inference failed ({e}) — falling back to the LLM path")
        return None
