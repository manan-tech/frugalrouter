"""Sentiment label from a purpose-trained classifier; the LLM only writes prose.

Measured: the 0.6B gets the LABEL wrong (it called a both-good-and-bad review
"Negative" when it was Mixed, and a flat factual line "Positive" when it was
Neutral) — and few-shot exemplars did not move it. But the *reason* it writes is
usually fine. So split the job: a roberta sentiment classifier decides the label,
and the LLM is asked only to justify that fixed label.

MIXED is not one of the model's classes, so it is derived: split the text into
clauses (sentences, plus contrast markers like "but"/"however") and classify each.
If some clause is clearly positive AND another is clearly negative, the text is
Mixed — which is exactly the shape the LLM kept mislabelling.

Failure-contained: if onnxruntime/tokenizers/model files are missing or anything
raises, classify() returns None and pipelines.sentiment() falls back to the pure
LLM path unchanged.
"""

import os
import re

from agent_core import BASE_DIR
from agent_core.util import log

MODEL_DIR = os.environ.get("SENTIMENT_ONNX_DIR",
                           os.path.join(BASE_DIR, "models", "sentiment"))
_SESSION = None
_TOKENIZER = None
_ID2LABEL = {}
_INIT_TRIED = False

# A clause is a sentence, or a span split off by a contrast marker — "The food
# was great BUT the service was slow" is one sentence holding both polarities.
_CLAUSE_SPLIT = re.compile(
    r"(?:(?<=[.!?])\s+)|(?:[,;]?\s+(?:but|however|although|though|whereas|"
    r"unfortunately|on the other hand)\s+)", re.IGNORECASE)

# An explicit contrast marker: the author is signalling a counterpoint even when
# the counterpoint carries no negative *words*.
_CONTRAST_RE = re.compile(
    r"\b(but|however|although|though|whereas|unfortunately|"
    r"on the other hand|that said)\b", re.IGNORECASE)

# A clause must be at least this confident to count as evidence for Mixed;
# otherwise a mildly-worded aside would flip a clear review to Mixed.
_STRONG = 0.60


def _init():
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
            log("sentiment_onnx: model files absent — using the LLM path")
            return
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1     # llama-server needs the other core
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(model, opts,
                                       providers=["CPUExecutionProvider"])
        with open(cfg, encoding="utf-8") as f:
            id2label = {int(k): v.lower()
                        for k, v in json.load(f)["id2label"].items()}
        _SESSION, _TOKENIZER, _ID2LABEL = session, Tokenizer.from_file(tok), id2label
        log(f"sentiment_onnx: loaded ({sorted(id2label.values())})")
    except Exception as e:  # noqa: BLE001
        log(f"sentiment_onnx: unavailable ({e}) — using the LLM path")


def available() -> bool:
    _init()
    return _SESSION is not None


def _score(text: str):
    """(label, confidence) for one span, via softmax over the model's logits."""
    import numpy as np

    enc = _TOKENIZER.encode(text)
    ids = enc.ids[:512]
    n = len(ids)
    feed = {}
    for inp in _SESSION.get_inputs():
        if inp.name == "input_ids":
            feed[inp.name] = np.array([ids], dtype=np.int64)
        elif inp.name == "attention_mask":
            feed[inp.name] = np.ones((1, n), dtype=np.int64)
        elif inp.name == "token_type_ids":
            feed[inp.name] = np.zeros((1, n), dtype=np.int64)
    logits = _SESSION.run(None, feed)[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    i = int(probs.argmax())
    return _ID2LABEL.get(i, "neutral"), float(probs[i])


def classify(text: str):
    """-> "Positive" | "Negative" | "Neutral" | "Mixed", or None if unavailable."""
    if not available():
        return None
    try:
        whole, _conf = _score(text)

        clauses = [c.strip() for c in _CLAUSE_SPLIT.split(text) if c and len(c.strip()) > 12]
        pos = neg = False
        labels = set()
        if len(clauses) > 1:
            for c in clauses:
                lab, cf = _score(c)
                labels.add(lab)
                if cf < _STRONG:
                    continue
                if lab == "positive":
                    pos = True
                elif lab == "negative":
                    neg = True

        # (a) both polarities stated outright -> unambiguously Mixed.
        if pos and neg:
            return "Mixed"

        # (b) an explicit contrast marker ("but", "however", "although") whose
        # clauses do NOT share one polarity. This catches drawbacks phrased
        # WITHOUT negative words — "The food was delicious BUT we waited nearly
        # an hour" scores the complaint as *neutral* (no lexical negativity), so
        # rule (a) misses it. The "but" is the signal. Requiring a real polarity
        # somewhere keeps flat factual sentences ("it weighs 300g but ships in a
        # box" -> all neutral) out of Mixed.
        if (_CONTRAST_RE.search(text) and len(labels) > 1
                and ("positive" in labels or "negative" in labels)):
            return "Mixed"

        return {"positive": "Positive", "negative": "Negative",
                "neutral": "Neutral"}.get(whole, "Neutral")
    except Exception as e:  # noqa: BLE001
        log(f"sentiment_onnx: inference failed ({e}) — using the LLM path")
        return None
