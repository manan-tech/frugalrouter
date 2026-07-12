"""Semantic task router — embeddings, not keyword regex.

WHY THIS EXISTS. The regex classifier is the single point of failure for the
whole agent: every pipeline, the ONNX tagger and the sentiment classifier are all
gated behind it, and it can only recognise phrasings someone thought to enumerate.
Measured: 7 of 18 plausible paraphrases ("Identify the people, organizations and
dates", "What is the overall tone?") fell through to the factual catch-all, so the
task skipped its pipeline AND got escalated in the wrong output format. The finals
re-run on REPHRASED prompts, which is exactly the case regex cannot cover.

Asking the local 0.6B to classify was measured and is WORSE than useless (4/9; it
turned "What is the capital of Australia?" into sentiment) — see classify.py.

Embeddings are the right tool: same meaning, different words, close vectors.
all-MiniLM-L6-v2 (int8 ONNX, 23 MB, ~5ms) embeds the prompt and we take the
category whose exemplars it is closest to.

Structural categories stay with the regex, because embeddings are weak at them and
regex is near-perfect: code_debug/code_gen hinge on literal code being present, and
math on digits plus a computation ask. Those signals are syntactic, not semantic.

Failure-contained: if the model or onnxruntime is missing, route() returns None and
main.py keeps using classify().
"""

import os

from .util import log

MODEL_DIR = os.environ.get("ROUTER_ONNX_DIR", "/models/router")
_SESSION = None
_TOKENIZER = None
_PROTOS = None          # (labels[list[str]], matrix[n, dim] L2-normalised)
_INIT_TRIED = False

# Below this cosine similarity we do not trust the embedding verdict and defer to
# the regex. 0.30 clipped real hits ("Trim this down to the essentials" -> summary
# scored exactly 0.30 and got dropped). Lowering to 0.22 is safe because route() is
# only ever consulted on the regex's factual CATCH-ALL, and a genuine factual still
# matches its own exemplars far higher (0.42-0.52) than any rival category.
_MIN_SIM = 0.22
# The embedding must beat the regex's incumbent "factual" verdict by THIS much
# before we override it. Real paraphrases win by 0.2-0.4; coin flips (f3: 0.005)
# do not. Platform float noise cannot cross a gap this wide.
_MARGIN = 0.06

# Exemplars per category — deliberately varied WORDING for the same INTENT, since
# that is precisely what the router must generalise over.
EXEMPLARS = {
    "ner": [
        "Extract all named entities from the text and label each one.",
        "Identify the people, organizations, locations and dates mentioned.",
        "List every entity in this passage along with its type.",
        "Tag the proper nouns in the following sentence and classify them.",
        "Pull out the names of companies, places and people from this text.",
        "Who and what is mentioned in the passage below? Give each with its kind.",
        "Note down every company, city and person this paragraph refers to.",
    ],
    "sentiment": [
        "Classify the sentiment of this review.",
        "Is the following customer feedback positive, negative or neutral?",
        "What is the overall tone of this statement?",
        "Determine how the writer feels about the product.",
        "Was the customer happy or upset in this message?",
        "How would you say the reviewer feels here?",
        "Judge the opinion expressed in the text below.",
    ],
    "summary": [
        "Summarize the passage in exactly two sentences.",
        "Give a brief overview of the article below.",
        "Condense the following into three bullet points.",
        "Put the following article in a nutshell.",
        "Boil this passage down to its key points.",
        "Write a short recap of the text.",
    ],
    "factual": [
        "What is the difference between RAM and ROM?",
        "Who developed the theory of general relativity?",
        "What is the capital of Australia?",
        "Explain what machine learning is.",
        "What are the primary colours of light and why?",
        "Describe how a refrigerator works.",
    ],
    "logic": [
        "Three friends each play a different sport. Ann does not swim. Who swims?",
        "Four boxes are labelled and every label is wrong. Work out the contents.",
        "Each person lives in a different house and owns a different pet. Deduce who owns the cat.",
        "Solve this deduction puzzle from the constraints given.",
    ],
    "math": [
        "A shop sells 240 units at $3 each. What is the total revenue?",
        "If a train travels 120 km in 2 hours, what is its average speed?",
        "A baker had 50 buns and sold 18. How many remain?",
        "Calculate the final price after a 20 percent discount.",
    ],
    "code_gen": [
        "Write a Python function that reverses a linked list.",
        "Implement a function to check whether a string is a palindrome.",
        "Create a program that merges two sorted lists.",
    ],
    "code_debug": [
        "Fix the bug in this code.",
        "This function returns the wrong output. Correct it.",
        "The code below crashes on an empty list. Debug it.",
    ],
}


def _init():
    global _SESSION, _TOKENIZER, _PROTOS, _INIT_TRIED
    if _INIT_TRIED:
        return
    _INIT_TRIED = True
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model = os.path.join(MODEL_DIR, "model.onnx")
        tok = os.path.join(MODEL_DIR, "tokenizer.json")
        if not (os.path.exists(model) and os.path.exists(tok)):
            log("router: model files absent — regex routing only")
            return
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1     # llama-server needs the other core
        opts.inter_op_num_threads = 1
        _SESSION = ort.InferenceSession(model, opts,
                                        providers=["CPUExecutionProvider"])
        _TOKENIZER = Tokenizer.from_file(tok)

        labels, vecs = [], []
        for cat, texts in EXEMPLARS.items():
            for t in texts:
                labels.append(cat)
                vecs.append(_embed(t))
        _PROTOS = (labels, np.vstack(vecs))
        log(f"router: loaded ({len(labels)} exemplars, {len(EXEMPLARS)} categories)")
    except Exception as e:  # noqa: BLE001
        _SESSION = None
        log(f"router: unavailable ({e}) — regex routing only")


def _embed(text: str):
    """Mean-pooled, L2-normalised sentence embedding.

    The tokenizer PADS to a fixed width, so the real attention_mask is essential:
    masking with ones() averages ~120 [PAD] embeddings into every vector, the
    padding dominates, and every sentence ends up ~0.9 cosine from every other —
    which silently destroys the whole router. Mask, then divide by the true token
    count.
    """
    import numpy as np

    enc = _TOKENIZER.encode(text)
    ids = enc.ids[:256]
    att = enc.attention_mask[:256]              # 1 for real tokens, 0 for [PAD]
    n = len(ids)
    mask = np.array([att], dtype=np.int64)
    feed = {}
    for inp in _SESSION.get_inputs():
        if inp.name == "input_ids":
            feed[inp.name] = np.array([ids], dtype=np.int64)
        elif inp.name == "attention_mask":
            feed[inp.name] = mask
        elif inp.name == "token_type_ids":
            feed[inp.name] = np.zeros((1, n), dtype=np.int64)
    out = _SESSION.run(None, feed)[0]           # (1, seq, dim)
    m = mask[..., None].astype(np.float32)
    vec = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)   # mean-pool
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return (vec / np.clip(norm, 1e-9, None)).astype(np.float32)


def available() -> bool:
    _init()
    return _SESSION is not None and _PROTOS is not None


def _instruction(prompt: str) -> str:
    """Just the ASK, not the payload.

    Real tasks are "<instruction>: <long passage>". Embedding the whole thing lets
    the passage dominate the vector and the instruction signal washes out (measured:
    13/19 on the real suites vs 19/19 for regex). Cutting at the first colon /
    newline / sentence end recovers the instruction, which is what we match on.
    """
    head = prompt.strip()
    for sep in (":", "\n"):
        i = head.find(sep)
        if 12 < i < 200:
            head = head[:i]
            break
    else:
        i = head.find(". ")
        if 12 < i < 200:
            head = head[:i]
    return head[:200]


def route(prompt: str):
    """-> (category, similarity) or None when the router can't decide/isn't loaded."""
    if not available():
        return None
    try:
        import numpy as np

        v = _embed(_instruction(prompt))             # (1, dim), normalised
        labels, protos = _PROTOS
        sims = (protos @ v.T).ravel()                # cosine, both normalised

        # best similarity PER CATEGORY (each has several exemplars)
        best = {}
        for lab, s in zip(labels, sims):
            f = float(s)
            if f > best.get(lab, -1.0):
                best[lab] = f
        top = max(best, key=best.get)
        if best[top] < _MIN_SIM:
            return None                              # too far from everything

        # MARGIN RULE. route() is only consulted AFTER the regex said "factual",
        # so factual is the incumbent — override it only on a CLEAR win, never on
        # a rounding error. Measured: f3 ("What does HTTP stand for, and what is
        # it used for?") scored summary 0.215 vs factual 0.210 — a 0.005 gap, and
        # amd64-vs-arm64 float noise flipped it, misrouting a factual task into
        # the summary pipeline. A genuine paraphrase wins by a mile (ner and
        # sentiment beat factual by 0.2-0.4), so the margin costs real rescues
        # nothing and kills the coin flips.
        if top != "factual" and (best[top] - best.get("factual", 0.0)) < _MARGIN:
            return None                              # too close to call — keep factual
        return top, best[top]
    except Exception as e:  # noqa: BLE001
        log(f"router: inference failed ({e}) — regex routing only")
        return None
