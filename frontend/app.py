"""FrugalRouter demo Space — the agent's zero-token local machinery, interactive.

Four tabs, each backed by the exact vendored module the competition agent runs
inside its container (see agent_core/):

  Router          agent_core.router          MiniLM int8 embeddings over category
                                             exemplars (adjudicates the regex
                                             catch-all in production)
  NER             agent_core.ner_onnx        OntoNotes BERT int8, char-offset BIO
                                             decoding + date backstop
  Sentiment       agent_core.sentiment_onnx  RoBERTa int8 + clause-level Mixed rule
  Code Retrieval  agent_core.rag             MMR-diverse retrieval over 204 code
                                             exemplars, embedded at startup with
                                             the SAME MiniLM session as the router

Everything runs on CPU with 1 onnxruntime thread per session (free-tier safe).
Models are fetched on first boot by download_models.ensure_models(); if any
model is missing, that tab degrades to a clear failure message instead of
crashing the app.
"""

import json
import os
import sys
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ---- fetch models before the agent modules probe for them (idempotent) -------
try:
    from download_models import ensure_models

    ensure_models()
except Exception as e:  # noqa: BLE001 — each tab reports its own absence
    print(f"[app] model download failed: {e}", file=sys.stderr)

import gradio as gr
import numpy as np

from agent_core import ner_onnx, rag, router, sentiment_onnx

CHUNKS_PATH = os.path.join(BASE_DIR, "data", "chunks.jsonl")


def _missing(name: str, model_dir: str) -> str:
    return (f"**{name} model is not loaded.** Expected `model.onnx` (+ tokenizer) "
            f"under `{os.path.relpath(model_dir, BASE_DIR)}/`.\n\n"
            f"Run `python download_models.py`, then restart the app. On a Space, "
            f"check the build/runtime logs for `[models] ... FAILED` lines.")


# ============================== Tab 1: Router =================================

def route_demo(prompt: str):
    prompt = (prompt or "").strip()
    if not prompt:
        return "Type a task prompt above, then press **Route**.", []
    if not router.available():
        return _missing("Router (all-MiniLM-L6-v2)", router.MODEL_DIR), []

    head = router._instruction(prompt)
    v = router._embed(head)                      # (1, 384) unit-norm
    labels, protos = router._PROTOS
    sims = (protos @ v.T).ravel()

    best = {}
    for lab, s in zip(labels, sims):
        f = float(s)
        if f > best.get(lab, -1.0):
            best[lab] = f
    ranked = sorted(best.items(), key=lambda kv: -kv[1])
    top, top_sim = ranked[0]

    # What production route() would return (it applies _MIN_SIM and the margin
    # rule, because in the real agent it is only consulted on the regex's
    # "factual" catch-all).
    verdict = router.route(prompt)
    if verdict is None:
        if top_sim < router._MIN_SIM:
            why = (f"below `_MIN_SIM={router._MIN_SIM}` — too far from every "
                   f"category's exemplars")
        else:
            why = (f"margin vs `factual` is under `_MARGIN={router._MARGIN}` — "
                   f"too close to call, the incumbent regex verdict stands")
        prod = f"`None` → defer to the regex classifier ({why})"
    else:
        prod = f"**`{verdict[0]}`** (similarity {verdict[1]:.3f})"

    md = (f"### Predicted category: `{top}`  (similarity {top_sim:.3f})\n\n"
          f"Instruction head matched: `“{head}”`\n\n"
          f"Production `route()` verdict: {prod}\n\n"
          f"*In the agent, embeddings only adjudicate prompts the regex classifier "
          f"dumped into its `factual` catch-all — structural categories (code, "
          f"math) stay with the regex.*")
    rows = [[cat, f"{sim:.4f}"] for cat, sim in ranked[:3]]
    return md, rows


# ================================ Tab 2: NER ==================================

def ner_demo(text: str):
    text = (text or "").strip()
    if not text:
        return "Paste some text above, then press **Extract**.", []
    if not ner_onnx.available():
        return _missing("NER (OntoNotes BERT int8)", ner_onnx.MODEL_DIR), []

    ents = ner_onnx.extract(text)
    if ents is None:
        return ("NER inference failed — see the server log "
                "(`ner_onnx: inference failed`)."), []
    if not ents:
        return ("No entities found (after dropping types outside the agent's "
                "label vocabulary and spurious frequency-dates)."), []

    lines = "\n".join(f"{s} | {t}" for s, t in ents)
    md = (f"### {len(ents)} entities\n\n"
          f"Agent answer format:\n```text\n{lines}\n```")
    return md, [[s, t] for s, t in ents]


# ============================= Tab 3: Sentiment ===============================

def sentiment_demo(text: str):
    text = (text or "").strip()
    if not text:
        return "Paste a review above, then press **Classify**.", []
    if not sentiment_onnx.available():
        return _missing("Sentiment (twitter-roberta int8)",
                        sentiment_onnx.MODEL_DIR), []

    label = sentiment_onnx.classify(text)
    if label is None:
        return ("Sentiment inference failed — see the server log "
                "(`sentiment_onnx: inference failed`)."), []

    whole_lab, whole_conf = sentiment_onnx._score(text)
    rows = [["(whole text)", whole_lab, f"{whole_conf:.3f}"]]
    clauses = [c.strip() for c in sentiment_onnx._CLAUSE_SPLIT.split(text)
               if c and len(c.strip()) > 12]
    if len(clauses) > 1:
        for c in clauses:
            lab, cf = sentiment_onnx._score(c)
            rows.append([c, lab, f"{cf:.3f}"])

    md = f"### Label: **{label}**\n\n"
    if label == "Mixed":
        md += ("Derived by the clause rule: the model has no `Mixed` class, so the "
               "text was split at sentence boundaries and contrast markers "
               "(*but / however / although* …) and each clause classified. Opposing "
               f"strong polarities (confidence ≥ {sentiment_onnx._STRONG}) — or a "
               "contrast marker joining clauses that disagree — make the text "
               "**Mixed**.\n")
    else:
        md += ("Whole-text verdict from the classifier; the clause rule found no "
               "opposing strong polarities.\n")
    md += ("\n*In the agent this classifier decides the label and the LLM only "
           "writes the justification — measured, the 0.6B LLM kept getting the "
           "label itself wrong.*")
    return md, rows


# ========================== Tab 4: Code Retrieval =============================
# The corpus (204 code exemplars) is embedded at startup with the router's
# MiniLM session — router._embed is the ONE mean-pool + L2-normalise embedder
# (real attention_mask), exactly as agent_core/rag.py mandates.

_CORPUS = None            # (chunks: list[dict], emb: float32 (N, 384) unit-norm)
_CORPUS_LOCK = threading.Lock()


def _ensure_corpus():
    global _CORPUS
    with _CORPUS_LOCK:
        if _CORPUS is not None:
            return _CORPUS
        if not router.available():
            return None
        chunks = []
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        vecs = [rag.embed_passage(c["text"]) for c in chunks]
        if any(v is None for v in vecs):
            return None
        _CORPUS = (chunks, np.vstack(vecs).astype(np.float32))
        print(f"[app] corpus embedded: {len(chunks)} exemplars", flush=True)
        return _CORPUS


def retrieve_demo(query: str, lam: float):
    query = (query or "").strip()
    if not query:
        return "Describe a coding task above, then press **Retrieve**."
    corpus = _ensure_corpus()
    if corpus is None:
        return _missing("Retrieval embedder (all-MiniLM-L6-v2)", router.MODEL_DIR)

    chunks, emb = corpus
    v = rag.embed_query(query)                   # (384,) unit-norm
    if v is None:
        return "Query embedding failed — see the server log."
    sims = emb @ v

    pool = min(20, len(chunks))
    idx = np.argpartition(-sims, pool - 1)[:pool]
    idx = idx[np.argsort(-sims[idx])]
    order = rag.select_mmr(emb[idx], sims[idx].astype(np.float64), k=3, lam=lam)

    parts = [f"### Top 3 of {len(chunks)} exemplars "
             f"(MMR over a {pool}-candidate pool, λ = {lam:.2f})\n"]
    for rank, j in enumerate(order, 1):
        ch = chunks[int(idx[j])]
        role = "most relevant" if rank == 1 else "most distinct given the picks above"
        parts.append(
            f"**{rank}. `{ch.get('title', '')}`** — cosine {sims[int(idx[j])]:.3f} "
            f"({role})\n\n{ch.get('text', '')}\n\n"
            f"```python\n{ch.get('code', '')}\n```\n")
    parts.append("*Raw cosine shown, not the MMR score — the agent gates on the "
                 "max cosine as retrieval confidence, while MMR only re-orders "
                 "the pool for diversity.*")
    return "\n".join(parts)


# ================================== UI ========================================

_INTRO = """
# FrugalRouter — zero-token local machinery

Demo of the local (no-API-call) components of **FrugalRouter**, an AMD Hackathon
Track-1 agent that answers 19 mixed NL tasks while minimising billed tokens.
Everything on this page runs three small **int8 ONNX models on CPU** — the same
files the agent bakes into its container; in production, whatever these
components handle locally costs **zero** tokens.
"""

with gr.Blocks(title="FrugalRouter Demo") as demo:
    gr.Markdown(_INTRO)

    with gr.Tab("Router"):
        gr.Markdown(
            "Semantic task routing: the prompt's **instruction head** is embedded "
            "with all-MiniLM-L6-v2 (int8, 23 MB, ~5 ms) and matched against "
            "hand-written exemplars for 8 task categories.")
        r_in = gr.Textbox(label="Task prompt", lines=3,
                          placeholder="e.g. Identify the people, organizations and dates mentioned in the passage below: …")
        r_btn = gr.Button("Route", variant="primary")
        r_md = gr.Markdown()
        r_df = gr.Dataframe(headers=["category", "best exemplar similarity"],
                            label="Top-3 category similarities", interactive=False)
        gr.Examples(
            examples=[
                "Identify the people, organizations and dates mentioned in the passage below: Tim Cook visited Berlin in May 2023.",
                "What is the overall tone of this statement? The battery died after two days.",
                "Trim this down to the essentials: The committee met on Thursday to discuss the budget…",
                "Who developed the theory of general relativity?",
                "Note down every company, city and person this paragraph refers to.",
            ],
            inputs=r_in)
        r_btn.click(route_demo, inputs=r_in, outputs=[r_md, r_df])
        r_in.submit(route_demo, inputs=r_in, outputs=[r_md, r_df])

    with gr.Tab("NER"):
        gr.Markdown(
            "OntoNotes BERT (int8, 104 MB): BIO decoding over **character offsets** "
            "(no subword-merge bugs), native DATE class for relative dates "
            "(“last April”), regex backstop only for numeric dates the "
            "model missed, spurious frequency-dates dropped.")
        n_in = gr.Textbox(label="Text", lines=4,
                          placeholder="Paste a passage with names, places, dates…")
        n_btn = gr.Button("Extract", variant="primary")
        n_md = gr.Markdown()
        n_df = gr.Dataframe(headers=["Entity", "Type"],
                            label="Entities", interactive=False)
        gr.Examples(
            examples=[
                "Tim Cook unveiled the iPhone 15 at Apple Park in Cupertino last September, joined by executives from Disney and the WHO.",
                "The Berlin Wall fell on 9 November 1989, three years before the Maastricht Treaty created the European Union.",
                "Dr. Sarah Chen of Stanford University will present at NeurIPS in Vancouver on 12/10/2024.",
            ],
            inputs=n_in)
        n_btn.click(ner_demo, inputs=n_in, outputs=[n_md, n_df])
        n_in.submit(ner_demo, inputs=n_in, outputs=[n_md, n_df])

    with gr.Tab("Sentiment"):
        gr.Markdown(
            "twitter-roberta (int8, 126 MB) picks Positive / Negative / Neutral. "
            "**Mixed** is derived: split into clauses at sentence boundaries and "
            "contrast markers, classify each — opposing strong polarities (or a "
            "contrast marker joining disagreeing clauses) ⇒ Mixed.")
        s_in = gr.Textbox(label="Review / statement", lines=4,
                          placeholder="e.g. The food was delicious but we waited nearly an hour.")
        s_btn = gr.Button("Classify", variant="primary")
        s_md = gr.Markdown()
        s_df = gr.Dataframe(headers=["Clause", "model label", "confidence"],
                            label="Clause-level scores", interactive=False)
        gr.Examples(
            examples=[
                "The food was delicious but we waited nearly an hour for a table.",
                "Absolutely love this laptop — fast, light, and the battery lasts all day.",
                "The package arrived on Tuesday and contained all three items.",
                "Terrible experience. The screen cracked within a week and support never replied.",
            ],
            inputs=s_in)
        s_btn.click(sentiment_demo, inputs=s_in, outputs=[s_md, s_df])
        s_in.submit(sentiment_demo, inputs=s_in, outputs=[s_md, s_df])

    with gr.Tab("Code Retrieval"):
        gr.Markdown(
            "204 code exemplars (task description + implementation), embedded at "
            "startup with the **same MiniLM session as the router**. Query → "
            "cosine over all exemplars → top-20 pool → **MMR re-rank** to 3 "
            "relevant-yet-distinct picks. Slide λ toward 0 for diversity, toward "
            "1 for pure relevance.")
        c_in = gr.Textbox(label="Coding task", lines=2,
                          placeholder="e.g. check whether a string is a palindrome ignoring punctuation")
        c_lam = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                          label="λ (relevance ↔ diversity trade-off)")
        c_btn = gr.Button("Retrieve", variant="primary")
        c_md = gr.Markdown()
        gr.Examples(
            examples=[
                "Check whether a string is a palindrome, ignoring punctuation.",
                "Count how often each word appears in a piece of text.",
                "Merge two sorted lists into one sorted list.",
                "Find the largest number in a list without using max().",
            ],
            inputs=c_in)
        c_btn.click(retrieve_demo, inputs=[c_in, c_lam], outputs=c_md)
        c_in.submit(retrieve_demo, inputs=[c_in, c_lam], outputs=c_md)


if __name__ == "__main__":
    # Embed the retrieval corpus at startup (a few seconds for 204 chunks); if
    # models are absent this is a no-op and the tabs report what is missing.
    try:
        _ensure_corpus()
    except Exception as e:  # noqa: BLE001
        print(f"[app] startup corpus embed failed: {e}", file=sys.stderr)
    demo.launch()
