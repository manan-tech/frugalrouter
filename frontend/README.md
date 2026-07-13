---
title: FrugalRouter Demo
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
---

# FrugalRouter — zero-token local machinery, interactive

FrugalRouter is an AMD Hackathon Track-1 agent that answers 19 mixed NL tasks
(8 categories) while minimising tokens billed through an API proxy — its core
trick is doing everything *verifiable* locally, for zero tokens. This Space
demos the four local components exactly as they run inside the agent's
container, on three small **int8 ONNX models on CPU** (~253 MB total, no GPU,
no API keys): a **semantic task router** (MiniLM embeddings over category
exemplars, with the production min-similarity and margin rules), an **NER
tagger** (OntoNotes BERT with char-offset BIO decoding and a regex date
backstop), a **sentiment classifier** (twitter-roberta with a clause-level
rule that derives the `Mixed` label the model doesn't have), and **code-exemplar
retrieval** (204 exemplars embedded at startup with the router's own MiniLM
session, MMR-re-ranked for diversity). The `agent_core/` modules are vendored
verbatim from the agent — only imports and model paths were adjusted.

## Run locally

```bash
cd frontend
pip install -r requirements.txt
python download_models.py      # ~253 MB into ./models/{router,ner,sentiment}
python app.py                  # http://127.0.0.1:7860
```

`app.py` also calls the downloader at startup, so `python app.py` alone works;
running `download_models.py` first just keeps the first launch snappy.

## Deploy to Hugging Face Spaces

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                       # paste a WRITE token from hf.co/settings/tokens

# 1. create the Space (replace YOUR_USERNAME)
huggingface-cli repo create frugalrouter-demo --repo-type space --space_sdk gradio

# 2. upload this folder (from inside frontend/) — exclude any locally
#    downloaded models; the Space fetches them itself on first boot
cd frontend
huggingface-cli upload YOUR_USERNAME/frugalrouter-demo . . \
    --repo-type space --exclude "models/*" --exclude "__pycache__/*"
```

The Space then builds from `requirements.txt`, and on first startup `app.py`
downloads the three ONNX models (~253 MB) and embeds the 204-exemplar retrieval
corpus (a few seconds). Free CPU tier is enough: every onnxruntime session is
pinned to 1 intra-op thread, models load lazily, and any tab whose model is
missing shows a clear message instead of crashing.

## Layout

```
app.py                 Gradio UI: Router / NER / Sentiment / Code Retrieval tabs
download_models.py     stdlib-urllib fetch of the 3 public int8 ONNX models
agent_core/            vendored agent modules (router, ner_onnx, sentiment_onnx,
                       rag, util) — inference logic unchanged
data/chunks.jsonl      204 code exemplars {id, title, text, code}
models/                created at first run (gitignored / never uploaded)
```

## Model credits

- [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) (router + retrieval embedder, int8)
- [zencrazycat/ner-bert-base-cased-ontonotesv5-englishv4-onnx](https://huggingface.co/zencrazycat/ner-bert-base-cased-ontonotesv5-englishv4-onnx) (NER, int8)
- [Xenova/twitter-roberta-base-sentiment-latest](https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest) (sentiment, int8)
