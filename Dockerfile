# FrugalRouter — AMD Hackathon ACT II Track 1
# Final image: python slim + llama.cpp CPU server + two small GGUF models.
# Build:  docker buildx build --platform linux/amd64 -t frugalrouter .

FROM --platform=linux/amd64 python:3.12-slim AS dl
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

ARG LLAMA_TAG=b9959
RUN mkdir -p /opt/llama && \
    curl -fL --retry 3 -o /tmp/llama.tar.gz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" && \
    tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1 \
      --wildcards "llama-${LLAMA_TAG}/llama-server" "llama-${LLAMA_TAG}/*.so*" && \
    ls -la /opt/llama && test -f /opt/llama/llama-server

ARG GENERAL_URL="https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
ARG CODER_URL="https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
RUN mkdir -p /models && \
    curl -fL --retry 3 -o /models/general.gguf "$GENERAL_URL" && \
    curl -fL --retry 3 -o /models/coder.gguf "$CODER_URL" && \
    ls -la /models

# Purpose-trained NER: OntoNotes-v5 BERT (int8 ONNX, ~104MiB). Chosen over the
# CoNLL ports because it has a NATIVE DATE class (catches "last April",
# "three years ago", "Yesterday" — which no regex enumerates) plus EVENT and
# PRODUCT, exactly the label set our answer format needs. Zero tokens, ~10ms.
# The repo has NO model.onnx — model_quantized.onnx is the only ONNX file.
ARG NER_REPO="https://huggingface.co/zencrazycat/ner-bert-base-cased-ontonotesv5-englishv4-onnx/resolve/main"
RUN mkdir -p /models/ner && \
    curl -fL --retry 3 -o /models/ner/model.onnx     "$NER_REPO/onnx/model_quantized.onnx" && \
    curl -fL --retry 3 -o /models/ner/tokenizer.json "$NER_REPO/tokenizer.json" && \
    curl -fL --retry 3 -o /models/ner/config.json    "$NER_REPO/config.json" && \
    ls -la /models/ner

# Sentiment classifier (roberta, int8 ONNX ~126MB). The 0.6B reliably mislabels
# sentiment (Mixed->Negative, Neutral->Positive) and few-shot did NOT fix it, but
# it writes a fine REASON. So the classifier decides the label and the LLM only
# justifies it. Mixed is derived clause-wise (some clause positive, another negative).
ARG SENT_REPO="https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest/resolve/main"
RUN mkdir -p /models/sentiment && \
    curl -fL --retry 3 -o /models/sentiment/model.onnx     "$SENT_REPO/onnx/model_int8.onnx" && \
    curl -fL --retry 3 -o /models/sentiment/tokenizer.json "$SENT_REPO/tokenizer.json" && \
    curl -fL --retry 3 -o /models/sentiment/config.json    "$SENT_REPO/config.json" && \
    ls -la /models/sentiment

# Semantic ROUTER (all-MiniLM-L6-v2, int8 ONNX, 23MB). The regex classifier is the
# single point of failure for the whole agent — every pipeline sits behind it — and
# it only knows phrasings someone enumerated (measured 8/16 on paraphrases). The
# finals re-run on REPHRASED prompts. Embeddings adjudicate the regex's `factual`
# catch-all, which is where every misroute lands: hybrid scores 16/16 on
# paraphrases while staying 19/19 + 10/10 on the real suites.
ARG ROUTER_REPO="https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main"
RUN mkdir -p /models/router && \
    curl -fL --retry 3 -o /models/router/model.onnx     "$ROUTER_REPO/onnx/model_int8.onnx" && \
    curl -fL --retry 3 -o /models/router/tokenizer.json "$ROUTER_REPO/tokenizer.json" && \
    ls -la /models/router

FROM --platform=linux/amd64 python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/manan-tech/frugalrouter" \
      org.opencontainers.image.description="FrugalRouter - local-first token-efficient routing agent (AMD Hackathon ACT II Track 1)"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libcurl4 && \
    rm -rf /var/lib/apt/lists/*
# onnxruntime + tokenizers power the local NER extractor (agent/ner_onnx.py).
# Both are optional at runtime: if the import fails, ner() falls back to the LLM.
RUN pip install --no-cache-dir onnxruntime==1.27.0 tokenizers==0.23.1 && \
    rm -rf /root/.cache/pip

COPY --from=dl /opt/llama /opt/llama
COPY --from=dl /models /models
ENV PATH="/opt/llama:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app
COPY agent /app/agent

CMD ["python", "-m", "agent.main"]
