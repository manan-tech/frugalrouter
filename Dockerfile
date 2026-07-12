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

# v15 single-model experiment: ONE mid-size general model serves every
# category (coder proxies to it via SINGLE_MODEL=1). Qwen2.5-3B-Instruct
# Q4_K_M = 1.96 GB, fits 4 GB with room for KV; drops the separate coder.
ARG GENERAL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
RUN mkdir -p /models && \
    curl -fL --retry 3 -o /models/general.gguf "$GENERAL_URL" && \
    ls -la /models

FROM --platform=linux/amd64 python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/manan-tech/frugalrouter" \
      org.opencontainers.image.description="FrugalRouter - local-first token-efficient routing agent (AMD Hackathon ACT II Track 1)"
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libcurl4 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=dl /opt/llama /opt/llama
COPY --from=dl /models /models
ENV PATH="/opt/llama:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama" \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app
COPY agent /app/agent

CMD ["python", "-m", "agent.main"]
