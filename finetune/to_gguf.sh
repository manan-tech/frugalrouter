#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# finetune/to_gguf.sh — merged HF model -> GGUF Q4_K_M, the format our container
# serves with llama.cpp.  Run this on the GPU box right after train.py.
#
#   bash finetune/to_gguf.sh finetune/out/merged
#
# Produces:  finetune/out/general.gguf   -> bake as /models/general.gguf
#
# WHY THE DETAILS MATTER (do not "simplify" these away):
#
#  * LLAMA_TAG is pinned to b9959 — the SAME tag the Dockerfile downloads. The
#    converter and the runtime must agree: a GGUF written by a newer converter
#    can use metadata/tensor conventions an older llama-server refuses to load.
#    If you bump the tag here, bump it in the Dockerfile too, and vice versa.
#
#  * We do NOT build llama-quantize from source. The b9959 release tarball ships
#    llama-quantize + llama-cli + llama-server prebuilt, so the binary that
#    quantizes and smoke-tests the model is byte-identical to the one in the
#    image. That is both a stronger guarantee and ~5 minutes of GPU time saved.
#    (A cmake fallback is kept for when the tarball can't be fetched/executed.)
#
#  * At b9959 the converter was REFACTORED: model classes now live in the
#    `conversion/` package, NOT in convert_hf_to_gguf.py. So the obvious check
#    `grep Qwen3ForCausalLM convert_hf_to_gguf.py` finds nothing and would tell
#    you Qwen3 is unsupported — a lie. The authoritative check is
#        python3 convert_hf_to_gguf.py --print-supported-models
#    and note it prints via logger.error(), i.e. to STDERR, so you must 2>&1 it.
#    (Verified at b9959: conversion/qwen.py:153 registers "Qwen3ForCausalLM".)
#
#  * The chat template must survive into the GGUF. llama-server runs with
#    --jinja and agent/llm.py sends chat_template_kwargs={"enable_thinking":false};
#    that only works if the model's own Qwen3 template is embedded. If it is
#    missing, llama.cpp silently substitutes a DEFAULT template — the model still
#    answers, so a naive smoke test passes, but it is now being prompted
#    differently than it was trained and format fidelity rots. We assert the
#    template is present, at runtime, via llama-server's /props.
# ---------------------------------------------------------------------------
set -euo pipefail

MERGED="${1:-finetune/out/merged}"

LLAMA_TAG="${LLAMA_TAG:-b9959}"        # MUST match Dockerfile's ARG LLAMA_TAG
QUANT="${QUANT:-Q4_K_M}"
OUTNAME="${OUTNAME:-general.gguf}"
MIN_MB="${MIN_MB:-300}"                # 1.7B Q4_K_M ~1.1GB, 0.6B ~0.4GB
PY="${PYTHON:-python3}"
PORT="${PORT:-8099}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

# absolute paths everywhere; the script must work from any cwd
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -d "$MERGED" ] || { echo "FATAL: merged dir not found: $MERGED" >&2; exit 1; }
MERGED="$(cd "$MERGED" && pwd)"
OUT="$ROOT/finetune/out"
WORK="$ROOT/finetune/.build"           # clone + binaries live here (gitignored)
SRC="$WORK/llama.cpp"
BIN="$WORK/bin"
F16="$OUT/model-f16.gguf"
GGUF="$OUT/$OUTNAME"
mkdir -p "$OUT" "$WORK"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mFATAL: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
say "0/7  preflight: is $MERGED actually a merged dense model?"
# ---------------------------------------------------------------------------
[ -f "$MERGED/config.json" ] || die "no config.json in $MERGED"
if [ -f "$MERGED/adapter_config.json" ] && ! ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
  die "$MERGED looks like a raw LoRA ADAPTER, not a merged model.
     convert_hf_to_gguf.py needs merged dense weights.
     train.py writes them via merge_and_unload() -> finetune/out/merged."
fi
ls "$MERGED"/*.safetensors >/dev/null 2>&1 || die "no *.safetensors in $MERGED"

ARCH="$("$PY" - "$MERGED" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1] + "/config.json", encoding="utf-8"))
archs = cfg.get("architectures") or []
print(archs[0] if archs else "")
PY
)"
[ -n "$ARCH" ] || die "config.json has no architectures[] — cannot convert"
echo "architecture: $ARCH"

# The chat template is load-bearing (see header). Fail NOW, not after a 5-minute
# convert, if the tokenizer save dropped it. b9959's gguf-py picks it up from any
# of these three places (gguf-py/gguf/vocab.py:317-331).
HAS_TMPL="$("$PY" - "$MERGED" <<'PY'
import json, os, sys
d = sys.argv[1]
ok = os.path.isfile(os.path.join(d, "chat_template.jinja")) \
     or os.path.isfile(os.path.join(d, "chat_template.json"))
if not ok:
    p = os.path.join(d, "tokenizer_config.json")
    if os.path.isfile(p):
        ok = bool(json.load(open(p, encoding="utf-8")).get("chat_template"))
print("yes" if ok else "no")
PY
)"
[ "$HAS_TMPL" = "yes" ] || die "no chat template in $MERGED
     (looked for chat_template.jinja / chat_template.json / tokenizer_config.json:chat_template)
     train.py must tok.save_pretrained(<merged dir>) with the Qwen3 tokenizer.
     Without it, llama-server --jinja invents a default template and the model
     gets prompted differently than it was trained."
echo "chat template: present"

# ---------------------------------------------------------------------------
say "1/7  llama.cpp source @ $LLAMA_TAG (the converter)"
# ---------------------------------------------------------------------------
if [ ! -d "$SRC/.git" ]; then
  rm -rf "$SRC"
  git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$SRC"
else
  echo "reusing $SRC"
fi
[ -f "$SRC/convert_hf_to_gguf.py" ] || die "convert_hf_to_gguf.py missing from clone"

# ---------------------------------------------------------------------------
say "2/7  python deps for the converter"
# ---------------------------------------------------------------------------
# NOTE: llama.cpp's requirements-convert_hf_to_gguf.txt pins torch==2.11.0 from
# the CPU wheel index. Installed blindly in the TRAINING env, that rips out your
# CUDA torch. So: if torch/transformers/numpy already import, leave them alone.
if "$PY" -c "import torch, transformers, numpy" >/dev/null 2>&1 && [ "${FORCE_REQS:-0}" != "1" ]; then
  echo "torch/transformers/numpy already present — NOT touching them"
  echo "(FORCE_REQS=1 to install llama.cpp's pinned CPU requirements anyway)"
  "$PY" -c "import sentencepiece" >/dev/null 2>&1 || pip install -q sentencepiece protobuf || true
else
  pip install -q -r "$SRC/requirements/requirements-convert_hf_to_gguf.txt"
fi
# gguf-py is vendored in the clone; convert_hf_to_gguf.py puts it on sys.path itself.

# ---------------------------------------------------------------------------
say "3/7  ARCH GATE: does THIS converter support $ARCH?"
# ---------------------------------------------------------------------------
# print_registered_models() writes with logger.error -> STDERR. Capture both.
SUPPORTED="$("$PY" "$SRC/convert_hf_to_gguf.py" --print-supported-models 2>&1 || true)"
if ! printf '%s\n' "$SUPPORTED" | grep -qE "^[[:space:]]*-[[:space:]]*${ARCH}$"; then
  echo "--- qwen-ish architectures this converter knows ---" >&2
  printf '%s\n' "$SUPPORTED" | grep -i qwen >&2 || true
  die "$ARCH is NOT registered in convert_hf_to_gguf.py @ $LLAMA_TAG.
     The plan dies here — pick a base model whose arch IS listed above, or move
     LLAMA_TAG forward (and update the Dockerfile to the same tag)."
fi
echo "OK: $ARCH is registered at $LLAMA_TAG"

# ---------------------------------------------------------------------------
say "4/7  HF -> GGUF (f16)"
# ---------------------------------------------------------------------------
"$PY" "$SRC/convert_hf_to_gguf.py" "$MERGED" --outfile "$F16" --outtype f16
[ -s "$F16" ] || die "converter exited 0 but produced no f16 file"
echo "f16: $(du -h "$F16" | cut -f1)"

# ---------------------------------------------------------------------------
say "5/7  llama.cpp binaries @ $LLAMA_TAG (quantize + smoke test)"
# ---------------------------------------------------------------------------
# Same prebuilt tarball the Dockerfile pulls -> the quantizer and the server we
# test with are the exact binaries that will run in the grader's container.
if [ ! -x "$BIN/llama-quantize" ]; then
  mkdir -p "$BIN"
  TGZ="$WORK/llama-$LLAMA_TAG.tar.gz"
  curl -fL --retry 3 -o "$TGZ" \
    "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/llama-$LLAMA_TAG-bin-ubuntu-x64.tar.gz" \
    || echo "tarball fetch failed — will fall back to a source build"
  if [ -f "$TGZ" ]; then tar -xzf "$TGZ" -C "$BIN" --strip-components=1; fi
fi
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"

if ! "$BIN/llama-quantize" --help >/dev/null 2>&1; then
  echo "prebuilt binaries unusable here — building from source (slower)"
  cmake -S "$SRC" -B "$SRC/build" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$SRC/build" --target llama-quantize llama-server llama-cli -j"$(nproc)" >/dev/null
  BIN="$SRC/build/bin"
  export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
fi
"$BIN/llama-quantize" --help >/dev/null 2>&1 || die "no working llama-quantize"
echo "binaries: $BIN"

# ---------------------------------------------------------------------------
say "6/7  quantize -> $QUANT"
# ---------------------------------------------------------------------------
"$BIN/llama-quantize" "$F16" "$GGUF" "$QUANT"
[ -s "$GGUF" ] || die "llama-quantize produced no output"

SZ_MB=$(( $(wc -c < "$GGUF") / 1024 / 1024 ))
echo "quantized: ${SZ_MB} MB"
[ "$SZ_MB" -ge "$MIN_MB" ] || die "$GGUF is only ${SZ_MB} MB (< ${MIN_MB} MB) — truncated/garbage"
rm -f "$F16"    # ~3.4 GB of scratch, useless once quantized

# ---------------------------------------------------------------------------
say "7/7  VERIFY: does the container's own llama-server actually load it?"
# ---------------------------------------------------------------------------
if [ "$SKIP_SMOKE" = "1" ]; then
  echo "SKIP_SMOKE=1 — skipped. Do not ship an unverified GGUF."
else
  # EXACT production flags, copied from agent/llm.py LlamaServer.start()
  # (-c 2048 = config.GENERAL_CTX, -t 2 = config.LLM_THREADS = the grader's 2 vCPU).
  "$BIN/llama-server" -m "$GGUF" -t 2 -c 2048 --port "$PORT" --host 127.0.0.1 \
      --jinja --no-webui --parallel 1 --cache-ram 0 -b 512 -ub 256 \
      -fa on -ctk q8_0 -ctv q8_0 >"$WORK/server.log" 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT

  if ! "$PY" - "$PORT" "$SRV" <<'PY'
import json, os, sys, time, urllib.request

port, pid = sys.argv[1], int(sys.argv[2])
base = f"http://127.0.0.1:{port}"

# 1. health — did it load at all, under the tight production flags?
deadline = time.time() + 180
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit("server process died during load (see server.log)")
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as r:
            if r.status == 200:
                break
    except Exception:
        time.sleep(0.5)
else:
    sys.exit("server never became healthy within 180s")
print("load: OK (production flags: -c 2048 -fa on -ctk/-ctv q8_0 --cache-ram 0)")

# 2. is OUR chat template embedded, or did --jinja fall back to a default?
with urllib.request.urlopen(base + "/props", timeout=10) as r:
    props = json.loads(r.read().decode())
tmpl = props.get("chat_template") or ""
if "<|im_start|>" not in tmpl:
    sys.exit("the chat template in this GGUF is NOT the Qwen3 one (no <|im_start|>).\n"
             "llama-server substituted a default => the model would be prompted\n"
             "differently than it was trained. Fix the tokenizer save in train.py.")
print("chat template: embedded, Qwen3 ({} chars, enable_thinking={})".format(
    len(tmpl), "enable_thinking" in tmpl))

# 3. a real one-shot task, in the exact shape agent/llm.py sends
#    (system prompt byte-identical to train.py's SYS).
SYS = ("You answer the user's task directly and in the exact format requested. "
       "No preamble, no explanation of your process. Answer once, correctly.")
body = {
    "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content":
         "Extract the named entities from this text:\n"
         "Satya Nadella said Microsoft will open a new campus in Hyderabad next April."},
    ],
    "max_tokens": 160,
    "temperature": 0.0,
    "cache_prompt": True,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(base + "/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=180) as r:
    resp = json.loads(r.read().decode())

out = (resp["choices"][0]["message"].get("content") or "").strip()
if not out:
    sys.exit("model loaded but generated NOTHING")
tps = float((resp.get("timings") or {}).get("predicted_per_second") or 0.0)

print("\n--- one-shot NER sample (temp 0, thinking off) " + "-" * 22)
print(out)
print("-" * 68)
print(f"decode: {tps:.1f} tok/s at -t 2 on THIS box")
if tps and tps < 7.0:
    print("WARN: below config.TPS_DEAD (7.0), and this box is FASTER than the "
          "grader's contended one — consider the 0.6B base.")
print("\nEYEBALL THE SAMPLE: it must be 'Entity | TYPE' lines and nothing else.\n"
      "If there is a preamble or a <think> block, the SFT format did not take —\n"
      "fix that BEFORE rebuilding the image.")
PY
  then
    echo "--- server.log (tail) ---" >&2
    tail -40 "$WORK/server.log" >&2 || true
    die "llama-server could not serve $GGUF"
  fi

  kill $SRV 2>/dev/null || true
  trap - EXIT
fi

# ---------------------------------------------------------------------------
printf '\n\033[32m========================  DONE  ========================\033[0m\n'
cat <<EOF

  BAKE THIS FILE:  $GGUF
                   ($(du -h "$GGUF" | cut -f1), $QUANT, arch $ARCH, llama.cpp $LLAMA_TAG)

  IT GOES TO:      /models/general.gguf   (agent/config.py GENERAL_MODEL_PATH)

  1. copy it off the GPU box (it is gitignored via *.gguf — never commit it):
       scp <gpu-box>:$GGUF  finetune/out/general.gguf

  2. Dockerfile: stop downloading the stock model, copy ours instead.
     Drop general.gguf from the ARG GENERAL_URL / RUN curl block, and add to the
     FINAL stage (after 'COPY --from=dl /models /models'):

       COPY finetune/out/general.gguf /models/general.gguf

     (.dockerignore does not exclude finetune/, so it IS in the build context.)

  3. rebuild + validate:
       docker buildx build --platform linux/amd64 -t ghcr.io/manan-tech/frugalrouter:vN --load .
       gh workflow run eval -R manan-tech/frugalrouter -f tasks=eval/rehearsal19.json -f budget=0

  4. flip the agent to one-shot (finetune/README.md): TPS_DEAD=0.0,
     ESCALATION_BUDGET_TOKENS=0, single generation, thinking=False.

EOF
