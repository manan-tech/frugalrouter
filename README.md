# FrugalRouter

**A fine-tuned, locally-verified, token-frugal task agent — AMD Developer Hackathon: ACT II, Track 1**
**Final submission: `ghcr.io/manan-tech/frugalrouter:v25` — 84.2% accuracy on the hidden evaluation set at 3,388 Fireworks tokens.**

FrugalRouter answers general-purpose AI tasks (factual Q&A, math, sentiment, summarisation, NER, code debugging, logic puzzles, code generation) using the fewest external tokens possible. A **LoRA-fine-tuned Qwen3-1.7B** runs entirely inside the container on 2 CPU cores, answering most categories in a single generation on its own training distribution. Purpose-trained ONNX models (router / NER / sentiment) and a sandboxed Python executor provide zero-token verification. Only the categories a 1.7B provably can't be trusted on escalate to the Fireworks proxy — batched, format-hinted, under a hard token budget.

## The core idea: put accuracy in the weights, not in sampling

Our early builds compensated for a weak local model with sampling — 3 generations per task, majority voting, cross-verification. That quality lived in *wall-clock*, and the grader's contended box stripped it: identical images scored 36.8%–78.9% depending on how slow the box was that day.

The fix was a fine-tune. We generated a **2,708-example SFT dataset** (category-exact output formats, math answers as executable Python programs, every code example execution-verified), LoRA-trained Qwen3-1.7B (r=32, 3 epochs, completion-only loss), and switched the agent to **one-shot inference**: one generation per task, temperature 0, on the exact prompt distribution the model was trained on. A slow box now makes answers *later, never worse* — 19 tasks decode in ~170s at grader speed with 4× headroom.

## How it works

```
/input/tasks.json
   │
   ├─ 1. Classify (regex + MiniLM-embedding adjudication, zero cost)
   ├─ 2. Sort by category (llama.cpp prompt-prefix reuse; one shared system prompt)
   │
   ├─ 3. One-shot local answer per task  (fine-tuned Qwen3-1.7B Q4_K_M, llama.cpp CPU)
   │      ├─ math:  the trained answer IS a Python program → executed in a sandbox,
   │      │         its stdout is the shipped answer
   │      ├─ code:  3-sample behavioral majority vote over a deterministic input
   │      │         battery (defeats float-noise coin flips between CPUs)
   │      └─ all:   generic format verifier — detects stated constraints
   │                ("exactly two sentences", bullet caps), retries once, and
   │                routes still-violating answers to escalation
   │
   ├─ 4. Targeted escalation → Fireworks proxy (batched, format-hinted,
   │      reasoning_effort=low, hard budget) — only for categories the 1.7B
   │      measurably can't self-verify
   │
   └─ /output/results.json   (atomic write after every task; watchdog ladder
                              soft 445s / flush 505s / hard-exit 535s makes
                              TIMEOUT and invalid JSON structurally impossible)
```

### Local machinery (zero tokens)

| Component | What it does |
|---|---|
| Fine-tuned Qwen3-1.7B (Q4_K_M, ~1.1 GB) | One-shot answers in category-exact formats; single llama-server, single global lock, memory-flat flags for the 4 GB cgroup |
| OntoNotes BERT NER (int8 ONNX, 104 MB) | Purpose-trained tagger with a native DATE class ("last April", "three years ago") |
| RoBERTa sentiment (int8 ONNX, 126 MB) | Label classification with a clause-level Mixed rule |
| MiniLM router (int8 ONNX, 23 MB) | Embedding adjudication of the regex classifier's catch-all — survives paraphrased prompts (16/16 vs 8/16 regex-only) |
| Python sandbox | Executes model-written math programs and code tests; hard timeouts |
| Code-exemplar retrieval (204 exemplars, MMR-diversified) | Execution-verified reference implementations the model can adapt instead of writing from scratch |

### Verification, not trust

Every locally-shipped answer is checked by something *deterministic*: math runs, code is behaviorally fingerprinted across samples on a fixed input battery, formats are programmatically enforced. Confidence scores derive from verification outcomes; anything below its category threshold escalates. The failure containment is layered — per-task exception isolation, a dead-streak detector for mid-run local death, mass-fallback promotion to an emergency budget, and a zero-model deterministic floor so every task always gets *an* answer.

### The token economics

Rank = ascending billed tokens among entries above an 80% accuracy gate, so every design choice is "answer locally if verifiable, pay only for what verification can't certify." Escalations are batched per category (byte-identical system prompt for Fireworks prompt-cache pricing), shrink-to-fit on budget-reservation failure, and degrade to individual calls rather than skipping tasks.

## Results

Validated on a dedicated c6i.large (2 vCPU / 4 GB / no swap — the exact grading spec), exact harness contract, LLM-judged against rubrics:

| Build | rehearsal19 | public10 | Fireworks tokens | Hidden set (grader) |
|---|---|---|---|---|
| Sampling build (pre-fine-tune) | 19/19 (CI) | 10/10 (CI) | 3,773–5,685 | 78.9% — collapsed on their box |
| Fine-tune, pure local | 15/19 | — | **0** | — (float-noise flips, gate fail) |
| Fine-tune + targeted escalation | 18/19 | 10/10 | ~1,500 | 78.9% |
| **+ format verifier (final, `:v25`)** | **19/19** | **10/10** | 3,212 / 771 | **84.2% — gate PASS** |

The journey is the result: six consecutive grader failures were diagnosed to wall-clock starvation and sampling collapse, and the stuck 78.9% (identical across three architecturally different builds) was cracked by the organizers' category breakdown plus a programmatic format-compliance net.

## Build

```bash
docker buildx build --platform linux/amd64 -t frugalrouter .
```

The multi-stage build downloads a portable llama.cpp CPU release, bakes the fine-tuned GGUF (`finetune/out/general.gguf`, produced by the pipeline below), three ONNX models, and the exemplar bundle. Image: ~1.4 GB.

## Run

The container follows the Track 1 harness contract: read `/input/tasks.json`, write `/output/results.json`, exit 0.

```bash
mkdir -p input output
docker run --rm --platform linux/amd64 --cpus=2 --memory=4g --memory-swap=4g \
  -e FIREWORKS_API_KEY=fw_... \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/minimax-m3 \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" \
  ghcr.io/manan-tech/frugalrouter:v25
```

Without a Fireworks key the agent still completes every task locally (measured 89.5% on rehearsal19 at zero tokens) — escalation is an optimization, not a dependency.

## Fine-tuning pipeline

```
finetune/
  sft.jsonl            2,708 examples (gitignored — competitive asset during the event)
  validate_dataset.py  Executes math programs, compiles code, checks formats
  train.py             LoRA SFT (r=32, α=64, completion-only loss, byte-exact
                       chat-template rendering) → merged weights
  to_gguf.sh           merged HF → GGUF Q4_K_M with the SAME llama.cpp tag the
                       image serves; smoke-tests with production flags
  rag/                 Corpus builders + the 204-exemplar code bundle
```

Trained on a g6e.2xlarge spot instance in ~11 minutes; the GGUF swap into the image is one `COPY` line.

## Evaluation harness

```bash
eval/run.sh eval/rehearsal19.json          # run under exact grading constraints
python3 eval/judge.py output/results.json eval/rehearsal19.json
```

`eval/rehearsal19.json` mirrors the real 19-task distribution; `eval/public10.json` holds the organizer sample shapes; `eval/ec2_test.sh` provisions the grader-spec EC2 replica used for every ship decision (CI runners hide the memory pressure that matters).

## Repository layout

```
agent/
  main.py        Orchestrator: contract I/O, watchdog, escalation, verdict parsing
  config.py      All tunables (baked; harness injects only the 3 contract vars)
  classify.py    Regex classifier + semantic routing
  router.py      MiniLM ONNX embedding adjudication
  llm.py         llama.cpp server management (single-model proxy mode)
  oneshot.py     One-shot inference on the SFT distribution; math execution;
                 behavioral code voting; exemplar retrieval priming
  pipelines.py   Legacy verification pipelines (crash fallback) + format verifier
  rag.py         int8 retrieval engine with MMR diversity re-rank
  ner_onnx.py    OntoNotes NER tagger
  sentiment_onnx.py  Sentiment classifier + Mixed rule
  fireworks.py   Escalation client: token budget, batching, model-order hardening
  sandbox.py     Sandboxed Python execution
  lastresort.py  Zero-model deterministic answer floor
finetune/        SFT dataset tooling, training, GGUF conversion, RAG corpus
eval/            Suites, judge, grader-spec EC2 runner
frontend/        Hugging Face Space demo (router / NER / sentiment / retrieval)
```

## Stack

llama.cpp (CPU) · Qwen3-1.7B + LoRA SFT · ONNX Runtime (MiniLM, BERT-NER, RoBERTa-sentiment) · Fireworks AI proxy · Python stdlib orchestrator · Docker

## License

MIT
