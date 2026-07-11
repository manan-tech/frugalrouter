# FrugalRouter

**A hybrid, token-efficient routing agent — AMD Developer Hackathon: ACT II, Track 1**

FrugalRouter answers general-purpose AI tasks (factual Q&A, math, sentiment, summarisation, NER, code debugging, logic puzzles, code generation) using the fewest external tokens possible. Two small quantized language models run **entirely inside the container on CPU**, wrapped in deterministic verification machinery. Only the answers the system cannot verify locally are escalated to the Fireworks AI API — cheapest-first, under a hard global token budget.

## How it works

```
/input/tasks.json
   │
   ├─ 1. Classify each task (regex heuristics, zero cost)
   ├─ 2. Sort by category (llama.cpp prompt-cache reuse)
   │
   ├─ 3. Per-category pipeline → verified answer + confidence score
   │      ├─ general model: Qwen3-1.7B Q4_K_M (llama.cpp, CPU)
   │      ├─ coder model:   Qwen2.5-Coder-1.5B-Instruct Q4_K_M
   │      └─ Python sandbox: execution, tests, brute-force solving
   │
   ├─ 4. Escalate lowest-confidence answers → Fireworks AI
   │      (ALLOWED_MODELS env, cheapest-first, hard token budget)
   │
   ├─ 5. Anytime loop: leftover wall-clock re-strengthens weak answers
   │
   └─ /output/results.json   (written atomically after every task)
```

### Verification, not trust

Small models are unreliable when prompted raw, so every category gets a deterministic harness:

| Category | Strategy |
|---|---|
| Math | Model writes a Python program → the program is **executed** → majority vote across samples on the computed number |
| Code generation | Two independently sampled implementations run against model-generated inputs → **agreement on observed behavior** required; disagreement triggers a third implementation and majority vote |
| Code debugging | The fix is behaviorally cross-checked against a from-scratch reference implementation; the bug description is derived from the actual before/after diff |
| Logic puzzles | Entities and constraints extracted to JSON → **every assignment brute-forced** → answer ships only if a unique solution is proven |
| Summarisation | Programmatic sentence/word-count and truncation checks with a regeneration loop |
| Sentiment | Grammar-constrained decoding (GBNF) — output cannot violate the `Label - justification` format; labels are majority-voted |
| NER | Two-sample agreement, verbatim-presence filtering against the source text, deterministic span re-expansion (`"2024"` → `"summer 2024"`) |
| Factual | Multi-sample self-consistency; escalates when the budget allows, since sample agreement cannot verify facts |

### Confidence-based escalation

Every answer carries a confidence score derived from its verification outcome (tests passed, solver proved, samples agreed). Tasks below the threshold escalate to the strongest model available through `FIREWORKS_BASE_URL`, cheapest-estimated-first, sequentially, under a hard global budget (`ESCALATION_BUDGET_TOKENS`, default 900). Escalation uses `reasoning_effort: low` and tight output caps to minimize token use. With the budget set to `0` the agent runs fully local and makes zero API calls.

### Reliability

- **Startup throughput probe + speed governor** — measures actual tok/s and reduces sampling depth (full → lean → panic) rather than risking the 10-minute limit
- **Atomic incremental output** — `/output/results.json` is valid JSON after every single task
- **Per-task exception isolation** — a failing task can never take down the run
- **Watchdog** — forced flush at 8.5 min, guaranteed clean exit before the harness timeout
- Pure-stdlib Python orchestrator; no pip dependencies

## Build

```bash
docker buildx build --platform linux/amd64 -t frugalrouter .
```

The multi-stage build downloads a portable llama.cpp CPU release (with per-CPU dispatch: SSE4.2 → Haswell → Skylake-X → Zen 4) and both GGUF models, producing a ~2.2 GB self-contained image.

## Run

The container follows the Track 1 harness contract: read `/input/tasks.json`, write `/output/results.json`, exit 0.

```bash
mkdir -p input output
cp tests/sample_input/tasks.json input/

docker run --rm --platform linux/amd64 --cpus=2 --memory=4g \
  -e FIREWORKS_API_KEY=fw_... \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/gpt-oss-120b \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  frugalrouter

cat output/results.json
```

Input format:

```json
[
  { "task_id": "t1", "prompt": "..." },
  { "task_id": "t2", "prompt": "..." }
]
```

Output format:

```json
[
  { "task_id": "t1", "answer": "..." },
  { "task_id": "t2", "answer": "..." }
]
```

Without a Fireworks key the agent still completes every task locally — escalation is an optimization, not a dependency.

## Evaluation harness

The repo includes a 40-task variant suite (5 per category) with grading rubrics, plus an LLM-judge script:

```bash
eval/run.sh eval/variants.json            # run under grading constraints
python3 eval/judge.py output/results.json # judge + accuracy report
```

A 19-task rehearsal set (`eval/rehearsal19.json`) mirrors the real task distribution for timing runs. A GitHub Actions workflow (`.github/workflows/eval.yml`) runs the full build + eval + judge on native x86 runners.

### Measured results (2 vCPU / 4 GB, native x86)

| Metric | Value |
|---|---|
| 19-task run, default time limits | 168–232 s wall clock |
| 40-task accuracy, with escalation | 97.5 % |
| 40-task accuracy, zero escalation | 90.0 % |
| Emergency mode (local inference disabled) | 95.0 % via batched escalation |
| Fireworks tokens, 19-task set (normal) | ~410–530 (measured) |
| Fireworks tokens per escalated task | ~80–160 batched (measured) |
| Image size | 2.24 GB |

## Configuration

All tunables live in `agent/config.py` and are baked in at build time. The grading harness injects only `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` (read at runtime; when absent, a curated serverless fallback list is used).

| Knob | Default | Meaning |
|---|---|---|
| `ESCALATION_BUDGET_TOKENS` | 900 | Hard global cap on Fireworks tokens (0 = fully local) |
| `ESCALATE_CONF_THRESHOLD` | 0.55 | Confidence below which a task escalates |
| `TPS_FULL` / `TPS_LEAN` | 10 / 5 | Speed-governor thresholds (tok/s) |
| `SOFT_NEW_WORK_S` / `FLUSH_S` / `HARD_EXIT_S` | 445 / 505 / 535 | Wall-clock guards |

## Repository layout

```
agent/
  main.py        Orchestrator: contract I/O, watchdog, escalation, anytime loop
  config.py      All tunables
  classify.py    Zero-cost category classifier
  llm.py         llama.cpp server management + chat client
  pipelines.py   Per-category verification pipelines
  fireworks.py   Escalation client with token budget
  sandbox.py     Sandboxed Python execution for verification
  util.py        Atomic writes, text normalization, helpers
eval/
  variants.json     40-task unseen-variant suite with rubrics
  rehearsal19.json  19-task timing rehearsal set
  judge.py          LLM-judge accuracy report
  run.sh            Run container under grading constraints
  debug_one.py      Trace a single pipeline verbosely
Dockerfile          Multi-stage linux/amd64 build
.github/workflows/  CI eval on native x86 runners
```

## Stack

llama.cpp (CPU) · Qwen3-1.7B · Qwen2.5-Coder-1.5B-Instruct · Fireworks AI API (gpt-oss-120b and other serverless models) · Python stdlib · Docker

## License

MIT
