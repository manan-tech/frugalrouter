# FrugalRouter — Hybrid Token-Efficient Routing Agent

**AMD Developer Hackathon: ACT II · Track 1**

A local-first hybrid agent that answers as many tasks as possible with **zero
Fireworks tokens** — two small quantized models run *inside the container* on
CPU, wrapped in deterministic verification — and escalates only its
lowest-confidence answers to Fireworks under a hard global token budget.

## Why this wins the token leaderboard

Rank = ascending total Fireworks tokens (above an 80% accuracy gate). Local
inference is explicitly legal and counts as **zero**. The hard part is passing
the accuracy gate with small models on the grading box (2 vCPU / 4 GB / 10
min). FrugalRouter solves that with *verification, not model size*:

| Category | Strategy |
|---|---|
| Math | Model writes a Python program → we **execute it** → majority vote on the computed number |
| Code generation | Model writes the function → auto-generated asserts **actually run** → repair loop on real tracebacks |
| Code debugging | Same verify-and-repair loop, plus a one-sentence bug statement |
| Logic puzzles | Model extracts entities/constraints → **brute-force solver proves** the unique answer |
| Summarisation | Programmatic sentence/word-count enforcement with regenerate loop |
| Sentiment | Grammar-constrained decoding (GBNF) — output physically cannot break format; label vote |
| NER | Two-sample agreement + verbatim-presence check against the source text |
| Factual | 3-sample self-consistency; divergence → escalation candidate |

Every task gets a confidence score (verification outcome + sample agreement).
The weakest tasks — and only those — escalate to the most token-frugal strong
model available (measured live: `gpt-oss-120b` with `reasoning_effort: low`),
under a **hard global budget** (default 500 tokens; `0` = pure zero-token mode).

Robustness: startup tok/s probe with a speed governor (degrades gracefully on
slow CPUs), atomic incremental `results.json` writes after every task, a
watchdog that flushes at 8.5 min and exits cleanly — the container cannot
time out, crash, or emit invalid JSON.

## Architecture

```
/input/tasks.json
   └─ classify (regex, zero-cost) ─ sort by category (prompt-cache reuse)
        └─ per-category pipeline ──────────────► verified answer + confidence
             ├─ general: Qwen3-1.7B  Q4_K_M (llama.cpp, CPU, ~1.1 GB)
             ├─ coder:   Qwen2.5-Coder-1.5B Q4_K_M (~1.1 GB)
             └─ Python sandbox (exec / asserts / brute-force solver)
        └─ lowest-confidence tasks ──► Fireworks (ALLOWED_MODELS, budget-capped)
        └─ anytime loop: leftover wall-clock re-strengthens weak answers
/output/results.json  (atomic write after every task)
```

Everything is Python stdlib — no pip dependencies.

## Build

```bash
docker buildx build --platform linux/amd64 -t frugalrouter .
```

The image (~2.3 GB) bakes in both GGUF models and a portable llama.cpp CPU
build with per-CPU dispatch (SSE4.2 → Zen4).

## Run (exactly like the grading harness)

```bash
mkdir -p input output
cp tests/sample_input/tasks.json input/
docker run --rm --platform linux/amd64 --cpus=2 --memory=4g \
  -e FIREWORKS_API_KEY=... \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=... \
  -v "$PWD/input:/input:ro" -v "$PWD/output:/output" \
  frugalrouter
cat output/results.json
```

Without a Fireworks key the agent still completes every task locally
(zero-token mode) — escalation is an optimization, not a dependency.

## Local evaluation harness

```bash
eval/run.sh eval/variants.json     # 40 unseen-variant tasks, graded env limits
python3 eval/judge.py output/results.json   # LLM-judge replica + 80% gate check
```

## Configuration

All tunables live in `agent/config.py` and are baked in at build time (the
harness injects only `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`,
`ALLOWED_MODELS`). Key knob: `ESCALATION_BUDGET_TOKENS` (500 default, 0 = never
call Fireworks).

## License

MIT
