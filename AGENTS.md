# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

FrugalRouter: a containerized agent for AMD Hackathon ACT II Track 1. It answers 19 mixed NL tasks (8 categories) under a harness contract — read `/input/tasks.json`, write `/output/results.json`, exit 0, within 2 vCPU / 4 GB RAM / 10 minutes — while minimizing tokens billed through the Fireworks proxy. Two quantized GGUF models run inside the container via llama.cpp (local inference bills zero); only low-confidence answers escalate to Fireworks.

If a competition round is still live: `HANDOFF.md`, `FINDINGS.md`, `EDITS.md`, `FAILURES.md`, `FUTURE.md` (all gitignored, local-only) carry the full operational context and must not be committed while the repo is public.

## Commands

```bash
# build (Apple Silicon hosts MUST cross-build; grader is linux/amd64)
docker buildx build --platform linux/amd64 -t ghcr.io/manan-tech/frugalrouter:vN --load .
docker push ghcr.io/manan-tech/frugalrouter:vN

# primary validation path — GitHub Actions, NOT local (local runs overheat the
# owner's laptop; CI runners are also closer to grader hardware)
gh workflow run eval -R manan-tech/frugalrouter -f tasks=eval/public10.json -f budget=1300
gh workflow run eval -R manan-tech/frugalrouter -f tasks=eval/rehearsal19.json \
  -f soft_s=445 -f flush_s=505 -f hard_s=535 -f budget=1300      # real-limits timing
gh workflow run eval -R manan-tech/frugalrouter -f tasks=eval/variants.json \
  -f budget=1300 -f force_emergency=1                             # escalate-all path
# fetch + judge a finished run
gh run download <run-id> -R manan-tech/frugalrouter -n eval-output -D /tmp/out
python3 eval/judge.py /tmp/out/output/results.json eval/<suite>.json

# local one-off (exact grading constraints), only when unavoidable
eval/run.sh eval/rehearsal19.json ghcr.io/manan-tech/frugalrouter:vN

# trace one category pipeline verbosely inside the container
docker run --rm -v "$PWD/agent:/app/agent:ro" -v "$PWD/eval:/app/eval:ro" \
  --entrypoint python <image> /app/eval/debug_one.py <category> "<prompt>"
```

Ship gate: `eval/public10.json` ≥ 8/10 AND `eval/rehearsal19.json` passes the 80% gate, both in CI, before any image push. `.env` holds `FIREWORKS_API_KEY` for dev/judging only — the grading harness injects its own.

## Architecture

Token economics drive everything: rank = ascending billed tokens among entries above an 80% LLM-judge accuracy gate. Every design choice is "answer locally if verifiable, pay Fireworks only for what verification can't certify."

**Flow** (`agent/main.py`): load tasks (schema-tolerant) → regex-classify into 8 categories (`classify.py`) → sort by category (llama.cpp prompt-prefix reuse) → per-category pipeline returns `Result(answer, confidence, …)` → below-threshold answers escalate in per-category batches → anytime loop re-strengthens weak local answers with leftover wall-clock → last-ditch individual escalations → zero-model deterministic floor (`lastresort.py`). Results are written atomically after every task; a watchdog ladder (soft 445s / flush 505s / hard-exit 535s) makes TIMEOUT and invalid-JSON structurally impossible.

**Verification, not trust** (`pipelines.py`): each category converts model output into something checkable — math executes model-written Python and requires the *general* model's independent boxed answer to agree (same-model agreement provably ratifies systematic miscodes); code generation runs two independently sampled implementations against generated inputs and compares observed behavior; code debugging cross-checks the fix against a from-scratch reference, with echo-of-buggy-code detection; logic extracts constraints to JSON and brute-forces all assignments (ships only a proven-unique solution); summaries/sentiment/NER get grammar-constrained decoding plus programmatic format enforcement; factual answers are *never* trusted locally — their confidence is capped at 0.50 so they always escalate when budget allows. A generic format-compliance verifier handles requirement shapes the parsers don't recognize.

**Local inference** (`llm.py`): two llama-server processes (Qwen3-1.7B general with selective thinking; Qwen2.5-Coder-1.5B), started sequentially with memory-flat flags (`--cache-ram 0`, small batches, q8 KV — the default 8 GiB host prompt-cache OOM-kills tight cgroups). A full-size warmup workout measures true decode speed from the server's own `timings`; below `TPS_DEAD` the local path is declared dead and everything escalates (measured: panic-grade local answers can't clear the gate). A single global lock serializes generation — 2 vCPUs are bandwidth-bound; concurrency only splits it.

**Escalation** (`fireworks.py`): `TokenBudget` uses reserve-by-estimate / commit-by-actual; batches shrink-to-fit on reservation failure and degrade to individual calls rather than skipping. `BATCH_SYS` must remain byte-identical (Fireworks prompt-cache pricing). HTTP 403/404 are per-model verdicts (skip to next model/id-form) — only 401 aborts; a 400 retries the same model with a parameter-minimal body. Language categories order code-named models last (kimi-k2p7-code emits `"..."` placeholders in language batches; junk answers are rejected before they can overwrite local fallbacks). Custom User-Agent is mandatory — Fireworks WAF 403s Python's default.

**Failure containment** (`main.py`): mid-run local death is detected two ways (3-task dead streak; mass-fallback share at escalation time) and promotes to `EMERGENCY_BUDGET_TOKENS` — the normal budget covers only ~7 escalations, which is how a mid-run death once scored 36.8%. Gate survival always outranks token rank.

**Config contract** (`config.py`): the grading harness injects only `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS`. Every other env read (wall-clock overrides, `TEST_FORCE_EMERGENCY`, `CATEGORY_THRESHOLDS_JSON`, `CALIBRATION_LOG_PATH`) exists for the CI harness only — baked defaults must always be production-correct. `ALLOWED_MODELS`, when present, takes strict precedence over `FALLBACK_MODELS`; calls to unlisted models invalidate the submission.

**Hard constraints**: `agent/` is pure Python stdlib (no pip installs in the image). Image must stay well under 5 GB compressed. Everything the container needs is baked at build time (`HF_HUB_OFFLINE=1`); the only permitted egress is the Fireworks proxy — any other network routing disqualifies the submission.
