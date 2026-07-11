# FrugalRouter — AMD Hackathon ACT II, Track 1 Design

**Date:** 2026-07-11 · **Deadline:** today, ~18:00 IST (6h) · **Goal:** top-3 on first submission

## Objective

Win Track 1 (Hybrid Token-Efficient Routing Agent): lowest total Fireworks tokens with LLM-judged
accuracy ≥ 80% (16/19 tasks). Current #1 = 1,377 tokens @ 89.5%. Target: **≤ 600 tokens @ ≥ 90%**,
with a budget knob that can go to 0 tokens for v2.

## Verified constraints (participant guide + Discord announcements)

- Grading env: **2 vCPU, 4 GB RAM, 10 min wall-clock**, linux/amd64, no GPU.
- Container contract: read `/input/tasks.json` (`[{task_id, prompt}]`), write `/output/results.json`
  (`[{task_id, answer}]`), exit 0. Malformed output = 0.
- Env injected at grading: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` (may be absent
  in dev; "any Fireworks serverless model" per organizers).
- **Local inference = 0 tokens, explicitly legal** (ZERO_API_CALLS flag is fine). All Fireworks calls
  must go through `FIREWORKS_BASE_URL`. Non-Fireworks external APIs ⇒ DQ. No hardcoding/caching
  answers; final rescoring uses **refreshed randomized prompt variants**.
- 19 fixed hidden tasks across 8 categories: factual, math, sentiment, summarisation, NER,
  code debugging, logic puzzles, code generation. Accuracy gate = 80%.
- Image ≤ 10 GB but **keep < 5 GB** (PULL_ERROR risk); 10 submissions/hr; ~1 h scoring turnaround.

## Architecture

Single Python-stdlib orchestrator (no pip deps) + two llama.cpp servers baked into the image:

- **General:** unsloth/Qwen3-1.7B-GGUF `Qwen3-1.7B-Q4_K_M.gguf` (1.11 GB) — thinking OFF by default,
  selectively ON (capped) for math/logic when wall-clock allows. Local thinking costs 0 tokens.
- **Coder:** Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF `q4_k_m` (1.12 GB) — code gen/debug (+ math codegen).
- Both resident (~2.6 GB with KV); requests serialized, `-t 2`, small ctx, per-category shared prompt
  prefixes with llama.cpp prompt cache.

### Flow

1. Read tasks; classify into 8 categories via regex/keyword heuristics (near-unambiguous wording);
   local-model tiebreak vote for ambiguous cases.
2. **Startup TPS probe** → speed governor: if measured tok/s below thresholds, reduce samples/thinking
   and raise escalation quota (never TIMEOUT).
3. Per-category pipelines with **deterministic verification** (the accuracy engine):
   - **Math:** model emits Python program → sandboxed exec → numeric answer; 2–3 samples,
     majority on the *number*; disagreement ⇒ escalation candidate.
   - **Code gen/debug:** coder model writes function → auto-derived sanity tests run in subprocess →
     ≤ 2 repair rounds with real tracebacks fed back.
   - **Logic:** model extracts entities/constraints as predicates → brute-force all assignments →
     proven answer; fallback = direct answer + self-consistency.
   - **Factual:** 3-sample self-consistency; divergence ⇒ escalation candidate.
   - **Sentiment/NER/Summary:** GBNF grammar-constrained or template outputs; programmatic format
     enforcement (sentence counter, label whitelist, entity→type lines).
4. **Confidence scoring** per task (verification outcome, sample agreement, category prior) →
   bottom-K escalate to strongest `ALLOWED_MODELS` entry (runtime-read; curated serverless fallback
   list; prefer Gemma IDs for language categories if present) — raw task prompt, terse system prompt,
   tight max_tokens, parallel calls, **hard global budget ≈ 500 tokens** (v1). Budget=0 ⇒ pure local.
5. **Anytime loop:** leftover wall-clock re-samples lowest-confidence answers.
6. **Safety:** per-task exception walls (always an answer); atomic incremental writes of
   `/output/results.json` (valid JSON after every task); hard watchdog flush at ~8.5 min; exit 0 always.

### Container

Multi-stage Dockerfile → final `python:3.12-slim` (linux/amd64) + `llama-server` binary
(portable CPU build from official ghcr.io/ggml-org/llama.cpp image) + 2 GGUFs. Target < 4 GB.
`HF_HUB_OFFLINE=1`; only outbound traffic is `FIREWORKS_BASE_URL`.

## Eval harness (pre-submission gate)

`eval/`: ~40 authored variant tasks (5/category, practice-difficulty, paraphrase-style) + LLM-judge
replica (Fireworks, own key, rubric: correctness vs expected intent + format compliance) + timing
runs under `docker run --platform linux/amd64 --cpus=2 --memory=4g` (Rosetta = conservative bound).
**Ship only if ≥ 17/19-equivalent (≥ 90%) across 3 runs and wall-clock ≤ 8.5 min emulated.**

## Submission strategy

- v1 at ~T+2.5h: budget ≈ 500 tokens (escalations = weakest tasks). Banks early timestamp
  (tie-break insurance), calibrates real-judge accuracy.
- v2 at ~T+4.5–5h only if locally proven better: tuned budget (possibly 0) informed by v1 scored
  accuracy + leaderboard. Resubmits can downgrade — only submit strict improvements.
- Registry: GHCR public under manan-tech. Repo: public GitHub with README (submission requirement).

## Risks & mitigations

- **Gate failure** → verification engine + local gate ≥ 90% before ship + escalation of weak tasks.
- **TIMEOUT** → 1.5–1.7B models only, TPS governor, anytime design, 8.5-min flush.
- **PULL_ERROR** → image < 4 GB, GHCR, push early, anonymous-pull check.
- **Judge format quirks** → per-category answer templates: direct answer first + one-line support.
- **ALLOWED_MODELS surprises** → strict runtime intersection; multi-candidate fallback chain on 404.
