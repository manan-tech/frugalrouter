# lablab.ai Submission — copy-paste fields

**Docker image (the field the grader pulls):**
```
ghcr.io/manan-tech/frugalrouter:v7
```

**Public GitHub repository:**
```
https://github.com/manan-tech/frugalrouter
```

**Project title:**
```
FrugalRouter — verified local-first routing, ~0 tokens
```

**Short description:**
```
A hybrid routing agent that answers almost every task with ZERO Fireworks tokens: two small quantized models (Qwen3-1.7B + Qwen2.5-Coder-1.5B) run entirely inside the container on 2 vCPUs, wrapped in deterministic verification — math is executed, code is behaviorally cross-checked, logic puzzles are brute-force proven. Only the lowest-confidence tasks escalate to the most token-frugal Fireworks model (gpt-oss-120b @ low reasoning), under a hard 900-token global budget.
```

**Long description:**
```
FrugalRouter inverts the routing problem: instead of predicting which cloud model is cheapest, it makes cloud calls unnecessary. Two small GGUF models are baked into the image and run CPU-only inside the grading container (llama.cpp, 2 threads, ~2.6 GB RAM) — local inference costs zero leaderboard tokens by the rules.

Small models fail raw prompting, so every category gets a verification harness instead of trust:
• Math → the model emits a Python program; we execute it and majority-vote on the computed number.
• Code generation → two independently sampled implementations run against model-generated inputs; agreement on observed behavior is required (asserts with guessed expected values are exactly what small models get wrong).
• Code debugging → the fix is cross-checked behaviorally against a from-scratch reference implementation; the bug description is generated from the actual before/after diff.
• Logic puzzles → entities and constraints are extracted to JSON and every assignment is brute-forced; the answer ships only if a unique solution is proven.
• Summarisation → programmatic sentence/word-count and truncation enforcement with regeneration.
• Sentiment → grammar-constrained decoding (GBNF): the output physically cannot violate the label+justification format; labels are majority-voted.
• NER → two-sample agreement, verbatim-presence filtering against the source text, and deterministic span re-expansion ("2024" → "summer 2024").
• Factual → the one category verification can't save (agreement ≠ truth), so factual tasks always escalate when budget allows — they're also the cheapest to escalate (~150 tokens each).

Every answer carries a confidence score from its verification outcome. The weakest answers escalate, cheapest-first, to the most token-frugal strong model measured live on Fireworks (gpt-oss-120b with reasoning_effort=low — 137 total tokens for a full word problem), under a hard global budget of 900 tokens. Set the budget to 0 and the agent is a pure zero-token, ZERO_API_CALLS submission.

Robustness: startup tok/s probe with a speed governor that degrades sampling depth instead of timing out, atomic incremental results.json writes after every task, per-task exception isolation, and a watchdog that flushes at 8.5 minutes — the container structurally cannot TIMEOUT, crash, or emit invalid JSON. Measured on native x86 CI at 2 vCPU / 4 GB: 40 tasks in 429 s (19-task sets finish in ~3.5 min), 95% accuracy on an unseen 40-task variant suite judged by an LLM-judge replica.

Stack: llama.cpp (CPU dispatch build), Qwen3-1.7B Q4_K_M, Qwen2.5-Coder-1.5B Q4_K_M, pure-stdlib Python orchestrator, Fireworks AI API for escalation and judging. Image 2.24 GB.
```

**Technology tags:** `llama.cpp`, `Qwen`, `Fireworks AI`, `Docker`, `Python`, `gpt-oss`

**Category tags:** `AI Agents`, `Routing`, `Token Efficiency`

---

## Pre-submit checklist

- [ ] Image pushed: `docker manifest inspect ghcr.io/manan-tech/frugalrouter:v7`
- [ ] Package set to **public** (anonymous pull verified)
- [ ] Repo public with README
- [ ] Image field contains ONLY `ghcr.io/manan-tech/frugalrouter:v7` (no digest, no extra text — per organizer announcement)
