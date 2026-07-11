# Slide deck + video script (build from this)

## Slides (7 slides, ~60–90s presentation)

### Slide 1 — Title
**FrugalRouter**
Hybrid Token-Efficient Routing Agent · AMD Developer Hackathon: ACT II, Track 1
`ghcr.io/manan-tech/frugalrouter:v1` · github.com/manan-tech/frugalrouter

### Slide 2 — The problem
- Enterprises want AI spend control without sacrificing quality
- Track 1 challenge: complete 19 varied tasks (math, code, logic, NER, …) using the fewest Fireworks tokens while staying above an accuracy gate
- Grading box: 2 vCPU, 4 GB RAM, 10 minutes — no GPU

### Slide 3 — The idea: don't route to the cheapest cloud model. Make cloud calls unnecessary.
- Two small quantized LLMs (Qwen3-1.7B + Qwen2.5-Coder-1.5B) baked into the container, running CPU-only via llama.cpp
- Local inference = 0 tokens
- Fireworks API reserved for the few answers local verification can't certify

### Slide 4 — Verification, not trust (the core innovation)
- Math → model writes Python, we EXECUTE it, majority-vote the number
- Code → two independent implementations must AGREE on observed behavior
- Logic → constraints extracted, every assignment BRUTE-FORCED, unique solution proven
- Formats → grammar-constrained decoding (GBNF), sentence-count enforcement
- Each answer gets a confidence score from its verification outcome

### Slide 5 — Smart escalation
- Lowest-confidence answers escalate, cheapest-first, sequentially
- Live-measured most frugal model: gpt-oss-120b @ reasoning_effort=low (137 tokens for a full word problem)
- Hard global budget: 900 tokens (0 = pure zero-API mode)
- Factual always escalates: sample agreement can't verify facts — and it's the cheapest category to fix

### Slide 6 — Engineering for a hostile clock
- Startup tok/s probe + speed governor (full → lean → panic modes)
- Atomic results.json after every task · per-task exception isolation
- Watchdog flush at 8.5 min → container structurally cannot TIMEOUT or emit invalid JSON
- 19 tasks in 232 s measured at grading constraints (2 vCPU / 4 GB, native x86)

### Slide 7 — Results
- 90.0% accuracy zero-escalation · 95.0% with escalation (40-task unseen-variant suite, LLM-judged)
- Expected total spend on a 19-task set: ~300–700 tokens
- Image: 2.24 GB · pure-stdlib Python · fully containerized
- Stack: llama.cpp · Qwen · Fireworks AI · Docker · GitHub Actions CI

## Video script (~75 seconds, screen-record the repo + a terminal run)

[0–10s, title slide]
"This is FrugalRouter, our entry for Track 1 — the hybrid token-efficient routing agent."

[10–25s, slide 3]
"Most routers try to pick the cheapest cloud model for each task. We inverted the problem: we make cloud calls unnecessary. Two small quantized models run entirely inside the grading container on two CPU cores — so most answers cost exactly zero tokens."

[25–45s, slide 4, or cut to terminal running eval/run.sh]
"Small models can't be trusted raw, so nothing ships unverified. Math answers come from Python programs we actually execute. Generated code only passes when two independent implementations agree on real observed behavior. Logic puzzles are solved by brute-forcing every possibility and proving the answer is unique."

[45–60s, slide 5]
"Every answer carries a confidence score. Only the weakest escalate to Fireworks — cheapest first, capped by a hard 900-token budget. We measured every serverless model and picked gpt-oss-120b at low reasoning effort: about 137 tokens for a full word problem."

[60–75s, slide 7]
"The result: ninety percent accuracy with zero external tokens, ninety-five with escalation, nineteen tasks in under four minutes on the grading hardware. FrugalRouter — verified answers, almost free."
