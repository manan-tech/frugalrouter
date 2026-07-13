# FrugalRouter — Presentation Content (final, v25)

Deck for AMD Developer Hackathon ACT II, Track 1 submission. 10 slides + optional appendix.
Each slide: TITLE / ON-SLIDE CONTENT / SPEAKER NOTES / VISUAL direction.
Numbers below are all real and defensible — grader-scored or measured on a dedicated
grader-spec EC2 replica (c6i.large, 2 vCPU / 4 GB / no swap, exact harness contract).

---

## Slide 1 — Title

**ON SLIDE**
- **FrugalRouter**
- Fine-tuned. Locally verified. Token-frugal.
- AMD Developer Hackathon — ACT II, Track 1 · Team TechMavericks
- Final submission: `ghcr.io/manan-tech/frugalrouter:v25`
- **84.2% hidden-set accuracy · 3,388 Fireworks tokens**

**SPEAKER NOTES**
One line: we built an agent that answers 8 categories of NL tasks on 2 CPU cores
inside a Docker container, and pays a remote API only for what it provably cannot
verify itself.

**VISUAL** Logo-style wordmark; the two headline numbers big.

---

## Slide 2 — The problem & the scoring game

**ON SLIDE**
- 19 hidden tasks, 8 categories: factual · math · sentiment · summary · NER · code-gen · code-debug · logic
- Hard box: 2 vCPU · 4 GB RAM · 10 minutes · Docker
- Scored by an LLM judge vs rubrics; **80% accuracy gate**
- **Rank = fewest API tokens among gate-passers**
- → every token you don't spend is rank; every point below 80% is elimination

**SPEAKER NOTES**
The scoring creates a knife-edge optimization: accuracy is binary (pass/fail gate),
tokens are the currency. The whole system design follows from this one sentence:
"answer locally if verifiable, pay only for what verification can't certify."

**VISUAL** Two-axis diagram: accuracy gate as a wall at 80%, token count as the
race beyond the wall.

---

## Slide 3 — Architecture (the money slide)

**ON SLIDE** (flow diagram)
```
tasks.json → classify (regex + MiniLM embeddings, 0 tokens)
          → fine-tuned Qwen3-1.7B, ONE generation per task (llama.cpp, CPU)
              · math: model writes Python → we EXECUTE it → stdout ships
              · code: 3-sample behavioral vote on a fixed input battery
              · all: format verifier (programmatic constraint checks)
          → targeted escalation (Fireworks, batched, budget-capped)
              · only categories a 1.7B measurably can't self-verify
          → results.json (atomic per task; watchdog ladder 445/505/535s)
```
- 4 local models in a 1.4 GB image: fine-tuned 1.7B + 3 ONNX (router/NER/sentiment)

**SPEAKER NOTES**
Everything that can be checked deterministically stays local and free: executed
math, behaviorally-fingerprinted code, programmatic format enforcement. The
remote API is a scalpel, not a crutch — batched, format-hinted, under a hard
token budget with reserve-by-estimate / commit-by-actual accounting.

**VISUAL** The flow as a vertical pipeline; local zone tinted green ("0 tokens"),
escalation zone tinted amber with a budget meter.

---

## Slide 4 — The core insight: accuracy in the weights, not in sampling

**ON SLIDE**
- v1–v20: weak local model + sampling (3 generations, majority voting, cross-checks)
- Sampling quality lives in **wall-clock** → the grader's contended box strips it
- Same image, different days: **36.8% → 68.4% → 47.4%** (identical bytes!)
- The fix: **LoRA fine-tune Qwen3-1.7B** on 2,708 category-exact examples → ONE generation per task
- Slow box now makes answers *later, never worse*: 19 tasks ≈ 170s at grader speed (4× headroom)

**SPEAKER NOTES**
This is the biggest lesson of the competition. We diagnosed six grader failures
to the same root cause: our accuracy was stored in compute-hungry sampling, and
the grading box's speed decided our score. A fine-tune moves the accuracy into
the weights where the box can't take it away.

**VISUAL** Two bars: "sampling build" quality melting as box slows; "fine-tuned
build" flat line. The 36.8/68.4/47.4 identical-digest scores as scattered dots.

---

## Slide 5 — The fine-tune, end to end (one afternoon)

**ON SLIDE**
- Dataset: **2,708 examples**, ~12.6% per category, agent-generated + machine-validated
  (every math answer EXECUTES; every code example passes its own tests)
- Trick: math answers ARE Python programs — a 1.7B can't do compound interest in
  its head, but it reliably writes the 4-line program that can
- LoRA r=32 · 3 epochs · completion-only loss · byte-exact chat-template rendering
- Trained in **~11 min** on a g6e.2xlarge spot → GGUF Q4_K_M (1.1 GB) → one `COPY` into the image
- Result: exact output formats, one-shot, temperature 0

**SPEAKER NOTES**
The dataset was generated and adversarially reviewed by a multi-agent workflow —
21 agents found the fatal bugs (a quantize script that died unconditionally, a
chat-template mismatch) before we spent GPU money. Formats took perfectly:
first smoke test output was byte-exact "Entity | TYPE" lines.

**VISUAL** Pipeline strip: dataset → validate → LoRA → merge → GGUF → Docker COPY,
with the 11-minute timer on the training step.

---

## Slide 6 — Zero-token specialists (ONNX)

**ON SLIDE**
- **Router**: MiniLM embeddings adjudicate the regex classifier — survives paraphrased
  finals prompts (16/16 vs 8/16 regex-only)
- **NER**: OntoNotes BERT with a native DATE class ("last April", "three years ago") — 6 ms/sentence
- **Sentiment**: RoBERTa classifier + clause-level Mixed rule ("delicious BUT we waited an hour")
- **Code exemplars**: 204 execution-verified reference implementations, retrieved by
  MMR-diversified similarity — the model adapts correct code instead of writing from scratch
- All int8 ONNX · 253 MB total · zero tokens · zero API dependency

**SPEAKER NOTES**
A purpose-trained 100 MB model beats a general 600 MB LLM at its own task. The
router matters most: classification is the single point of failure for every
pipeline behind it, and the finals re-run on rephrased prompts.

**VISUAL** Four cards with model sizes and one killer example each.

---

## Slide 7 — Verification, not trust

**ON SLIDE**
- Math: the program **runs** — no arithmetic hallucination possible
- Code: 3 samples, behaviorally fingerprinted on a fixed input battery →
  majority behavior ships (defeats CPU float-noise coin flips — measured: the same
  temp-0 prompt gives correct/wrong code on different thread counts!)
- Formats: programmatic checks for stated constraints ("exactly two sentences",
  "3 bullets under 15 words") with one guided retry, then escalation
- Failure containment: per-task isolation · dead-streak detector · emergency
  budget promotion · zero-model deterministic floor → an answer for every task, always

**SPEAKER NOTES**
The float-noise discovery is worth telling: identical prompts, temperature 0,
produce different code on 2 threads vs 4 threads — knife-edge logits flip on
accumulation order. Majority-of-3 behavioral voting makes shipping deterministic.

**VISUAL** A "trust nothing" checklist; the float-noise anecdote as a callout box.

---

## Slide 8 — The detective story: cracking the stuck 78.9%

**ON SLIDE**
- Three architecturally different builds → **identical 78.947% (15/19)** on the hidden set
- Organizer breakdown of our fails: 2 logic · 1 NER · 1 math — all local one-liners
- Response: format-compliance verifier on every one-shot answer + targeted escalation
- **Result: 84.2% — gate PASSED** (16/19 @ 3,388 tokens)
- Validated same-day on a dedicated grader-spec EC2: **19/19 rehearsal + 10/10 public**

**SPEAKER NOTES**
When three different systems fail identically, the failure isn't the system.
We used the organizers' category data to aim the fix precisely. Also honest:
we measured our "obvious improvements" — a RAG-verified factual mode and local
exemplar code — and they regressed accuracy or tokens on the validation box, so
they were cut. Measurement beat intuition every time this weekend.

**VISUAL** Timeline of grader scores: 36.8 → 0.0 → 57.9 → 57.9 → 52.6 → 78.9 →
78.9 → 78.9 → **84.2** with the gate line at 80%.

---

## Slide 9 — Results

**ON SLIDE**

| | Accuracy | Tokens |
|---|---|---|
| Hidden set (grader, final) | **84.2% — PASS** | **3,388** |
| rehearsal19 (grader-spec EC2) | **19/19 = 100%** | 3,212 |
| public10 (grader-spec EC2) | **10/10 = 100%** | 771 |
| Zero-token mode (no API key) | 89.5% rehearsal | **0** |

- Wall-clock: 170–195 s of the 600 s limit · Peak RSS ~1.5 GB of 4 GB · Image 1.4 GB
- Every task answered on every run: no timeouts, no invalid JSON, ever (watchdog ladder)

**SPEAKER NOTES**
The zero-token line shows the ceiling of the local-only path — it passes our gate
locally and is the roadmap item (see next slide). The 771-token public10 shows
what targeted escalation looks like when the local model is strong: only factual paid.

**VISUAL** Big table; a small "reliability: 100% completion" badge.

---

## Slide 10 — What's next

**ON SLIDE**
- **~1,000-token build**: exemplar-adapted local code + verify-only factual — all
  components built & committed, needs one more validation loop
- **RAG knowledge base**: 234,796 Wikipedia chunks embedded (90 MB int8) + MMR
  retrieval engine — needs an answer-in-context verification gate (cosine alone
  can't separate right from wrong retrieval — measured)
- Live demo: Hugging Face Space — router, NER, sentiment, exemplar retrieval, interactive
- Everything reproducible: dataset tooling → LoRA → GGUF → image, one afternoon end-to-end

**SPEAKER NOTES**
Close on the thesis: small models + deterministic verification + surgical API use
is a general recipe for cheap, reliable agents — not just a competition trick.

**VISUAL** Roadmap arrows; QR code / link to the repo and the HF Space.

---

## Appendix slides (optional, for Q&A)

**A1 — Token economics detail**: reserve-by-estimate/commit-by-actual budget,
batched escalation with byte-identical system prefix (Fireworks prompt-cache
pricing), shrink-to-fit on reservation failure, per-category output caps.

**A2 — Memory engineering**: default llama.cpp host prompt-cache (8 GiB!)
OOM-kills a 4 GB cgroup — `--cache-ram 0`, small ubatch, q8_0 KV cache; single
server, coder proxied to general (never load the same GGUF twice).

**A3 — What we measured and killed**: few-shot sentiment (0/2), a 0.6B
classifier (4/9), a 3B local model (blows the wall-clock), cosine-gated
factual RAG (wrong article outscored right article), remote NER escalation
(overwrote a correct local answer). Every cut was a measurement, not an opinion.

**A4 — Multi-agent engineering process**: dataset generated and reviewed by
agent workflows (4 generators + execution verifier: 205/205 exemplars passed);
21-agent adversarial review found the fatal GGUF-conversion bug pre-GPU;
grader-spec EC2 replica for every ship decision.
