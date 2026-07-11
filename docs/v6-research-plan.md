# FrugalRouter — Final Prioritized Implementation Plan

## 1. Executive Summary

Your rank is decided almost entirely by escalation tokens, and you already sit ~11–13pp above the 16/19 (84.2%) gate — so the winning move is to **cut escalations and make the survivors cheap, without touching the ~2–4pp of true-accuracy slack you actually have** (binomial math: clearing the n=19 gate at 95% confidence needs only ~93% true accuracy, not the ~96% you run at). The three biggest safe levers are (a) replacing the global 0.55 cutoff with **per-category calibrated, risk-certified thresholds fit offline on your 40-task suite** (0 tokens, provably gate-safe, ~30–50% fewer escalations), (b) making every surviving escalation cheaper via **batching into one gpt-oss call + stable-prefix prompt caching + per-category `reasoning_effort`/`max_tokens` + terse output** (~40–60% off the residual bill), and (c) **decoding/sampling hygiene** (Qwen3 official params, never greedy, thinking-gated by category, min-p for self-consistency, `\boxed{}`) which removes silent repetition-loop and parse failures that currently cause both wrong answers and wasted escalations. Your one measured *accuracy* hole — **code_debug** — is closed by a local execute→traceback→repair loop, and **logic** is hardened for free by a solver-cardinality certificate (unique-solution == valid). The one thing to be careful with is your other weak spot, **factual recall**: naively keeping factual answers local via lexical agreement is dangerous because small models are confidently wrong, so that lever is P1 and hard-gated behind an NLI + self-certainty check, validated on the suite before shipping.

---

## 2. Ranked Plan (P0 / P1 / P2)

Deltas are expectations on the **19-task hidden set**; token deltas are relative to your current 400–900-token bill. "Item" = expected whole tasks out of 19.

### P0 — do these first (safe, highest leverage)

**P0-1 · Decoding & thinking-mode hygiene** *(bundle: Qwen3 card, min-p arXiv:2407.01082, CRANE-adjacent, overthinking arXiv:2510.07880)*
- **Change:** In the llama.cpp router, split sampler config by model/mode: Qwen3 thinking (math/logic/code_debug) `temp 0.6 / top_p 0.95 / top_k 20 / min_p 0`; Qwen3 non-thinking (sentiment/NER/summary/factual) `0.7 / 0.8 / 20`. **Never greedy in thinking mode.** For the self-consistency sampling endpoints only (math vote, NER 2-sample, code agreement) switch to `temp 0.9 / min_p 0.07 / top_p off / top_k 20` for diverse-but-coherent paths. Add per-category `enable_thinking` flag (OFF for sentiment/NER/summary/factual, ON with a ~1–2k token budget for math/logic/debug). Math prompt: append "put your final answer within `\boxed{}`" and parse from it.
- **Accuracy Δ:** +0 to +1 item (eliminates repetition-loop failures on math/logic/code and overthinking-flips on easy categories; kills silent parse failures).
- **Token Δ:** net-negative, ~−50 to −150 (fewer garbage-answer escalations); frees wall-clock.
- **Effort:** 6h

**P0-2 · Per-category calibrated + risk-certified escalation thresholds** *(UCCI isotonic arXiv:2605.18796; self-certainty arXiv:2502.18581; token-margin/Kadavath; LTT arXiv:2110.01052; binomial gate math)*
- **Change:** Replace the single 0.55 gate. Base signals, read free from llama.cpp logprobs: **self-certainty (KL-to-uniform)** for free-form, **token-margin (top1−top2 at the answer token)** for structured (sentiment GBNF label, NER boundaries, MCQ factual). Fit **one isotonic curve per category** on the 40-task suite → calibrated P(wrong); then run **fixed-sequence LTT ordered cheapest-escalation-first** to certify the frugalest per-category thresholds that hold `P(acc ≥ 80%) ≥ 0.95`, targeting a **~0.93 true-accuracy floor** (not 0.97). Ship 8 small lookup tables as static config; no runtime model change.
- **Accuracy Δ:** −0 to −0.5 item (certified to stay above gate).
- **Token Δ:** large, **−30 to −50% of escalation count** — the biggest *safe* lever.
- **Effort:** 10h

**P0-3 · Cheaper escalations: two-phase batch + prompt-cache + per-category effort caps + terse output** *(batch-prompting arXiv:2511.04108/2301.08721; Fireworks prompt caching; gpt-oss `reasoning_effort`; extractive-judge arXiv:2504.11972)*
- **Change:** Refactor to two phases. **Phase 1** runs all local inference/verification and fills **per-category escalation queues** (factual is the ideal homogeneous batch). **Phase 2** flushes each queue as **1–2 batched gpt-oss-120b calls** (size 4–8, numbered-JSON out, re-ask only parse-fails). Put the **static instruction/schema/exemplar block first** (byte-identical → Fireworks cache hit at ~10%), variable question last. Set `reasoning_effort` + `max_tokens` **per category** (sentiment/NER ~150–300, factual ~300–500, summary ~400–700, math/logic ~900–1300). **Terse extractive output** + stop sequence for all categories **except summarization** (keep its length contract).
- **Accuracy Δ:** ~0 (batching preserves per-answer quality within ~2.4%).
- **Token Δ:** **−40 to −60% of the residual escalation bill** (batching cuts per-item reasoning ~74% + amortizes the shared prefix; caching discounts the rest ~90%).
- **Effort:** 10h

**P0-4 · Local self-debug execution-repair loop for code_debug** *(Self-Debugging arXiv:2304.05128; FLARE arXiv:2606.03852; FlexFL) — targets your measured weakness*
- **Change:** In the code pipeline, when the two implementations disagree/fail on a generated or shipped failing input, **run the candidate, capture the real traceback (+ offending input→observed vs expected)**, feed it back to Qwen2.5-Coder-1.5B and retry ≤2 rounds before escalating. Add cheap coarse fault-localization (traceback line/function) to narrow the repair prompt. Lower the local-confidence prior for non-Python/JS/TS languages (escalate those earlier, where the 1.5B is weakest).
- **Accuracy Δ:** **+0.5 to +1 item on code_debug** (one execution-feedback loop ≈ ~10 blind resamples).
- **Token Δ:** net-negative (more debug items resolve locally); ~−50 to −150.
- **Effort:** 7h

**P0-5 · Logic solver-cardinality certificate + constraint back-translation** *(ZebraLogic arXiv:2502.01100; Logic.py arXiv:2502.15776; formalizer reality-check arXiv:2505.13252)*
- **Change:** Have the brute-force solver return a **solution count**. `count == 1` → emit locally, **skip the confidence gate** (provably valid). `count == 0` (over-constrained) or `> 1` (under-constrained) → the extraction is faulty → re-extract once, then escalate. Optionally add a cheap **back-translation** of the extracted JSON constraints to NL + yes/no "same puzzle?" check; mismatch → low-confidence.
- **Accuracy Δ:** +0 to +0.5 item (eliminates confident-but-wrong logic answers that pass text heuristics).
- **Token Δ:** slightly net-negative (unique-solution items become un-escalatable).
- **Effort:** 4h (cardinality 2h is the high-EV core; back-translation +2h)

### P1 — do next

**P1-6 · Batched shared-prefix self-consistency + CPU baseline** *(llama.cpp `-np`/`is_sp_shared`/`cache_prompt`; thread-tuning)*
- **Change:** First lock the throughput floor on the **actual 2-vCPU host** (`-t 2`, no SMT, `-tb 2`, `llama-bench`). Then convert math majority-vote, 2-impl code, and 2-sample NER from N sequential `/completion` calls to **one prefill + N decode slots** over the shared prefix. On a bandwidth-bound box this amortizes weight streaming across samples (~2–3.5× per-sample).
- **Accuracy Δ:** +0 to +0.5 item (affords SC width ~3→~7 on hard math/logic inside 10 min).
- **Token Δ:** net-negative (fewer time-forced escalations).
- **Effort:** 8h

**P1-7 · Inline-markup NER + reason-then-constrain (CRANE)** *(inline-NER arXiv:2601.17898; "Let Me Speak Freely" arXiv:2408.02442; CRANE arXiv:2502.09061)*
- **Change:** Replace JSON-with-offsets NER with **inline tagging** — re-emit the sentence with entities wrapped (`[PER Alice] flew to [LOC Paris]`), parse spans deterministically; keep your verbatim-presence filter + span-expansion. Let the model emit a short **unconstrained rationale first**, then switch the GBNF grammar on only for the final tags/label (fixes the up-to-−27pp reasoning-flavored constraint penalty on NER and hard/sarcastic sentiment). Also **balance few-shot label counts and put the most representative shot last** for sentiment/NER.
- **Accuracy Δ:** +0.5 to +1 item on NER/sentiment (near-zero parse failures + recovered boundary accuracy).
- **Token Δ:** net-negative (fewer malformed-output escalations).
- **Effort:** 6h

**P1-8 · Verifier-gated reasoning-effort laddering + non-reasoning tier** *(gpt-oss `reasoning_effort`; cascade practice)*
- **Change:** For verifiable categories (math executor, code behavioral check, logic solver), call gpt-oss at `reasoning_effort=low` (or a non-reasoning tier for short-answer extractive categories), **run your local verifier on the result, and ladder up to medium only on verify-fail** (cap 2 rungs). Apply **Chain-of-Draft** ("≤5-word draft per step") to **math/logic/factual escalations only**.
- **Accuracy Δ:** ~0 (verifier guards correctness).
- **Token Δ:** −20 to −40% on the verifiable-category escalations (reasoning tokens dominate gpt-oss cost).
- **Effort:** 6h

**P1-9 · Judge-aligned formatting of escalated answers** *(rubric/verbosity-bias analyses; anti-hedging arXiv:2507.21919)*
- **Change:** Output-templating layer (no new calls): **answer-first, mirror requested format/units, strip hedging** ("I think", "as an AI"), commit to a single answer. Use **terse for extractive** categories but **"answer + 1–2 sentence why"** for factual/free-form where the judge scores holistically (verbosity bias is real there). Especially valuable for factual (always escalates). Keep summary length contract untouched.
- **Accuracy Δ:** +0 to +0.5 item (recovers correct-but-terse/hedged answers at the gate).
- **Token Δ:** ~0 to slightly negative.
- **Effort:** 4h

**P1-10 · Gated factual keep-local via self-consistency + NLI** *(SelfCheckGPT arXiv:2303.08896; SelfCheckGPT-NLI; SAC3) — high token upside, guarded because factual recall is a measured weakness*
- **Change:** Draw 3–5 non-thinking short-answer samples; keep local **only if unanimous AND high self-certainty AND a quantized DeBERTa-v3-small NLI (~140MB) finds no contradiction**; otherwise escalate. **Set the threshold conservatively and validate on the 40-task suite that kept-local factual accuracy stays ~100% before shipping.** Also feed judge-noise (Rogan-Gladen) estimates so you skip escalations gpt-oss can't flip.
- **Accuracy Δ:** −0.5 to +0 item **if gated correctly** (risk of confident-wrong facts if not).
- **Token Δ:** potentially **large negative** — factual is your only always-escalate category.
- **Effort:** 8h

### P2 — if time remains

**P2-11 · CodeT consensus for code_gen + LEVER execution-marginalization for math** *(CodeT arXiv:2207.10397; LEVER arXiv:2302.08468)*
- **Change:** Upgrade 2-impl agreement to **K programs × K generated tests, consensus-set scoring** (dual-agreement), reusing your sandbox. In math, **marginalize votes over execution *results*** across syntactically different programs so semantically-equal solutions pool mass.
- **Accuracy Δ:** +0 to +0.5 item (code_gen + math).  **Token Δ:** net-negative.  **Effort:** 8h

**P2-12 · Wall-clock watchdog / time-budget banking** *(compute-optimal TTS arXiv:2502.06703; cont-batching)*
- **Change:** Make the escalation trigger **remaining wall-clock**, not just confidence: bank time from fast easy items into a pool the hard items draw on for wider SC; escalate only when the global 600s budget is genuinely threatened (never DNF).
- **Accuracy Δ:** +0 to +0.5 item.  **Token Δ:** net-negative (1–3 fewer premature escalations).  **Effort:** 8h

---

## 3. If We Could Only Do 3 Things Tonight (highest EV-per-hour)

1. **Decoding & thinking hygiene (subset of P0-1: no-greedy + Qwen3 params + `\boxed{}` + min-p, ~3–6h).** Near-zero risk, tiny hours, removes silent repetition-loop and parse failures that cause *both* wrong answers and wasted escalations. Pure upside.
2. **Logic solver-cardinality certificate (P0-5 core, 2h).** A deterministic check that can only help — `count==1` items become un-escalatable and provably correct; `count≠1` catches a whole class of confident-wrong logic answers. Highest EV/hour in the plan.
3. **Batch escalations into one gpt-oss call + stable-prefix prompt caching (P0-3 core, ~6–8h).** Immediately shrinks the exact quantity that sets your rank — the residual escalation bill — by ~40–60%, with ~0 accuracy cost. Factual (always-escalate, homogeneous) is the ideal first batch.

*(If a 4th slot opens, start the offline calibration in P0-2 — it's the single biggest safe token lever, just longer to land.)*

---

## 4. DO-NOT-DO (popular techniques that would hurt us)

- **Greedy decoding in thinking mode.** Qwen3 card explicitly forbids it — causes repetition loops = wasted wall-clock, truncated answers, and needless escalations. (This is why P0-1 is P0.)
- **Naively flipping "factual = always escalate" with lexical agreement / plain self-consistency.** Factual recall is your *measured* weak spot and 1.7B models are confidently wrong, so N samples happily agree on a wrong fact. Only the hard-gated NLI + self-certainty variant (P1-10), conservatively thresholded and suite-validated, is safe.
- **Chain-of-Draft on code tasks.** The SWE follow-up (arXiv:2506.10987) shows concise-reasoning *degrades* code — and code_debug is already your weakest category. Apply CoD to math/logic/factual only.
- **Constraining generation from token 0 on NER / ambiguous sentiment.** "Let Me Speak Freely" (EMNLP'24) reports up to −27pp on reasoning-flavored generation. Reason free-form first, constrain only the final span/label (P1-7).
- **Over-sampling self-consistency (>10–15 paths).** Gains plateau and can *decline* (arXiv:2511.00751), and on 2 vCPU / 10 min the wasted compute *forces* time-driven escalations. Cap ~5–8, early-stop on a clear majority.
- **Global `reasoning_effort` / global `max_tokens` cap.** Too tight truncates a math/logic derivation (gate risk); too loose wastes tokens on lookups. Must be per-category (P0-3).
- **Quantizing below Q4 (Q3/IQ3).** Quality cliff — up to −7% on hard code, subtle multi-step logic errors that your verifier may not catch. Stay Q4–Q5; use Q4_K_M only where a hard verifier catches errors, Q5_K_M for unverified reasoning.
- **Trained verifiers / PRM-guided search / learned rerankers / heavy routers (full LEVER, RouteLLM).** Their headline gains need labelled data + training you cannot produce under CPU-only/10-min. Adopt only the training-free cores (execution-result marginalization, your existing deterministic verifiers as the reward).
- **Semantic-entropy hidden-state probes (arXiv:2406.15927).** ~12h of fiddly llama.cpp hidden-state plumbing + a trained probe, for a signal token-margin/self-certainty already give you free from logprobs. Skip unless you become CPU-bound.
- **Verbose or hedged escalation output** (restating the question, "as an AI…", "possibly"). Pure token waste on extractive categories; commit and lead with the answer (P1-9). (Do *not* over-correct into one-word answers on free-form factual/summary, where judge verbosity bias penalizes terseness.)
- **Blindly stacking speculative decoding on top of wide batched self-consistency.** On 2 shared cores the draft contends with the target and with the batch; the textbook 2–2.5× is a GPU number (realistic ~1.3–1.8×, structured output only). Profile per category; batched SC is the primary throughput win, spec-decode is a conditional add-on.
- **P(True) self-evaluation as a standalone gate.** 1.7B is systematically overconfident; use it only as one fused feature inside the calibrated router, never as the decision.