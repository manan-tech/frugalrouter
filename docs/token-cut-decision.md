# FrugalRouter — Token-Cut Decision Document

*Scope: what to ship for the hidden 19-task rescoring. Gate = 16/19 correct (disqualification, not rank penalty, if missed). Rank among gate-passers is ascending total Fireworks tokens. Leaderboard to beat: #1 = 1,377 tok, #3 = 1,797 tok. All four candidate policies below already clear #1 comfortably — so the real contest is **gate survival**, not rank.*

*Measurement provenance: five judged runs — `r438` (only 0-token/local-only run, 40 tasks), `rNorm2` (40 tasks, normal mode), `r29155126446` (19-task run, the composition mirror of the hidden set), `rEm2` (emergency escalate-all, 40 tasks), `r29155125266` (emergency at budget 4500). Numbers tagged **[M]** are measured from those logs; **[P]** are Poisson-binomial projections onto the hidden 19.*

---

## 1. THE LEDGER — where the tokens go and what each escalation bought

**Normal-mode lifetime spend: 4,023 tok across 5 runs; ~2,735 (68%) bought zero judge flips. [M]** Only two categories ever escalate in normal mode — everything else sits at local conf ≥ 0.55 and never triggers.

### The two runs in the brief

**`rNorm2` (40-task, 1,045 tok) [M]**

| Category | Escalation | Tokens | Judge flips bought | Verdict |
|---|---|---|---|---|
| factual | 5 **individual** calls (286+152+148+167+161) | **914** | **+2** (f3, f5) | f1/f2/f4 already PASS locally → 3 of 5 calls (~548 tok) re-bought passing answers |
| logic | 1 call (l1) | **131** | **0** | l1–l5 are 5/5 local in r438 — pure waste |
| — | (d1 stayed FAIL at conf 0.90) | 0 | — | invisible to escalation; not a token problem |

The 914-tok factual bill is inflated: identical work in `rEm2` batched at **387 tok [M]**. The +527 gap is the **batch-reservation bug** — the run logged `batch[factual] cannot reserve est=1857 n=5 … degrading to individual calls`. Those extra tokens then *starved* two conf-0.35 tasks (d2, g5) that shipped unescalated.

**`r29155126446` (19-task, 528 tok) — the hidden-set mirror [M]**

| Category | Escalation | Tokens | Judge flips bought |
|---|---|---|---|
| factual | 1 **batch** n=3 (f1,f2,f3) | **528** | **0** — f3 FAILED even escalated; f1/f2 pass locally |
| all others | none (local) | 0 | — |

This run's 18/19 (94.7%) is **byte-identical to what a 0-token run would have scored** — the 528 tok bought nothing. **But** re-running the *same* n=3 batch with the factual completeness hint live drops it to **409 tok and flips f3 to PASS → 19/19 [M, single temp-0.2 sample]**. That is the one high-value edit in the whole system.

### Lifetime per-category ledger [M]

- **factual** — 23 escalations (20 individual @ ~161 avg + 1 batch n=3 @ 528). Flips: **+2 per 40-task run** (f3, f5); **0** on the 19-task run. Local baseline (r438): **3/5** (f1/f2/f4 PASS, f3/f5 FAIL). → factual is simultaneously the **only accuracy hole** and **essentially the only token cost**.
- **logic** — 2 escalations (131, 144 tok). Flips: **0**. Local is 5/5. Remote is actively *worse*: rEm2's remote got l4 WRONG.
- **6 other categories** — 0 normal-mode escalations, ever. Local ceilings: code_gen/logic/math/ner/summary ≈ 1.00, sentiment 0.94, code_debug 0.82. [M]

**Bottom line:** every token above zero is really "how much of factual do we escalate." Beyond factual, added tokens buy ≈ 0 accuracy.

---

## 2. DECISION MATRIX

*P(gate fail) is projected [P] via Poisson-binomial on the r29 category mix (cd2 cg3 f3 lo2 ma3 ner2 se2 su2 — itself an assumption that the hidden set mirrors r29). "Base" uses measured local pass-rates; "judge-noise" is Analyst 4's adverse column where a ±1 stricter judge effectively raises the gate to 17/19. The base-P spread (0.5%–5.3%) reflects two defensible per-category models; treat it as a range, not a point.*

| Policy | Exp. tokens (hidden 19) | P(gate fail) base / judge-noise | Config + code delta | Choose when |
|---|---|---|---|---|
| **Zero-token** | **0** [P] | **18% / 33–42%** [P] | `ESCALATION_BUDGET_TOKENS=0` (or factual thr < 0.50) | **Never.** Disqualification suicide. Local factual is a *measured* 3/5 (r438) — f5 is confidently-wrong ("North America"), which no NLI gate catches. Rank you'd win is worthless if you can't clear the gate. |
| **Frugal — escalate factual only** ✅ | **~409–528** [M] | **~0.5% / ~18%** [P] | `CATEGORY_THRESHOLDS = {'factual':0.55, all others:0.40}`; right-size `ESC_CAPS`; keep factual hint; force batch | **Default ship.** Frugalest gate-safe policy on today's code. Beats #1 (1,377) by ~2.5–3.4×. Extra tokens above this buy zero accuracy. |
| **Balanced — factual + ALL code_debug** | **~828** [P] | **~0.12% / lower** [P] | Frugal + `'code_debug':0.95` (whole category; its failures are at conf 0.90, so any threshold *below* 0.90 misses them) + `'logic':0.55` | Closes the one measured accuracy hole (code_debug 0.82, escalated 5/5 in rEm2) for +300 tok. Still 1.7× under #1. Choose if a randomized variant surfaces d1/d3-class code_debug in the hidden mix. |
| **Current — global 0.55** | **528–1,045** [M] | **~0.5% / ~18%** [P] | (no change) | **Do-nothing baseline only.** Escalates the *same* factual set as Frugal, same accuracy — but carries the +517-tok individual-call tail (rNorm2's 1,045) from the reservation bug. Strictly dominated by Frugal. |

**Emergency budget is orthogonal and settled:** keep `EMERGENCY_BUDGET_TOKENS=12000`. Measured: 4,500 → 52.5% GATE FAIL [M]; 12,000 → full coverage at 5,055 tok, 95.0% PASS [M]. Never lower below ~6,000.

---

## 3. RECOMMENDED SHIP — Frugal (escalate factual only, batched + completeness hint)

**Rationale:** The critic adjudication resolves the analyst spread decisively. (a) Local factual is **measured at 0.60**, not unknown — Analyst 2's "unmeasured" premise was a data-selection artifact (r438 was excluded from its records). 0.60 is genuinely weak, so factual must keep escalating. (b) "Batching degrades accuracy" is **overstated** — f3 batch-PASSED in rEm2 and only failed in r29's *hint-less* batch; the completeness hint (already present at `fireworks.py:182-183`) is the fix, so we keep batching for the ~26% token saving. (c) Analyst 1's aggressive 0–160 target loses the f5-class flip and trades measured gate insurance for rank we already own — **negative EV**.

### Exact diff to apply

**`agent/config.py`**
```python
# line 47-56 — right-size ESC_CAPS so factual n=5 reserves 160*5+~360=1160 < 1300 (batches cleanly, no degrade)
ESC_CAPS = {
    "factual":    ("low", 160),   # was 300  (measured actual 77-176/item)
    "sentiment":  ("low", 120),   # was 200  (91)
    "ner":        ("low", 150),   # was 250  (97)
    "math":       ("low", 200),   # was 400  (111)
    "logic":      ("low", 220),   # was 400  (122)
    "summary":    ("low", 320),   # was 500  (219)
    "code_gen":   ("low", 500),   # was 900  (155)
    "code_debug": ("low", 500),   # was 900  (139)
}

# line 60 — replace the flat 0.55 default with an explicit per-category map
CATEGORY_THRESHOLDS = {
    "factual": 0.55,      # conf capped at 0.50 in pipeline → always escalates (intended)
    "code_debug": 0.40, "code_gen": 0.40, "logic": 0.40,
    "math": 0.40, "ner": 0.40, "sentiment": 0.40, "summary": 0.40,
}
# 0.40 keeps a safety net for a genuinely-broken conf=0.35 local answer while
# leaving every reliable 0.70-0.92 answer local (zero tokens).

# UNCHANGED (validated): ESCALATION_BUDGET_TOKENS=1300, EMERGENCY_BUDGET_TOKENS=12000, BATCH_ESCALATION=True
```

**`agent/fireworks.py`**
- **`_CATEGORY_HINTS['factual']` — confirm present** at lines 182-183 (already in the working tree; it is the accuracy safeguard that makes factual batching safe — do not remove). Keep the `esc_suffix` wording "covering every part asked" verbatim; f3 fails on completeness omissions.
- **Batch reservation shrink-to-fit** (kills the +517-tok tail permanently). In `batch_chat`, on `try_reserve` failure, retry with `max_tokens` shrunk to fit the remaining budget *before* degrading to individual calls:
```python
# replace the immediate "degrade to individual" at line ~299-305
if not BUDGET.try_reserve(est):
    floor = est_tokens(BATCH_SYS + numbered)
    fit = BUDGET.total - BUDGET.spent - floor
    if fit >= len(items) * 60:                       # enough for a terse answer each
        max_tokens = min(max_tokens, fit)
        est = floor + max_tokens
    if not BUDGET.try_reserve(est):
        log(f"batch[{category}] cannot reserve est={est} … degrading to individual calls")
        return _batch_individual(items, category, per_item)
```
`commit()` already reconciles to actuals, so a tighter reservation can never overspend — this is a pure win.

*Optional, second-order (defer): shorten `BATCH_SYS` (411→~170 chars, −50–70 tok/batch) and hoist the factual `esc_suffix` into the shared hint. Skip for this submission to avoid pre-rescoring churn.*

### Validation runs required before resubmission

1. **Full 40-task CI at shipped config.** Grep `agent.log`: must show `batch escalated[factual]` as a **single call**, never `degrading to individual calls`. Assert judge ≥ 39/40 and **f3 PASS**.
2. **19-task rehearsal (`rehearsal19`) at shipped config.** Confirm token bill lands **~409 (≤ 528)** and score **19/19** (floor 17/19). [Reproduces Analyst 5's measured result.]
3. **Truncation check.** Grep for batch-parse re-ask lines across all batches: expect **0** unparsed (observed 0/9 historically at the old caps; the new caps keep ≥2.2× measured per-item need).
4. **Emergency-path smoke test** (`TEST_FORCE_EMERGENCY=1`): confirm the reservation fix lets all 8 category batches reserve under 12,000 and reach full coverage (~5,055 tok) — the r29155125266 52.5% FAIL must not reproduce.

**Do NOT** ship a factual-local (0-token) config without first running a clean pure-local calibration (`ESCALATION_BUDGET_TOKENS=0`) across the full suite. r438 gives only 3/5 factual; that is not enough evidence to trust dropping factual escalation.

---

## 4. TRIPWIRES — what flips the decision

**Escalate up (Frugal → Balanced), never down, on any of these:**
- **Gate failure / disqualification on the leaderboard** (score 0). Immediate move to Balanced; investigate before any further token cut.
- **A randomized rescoring variant surfaces a d1/d3-class code_debug item** and it fails at conf 0.90. code_debug is the one measured accuracy hole (0.82); Balanced's `code_debug:0.95` (escalate whole category, +300 tok) is the insurance.
- **f3 or any factual FAILS in a batched CI run.** The completeness hint has regressed — block resubmit until `_CATEGORY_HINTS['factual']` is restored and f3 re-PASSES.

**Block resubmit (config didn't take):**
- CI logs `batch[factual] … degrading to individual calls`, or the factual bill spikes toward ~914–1,045. The ESC_CAPS resize / shrink-to-fit fix isn't live; the token tail is back.
- Any batch produces unparsed items (truncation) — the new ESC_CAPS were cut too aggressively for that category.

**Hold position (no change needed):**
- Leaderboard #1 drops below ~1,000 tok: we still win at 409. Do **not** chase sub-400 by loosening factual — the critic-adjudicated EV is negative (loses the f5-class flip; re-opens the 18%/33–42% gate-fail tail). Only consider going more aggressive after **multiple** randomized variants confirm a comfortable (≥2-task) gate margin with factual local.
- Emergency path fires (local dead/slow): expected and fine at 12,000 budget — rank is already sacrificed for gate survival there. Never lower `EMERGENCY_BUDGET_TOKENS` below ~6,000 (4,500 = measured 52.5% catastrophic FAIL).

---

**One-line ship call:** Frugal — escalate factual only, batched with the completeness hint, ESC_CAPS right-sized and reservation shrunk-to-fit. Expected **~409 tok** on the hidden 19 (measured), **P(gate fail) ~0.5% base** (projected), **~3.4× under leaderboard #1** — with a 2-task gate cushion and no accuracy left on the table.