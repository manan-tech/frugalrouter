# Fine-tuning runbook (ACT II endgame)

## Why

Our accuracy lived in **sampling** — 3 generations, majority voting, executed verification.
That machinery costs wall-clock, and on the grader's contended box the mode ladder strips it
away, so quality collapsed: **74% in CI -> ~50% there**. Six graded runs across three models
told the same story (36.8, 68.4, 47.4, 57.9, 52.6, 63, 78.9%).

The four teams scoring **84-94% at ZERO tokens** fine-tuned instead. Their accuracy is in the
**weights**, so a slow box makes them *later*, never *worse*. That is the whole difference.

**Goal: the model answers correctly, in the exact required format, in ONE shot.** No sampling,
no verification, no escalation. Zero tokens.

## Model choice

**Qwen3-1.7B** — proven to RUN on the grader box (v9/v10 produced local answers there at
6-8.5 tok/s). Do NOT go bigger: our 3B build **died** on their box (52.6%), and 4-bit 2-3B is
the documented ceiling for 4 GB.

Fallback if 1.7B is too slow one-shot: **Qwen3-0.6B** (14-25 tok/s there). Train both if the
GPU hour allows; pick on measured wall-clock.

## GPU

**AWS g6e.2xlarge** (L40S, 48 GB VRAM) — 4x more than a 1.7B LoRA needs (~10-12 GB). ~$2/hr,
~1-1.5h of work.

**Do NOT use Fireworks fine-tuning**: it hosts the model on *their* servers, so we would keep
paying tokens per call — the exact thing we are escaping. We need downloadable weights to
convert to GGUF and bake into the image.

## Run it

```bash
# on the GPU box
pip install torch transformers peft trl datasets accelerate bitsandbytes
git clone <this repo> && cd amd2

python finetune/train.py --data finetune/sft.jsonl --base Qwen/Qwen3-1.7B
bash  finetune/to_gguf.sh finetune/out/merged      # -> finetune/out/general.gguf
```

Copy `general.gguf` back, bake it as `/models/general.gguf`, rebuild the image.

## The two things that decide whether this works

1. **Prompt diversity in the training data.** The finals **rephrase every prompt**. If the SFT
   set says "Extract all named entities" 100 times, the model overfits that phrasing and dies on
   "Which organisations does this article mention?". This is the single biggest risk, and it is
   probably how the 0-token teams will crack at finals. Our generator varies instruction wording
   heavily on purpose.

2. **Format exactness.** The judge scores against a rubric. `Entity | TYPE` lines, `Label - reason`,
   "exactly two sentences" — the model must emit these verbatim, with no "Sure, here is..." preamble.

## After training: flip the agent to one-shot

A fine-tuned model does not need our sampling machinery, and that machinery is what a slow box
confiscates. Once the model is in:

- `TPS_DEAD = 0.0` — never abandon local; there is nothing to escalate to.
- `ESCALATION_BUDGET_TOKENS = 0` — zero tokens is the whole point.
- Pipelines: single generation, `thinking=False`. Delete the self-consistency loops.
- Keep the deterministic verifiers ONLY where they are free (executing generated Python costs no
  model time and catches arithmetic slips).
