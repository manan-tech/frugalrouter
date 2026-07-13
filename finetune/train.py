#!/usr/bin/env python3
"""LoRA SFT for FrugalRouter's local model (Qwen3-1.7B) — one-shot, zero-token.

WHY THIS EXISTS
    Our accuracy lived in SAMPLING: 3 generations, majority voting, executed
    verification. That machinery costs wall-clock, and on the grader's contended
    box the mode ladder strips it away — quality collapsed (74% in CI -> ~50%
    there). Teams scoring 84-94% at ZERO tokens put their accuracy in the
    WEIGHTS instead, so a slow box makes them later, never worse.

    Goal: the model answers CORRECTLY and IN THE EXACT FORMAT in ONE shot.
    No sampling, no verification, no escalation.

VERSIONS TARGETED
    torch >= 2.3, transformers >= 4.44 (incl. v5.x), peft >= 0.11, accelerate.
    No trl, no datasets, no bitsandbytes — nothing here needs them.
    Exercised end-to-end on transformers 4.57.2 / torch 2.11 (real Qwen3 arch,
    real Qwen3-1.7B tokenizer): template, loss mask, LoRA, merge, save, smoke test.
    Worth knowing: on 4.57 Trainer's `tokenizer=` kwarg is GONE (processing_class
    only), so the tutorial spelling TypeErrors — we probe the signature instead.
    NOTE: this deliberately does NOT use trl.SFTTrainer. trl renamed/moved the
    exact things we depend on across versions — TrainingArguments -> SFTConfig,
    max_seq_length moved into SFTConfig then deprecated, tokenizer= ->
    processing_class=, dataset_text_field semantics, and DataCollatorFor-
    CompletionOnlyLM was removed in 0.13. The only thing we actually wanted from
    SFTTrainer is completion-only loss masking, which is ~15 lines done by hand
    (see build_example). peft + transformers.Trainer is the stable surface, and
    a crash here costs GPU time we do not have. Every version-fragile kwarg
    below is probed with inspect.signature before use.
    (If you must use trl: SFTConfig(..., max_length=N, packing=False) +
     SFTTrainer(model=, args=, train_dataset=, peft_config=, processing_class=).)

THINKING MUST BE OFF, AND THE TOKENS MUST MATCH (verified, not assumed)
    At inference agent/llm.py calls llama-server /v1/chat/completions with
    --jinja and chat_template_kwargs={"enable_thinking": false}. That takes the
    template's generation-prompt branch:

        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\\n\\n</think>\\n\\n' }}

    so the model is ALWAYS fed  `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`
    and must continue straight into the answer. Training text has to reproduce
    that prefix exactly or every example is skewed from what we serve.

    We therefore render the PROMPT with add_generation_prompt=True +
    enable_thinking=False and concatenate the answer + <|im_end|> ourselves. This
    is exact by construction, and it hands us the prompt/completion boundary the
    loss mask needs anyway.

    Checked against the real Qwen/Qwen3-1.7B template (transformers 4.57): our
    construction is BYTE-IDENTICAL to apply_chat_template() over the full
    conversation. Note *why*, because it is a trap for whoever edits this next:
    the full-conversation render only agrees by coincidence. Its assistant branch
    emits '\\n<think>\\n' + reasoning_content.strip() + '\\n</think>\\n\\n', which
    collapses to the same empty block when reasoning_content is '' — it is the
    reasoning-content wrapper doing that, NOT enable_thinking (which the template
    consults only under add_generation_prompt). That coincidence is revision- and
    fork-dependent; the generation-prompt path is the one llama.cpp actually
    walks, so we build from it and assert on it at startup.

Usage (GPU box):
    pip install -U torch transformers peft accelerate
    python finetune/train.py --dry-run --data finetune/sft.jsonl   # CPU, no model load
    python finetune/train.py --data finetune/sft.jsonl
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import sys
import time

DEFAULT_BASE = "Qwen/Qwen3-1.7B"  # proven to RUN on the grader box (v9/v10). Do
                                  # NOT go bigger: the 3B build died there.

# Qwen3's non-thinking prefix. Byte-identical to what the template emits and to
# what llama.cpp will feed the model at inference.
EMPTY_THINK = "<think>\n\n</think>\n\n"

# The system prompt the model is trained under. agent/llm.py MUST send this
# string verbatim at inference — a fine-tuned model is conditioned on it, and
# drifting it is a silent accuracy leak. Rows may override with a "system" key.
SYSTEM_PROMPT = (
    "You answer the user's task directly and in the exact format requested. "
    "No preamble, no restating the question, no explanation of your process. "
    "Answer once, correctly."
)

CATEGORIES = ("ner", "sentiment", "summary", "math", "logic", "code_gen",
              "code_debug", "factual")

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


# ---------------------------------------------------------------- data loading

def load_rows(path):
    """Schema-tolerant loader. Accepts, per JSONL line, either:
         {"messages": [{"role": ..., "content": ...}, ...]}          (canonical)
         {"prompt"|"instruction"|"input": str,
          "answer"|"completion"|"output"|"response": str,
          "system": str (optional), "category": str (optional)}
    The dataset builder is a sibling script; do not make it guess our key names.
    """
    rows = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"{path}:{lineno}: bad JSON: {e}")

            system, user, answer = None, None, None
            if isinstance(obj.get("messages"), list) and obj["messages"]:
                msgs = obj["messages"]
                # last assistant turn is the target; everything before it is context
                ai = max((i for i, m in enumerate(msgs)
                          if m.get("role") == "assistant"), default=-1)
                if ai < 0:
                    sys.exit(f"{path}:{lineno}: messages[] has no assistant turn")
                answer = msgs[ai].get("content") or ""
                prior = msgs[:ai]
                system = next((m["content"] for m in prior
                               if m.get("role") == "system"), None)
                user = "\n\n".join(m.get("content") or "" for m in prior
                                   if m.get("role") == "user")
            else:
                for k in ("prompt", "instruction", "input", "question", "text"):
                    if isinstance(obj.get(k), str) and obj[k].strip():
                        user = obj[k]
                        break
                for k in ("answer", "completion", "output", "response", "target"):
                    if isinstance(obj.get(k), str) and obj[k].strip():
                        answer = obj[k]
                        break
                system = obj.get("system")

            if not user or not answer or not answer.strip():
                sys.exit(f"{path}:{lineno}: could not find a prompt/answer pair "
                         f"in keys {sorted(obj)}")

            # A teacher model may have leaked a think block into the target. If we
            # train on it, we train the model to think — wall-clock suicide at
            # ~7 tok/s. Strip it and say so.
            cleaned = _THINK_RE.sub("", answer).strip()
            if cleaned != answer.strip():
                print(f"  ! line {lineno}: stripped a <think> block from the target")
            rows.append({
                "system": system or SYSTEM_PROMPT,
                "user": user,
                "answer": cleaned,
                "category": obj.get("category") or "unknown",
            })
    return rows


# ------------------------------------------------------------ prompt rendering

def render_prompt(tok, system, user):
    """Exactly the string llama.cpp will hand the model at inference time."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:  # template/tokenizer without the kwarg
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    # Non-negotiable: match llama.cpp's non-thinking prefix. If the installed
    # template honoured enable_thinking it is already there; if it did not, we
    # append it, because llama-server WILL feed it at inference. The generation
    # prompt ends with "<|im_start|>assistant\n", so a plain append is exact.
    if not text.endswith(EMPTY_THINK):
        text += EMPTY_THINK
    return text


def build_example(tok, row, maxlen, eot):
    """-> {"input_ids", "labels"} with the PROMPT masked to -100.

    Loss on the assistant turn only: we want the model to learn to ANSWER, not
    to reproduce our prompts (and the prompts are the bulk of the tokens).
    """
    prompt_text = render_prompt(tok, row["system"], row["user"])
    completion_text = row["answer"].strip() + eot  # must learn to STOP

    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion_text, add_special_tokens=False)["input_ids"]

    total = len(p_ids) + len(c_ids)
    if total > maxlen:
        # Never right-truncate: that would eat the ANSWER, i.e. exactly the
        # format we are paying to teach. Drop the example and report it.
        return None, total
    return {"input_ids": p_ids + c_ids,
            "labels": [-100] * len(p_ids) + list(c_ids)}, total


class Collator:
    """Pad right, mask labels with -100, pad length to a multiple of 8."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        import torch
        n = max(len(f["input_ids"]) for f in feats)
        n = (n + 7) // 8 * 8
        ids, labels, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long)}


def _supported(fn, **kw):
    """Keep only kwargs this installed version actually accepts."""
    ok = set(inspect.signature(fn).parameters)
    return {k: v for k, v in kw.items() if k in ok}


# --------------------------------------------- upstream bug: transformers 4.57.2
# tokenization_utils_base.py:293 does `_config.model_type` on a plain DICT inside
# a Mistral-regex fixup path. It raises AttributeError for ANY tokenizer load that
# is simultaneously:
#     vocab_size > 100_000   (Qwen3 = 151_936  -> yes)
#     a fast tokenizer       (yes)
#     from a LOCAL directory (_is_local)
#     whose config.json has transformers_version <= 4.57.2
# Loading the concrete class instead of AutoTokenizer does NOT help — the bug is
# in the shared base class. The only clean defusal is the config.json key itself:
# `_config.get("transformers_version")` returning None short-circuits the branch.
#
# This matters far beyond our own load: llama.cpp's convert_hf_to_gguf.py calls
# AutoTokenizer.from_pretrained(<merged dir>) — a local dir with a config.json we
# just wrote — so an un-defused merged dir kills to_gguf.sh AFTER training, which
# is the most expensive possible place to discover it. We strip the key from the
# dir we create. It is optional provenance metadata; nothing (transformers,
# convert_hf_to_gguf, llama.cpp) reads it. DO NOT "restore" it.

def defuse_local_tokenizer_bug(d):
    p = os.path.join(d, "config.json")
    try:
        with open(p, encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg.pop("transformers_version", None) is None:
            return
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        print("   dropped config.json:transformers_version "
              "(defuses the transformers 4.57.2 local-tokenizer AttributeError)")
    except Exception as e:  # never let a cosmetic fixup kill a finished run
        print(f"   ! could not defuse config.json ({e}) — if to_gguf.sh dies with "
              f"\"'dict' object has no attribute 'model_type'\", delete the "
              f"transformers_version key from {p}")


def load_tokenizer(base):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    except AttributeError as e:
        if "model_type" not in str(e):
            raise
        sys.exit(
            f"\nFATAL: hit the transformers 4.57.2 local-tokenizer bug loading "
            f"{base!r}\n  ('dict' object has no attribute 'model_type' — "
            f"tokenization_utils_base.py:293)\n"
            f"  It fires on a LOCAL model dir whose config.json declares "
            f"transformers_version <= 4.57.2.\n"
            f"  Fix in 5 seconds — drop that optional key:\n"
            f"    python3 -c \"import json;p='{base}/config.json';"
            f"c=json.load(open(p));c.pop('transformers_version',None);"
            f"json.dump(c,open(p,'w'),indent=2)\"\n"
            f"  ...or just pass the hub id (--base Qwen/Qwen3-1.7B), which is not "
            f"affected.\n")


# ------------------------------------------------------------------------ main

def main():
    global SYSTEM_PROMPT  # must precede any read of it in this scope (SyntaxError otherwise)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="finetune/sft.jsonl")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default="finetune/out")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)     # effective batch 16
    ap.add_argument("--maxlen", type=int, default=2048)  # == GENERAL_CTX
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last checkpoint in --out (spot recovery)")
    ap.add_argument("--system", default=SYSTEM_PROMPT,
                    help="system prompt for rows that don't carry their own")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="faster, more VRAM. Default ON: an OOM crash costs more "
                         "GPU time than a 2x slower 150-step run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="render + tokenize + print stats, load no model. RUN THIS "
                         "FIRST — it catches data bugs for free.")
    a = ap.parse_args()
    SYSTEM_PROMPT = a.system

    random.seed(a.seed)

    print(f"== loading {a.data}")
    rows = load_rows(a.data)
    if not rows:
        sys.exit(f"{a.data} has no usable rows")
    by_cat = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"{len(rows)} rows; per category: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))
    missing = [c for c in CATEGORIES if c not in by_cat]
    if missing:
        print(f"  ! WARNING: no training rows for: {', '.join(missing)}")

    tok = load_tokenizer(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    eot = tok.eos_token or "<|im_end|>"

    # Eyeball example 0 before three epochs burn. This is the whole ballgame.
    demo = render_prompt(tok, rows[0]["system"], rows[0]["user"])
    print("\n== rendered prompt (example 0), repr of the tail:")
    print("   " + repr(demo[-160:]))
    if EMPTY_THINK not in demo:
        sys.exit("FATAL: the empty <think> block is missing from the rendered "
                 "prompt. llama.cpp WILL insert it at inference (enable_thinking="
                 "false) — training without it skews every example. Fix "
                 "render_prompt() before spending GPU time.")
    print(f"   thinking: DISABLED (empty think block present)  eos: {eot!r}")

    feats, lens, dropped = [], [], []
    for i, r in enumerate(rows):
        ex, total = build_example(tok, r, a.maxlen, eot)
        lens.append(total)
        if ex is None:
            dropped.append((i, r["category"], total))
        else:
            feats.append(ex)
    lens.sort()
    p = lambda q: lens[min(len(lens) - 1, int(len(lens) * q))]  # noqa: E731
    print(f"\n== token lengths: p50={p(.5)} p95={p(.95)} max={lens[-1]} "
          f"(maxlen={a.maxlen})")
    if dropped:
        print(f"  ! DROPPED {len(dropped)} rows longer than --maxlen "
              f"(would have truncated the ANSWER):")
        for i, c, t in dropped[:10]:
            print(f"      row {i} [{c}] {t} tokens")
        print(f"  ! raise --maxlen (VRAM allows) or shorten those prompts")

    supervised = sum(sum(1 for x in f["labels"] if x != -100) for f in feats)
    print(f"== {len(feats)} usable examples, {supervised} supervised tokens "
          f"(prompt tokens are masked to -100)")

    if a.dry_run:
        print("\n== dry run: data is valid. Now run without --dry-run.")
        return

    # ---------------------------------------------------------------- training
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    random.shuffle(feats)
    n_val = int(len(feats) * a.val_frac) if a.val_frac > 0 else 0
    val, train = feats[:n_val], feats[n_val:]
    print(f"== train={len(train)}  val={len(val)}")

    print(f"== loading {a.base} in bf16")
    load_kw = dict(trust_remote_code=True, attn_implementation="sdpa")
    # transformers >= 4.56 renamed torch_dtype -> dtype. from_pretrained swallows
    # unknown kwargs instead of raising, so the WRONG name does not crash — it
    # silently loads fp32. Try the new name, fall back, then VERIFY the actual
    # parameter dtype and coerce. Never trust the kwarg took.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            a.base, dtype=torch.bfloat16, **load_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            a.base, torch_dtype=torch.bfloat16, **load_kw)
    if next(model.parameters()).dtype != torch.bfloat16:
        print("  ! dtype kwarg did not take — coercing to bf16 explicitly")
        model = model.to(torch.bfloat16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = False  # required with gradient checkpointing

    peft_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        # every projection: a 1.7B needs the capacity to absorb eight distinct
        # output formats without trading one off against another
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    grad_ckpt = not a.no_grad_ckpt
    if grad_ckpt:
        model.enable_input_require_grads()  # LoRA + ckpt: inputs need grads


    # ------------------------------------------------------------ health log
    # SPOT INSTANCE: the box can vanish mid-run. Everything needed to judge the
    # run's health streams to train_health.jsonl (one JSON object per log step)
    # so `tail -f` / scp gives loss curve, lr, grad-norm, GPU memory, throughput
    # — and a resumable checkpoint always exists (save_steps below).
    from transformers import TrainerCallback

    class HealthCallback(TrainerCallback):
        def __init__(self, path, total_steps):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.path = path
            self.t0 = time.time()
            self.last_t = self.t0
            self.last_step = 0
            self.total = total_steps
            open(self.path, "w").close()

        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            now = time.time()
            steps = state.global_step - self.last_step
            step_s = (now - self.last_t) / steps if steps > 0 else 0.0
            gpu_gb = 0.0
            try:
                import torch as _t
                if _t.cuda.is_available():
                    gpu_gb = _t.cuda.max_memory_allocated() / 1e9
            except Exception:
                pass
            rec = {
                "t": round(now - self.t0, 1),
                "step": state.global_step,
                "of": self.total,
                "epoch": round(state.epoch or 0, 3),
                "loss": logs.get("loss"),
                "eval_loss": logs.get("eval_loss"),
                "lr": logs.get("learning_rate"),
                "grad_norm": logs.get("grad_norm"),
                "sec_per_step": round(step_s, 2),
                "gpu_mem_gb": round(gpu_gb, 2),
                "eta_min": round((self.total - state.global_step) * step_s / 60, 1)
                           if step_s and self.total else None,
            }
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            bits = [f"step {rec['step']}/{rec['of']}", f"ep {rec['epoch']}"]
            if rec["loss"] is not None:      bits.append(f"loss {rec['loss']:.4f}")
            if rec["eval_loss"] is not None: bits.append(f"EVAL {rec['eval_loss']:.4f}")
            if rec["grad_norm"] is not None: bits.append(f"gnorm {rec['grad_norm']:.2f}")
            bits.append(f"{rec['gpu_mem_gb']}GB")
            if rec["eta_min"] is not None:   bits.append(f"eta {rec['eta_min']}m")
            print("  [health] " + "  ".join(bits), flush=True)

    steps_per_epoch = max(1, len(train) // (a.bsz * a.accum))
    total_steps = int(steps_per_epoch * a.epochs)
    health = HealthCallback(os.path.join(a.out, "train_health.jsonl"), total_steps)

    targs = TrainingArguments(**_supported(
        TrainingArguments.__init__,
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.bsz,
        gradient_accumulation_steps=a.accum,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        # SPOT-SAFE: checkpoint every 25 steps, not per-epoch — an interruption
        # costs at most ~25 steps. Resume with --resume (passes the last
        # checkpoint dir to trainer.train()).
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        eval_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=True,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        remove_unused_columns=False,
        # 0, deliberately. The dataset is a small in-memory list of ALREADY
        # tokenized dicts, so a worker process would parallelize nothing but the
        # collator's list padding — zero throughput to gain, while forking a
        # process that re-imports __main__ and re-forks the tokenizer is a real
        # crash mode. Keep it single-process.
        dataloader_num_workers=0,
        report_to="none",
        seed=a.seed,
    ))

    # Trainer renamed tokenizer= -> processing_class= (4.46 deprecated it, v5
    # removed it). Both exist in the overlap window — pass exactly one.
    params = set(inspect.signature(Trainer.__init__).parameters)
    tkw = ({"processing_class": tok} if "processing_class" in params
           else {"tokenizer": tok})

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train,
        eval_dataset=val or None,
        data_collator=Collator(tok.pad_token_id),
        callbacks=[health],
        **tkw)

    t0 = time.time()
    ckpt = None
    if a.resume:
        cks = sorted((d for d in os.listdir(a.out) if d.startswith("checkpoint-")),
                     key=lambda d: int(d.split("-")[1])) if os.path.isdir(a.out) else []
        ckpt = os.path.join(a.out, cks[-1]) if cks else None
        print(f"== resume from: {ckpt or 'no checkpoint found — fresh run'}")
    trainer.train(resume_from_checkpoint=ckpt)
    print(f"== trained in {(time.time() - t0) / 60:.1f} min")

    if val:
        # No eval_strategy during training (that kwarg was renamed across
        # versions); one evaluate() at the end is enough to spot overfit.
        loss = trainer.evaluate().get("eval_loss")
        print(f"== held-out loss: {loss:.4f}" if loss is not None
              else "== held-out loss: unavailable")

    # Adapter first: cheap insurance if the merge OOMs.
    adir = os.path.join(a.out, "adapter")
    trainer.model.save_pretrained(adir)
    print(f"== adapter -> {adir}")

    print("== merging LoRA into the base (llama.cpp cannot convert a bare adapter)")
    merged = trainer.model.merge_and_unload()
    merged.config.use_cache = True
    mdir = os.path.abspath(os.path.join(a.out, "merged"))
    merged.save_pretrained(mdir, safe_serialization=True)
    tok.save_pretrained(mdir)  # carries the chat template into the GGUF
    defuse_local_tokenizer_bug(mdir)  # or convert_hf_to_gguf.py dies on this dir
    with open(os.path.join(mdir, "frugal_sft_meta.json"), "w") as fh:
        json.dump({"base": a.base, "system_prompt": SYSTEM_PROMPT,
                   "enable_thinking": False, "epochs": a.epochs, "lr": a.lr,
                   "lora": {"r": 32, "alpha": 64}, "maxlen": a.maxlen,
                   "n_train": len(train), "categories": by_cat}, fh, indent=2)
    print(f"== MERGED MODEL -> {mdir}")

    # ------------------------------------------------------------ smoke test
    # Greedy, one-shot, production template. These prompts are from the training
    # distribution, so this is a FORMAT check, not an accuracy check — but format
    # is what the judge scores, and catching "Sure, here is..." preamble or a
    # leaked <think> now saves an hour of GGUF + docker + CI to learn the same.
    print("\n== smoke test (greedy, one-shot, production template)")
    merged.eval()
    seen, shown = set(), 0
    for r in rows:
        if r["category"] in seen or shown >= 4:
            continue
        seen.add(r["category"])
        shown += 1
        text = render_prompt(tok, r["system"], r["user"])
        ids = tok(text, return_tensors="pt",
                  add_special_tokens=False).to(merged.device)
        with torch.no_grad():
            out = merged.generate(**ids, max_new_tokens=384, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
        gen = tok.decode(out[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        print(f"\n--- [{r['category']}] {r['user'][:70]!r}")
        print(gen[:500])
        if "<think>" in gen:
            print("  ! LEAKED <think> — check the data for think blocks")
        if re.match(r"^(sure|here|okay|certainly)\b", gen, re.I):
            print("  ! PREAMBLE — the model is not answering one-shot")

    print(f"""
================================================================================
NEXT COMMAND (run it now):

    bash finetune/to_gguf.sh {mdir}

then copy finetune/out/general.gguf back to the repo, bake it as
/models/general.gguf, and flip the agent to one-shot:
    config.py  TPS_DEAD = 0.0            (never abandon local — nothing to escalate to)
    config.py  ESCALATION_BUDGET_TOKENS = 0
    llm.py     send this system prompt VERBATIM, enable_thinking=False, temp 0:
               {SYSTEM_PROMPT!r}
================================================================================""")


if __name__ == "__main__":
    main()
