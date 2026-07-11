"""All tunables. Baked-in constants — the grading harness injects only
FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS, so every knob here
must hold its final value at image-build time."""

import os

# ---- paths (grading contract) ----
INPUT_PATH = os.environ.get("INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output/results.json")

# ---- local model servers ----
GENERAL_MODEL_PATH = "/models/general.gguf"   # Qwen3-1.7B Q4_K_M
CODER_MODEL_PATH = "/models/coder.gguf"       # Qwen2.5-Coder-1.5B-Instruct Q4_K_M
GENERAL_PORT = 8091
CODER_PORT = 8092
GENERAL_CTX = 2048   # long summarisation/NER passages must fit
CODER_CTX = 1536     # code prompts are short; saves KV memory
LLM_THREADS = 2
SERVER_START_TIMEOUT_S = 60

# ---- wall clock (10-min hard limit upstream; env overrides are for the
# local eval harness only — the grading harness never sets these) ----
SOFT_NEW_WORK_S = int(os.environ.get("SOFT_NEW_WORK_S", "445"))
FLUSH_S = int(os.environ.get("FLUSH_S", "505"))
HARD_EXIT_S = int(os.environ.get("HARD_EXIT_S", "535"))

# ---- escalation ----
# Total Fireworks tokens we are willing to spend (input+output, from usage).
# 0 => pure local zero-token mode.
# cap, not target: healthy runs spend ~400-700. Sized so that even a
# harder-than-expected task set can escalate every weak answer and still
# undercut the leaderboard's ~1,400-token top entries.
ESCALATION_BUDGET_TOKENS = int(os.environ.get("ESCALATION_BUDGET_TOKENS", "1300"))
# When local inference is dead or unusably slow, passing the accuracy gate
# outranks token frugality: emergency budget covers escalating every task.
EMERGENCY_BUDGET_TOKENS = int(os.environ.get("EMERGENCY_BUDGET_TOKENS", "12000"))
# below this, local quality/speed can't clear the gate — panic-grade local
# answers score ~30%, so slow counts as dead and we escalate everything
TPS_DEAD = 4.5
ESCALATE_CONF_THRESHOLD = 0.55   # tasks below this confidence are candidates
ESCALATION_TIMEOUT_S = 30
ESCALATION_WORKERS = 4

# Per-category escalation caps: (reasoning_effort, max_tokens). Keeps remote
# spend tight — every category runs low reasoning; token ceilings scale with
# how verbose a correct answer needs to be (code needs the most).
ESC_CAPS = {
    "factual": ("low", 160),
    "sentiment": ("low", 120),
    "ner": ("low", 150),
    "summary": ("low", 320),
    "math": ("low", 200),
    "logic": ("low", 220),
    "code_gen": ("low", 500),
    "code_debug": ("low", 500),
}
# Per-category escalation-confidence thresholds. Defaults to the global
# ESCALATE_CONF_THRESHOLD for every category; calibration overwrites these
# later. Kept in sync with ESC_CAPS keys.
# factual: conf is capped at 0.50 in its pipeline, so 0.55 = always escalate
# (the only measured accuracy hole escalation actually fixes). Everything else
# keeps only a 0.40 safety net for genuinely-broken local answers — measured
# reliable confs (0.70-0.92) stay local at zero tokens.
CATEGORY_THRESHOLDS = {"factual": 0.55, "code_debug": 0.40, "code_gen": 0.40,
                       "logic": 0.40, "math": 0.40, "ner": 0.40,
                       "sentiment": 0.40, "summary": 0.40}
# Batch multiple escalation questions into one remote chat when possible.
BATCH_ESCALATION = True
# ceiling on any single batched call's max_tokens (truncation-safety)
BATCH_MAX_TOKENS_CLAMP = 2200

# Strong serverless models, ranked by measured total-token frugality on a
# representative escalation (verified live 2026-07-11: gpt-oss-120b w/
# reasoning_effort=low → 137 tok; glm-5p2 → 165; kimi-k2p6 → 180;
# deepseek-v4-pro → 189; glm-5p1 → 237). Used when ALLOWED_MODELS env is
# absent/empty; otherwise strictly intersected with it.
FALLBACK_MODELS = [
    "accounts/fireworks/models/gpt-oss-120b",
    "accounts/fireworks/models/glm-5p2",
    "accounts/fireworks/models/kimi-k2p6",
    "accounts/fireworks/models/deepseek-v4-pro",
    "accounts/fireworks/models/glm-5p1",
]
# Preferred for language-category escalations when present in ALLOWED_MODELS
# (Gemma sub-prize optionality; never used unless explicitly allowed).
GEMMA_HINT = "gemma"

# ---- sampling ----
# Self-consistency sampling (diverse local samples for majority voting).
SC_TEMP = 0.9
SC_MIN_P = 0.07
GEN_TEMP = 0.7
GEN_TOP_P = 0.8
GEN_TOP_K = 20
THINK_TEMP = 0.6
THINK_TOP_P = 0.95
CODE_TEMP = 0.3

# ---- speed governor thresholds (general-model decode tok/s) ----
TPS_FULL = 10.0   # >=: full pipeline (self-consistency + selective thinking)
TPS_LEAN = 5.0    # >=: lean (2 samples max, thinking only for logic)
                  # <: panic (single-sample, no thinking, escalate weakest)

MAX_LOCAL_RETRIES = 2

# ---- local-eval test/calibration hooks (grading harness never sets these;
# baked-in defaults keep them inert in production) ----
# "1" => pretend local inference is dead and take the emergency
# escalate-everything path (CI knob: eval.yml force_emergency input).
TEST_FORCE_EMERGENCY = os.environ.get("TEST_FORCE_EMERGENCY", "0") == "1"
# Non-empty => append one JSONL record per scored task:
#   {"task_id", "category", "confidence", "signals"}
# Join judge results as "judge_pass" to build the records eval/calibrate.py
# consumes. Empty (default) disables all signal capture and logging.
CALIBRATION_LOG_PATH = os.environ.get("CALIBRATION_LOG_PATH", "")
