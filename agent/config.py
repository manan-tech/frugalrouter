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
GENERAL_CTX = 2048
CODER_CTX = 2048
LLM_THREADS = 2
SERVER_START_TIMEOUT_S = 90

# ---- wall clock (10-min hard limit upstream) ----
SOFT_NEW_WORK_S = 445    # no new improvement work after this
FLUSH_S = 505            # watchdog forces a results flush
HARD_EXIT_S = 535        # watchdog force-exits process (exit 0)

# ---- escalation ----
# Total Fireworks tokens we are willing to spend (input+output, from usage).
# 0 => pure local zero-token mode.
ESCALATION_BUDGET_TOKENS = int(os.environ.get("ESCALATION_BUDGET_TOKENS", "500"))
ESCALATE_CONF_THRESHOLD = 0.55   # tasks below this confidence are candidates
ESCALATION_TIMEOUT_S = 30
ESCALATION_WORKERS = 4

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
