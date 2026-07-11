"""Fireworks escalation client (stdlib only).

Every call goes through FIREWORKS_BASE_URL (the judging proxy) and spends from
a single global token budget. Model choice: ALLOWED_MODELS env when present
(strict — anything else risks MODEL_VIOLATION), else a curated serverless
fallback chain with 404 fall-through."""

import json
import os
import threading
import urllib.error
import urllib.request

from . import config
from .util import log


class TokenBudget:
    def __init__(self, total: int):
        self.total = total
        self.spent = 0
        self._lock = threading.Lock()

    def try_reserve(self, est: int) -> bool:
        with self._lock:
            if self.spent + est > self.total:
                return False
            self.spent += est  # provisional; corrected on commit
            self._pre = est
            return True

    def commit(self, est: int, actual: int):
        with self._lock:
            self.spent += actual - est

    def refund(self, est: int):
        with self._lock:
            self.spent -= est


BUDGET = TokenBudget(config.ESCALATION_BUDGET_TOKENS)


def raise_budget(total: int):
    """Emergency mode: gate survival outranks token rank."""
    with BUDGET._lock:
        if total > BUDGET.total:
            BUDGET.total = total
            log(f"escalation budget raised to {total} (emergency)")


def _base_urls():
    base = (os.environ.get("FIREWORKS_BASE_URL")
            or "https://api.fireworks.ai/inference/v1").rstrip("/")
    urls = [base + "/chat/completions"]
    if not base.endswith("/v1"):
        urls.append(base + "/v1/chat/completions")
    return urls


def allowed_models():
    """ALLOWED_MODELS env (strict) or curated fallback chain."""
    env = os.environ.get("ALLOWED_MODELS", "").strip()
    if env:
        listed = [m.strip() for m in env.split(",") if m.strip()]
        ranked = [m for m in config.FALLBACK_MODELS if m in listed]
        ranked += [m for m in listed if m not in ranked]
        return ranked, True
    return list(config.FALLBACK_MODELS), False


def pick_models(category: str):
    models, strict = allowed_models()
    if strict and category in ("factual", "sentiment", "summary", "ner"):
        gemma = [m for m in models if config.GEMMA_HINT in m.lower()]
        if gemma:
            return gemma + [m for m in models if m not in gemma]
    return models


def est_tokens(text: str) -> int:
    return max(8, int(len(text) / 3.4))


def chat(prompt: str, category: str, max_tokens: int = 120):
    """One escalation call. Returns (answer or None, tokens_spent)."""
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        log("escalation skipped: no FIREWORKS_API_KEY")
        return None, 0

    sys_prompt = "Answer directly and concisely."
    est = est_tokens(sys_prompt + prompt) + max_tokens
    if not BUDGET.try_reserve(est):
        log(f"escalation skipped (budget): est={est} spent={BUDGET.spent}/{BUDGET.total}")
        return None, 0

    body_base = {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        # all current serverless models are hybrid reasoners; 'low' minimizes
        # billed reasoning tokens and is accepted (or ignored) by every model
        "reasoning_effort": "low",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # default urllib UA gets WAF-blocked (403) — any custom UA passes
        "User-Agent": "frugalrouter/1.0",
    }

    for model in pick_models(category):
        body = dict(body_base, model=model)
        data = json.dumps(body).encode()
        for url in _base_urls():
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=config.ESCALATION_TIMEOUT_S) as r:
                    resp = json.loads(r.read().decode())
                content = (resp["choices"][0]["message"].get("content") or "").strip()
                usage = resp.get("usage", {}) or {}
                actual = int(usage.get("total_tokens")
                             or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
                             or est)
                BUDGET.commit(est, actual)
                log(f"escalated[{category}] via {model.rsplit('/', 1)[-1]}: "
                    f"{actual} tok (total {BUDGET.spent}/{BUDGET.total})")
                if content:
                    return content, actual
                return None, actual
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue  # wrong path or unknown model — next url, then next model
                if e.code in (401, 403):
                    log(f"escalation auth failure ({e.code}) — giving up")
                    BUDGET.refund(est)
                    return None, 0
                log(f"escalation http {e.code} on {model.rsplit('/', 1)[-1]}")
                break  # server error on this model — try next model
            except Exception as e:  # noqa: BLE001
                log(f"escalation error on {model.rsplit('/', 1)[-1]}: {e}")
                break
    BUDGET.refund(est)
    return None, 0
