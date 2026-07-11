"""Fireworks escalation client (stdlib only).

Every call goes through FIREWORKS_BASE_URL (the judging proxy) and spends from
a single global token budget. Model choice: ALLOWED_MODELS env when present
(strict — anything else risks MODEL_VIOLATION), else a curated serverless
fallback chain with 404 fall-through."""

import json
import os
import re
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


def _headers():
    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # default urllib UA gets WAF-blocked (403) — any custom UA passes
        "User-Agent": "frugalrouter/1.0",
    }


def _request(body_base: dict, category: str, est: int):
    """Run the model/url fallback loop for one already-reserved `est`.
    Commits or refunds `est` against BUDGET exactly once. Returns
    (content, actual_tokens, model) on success, else (None, 0, None)."""
    headers = _headers()
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
                return content, actual, model
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue  # wrong path or unknown model — next url, then next model
                if e.code in (401, 403):
                    log(f"escalation auth failure ({e.code}) — giving up")
                    BUDGET.refund(est)
                    return None, 0, None
                log(f"escalation http {e.code} on {model.rsplit('/', 1)[-1]}")
                break  # server error on this model — try next model
            except Exception as e:  # noqa: BLE001
                log(f"escalation error on {model.rsplit('/', 1)[-1]}: {e}")
                break
    BUDGET.refund(est)
    return None, 0, None


def chat(prompt: str, category: str, max_tokens: int = 120):
    """One escalation call. Returns (answer or None, tokens_spent)."""
    if not os.environ.get("FIREWORKS_API_KEY", ""):
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
    content, actual, model = _request(body_base, category, est)
    if model is None:
        return None, 0
    log(f"escalated[{category}] via {model.rsplit('/', 1)[-1]}: "
        f"{actual} tok (total {BUDGET.spent}/{BUDGET.total})")
    return (content or None), actual


# --------------------------------------------------------------------------
# batched escalation: one call carries a whole category's weak questions
# --------------------------------------------------------------------------
# Byte-stable instruction block (cache-friendly prefix across calls). Do NOT
# interpolate anything into this string — its bytes must stay identical.
BATCH_SYS = (
    "You answer a numbered list of independent questions. "
    "Reply with ONLY a JSON array and nothing else — no prose before or "
    "after, no code fences. Use exactly this shape: "
    '[{"id": <question number>, "answer": "<answer text>"}]. '
    "Emit one object per question, reusing the given id numbers. "
    "Each answer must be complete but terse: lead with the answer itself, "
    "do not hedge, do not restate the question, add no preamble."
)


def _esc_cap(category: str):
    """(reasoning_effort, max_tokens_per_item) for a category, baked default."""
    caps = getattr(config, "ESC_CAPS", {}) or {}
    val = caps.get(category)
    if not val:
        return "low", 160
    reasoning, per_item = val
    return reasoning, int(per_item)


def _extract_json_array(text: str):
    """Pull the first balanced top-level JSON array out of `text`, tolerating
    reasoning preambles / <think> blocks / code fences. Returns a list or None."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    for s in (i for i, ch in enumerate(text) if ch == "["):
        depth = 0
        in_str = False
        esc = False
        for j in range(s, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        val = json.loads(text[s:j + 1])
                    except (json.JSONDecodeError, ValueError):
                        break  # unbalanced/garbage — try the next '['
                    if isinstance(val, list):
                        return val
                    break
    return None


def _parse_batch(content: str, n: int) -> dict:
    """Map answer objects to their 1-based ids. Returns {id: answer_str}."""
    out = {}
    arr = _extract_json_array(content)
    if arr is None:
        return out
    for obj in arr:
        if not isinstance(obj, dict) or "id" not in obj:
            continue
        try:
            idx = int(obj["id"])
        except (TypeError, ValueError):
            continue
        ans = obj.get("answer")
        if isinstance(ans, (int, float)):
            ans = str(ans)
        if isinstance(ans, str) and ans.strip():
            out[idx] = ans.strip()
    return out


def _batch_individual(items, category: str, per_item: int) -> dict:
    """Per-item fallback: re-ask each question with a normal single chat()."""
    out = {}
    for tid, prompt in items:
        ans, _spent = chat(prompt, category, max_tokens=per_item)
        if ans and ans.strip():
            out[tid] = ans.strip()
    return out


def batch_chat(items, category: str) -> dict:
    """One escalation call for a batch of same-category weak questions.

    items: iterable of (task_id, prompt). Returns {task_id: answer}.
    Budget accounting matches chat(): est = prefix + sum(questions) +
    max_tokens, where max_tokens = per-item cap * batch size. Parse failures
    (and whole-call failures) fall back to individual chat() re-asks."""
    results = {}
    items = list(items)
    if not items:
        return results
    if not os.environ.get("FIREWORKS_API_KEY", ""):
        log("batch escalation skipped: no FIREWORKS_API_KEY")
        return results

    reasoning, per_item = _esc_cap(category)
    max_tokens = per_item * len(items)  # scales with batch size
    numbered = "\n".join(f"{i}. {p}" for i, (_tid, p) in enumerate(items, 1))
    est = est_tokens(BATCH_SYS + numbered) + max_tokens
    if not BUDGET.try_reserve(est):
        log(f"batch escalation skipped (budget): est={est} n={len(items)} "
            f"spent={BUDGET.spent}/{BUDGET.total}")
        return results

    body_base = {
        "messages": [
            {"role": "system", "content": BATCH_SYS},
            {"role": "user", "content": numbered},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "reasoning_effort": reasoning,
    }
    content, actual, model = _request(body_base, category, est)
    if model is None:
        log(f"batch[{category}] call failed — re-asking {len(items)} individually")
        return _batch_individual([(t, p) for t, p in items], category, per_item)

    log(f"batch escalated[{category}] n={len(items)} via "
        f"{model.rsplit('/', 1)[-1]}: {actual} tok (total {BUDGET.spent}/{BUDGET.total})")
    parsed = _parse_batch(content, len(items))
    missing = []
    for idx, (tid, prompt) in enumerate(items, 1):
        ans = parsed.get(idx)
        if ans and ans.strip():
            results[tid] = ans.strip()
        else:
            missing.append((tid, prompt))
    if missing:
        log(f"batch[{category}] {len(missing)}/{len(items)} unparsed — "
            f"re-asking individually")
        results.update(_batch_individual(missing, category, per_item))
    return results
