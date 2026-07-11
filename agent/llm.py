"""llama.cpp server management + chat client (pure stdlib).

Two servers (general + coder) stay resident; a single global lock serializes
generation so the 2 vCPUs are never oversubscribed. cache_prompt=True lets
same-category tasks reuse the shared prompt prefix."""

import json
import math
import subprocess
import threading
import time
import urllib.error
import urllib.request

from . import config
from .util import log, strip_think

_GEN_LOCK = threading.Lock()  # one local generation at a time, ever

# Calibration capture (local eval only): when True, every local completion
# requests logprobs and stashes its confidence signals in LAST_SIGNALS so
# main.py can emit calibration JSONL records without any pipeline plumbing.
# Enabled by main() iff config.CALIBRATION_LOG_PATH is set; the grading
# harness never sets it, so production requests stay byte-identical.
CAPTURE_SIGNALS = False
LAST_SIGNALS = {}  # signals from the most recent local completion (mutated in place)

# Number of top-logprob alternatives requested per position; also the fixed k
# used by _extract_signals' self_certainty ("relative to uniform over top-k").
_TOP_LOGPROBS_K = 5


def _extract_signals(choice) -> dict:
    """Compute confidence signals from a choice's per-token logprobs.

    Returns {"mean_logprob", "min_token_margin", "self_certainty"}, each None
    when the underlying data is unavailable (older server, empty generation).

    - mean_logprob:    mean of the chosen token logprobs across positions.
    - min_token_margin: min over positions of (top1.logprob - top2.logprob),
                        i.e. the least-decisive token step in the output.
    - self_certainty:  mean over positions of the top-1 probability expressed
                        relative to a uniform distribution over the top-k
                        requested (top1_prob / (1/k) == top1_prob * k, with
                        k fixed at _TOP_LOGPROBS_K); 1.0 means "no better than
                        uniform", larger means more peaked. k is fixed (not the
                        observed alternative count) so the signal stays
                        consistent even at truncated/end-of-vocab positions.
    """
    empty = {"mean_logprob": None, "min_token_margin": None,
             "self_certainty": None}
    lp = choice.get("logprobs") if isinstance(choice, dict) else None
    entries = lp.get("content") if isinstance(lp, dict) else None
    if not entries:
        return empty

    chosen_logprobs = []
    margins = []
    certainties = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        clp = entry.get("logprob")
        if isinstance(clp, (int, float)):
            chosen_logprobs.append(float(clp))
        top = entry.get("top_logprobs") or []
        # keep only well-formed alternatives, sorted most-probable first
        alts = [a.get("logprob") for a in top
                if isinstance(a, dict) and isinstance(a.get("logprob"), (int, float))]
        alts = sorted((float(x) for x in alts), reverse=True)
        if len(alts) >= 2:
            margins.append(alts[0] - alts[1])
        if alts:
            certainties.append(math.exp(alts[0]) * _TOP_LOGPROBS_K)

    return {
        "mean_logprob": (sum(chosen_logprobs) / len(chosen_logprobs)
                         if chosen_logprobs else None),
        "min_token_margin": min(margins) if margins else None,
        "self_certainty": (sum(certainties) / len(certainties)
                           if certainties else None),
    }


class LlamaServer:
    def __init__(self, name: str, model_path: str, port: int, ctx: int):
        self.name = name
        self.model_path = model_path
        self.port = port
        self.ctx = ctx
        self.proc = None

    def start(self, ctx_override=None):
        cmd = [
            "llama-server",
            "-m", self.model_path,
            "-t", str(config.LLM_THREADS),
            "-c", str(ctx_override or self.ctx),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--jinja",
            "--no-webui",
            # --parallel 1: llama.cpp splits -c across slots (per-slot ctx =
            # -c / parallel), so --parallel 2 would halve usable context to
            # 1024 and break the long summary/NER passages GENERAL_CTX=2048 was
            # sized for. _GEN_LOCK serializes all local generation anyway, so a
            # second slot would give zero concurrency benefit — keep it at 1.
            "--parallel", "1",
        ]
        log(f"starting {self.name}: {' '.join(cmd)}")
        # keep stderr: if the grader kills us, the harness logs show why
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)

    def wait_ready(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                log(f"{self.name} server died (rc={self.proc.returncode})")
                return False
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        log(f"{self.name} ready on :{self.port}")
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def ensure_alive(self) -> bool:
        if getattr(self, "proxied", False):
            return True  # routed to another live server — never respawn
        if self.proc and self.proc.poll() is None:
            return True
        log(f"{self.name} not running — restarting")
        self.start()
        return self.wait_ready(config.SERVER_START_TIMEOUT_S)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def chat(self, messages, max_tokens=160, temperature=None, top_p=None,
             top_k=None, grammar=None, thinking=False, timeout_s=120.0,
             min_p=None, return_signals=False):
        """Blocking chat completion against this local server.

        Returns content with any <think> block stripped. Raises on failure
        after one retry.

        min_p (float|None): passed through to llama-server as "min_p" when set.
        return_signals (bool): when True, requests token logprobs and returns a
            tuple (content, signals) instead of a plain string, where signals =
            {"mean_logprob", "min_token_margin", "self_certainty"} (any value may
            be None if logprobs were unavailable). Default False keeps the plain
            string return for backwards compatibility.
        """
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": config.THINK_TEMP if thinking else (
                temperature if temperature is not None else config.GEN_TEMP),
            "top_p": config.THINK_TOP_P if thinking else (
                top_p if top_p is not None else config.GEN_TOP_P),
            "top_k": top_k if top_k is not None else config.GEN_TOP_K,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }
        if min_p is not None:
            body["min_p"] = min_p
        if grammar:
            body["grammar"] = grammar
        if return_signals or CAPTURE_SIGNALS:
            # llama-server's OpenAI-compat endpoint returns per-token logprobs
            # with the top-k alternatives at each position.
            body["logprobs"] = True
            body["top_logprobs"] = _TOP_LOGPROBS_K
        data = json.dumps(body).encode()
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        last_err = None
        for attempt in (1, 2):
            try:
                with _GEN_LOCK:
                    req = urllib.request.Request(
                        url, data=data, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=timeout_s) as r:
                        resp = json.loads(r.read().decode())
                choice = resp["choices"][0]
                msg = choice["message"]
                content = strip_think(msg.get("content") or "")
                if return_signals or CAPTURE_SIGNALS:
                    signals = _extract_signals(choice)
                    LAST_SIGNALS.clear()
                    LAST_SIGNALS.update(signals)
                    if return_signals:
                        return content, signals
                return content
            except Exception as e:  # noqa: BLE001 — never let a task die silently
                last_err = e
                log(f"{self.name} chat error (attempt {attempt}): {e}")
                if not self.ensure_alive():
                    break
        raise RuntimeError(f"{self.name} chat failed: {last_err}")


GENERAL = LlamaServer("general", config.GENERAL_MODEL_PATH,
                      config.GENERAL_PORT, config.GENERAL_CTX)
CODER = LlamaServer("coder", config.CODER_MODEL_PATH,
                    config.CODER_PORT, config.CODER_CTX)


def _start_one(server) -> bool:
    server.start()
    if server.wait_ready(config.SERVER_START_TIMEOUT_S):
        return True
    # retry once with a smaller context (halves KV allocation)
    log(f"{server.name} failed to start — retrying with ctx=1024")
    server.stop()
    time.sleep(1)
    server.start(ctx_override=1024)
    return server.wait_ready(config.SERVER_START_TIMEOUT_S)


def start_all() -> bool:
    # sequential startup halves the peak memory/CPU spike of model loading —
    # the grading env is tighter than it looks (no swap headroom)
    ok_g = _start_one(GENERAL)
    ok_c = _start_one(CODER)
    if ok_g and not ok_c:
        log("coder dead — general will handle code categories")
        CODER.stop()
        CODER.proc = None
        CODER.port = GENERAL.port  # route coder calls to the general model
        CODER.proxied = True
        return True
    return ok_g and ok_c


def stop_all():
    GENERAL.stop()
    CODER.stop()


def probe_tps() -> float:
    """Measure decode speed with a short fixed generation."""
    t0 = time.monotonic()
    try:
        GENERAL.chat(
            [{"role": "user", "content": "Count from 1 to 30, comma separated."}],
            max_tokens=48, temperature=0.0, timeout_s=90)
    except Exception as e:
        log(f"tps probe failed: {e}")
        return 0.0
    dt = time.monotonic() - t0
    tps = 48.0 / dt if dt > 0 else 0.0
    log(f"tps probe: ~{tps:.1f} tok/s")
    return tps
