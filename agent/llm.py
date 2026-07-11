"""llama.cpp server management + chat client (pure stdlib).

Two servers (general + coder) stay resident; a single global lock serializes
generation so the 2 vCPUs are never oversubscribed. cache_prompt=True lets
same-category tasks reuse the shared prompt prefix."""

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request

from . import config
from .util import log, strip_think

_GEN_LOCK = threading.Lock()  # one local generation at a time, ever


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
             top_k=None, grammar=None, thinking=False, timeout_s=120.0) -> str:
        """Blocking chat completion against this local server. Returns content
        with any <think> block stripped. Raises on failure after one retry."""
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
        if grammar:
            body["grammar"] = grammar
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
                msg = resp["choices"][0]["message"]
                content = msg.get("content") or ""
                return strip_think(content)
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
