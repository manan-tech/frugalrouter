"""Fetch the three public int8 ONNX models the demo runs on (~253 MB total).

Same artifacts the competition Dockerfile bakes into the agent image:

  models/router/     Xenova/all-MiniLM-L6-v2                     (int8, ~23 MB)
  models/ner/        zencrazycat/ner-bert-base-cased-ontonotesv5 (int8, ~104 MB)
  models/sentiment/  Xenova/twitter-roberta-base-sentiment-latest(int8, ~126 MB)

Stdlib-only (urllib) — no huggingface_hub needed. Idempotent: files that
already exist with a non-zero size are skipped, so calling this at every Space
startup costs nothing after the first boot. Run manually with:

    python download_models.py
"""

import os
import sys
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_ROUTER = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main"
_NER = ("https://huggingface.co/zencrazycat/"
        "ner-bert-base-cased-ontonotesv5-englishv4-onnx/resolve/main")
_SENT = ("https://huggingface.co/Xenova/"
         "twitter-roberta-base-sentiment-latest/resolve/main")

# (subdir, filename-on-disk, url). The NER repo ships ONLY model_quantized.onnx
# (no model.onnx); the agent modules always look for "model.onnx", so every
# download is renamed on disk.
FILES = [
    ("router", "model.onnx", f"{_ROUTER}/onnx/model_int8.onnx"),
    ("router", "tokenizer.json", f"{_ROUTER}/tokenizer.json"),
    ("ner", "model.onnx", f"{_NER}/onnx/model_quantized.onnx"),
    ("ner", "tokenizer.json", f"{_NER}/tokenizer.json"),
    ("ner", "config.json", f"{_NER}/config.json"),
    ("sentiment", "model.onnx", f"{_SENT}/onnx/model_int8.onnx"),
    ("sentiment", "tokenizer.json", f"{_SENT}/tokenizer.json"),
    ("sentiment", "config.json", f"{_SENT}/config.json"),
]

_UA = "FrugalRouterDemo/1.0 (+https://huggingface.co/spaces)"


def _fetch(url: str, dest: str, retries: int = 3) -> None:
    """Download url -> dest atomically (tmp file + rename), with retries."""
    tmp = dest + ".part"
    last = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {url}: {last}")


def ensure_models(verbose: bool = True) -> bool:
    """Download any missing model file. Returns True iff all files are present."""
    ok = True
    for subdir, name, url in FILES:
        dest = os.path.join(MODELS_DIR, subdir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            if verbose:
                print(f"[models] {subdir}/{name}: present "
                      f"({os.path.getsize(dest) / 1e6:.1f} MB)")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if verbose:
            print(f"[models] {subdir}/{name}: downloading {url}", flush=True)
        try:
            _fetch(url, dest)
            if verbose:
                print(f"[models] {subdir}/{name}: done "
                      f"({os.path.getsize(dest) / 1e6:.1f} MB)", flush=True)
        except Exception as e:  # noqa: BLE001 — the app degrades gracefully per tab
            ok = False
            print(f"[models] {subdir}/{name}: FAILED ({e})", file=sys.stderr)
    return ok


if __name__ == "__main__":
    sys.exit(0 if ensure_models() else 1)
