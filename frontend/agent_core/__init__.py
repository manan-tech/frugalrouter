"""Vendored copies of FrugalRouter's zero-token local machinery.

These are the exact modules from the competition agent (agent/router.py,
agent/ner_onnx.py, agent/sentiment_onnx.py, agent/rag.py, agent/util.py) with
two demo-only adjustments:

  1. relative imports (`from .util import log`) became absolute package imports
     (`from agent_core.util import log`) so the Gradio app can import them from
     the Space's working directory;
  2. the default MODEL_DIR constants point at ./models/{router,ner,sentiment}
     next to app.py instead of the container's /models — the same env vars
     (ROUTER_ONNX_DIR, NER_ONNX_DIR, SENTIMENT_ONNX_DIR, RAG_DIR) still
     override them.

No inference logic was changed.
"""

import os

# frontend/ root — the directory that holds app.py, models/ and data/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
