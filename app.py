"""
Hugging Face Spaces entry point.

Environment variables (set as Space Secrets on HF):
  HF_MODEL_REPO_ID  — e.g. "your-username/swin-daf-chaos-mri"
  HF_MODEL_FILENAME — filename inside that repo (default: Best_MS-Dual-Guided.pth)
  ENCODER_NAME      — swin_tiny/small/base (default: swin_tiny_patch4_window7_224)
  GROQ_API_KEY      — free key from console.groq.com; enables LLM clinical reports
"""

import os
import sys

# Make src/ importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from huggingface_hub import hf_hub_download

from demo import build_ui, load_model, _cache as _model_cache

# ── Read config from env vars ─────────────────────────────────────────────────
HF_REPO_ID  = os.environ.get("HF_MODEL_REPO_ID",  "")
HF_FILENAME = os.environ.get("HF_MODEL_FILENAME",  "Best_MS-Dual-Guided.pth")
ENCODER     = os.environ.get("ENCODER_NAME",        "swin_tiny_patch4_window7_224")

# ── Pre-load model weights from HF Hub ───────────────────────────────────────
weights_path = ""
if HF_REPO_ID:
    print(f"[app] Downloading weights: {HF_REPO_ID}/{HF_FILENAME}")
    try:
        weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model_cache[(weights_path, ENCODER)] = load_model(weights_path, ENCODER, device)
        print(f"[app] Model ready on {device}.")
    except Exception as exc:
        print(f"[app] WARNING — could not load weights from HF Hub: {exc}")
        print("[app] Running in random-weight demo mode.")
else:
    print("[app] HF_MODEL_REPO_ID not set — running in random-weight demo mode.")

# ── Launch Gradio app ─────────────────────────────────────────────────────────
app = build_ui(default_weights=weights_path, default_encoder=ENCODER)
app.launch()
