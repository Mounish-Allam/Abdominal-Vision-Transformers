"""Ensure the repo root is importable so `rag.*` and `src.*`-style imports resolve."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
