"""Shared bootstrap for production CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path

PRODUCTION_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PRODUCTION_DIR.parent


def bootstrap() -> Path:
    """Ensure repo root is on sys.path for production + research imports."""
    root_str = str(REPO_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return REPO_ROOT
