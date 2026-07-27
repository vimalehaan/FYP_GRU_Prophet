"""Path discovery helpers for notebooks and scripts."""

from __future__ import annotations

from pathlib import Path

from production.utils.config import HYBRID_DIR, PRODUCTION_DIR, REPO_ROOT


def discover_repo_root(start: Path | None = None) -> Path:
    """Find repository root by locating production/hybrid/models/model.keras."""
    marker = Path("production") / "hybrid" / "models" / "model.keras"
    candidates = []
    if start is not None:
        candidates.append(start.resolve())
    candidates.extend([
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
        Path(__file__).resolve().parents[2],
        Path(__file__).resolve().parents[3],
    ])
    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        if (base / marker).exists():
            return base
    return REPO_ROOT


def discover_hybrid_dir(start: Path | None = None) -> Path:
    """Return production/hybrid artifact directory."""
    root = discover_repo_root(start)
    hybrid = root / "production" / "hybrid"
    if not hybrid.exists():
        raise FileNotFoundError(f"Production hybrid directory not found: {hybrid}")
    return hybrid


def ensure_on_syspath() -> Path:
    """Insert repo root on sys.path for research + production imports."""
    import sys

    root = discover_repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
