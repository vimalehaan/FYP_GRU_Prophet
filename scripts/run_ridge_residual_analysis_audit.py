#!/usr/bin/env python3
"""Run read-only audit of Ridge Residual Analysis experiment."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.rra_audit.runner import run_audit  # noqa: E402

RRA_EXP = REPO_ROOT / "experiments" / "ridge_residual_analysis_2026-07-26_161500"


def main(timestamp: str | None = None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    audit_dir = REPO_ROOT / "experiments" / f"ridge_residual_analysis_audit_{ts}"
    audit_dir.mkdir(parents=True, exist_ok=True)
    report = run_audit(RRA_EXP, audit_dir, REPO_ROOT)
    print(f"Audit complete: {audit_dir}")
    print(f"Scaling bug confirmed: {report['decision']['scaling_bug_in_rra_v1']}")
    print(f"Corrected ACF reduction: {report['corrected_summary']['corrected_acf_reduction']:.4f}")
    return audit_dir


if __name__ == "__main__":
    ts = sys.argv[1] if len(sys.argv) > 1 else None
    main(ts)
