"""LFHE evaluation pipeline reproduction via frozen Stage 2 B0 weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.hybrid_artifacts import load_hybrid_artifacts


def load_stage2_b0_reference(csrle_exp: Path) -> dict[str, float]:
    """Cohort means from frozen Stage 2 B0 residual_summary.json."""
    ref_path = (
        csrle_exp
        / "stage2_synthetic_gru"
        / "b0_lc"
        / "evaluation"
        / "residual_summary.json"
    )
    with ref_path.open() as fh:
        data = json.load(fh)
    res = data["residual"]
    return {
        "residual_pearson_r": res["residual_pearson_r"]["mean"],
        "residual_std_ratio": res["residual_std_ratio"]["mean"],
        "residual_mae_scaled": res["residual_mae_scaled"]["mean"],
        "residual_rmse_scaled": res["residual_rmse_scaled"]["mean"],
        "residual_r2": res["residual_r2"]["mean"],
    }


def load_stage2_b0_extended_csv(csrle_exp: Path) -> pd.DataFrame:
    path = (
        csrle_exp
        / "stage2_synthetic_gru"
        / "b0_lc"
        / "evaluation"
        / "extended_metrics.csv"
    )
    return pd.read_csv(path)


def check_reproduction_gate(
    ext_df: pd.DataFrame,
    reference: dict[str, float],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    """Compare LFHE evaluation cohort means to Stage 2 B0 reference."""
    actual = {
        "residual_pearson_r": float(ext_df["residual_pearson_r"].mean()),
        "residual_std_ratio": float(ext_df["residual_std_ratio"].mean()),
        "residual_mae_scaled": float(ext_df["residual_mae_scaled"].mean()),
        "residual_rmse_scaled": float(ext_df["residual_rmse_scaled"].mean()),
        "residual_r2": float(ext_df["residual_r2"].mean()),
    }
    checks: dict[str, Any] = {}
    for key in ("residual_pearson_r", "residual_std_ratio", "residual_mae_scaled"):
        tol = tolerances.get(key, tolerances.get(key.replace("_scaled", ""), 0.015))
        delta = abs(actual[key] - reference[key])
        checks[key] = {
            "lfhe_value": actual[key],
            "stage2_reference": reference[key],
            "abs_delta": delta,
            "tolerance": tol,
            "pass": delta <= tol,
        }
    for key in ("residual_rmse_scaled", "residual_r2"):
        checks[key] = {
            "lfhe_value": actual[key],
            "stage2_reference": reference[key],
            "abs_delta": abs(actual[key] - reference[key]),
            "pass": True,
            "informational": True,
        }

    passed = all(checks[k]["pass"] for k in ("residual_pearson_r", "residual_std_ratio", "residual_mae_scaled"))
    return {
        "pass": passed,
        "checks": checks,
        "n_containers": len(ext_df),
    }


def check_per_container_reproduction(
    lfhe_ext: pd.DataFrame,
    stage2_ext: pd.DataFrame,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Verify LFHE eval path on frozen Stage 2 weights matches Stage 2 CSV."""
    merged = lfhe_ext.merge(stage2_ext, on="container_id", suffixes=("_lfhe", "_stage2"))
    max_diffs: dict[str, float] = {}
    for col in (
        "residual_pearson_r",
        "residual_std_ratio",
        "residual_mae_scaled",
        "residual_rmse_scaled",
    ):
        diff = (merged[f"{col}_lfhe"] - merged[f"{col}_stage2"]).abs()
        max_diffs[col] = float(diff.max())
    passed = all(v <= atol for v in max_diffs.values())
    return {
        "pass": passed,
        "max_abs_diff": max_diffs,
        "tolerance": atol,
        "n_containers": len(merged),
    }


def write_reproduction_report(path: Path, gate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(gate, fh, indent=2)


def verify_frozen_stage2_model(
    csrle_exp: Path,
    container_ids: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    peak_thresholds: dict[str, float],
    tolerances: dict[str, float],
) -> dict[str, Any]:
    """
    Phase 2 reproduction: evaluate read-only Stage 2 B0 weights through LFHE pipeline.
    """
    from utils.csrle.stage2_training import build_condition_scalers
    from utils.lfhe.evaluation import run_lfhe_evaluation

    model_dir = csrle_exp / "stage2_synthetic_gru" / "b0_lc" / "models"
    model, res_mean, res_std, _ = load_hybrid_artifacts(
        model_dir / "hybrid_gru.keras",
        model_dir / "residual_stats.pkl",
    )
    scalers = build_condition_scalers(train_df)
    eval_out = run_lfhe_evaluation(
        container_ids, train_df, val_df, scalers, model, res_mean, res_std, peak_thresholds,
    )
    ext_df = eval_out["extended_metrics"]
    ref = load_stage2_b0_reference(csrle_exp)
    cohort_gate = check_reproduction_gate(ext_df, ref, tolerances)
    stage2_csv = load_stage2_b0_extended_csv(csrle_exp)
    per_container_gate = check_per_container_reproduction(ext_df, stage2_csv)

    return {
        "pass": cohort_gate["pass"] and per_container_gate["pass"],
        "cohort_gate": cohort_gate,
        "per_container_gate": per_container_gate,
        "eval_out": eval_out,
        "reference": ref,
        "model_source": str(model_dir),
    }
