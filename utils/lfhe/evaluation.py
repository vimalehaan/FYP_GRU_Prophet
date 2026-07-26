"""LFHE evaluation — wraps frozen CSRLE Stage 2 metric pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.csrle.stage2_baselines import train_ridge_residual_model
from utils.csrle.stage2_metrics import (
    actual_predicted_residual_arrays,
    run_condition_evaluation,
)
from utils.csrle.stage2_stats import cohort_summary, paired_bootstrap_diff, residual_cohort_summary
from utils.hybrid_config import DEFAULT_INPUT_WINDOW


def run_lfhe_evaluation(
    container_ids: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scalers: dict[str, Any],
    model: Any,
    res_mean: float,
    res_std: float,
    peak_thresholds: dict[str, float],
) -> dict[str, Any]:
    """Standard CSRLE-style evaluation for one LFHE arm."""
    ridge = train_ridge_residual_model(
        train_df, res_mean, res_std, input_window=DEFAULT_INPUT_WINDOW,
    )
    eval_out = run_condition_evaluation(
        container_ids,
        train_df,
        val_df,
        scalers,
        model,
        res_mean,
        res_std,
        ridge,
        peak_thresholds,
        compute_recovery=False,
    )
    ext_df = eval_out["extended_metrics"]
    per_container: dict[str, tuple] = {}
    for cid, result in eval_out["inference"].items():
        per_container[cid] = actual_predicted_residual_arrays(result)

    return {
        **eval_out,
        "per_container_residuals": per_container,
        "residual_summary": residual_cohort_summary(ext_df),
        "ridge_model": ridge,
    }


def aggregate_horizon_cohort(horizon_pc: pd.DataFrame) -> pd.DataFrame:
    if horizon_pc.empty:
        return pd.DataFrame()
    return (
        horizon_pc.groupby("horizon", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            std_ratio=("std_ratio", "mean"),
        )
    )


def paired_primary_metrics(
    da_ext: pd.DataFrame,
    mse_ext: pd.DataFrame,
    bootstrap_seed: int = 12_345,
) -> dict[str, Any]:
    """Paired bootstrap for primary hypothesis metrics."""
    idx = "container_id"
    out: dict[str, Any] = {}
    for metric in ("residual_pearson_r", "residual_std_ratio", "residual_mae_scaled"):
        da_s = da_ext.set_index(idx)[metric]
        mse_s = mse_ext.set_index(idx)[metric]
        out[metric] = paired_bootstrap_diff(
            da_s, mse_s, seed=bootstrap_seed,
        )
    return out


def evaluate_hypothesis_verdict(
    da_ext: pd.DataFrame,
    mse_ext: pd.DataFrame,
    paired: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Apply frozen success_failure_criteria (mechanical)."""
    da_r = float(da_ext["residual_pearson_r"].mean())
    da_std = float(da_ext["residual_std_ratio"].mean())
    mse_std = float(mse_ext["residual_std_ratio"].mean())
    mae_delta = float(da_ext["residual_mae_scaled"].mean() - mse_ext["residual_mae_scaled"].mean())

    s1 = (
        da_std >= thresholds["mean_std_ratio_da_mse"]
        and paired["residual_std_ratio"]["ci_low"] > 0
        and paired["residual_std_ratio"]["fraction_left_better"] >= 0.60
    )
    s2 = (
        paired["residual_pearson_r"]["ci_low"] > 0
        and da_r >= thresholds.get("mean_pearson_r_da_mse", 0.10)
    )
    s3 = mae_delta <= thresholds["mae_increase_max"]

    if s1 and s2 and s3:
        verdict = "support_H1"
    elif paired["residual_std_ratio"]["ci_high"] < 0 and paired["residual_pearson_r"]["ci_high"] < 0:
        verdict = "reject_H1"
    elif s1 and not s2:
        verdict = "inconclusive_I1"
    elif s2 and not s1:
        verdict = "inconclusive_I2"
    elif (s1 or s2) and not s3:
        verdict = "inconclusive_I3"
    else:
        verdict = "reject_H1"

    return {
        "verdict": verdict,
        "S1_dispersion": s1,
        "S2_correlation": s2,
        "S3_mae_guardrail": s3,
        "cohort_mean_std_ratio_da_mse": da_std,
        "cohort_mean_std_ratio_mse": mse_std,
        "cohort_mean_pearson_r_da_mse": da_r,
        "cohort_mean_mae_delta": mae_delta,
        "paired_bootstrap": paired,
    }
