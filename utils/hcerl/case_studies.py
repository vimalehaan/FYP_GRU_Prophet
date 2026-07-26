"""Automatic HCERL case study selection and narrative."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.hybrid_inference import HybridInferenceResult


def select_case_study_containers(
    v0_df: pd.DataFrame,
    best_df: pd.DataFrame,
    v0_id: str = "v0_baseline",
    best_id: str = "v4_full",
) -> dict[str, str]:
    """
    Pick containers for four case study categories based on Δ MAE (best - v0).

    Negative delta = improvement with best variant.
    """
    merged = v0_df.merge(
        best_df, on="container_id", suffixes=("_v0", "_best"),
    )
    merged["delta_mae"] = merged["day1_mae_best"] - merged["day1_mae_v0"]

    valid = merged.dropna(subset=["delta_mae"])
    if valid.empty:
        return {}

    largest_improvement = valid.loc[valid["delta_mae"].idxmin(), "container_id"]
    largest_degradation = valid.loc[valid["delta_mae"].idxmax(), "container_id"]
    median_improvement = valid.iloc[
        (valid["delta_mae"] - valid["delta_mae"].median()).abs().idxmin()
    ]["container_id"]
    no_improvement = valid.loc[
        valid["delta_mae"].abs().idxmin(), "container_id",
    ]

    return {
        "largest_improvement": str(largest_improvement),
        "median_improvement": str(median_improvement),
        "no_improvement": str(no_improvement),
        "performance_degradation": str(largest_degradation),
    }


def build_case_study_narratives(
    case_ids: dict[str, str],
    variant_eval_dfs: dict[str, pd.DataFrame],
    inference_by_variant: dict[str, dict[str, HybridInferenceResult]],
    variant_order: tuple[str, ...],
) -> dict[str, Any]:
    """Build trajectory data and short explanations for each case."""
    narratives: dict[str, Any] = {"cases": {}, "trajectories": {}}

    for category, cid in case_ids.items():
        v0_mae = float(
            variant_eval_dfs["v0_baseline"]
            .loc[variant_eval_dfs["v0_baseline"]["container_id"] == cid, "day1_mae"]
            .iloc[0]
        )
        best_vid = variant_order[-1]
        best_mae = float(
            variant_eval_dfs[best_vid]
            .loc[variant_eval_dfs[best_vid]["container_id"] == cid, "day1_mae"]
            .iloc[0]
        )
        delta = best_mae - v0_mae

        v0_r = float(
            variant_eval_dfs["v0_baseline"]
            .loc[variant_eval_dfs["v0_baseline"]["container_id"] == cid, "residual_pearson_r"]
            .iloc[0]
        )
        best_r = float(
            variant_eval_dfs[best_vid]
            .loc[variant_eval_dfs[best_vid]["container_id"] == cid, "residual_pearson_r"]
            .iloc[0]
        )

        explanation = _explain_case(category, delta, v0_r, best_r, v0_mae, best_mae)
        narratives["cases"][category] = {
            "container_id": cid,
            "v0_mae": v0_mae,
            "best_mae": best_mae,
            "delta_mae": delta,
            "v0_pearson_r": v0_r,
            "best_pearson_r": best_r,
            "explanation": explanation,
        }

        actual = np.ravel(
            inference_by_variant["v0_baseline"][cid].actual_day1_real,
        )
        predictions = {
            vid: np.ravel(inference_by_variant[vid][cid].day1_final_real)
            for vid in variant_order
            if cid in inference_by_variant.get(vid, {})
        }
        narratives["trajectories"][category] = {
            "container_id": cid,
            "actual": actual,
            "predictions": predictions,
        }

    return narratives


def case_studies_for_json(narratives: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe subset (exclude ndarray trajectories)."""
    return {"cases": narratives.get("cases", {})}


def _explain_case(
    category: str,
    delta_mae: float,
    v0_r: float,
    best_r: float,
    v0_mae: float,
    best_mae: float,
) -> str:
    if category == "largest_improvement":
        return (
            f"Largest MAE gain (Δ={delta_mae:.3f} pp): context features likely helped "
            f"this container's regime-dependent corrections. Pearson r moved "
            f"{v0_r:.3f}→{best_r:.3f}."
        )
    if category == "median_improvement":
        return (
            f"Typical cohort response (Δ MAE={delta_mae:.3f}): moderate context benefit "
            f"without outlier behaviour."
        )
    if category == "no_improvement":
        return (
            f"Near-zero MAE change (Δ={delta_mae:.3f}): context channels did not alter "
            f"forecasts materially; residual signal may remain weak (V0 MAE={v0_mae:.3f})."
        )
    return (
        f"Performance degraded (Δ MAE={delta_mae:.3f}): added complexity may overfit or "
        f"conflict with Prophet level for this workload (V4 MAE={best_mae:.3f})."
    )
