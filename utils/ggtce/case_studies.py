"""GGTCE case study selection and narratives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.global_inference import GlobalInferenceResult


def select_case_study_containers(
    g96_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    left_id: str = "g96",
    right_id: str = "g288",
) -> dict[str, str]:
    """Select largest/median improvement and largest degradation vs G96."""
    merged = g96_df.merge(
        compare_df, on="container_id", suffixes=("_left", "_right"),
    )
    merged["delta_mae"] = merged["day1_mae_right"] - merged["day1_mae_left"]
    valid = merged.dropna(subset=["delta_mae"])
    if valid.empty:
        return {}

    return {
        "largest_improvement": str(valid.loc[valid["delta_mae"].idxmin(), "container_id"]),
        "median_improvement": str(
            valid.iloc[(valid["delta_mae"] - valid["delta_mae"].median()).abs().idxmin()][
                "container_id"
            ],
        ),
        "largest_degradation": str(valid.loc[valid["delta_mae"].idxmax(), "container_id"]),
    }


def build_case_study_narratives(
    case_ids: dict[str, str],
    variant_eval_dfs: dict[str, pd.DataFrame],
    inference_by_variant: dict[str, dict[str, GlobalInferenceResult]],
    memory_merged: pd.DataFrame | None,
    variant_order: tuple[str, ...],
    baseline_id: str = "g96",
    compare_id: str = "g288",
) -> dict[str, Any]:
    """Build case narratives with memory context."""
    narratives: dict[str, Any] = {"cases": {}}

    mem_lookup = {}
    if memory_merged is not None and not memory_merged.empty:
        mem_lookup = memory_merged.set_index("container_id").to_dict("index")

    for category, cid in case_ids.items():
        base_mae = float(
            variant_eval_dfs[baseline_id]
            .loc[variant_eval_dfs[baseline_id]["container_id"] == cid, "day1_mae"]
            .iloc[0]
        )
        cmp_mae = float(
            variant_eval_dfs[compare_id]
            .loc[variant_eval_dfs[compare_id]["container_id"] == cid, "day1_mae"]
            .iloc[0]
        )
        delta = cmp_mae - base_mae

        mem = mem_lookup.get(cid, {})
        explanation = _explain_window_case(
            category, delta, base_mae, cmp_mae, mem, baseline_id, compare_id,
        )

        actual = np.ravel(
            inference_by_variant[baseline_id][cid].actual_day1_real,
        )
        predictions = {
            vid: np.ravel(inference_by_variant[vid][cid].day1_pred_real)
            for vid in variant_order
            if cid in inference_by_variant.get(vid, {})
        }

        narratives["cases"][category] = {
            "container_id": cid,
            "baseline_mae": base_mae,
            "compare_mae": cmp_mae,
            "delta_mae": delta,
            "memory_score": mem.get("memory_score"),
            "memory_class": mem.get("memory_class"),
            "acf_lag_96": mem.get("acf_lag_96"),
            "acf_lag_192": mem.get("acf_lag_192"),
            "acf_lag_288": mem.get("acf_lag_288"),
            "explanation": explanation,
            "actual_day1_real": actual.tolist(),
            "predictions_by_variant": {
                k: v.tolist() for k, v in predictions.items()
            },
        }

    return narratives


def case_studies_for_json(narratives: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe case studies (trajectories as lists, not ndarrays)."""
    safe: dict[str, Any] = {"cases": {}}
    for cat, case in narratives.get("cases", {}).items():
        safe["cases"][cat] = {
            k: v for k, v in case.items()
            if k not in {"actual_day1_real", "predictions_by_variant"}
        }
    return safe


def _explain_window_case(
    category: str,
    delta_mae: float,
    base_mae: float,
    cmp_mae: float,
    mem: dict[str, Any],
    baseline_id: str,
    compare_id: str,
) -> str:
    mclass = mem.get("memory_class", "unknown")
    mscore = mem.get("memory_score")
    acf96 = mem.get("acf_lag_96")
    acf288 = mem.get("acf_lag_288")

    mem_note = (
        f" Memory class={mclass}, score={mscore:.3f}, ACF@96={acf96:.3f}, ACF@288={acf288:.3f}."
        if mscore is not None and acf96 is not None
        else ""
    )

    if category == "largest_improvement":
        return (
            f"Longer window ({compare_id}) reduced MAE by {abs(delta_mae):.3f} pp vs {baseline_id} "
            f"({base_mae:.3f}→{cmp_mae:.3f}).{mem_note} "
            "Extended temporal context likely captured slow cycles missed by G96."
        )
    if category == "median_improvement":
        return (
            f"Representative cohort shift (Δ MAE={delta_mae:.3f}).{mem_note} "
            "Typical trade-off between extra history and fewer training sequences."
        )
    return (
        f"Longer window hurt MAE by {delta_mae:.3f} pp ({base_mae:.3f}→{cmp_mae:.3f}).{mem_note} "
        "Additional context may add noise or amplify overfitting with reduced sequence count."
    )
