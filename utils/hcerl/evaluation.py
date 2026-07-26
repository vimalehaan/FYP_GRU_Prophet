"""HCERL evaluation — extended Hybrid metrics with energy recovery."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.csrle.control_reproduction import compute_container_extended_metrics
from utils.csrle.stage2_metrics import actual_predicted_residual_arrays
from utils.csrle.stage2_stats import cohort_summary, residual_cohort_summary
from utils.hcerl.inference import run_hcerl_inference
from utils.hcerl.features import FeatureStats
from utils.hybrid_evaluation import (
    _evaluable_container_ids,
    compute_day1_mape,
)
from utils.hybrid_inference import HybridInferenceResult


SIGMA_FLOOR = 1e-8


def _energy_recovery_ratio(actual: np.ndarray, pred: np.ndarray) -> float:
    e_a = float(np.sum(np.square(actual)))
    e_p = float(np.sum(np.square(pred)))
    return e_p / e_a if e_a > SIGMA_FLOOR else float("nan")


def compute_hcerl_container_metrics(
    result: HybridInferenceResult,
    peak_threshold: float,
) -> dict[str, float]:
    """Extended container metrics including MAPE and energy recovery."""
    base = compute_container_extended_metrics(result, peak_threshold)
    base["day1_mape"] = compute_day1_mape(
        result.actual_day1_real,
        result.day1_final_real,
    )
    actual_res, pred_res = actual_predicted_residual_arrays(result)
    base["energy_recovery_ratio"] = _energy_recovery_ratio(actual_res, pred_res)
    base["energy_actual"] = float(np.sum(np.square(actual_res)))
    base["energy_predicted"] = float(np.sum(np.square(pred_res)))
    return base


def evaluate_hcerl_variant(
    selected_containers: np.ndarray | list[str],
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    hybrid_gru_model: Any,
    scalers: dict[str, Any],
    stats: FeatureStats,
    features: list[str],
    peak_thresholds: dict[str, float],
    input_window: int,
    variant_id: str,
) -> tuple[
    pd.DataFrame,
    dict[str, HybridInferenceResult],
    dict[str, tuple[np.ndarray, np.ndarray]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """Evaluate one HCERL variant across the selected cohort."""
    rows: list[dict[str, Any]] = []
    inference_results: dict[str, HybridInferenceResult] = {}
    residual_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    failures: list[tuple[str, str]] = []

    evaluable_ids, skipped = _evaluable_container_ids(
        selected_containers=selected_containers,
        global_train=global_train,
        global_val=global_val,
        scalers=scalers,
    )

    for container_id in evaluable_ids:
        try:
            result = run_hcerl_inference(
                container_id=container_id,
                global_train=global_train,
                global_val=global_val,
                hybrid_gru_model=hybrid_gru_model,
                scalers=scalers,
                stats=stats,
                features=features,
                input_window=input_window,
            )
            metrics = compute_hcerl_container_metrics(
                result,
                peak_thresholds[container_id],
            )
            metrics["variant_id"] = variant_id
            rows.append(metrics)
            inference_results[container_id] = result
            residual_arrays[container_id] = actual_predicted_residual_arrays(result)
        except Exception as exc:
            failures.append((container_id, str(exc)))

    evaluation_df = pd.DataFrame(rows)
    if not evaluation_df.empty:
        evaluation_df = evaluation_df.sort_values("container_id").reset_index(drop=True)

    return evaluation_df, inference_results, residual_arrays, failures, skipped


def summarize_variant_cohort(evaluation_df: pd.DataFrame) -> dict[str, Any]:
    """Cohort summaries for primary and residual metrics."""
    summary: dict[str, Any] = {"n_containers": int(len(evaluation_df))}
    for col in evaluation_df.columns:
        if col in {"container_id", "variant_id"}:
            continue
        if pd.api.types.is_numeric_dtype(evaluation_df[col]):
            summary[col] = cohort_summary(evaluation_df[col])
    summary["residual_block"] = residual_cohort_summary(evaluation_df)
    return summary


def save_inference_cache(
    inference_results: dict[str, HybridInferenceResult],
    path: Path,
) -> None:
    """Save Day-1 arrays for downstream diagnostics."""
    cache = {
        cid: {
            "actual_day1_real": np.ravel(res.actual_day1_real),
            "day1_final_real": np.ravel(res.day1_final_real),
        }
        for cid, res in inference_results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(cache, handle)
