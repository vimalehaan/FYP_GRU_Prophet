"""RRE evaluation and optimisation metrics."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.csrle.control_reproduction import compute_container_extended_metrics
from utils.csrle.stage2_stats import cohort_summary, residual_cohort_summary
from utils.hybrid_evaluation import _evaluable_container_ids
from utils.hybrid_inference import HybridInferenceResult
from utils.rre.config import TRAINING_CONFIG
from utils.rre.inference import run_rre_inference
from utils.rre.scaling import ScalingStats


def evaluate_rre_variant(
    selected_containers: np.ndarray | list[str],
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    hybrid_gru_model: Any,
    scalers: dict[str, Any],
    scaling_stats: ScalingStats,
    peak_thresholds: dict[str, float],
    input_window: int,
    variant_id: str,
) -> tuple[pd.DataFrame, dict[str, HybridInferenceResult], list[tuple[str, str]], list[tuple[str, str]]]:
    """Evaluate one RRE variant across the cohort."""
    rows: list[dict[str, Any]] = []
    inference_results: dict[str, HybridInferenceResult] = {}
    failures: list[tuple[str, str]] = []

    evaluable_ids, skipped = _evaluable_container_ids(
        selected_containers=selected_containers,
        global_train=global_train,
        global_val=global_val,
        scalers=scalers,
    )

    for container_id in evaluable_ids:
        try:
            result = run_rre_inference(
                container_id=container_id,
                global_train=global_train,
                global_val=global_val,
                hybrid_gru_model=hybrid_gru_model,
                scalers=scalers,
                scaling_stats=scaling_stats,
                input_window=input_window,
            )
            metrics = compute_container_extended_metrics(
                result,
                peak_thresholds[container_id],
            )
            metrics["variant_id"] = variant_id
            rows.append(metrics)
            inference_results[container_id] = result
        except Exception as exc:
            failures.append((container_id, str(exc)))

    evaluation_df = pd.DataFrame(rows)
    if not evaluation_df.empty:
        evaluation_df = evaluation_df.sort_values("container_id").reset_index(drop=True)
    return evaluation_df, inference_results, failures, skipped


def summarize_variant_cohort(evaluation_df: pd.DataFrame) -> dict[str, Any]:
    """Cohort summaries for forecast and residual metrics."""
    summary: dict[str, Any] = {"n_containers": int(len(evaluation_df))}
    for col in evaluation_df.columns:
        if col in {"container_id", "variant_id"}:
            continue
        if pd.api.types.is_numeric_dtype(evaluation_df[col]):
            summary[col] = cohort_summary(evaluation_df[col])
    summary["residual_block"] = residual_cohort_summary(evaluation_df)
    return summary


def optimization_summary(train_meta: dict[str, Any]) -> dict[str, Any]:
    """Extract optimisation-focused training metrics."""
    return {
        "variant_id": train_meta["variant_id"],
        "best_epoch": train_meta["best_epoch"],
        "epochs_run": train_meta["epochs_run"],
        "best_train_loss": train_meta["best_train_loss"],
        "best_val_loss": train_meta["best_val_loss"],
        "final_train_loss": train_meta["final_train_loss"],
        "final_val_loss": train_meta["final_val_loss"],
        "generalisation_gap": train_meta["generalisation_gap"],
        "convergence_speed": train_meta["convergence_speed"],
        "early_stopped": train_meta["epochs_run"] < int(TRAINING_CONFIG["epochs"]),
        "scaling_stats": train_meta["scaling_stats"],
    }


def save_inference_cache(
    inference_results: dict[str, HybridInferenceResult],
    path: Path,
) -> None:
    cache = {
        cid: {
            "actual_day1_real": np.ravel(res.actual_day1_real),
            "day1_final_real": np.ravel(res.day1_final_real),
            "day1_prophet_real": np.ravel(res.prophet_day1_real),
        }
        for cid, res in inference_results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(cache, handle)
