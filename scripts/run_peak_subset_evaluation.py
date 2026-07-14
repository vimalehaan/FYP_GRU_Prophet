#!/usr/bin/env python3
"""Compute per-container peak-subset metrics for treatment and baseline."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.hybrid_evaluation import (  # noqa: E402
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)
from utils.peak_config import (  # noqa: E402
    BASELINE_REFERENCE_DIR,
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    INPUT_WINDOW,
    PEAK_AWARE_MODEL_FILENAME,
    RESIDUAL_STATS_FILENAME,
)
from utils.peak_detection import load_peak_thresholds  # noqa: E402
from utils.peak_evaluation import (  # noqa: E402
    build_peak_subset_comparison_per_container,
    compute_peak_subset_metrics_batch,
    compute_peak_subset_metrics_from_cache,
    summarize_peak_subset_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-container peak-subset Day-1 metrics.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=REPO_ROOT / "experiments/peak_aware_2026-07-14_164518",
    )
    parser.add_argument(
        "--baseline-model-dir",
        type=Path,
        default=BASELINE_REFERENCE_DIR / "models",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save_inference_cache(inference_results: dict, path: Path) -> None:
    cache: dict[str, dict[str, np.ndarray]] = {}
    for container_id, result in inference_results.items():
        cache[container_id] = {
            "actual_day1_real": np.ravel(result.actual_day1_real),
            "day1_final_real": np.ravel(result.day1_final_real),
        }
    with open(path, "wb") as handle:
        pickle.dump(cache, handle)


def main() -> None:
    args = _parse_args()
    experiment_dir = args.experiment_dir.resolve()
    evaluation_dir = experiment_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    treatment_cache_path = evaluation_dir / "treatment_inference_cache.pkl"
    baseline_cache_path = evaluation_dir / "baseline_inference_cache.pkl"
    treatment_peak_path = evaluation_dir / "treatment_peak_subset_per_container.csv"
    baseline_peak_path = evaluation_dir / "baseline_peak_subset_per_container.csv"
    peak_subset_metrics_path = evaluation_dir / "peak_subset_metrics.csv"
    comparison_path = evaluation_dir / "peak_subset_per_container_comparison.csv"
    baseline_eval_df_path = evaluation_dir / "baseline_evaluation_df.csv"
    baseline_eval_summary_path = evaluation_dir / "baseline_evaluation_summary.csv"
    pooled_summary_path = evaluation_dir / "peak_subset_summary.csv"
    metadata_path = evaluation_dir / "peak_subset_metadata.json"

    thresholds = load_peak_thresholds(
        experiment_dir / "peak" / "peak_thresholds.pkl"
    )

    print("=" * 72)
    print("Peak-Subset Evaluation — Per-Container Metrics")
    print("=" * 72)

    # --- Treatment (reuse cache from Task 2 when available) ---
    if treatment_cache_path.exists():
        print("Loading treatment inference cache...")
        with open(treatment_cache_path, "rb") as handle:
            treatment_cache = pickle.load(handle)
        treatment_peak_df = compute_peak_subset_metrics_from_cache(
            treatment_cache, thresholds
        )
    else:
        raise FileNotFoundError(
            f"Treatment cache not found: {treatment_cache_path}. "
            "Run scripts/run_peak_aware_evaluation.py first."
        )

    treatment_peak_df.to_csv(treatment_peak_path, index=False)
    print(f"Treatment per-container peak metrics: {len(treatment_peak_df)} rows")

    # --- Baseline (run inference for per-timestep arrays) ---
    train_df = pd.read_parquet(DATA_TRAIN)
    val_df = pd.read_parquet(DATA_VAL)
    selected_containers = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[
        train_df["container_id"].isin(selected_containers)
    ].copy()
    global_val = val_df[
        val_df["container_id"].isin(selected_containers)
    ].copy()

    baseline_stats_path = args.baseline_model_dir / RESIDUAL_STATS_FILENAME
    baseline_model_path = args.baseline_model_dir / "hybrid_gru.keras"

    with open(baseline_stats_path, "rb") as handle:
        baseline_stats = pickle.load(handle)

    import tensorflow as tf

    baseline_model = tf.keras.models.load_model(str(baseline_model_path))

    print("Running baseline inference for peak-subset metrics...")
    t0 = time.perf_counter()
    baseline_eval_df, baseline_inference, failures, skipped = (
        evaluate_selected_containers(
            selected_containers=selected_containers,
            global_train=global_train,
            global_val=global_val,
            hybrid_gru_model=baseline_model,
            scalers=scalers,
            res_mean=float(baseline_stats["res_mean"]),
            res_std=float(baseline_stats["res_std"]),
            input_window=int(baseline_stats.get("input_window", INPUT_WINDOW)),
        )
    )
    baseline_runtime = time.perf_counter() - t0

    if failures:
        raise RuntimeError(f"Baseline inference failures: {failures}")

    baseline_eval_df.to_csv(baseline_eval_df_path, index=False)
    summarize_evaluation_metrics(baseline_eval_df).round(6).to_csv(
        baseline_eval_summary_path
    )

    _save_inference_cache(baseline_inference, baseline_cache_path)
    baseline_peak_df = compute_peak_subset_metrics_batch(
        baseline_inference, thresholds
    )
    baseline_peak_df.to_csv(baseline_peak_path, index=False)
    print(f"Baseline per-container peak metrics : {len(baseline_peak_df)} rows")

    comparison_df = build_peak_subset_comparison_per_container(
        treatment_peak_df, baseline_peak_df
    )
    comparison_df.to_csv(comparison_path, index=False)

    treatment_long = treatment_peak_df.copy()
    treatment_long.insert(0, "model", "peak_aware")
    baseline_long = baseline_peak_df.copy()
    baseline_long.insert(0, "model", "baseline")
    peak_subset_metrics_df = pd.concat(
        [baseline_long, treatment_long],
        ignore_index=True,
    ).sort_values(["container_id", "model"]).reset_index(drop=True)
    peak_subset_metrics_df.to_csv(peak_subset_metrics_path, index=False)

    treatment_pooled = _pooled_from_cache(treatment_cache, thresholds)
    baseline_pooled = summarize_peak_subset_metrics(
        baseline_inference, thresholds
    )

    pooled_rows = [
        {"model": "baseline", **baseline_pooled},
        {"model": "peak_aware", **treatment_pooled},
    ]
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(pooled_summary_path, index=False)

    metadata = {
        "computed_at": _iso_timestamp(),
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "containers_treatment": int(len(treatment_peak_df)),
        "containers_baseline": int(len(baseline_peak_df)),
        "skipped_containers": [
            {"container_id": cid, "reason": reason}
            for cid, reason in skipped
        ],
        "baseline_runtime_seconds": round(baseline_runtime, 2),
        "outputs": {
            "baseline_evaluation_df": str(
                baseline_eval_df_path.relative_to(experiment_dir)
            ),
            "baseline_evaluation_summary": str(
                baseline_eval_summary_path.relative_to(experiment_dir)
            ),
            "peak_subset_metrics": str(
                peak_subset_metrics_path.relative_to(experiment_dir)
            ),
            "treatment_peak_subset_per_container": str(
                treatment_peak_path.relative_to(experiment_dir)
            ),
            "baseline_peak_subset_per_container": str(
                baseline_peak_path.relative_to(experiment_dir)
            ),
            "peak_subset_per_container_comparison": str(
                comparison_path.relative_to(experiment_dir)
            ),
            "peak_subset_summary": str(
                pooled_summary_path.relative_to(experiment_dir)
            ),
        },
        "pooled_peak_step_fraction": float(
            baseline_pooled["peak_steps"]
            / baseline_pooled["total_day1_steps"]
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("-" * 72)
    print("Pooled peak-subset summary:")
    print(pooled_df.round(4).to_string(index=False))
    print()
    print("Saved:")
    print(f"  {baseline_eval_df_path}")
    print(f"  {peak_subset_metrics_path}")
    print(f"  {treatment_peak_path}")
    print(f"  {baseline_peak_path}")
    print(f"  {comparison_path}")
    print(f"  {pooled_summary_path}")

    workspace_path = evaluation_dir / "evaluation_workspace.json"
    if workspace_path.exists():
        workspace = json.loads(workspace_path.read_text())
        workspace.setdefault("tasks", {})["task_3"] = "complete"
        workspace["peak_subset_metadata"] = str(
            metadata_path.relative_to(REPO_ROOT)
        )
        workspace_path.write_text(json.dumps(workspace, indent=2))


def _pooled_from_cache(
    inference_cache: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
) -> dict[str, float | int]:
    """Pooled peak-subset summary from array cache."""
    from utils.peak_evaluation import label_day1_peak_timesteps

    peak_actual_chunks: list[np.ndarray] = []
    peak_predicted_chunks: list[np.ndarray] = []
    non_peak_actual_chunks: list[np.ndarray] = []
    non_peak_predicted_chunks: list[np.ndarray] = []
    peak_steps = 0
    non_peak_steps = 0

    for container_id, arrays in inference_cache.items():
        actual = np.ravel(arrays["actual_day1_real"])
        predicted = np.ravel(arrays["day1_final_real"])
        peak_mask = label_day1_peak_timesteps(actual, thresholds[container_id])
        peak_actual_chunks.append(actual[peak_mask])
        peak_predicted_chunks.append(predicted[peak_mask])
        non_peak_actual_chunks.append(actual[~peak_mask])
        non_peak_predicted_chunks.append(predicted[~peak_mask])
        peak_steps += int(peak_mask.sum())
        non_peak_steps += int((~peak_mask).sum())

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    peak_actual = np.concatenate(peak_actual_chunks)
    peak_predicted = np.concatenate(peak_predicted_chunks)
    non_peak_actual = np.concatenate(non_peak_actual_chunks)
    non_peak_predicted = np.concatenate(non_peak_predicted_chunks)

    return {
        "containers_evaluated": len(inference_cache),
        "total_day1_steps": peak_steps + non_peak_steps,
        "peak_steps": peak_steps,
        "non_peak_steps": non_peak_steps,
        "peak_step_fraction": peak_steps / (peak_steps + non_peak_steps),
        "peak_mae": float(mean_absolute_error(peak_actual, peak_predicted)),
        "peak_rmse": float(
            np.sqrt(mean_squared_error(peak_actual, peak_predicted))
        ),
        "non_peak_mae": float(
            mean_absolute_error(non_peak_actual, non_peak_predicted)
        ),
        "non_peak_rmse": float(
            np.sqrt(mean_squared_error(non_peak_actual, non_peak_predicted))
        ),
    }


if __name__ == "__main__":
    main()
