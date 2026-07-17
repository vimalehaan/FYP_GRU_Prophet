#!/usr/bin/env python3
"""Run Peak-Aware Hybrid Day 1 primary evaluation (treatment model)."""

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
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    INPUT_WINDOW,
    PEAK_AWARE_MODEL_FILENAME,
    RESIDUAL_STATS_FILENAME,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Peak-Aware Hybrid GRU on the frozen validation cohort.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=REPO_ROOT / "experiments/peak_aware_2026-07-14_164518",
        help="Peak-aware experiment directory containing trained model artifacts",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save_inference_cache(
    inference_results: dict,
    path: Path,
) -> None:
    """Save minimal Day-1 arrays for Task 3 peak-subset reuse."""
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

    model_path = experiment_dir / "models" / PEAK_AWARE_MODEL_FILENAME
    stats_path = experiment_dir / "models" / RESIDUAL_STATS_FILENAME
    eval_df_path = evaluation_dir / "evaluation_df.csv"
    eval_summary_path = evaluation_dir / "evaluation_summary.csv"
    eval_metadata_path = evaluation_dir / "evaluation_metadata.json"
    inference_cache_path = evaluation_dir / "treatment_inference_cache.pkl"

    print("=" * 72)
    print("Peak-Aware Hybrid — Treatment Primary Evaluation (Day 1)")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print(f"Model                : {model_path}")
    print()

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

    with open(stats_path, "rb") as handle:
        residual_stats = pickle.load(handle)

    res_mean = float(residual_stats["res_mean"])
    res_std = float(residual_stats["res_std"])
    input_window = int(residual_stats.get("input_window", INPUT_WINDOW))

    import tensorflow as tf

    hybrid_gru_model = tf.keras.models.load_model(str(model_path))

    started_at = _iso_timestamp()
    t0 = time.perf_counter()

    evaluation_df, inference_results, failures, skipped = (
        evaluate_selected_containers(
            selected_containers=selected_containers,
            global_train=global_train,
            global_val=global_val,
            hybrid_gru_model=hybrid_gru_model,
            scalers=scalers,
            res_mean=res_mean,
            res_std=res_std,
            input_window=input_window,
        )
    )

    runtime_seconds = time.perf_counter() - t0
    summary = summarize_evaluation_metrics(evaluation_df)

    evaluation_df.to_csv(eval_df_path, index=False)
    summary.round(6).to_csv(eval_summary_path)
    _save_inference_cache(inference_results, inference_cache_path)

    metadata: dict[str, object] = {
        "evaluation_type": "peak_aware_hybrid_treatment_primary",
        "phase": 5,
        "task": 2,
        "model_variant": residual_stats.get(
            "model_variant", "hybrid_prophet_gru_peak_aware_v1"
        ),
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "residual_stats_path": str(stats_path.relative_to(REPO_ROOT)),
        "res_mean": res_mean,
        "res_std": res_std,
        "input_window": input_window,
        "selected_containers": int(len(selected_containers)),
        "evaluated_containers": int(len(evaluation_df)),
        "skipped_containers": [
            {"container_id": cid, "reason": reason}
            for cid, reason in skipped
        ],
        "failures": [
            {"container_id": cid, "error": error}
            for cid, error in failures
        ],
        "evaluation_module": "utils.hybrid_evaluation.evaluate_selected_containers",
        "inference_module": "utils.hybrid_inference.run_hybrid_inference",
        "started_at": started_at,
        "completed_at": _iso_timestamp(),
        "runtime_seconds": round(runtime_seconds, 2),
        "outputs": {
            "evaluation_df": str(eval_df_path.relative_to(experiment_dir)),
            "evaluation_summary": str(
                eval_summary_path.relative_to(experiment_dir)
            ),
            "treatment_inference_cache": str(
                inference_cache_path.relative_to(experiment_dir)
            ),
        },
        "aggregate_summary": summary.round(6).to_dict(),
    }
    eval_metadata_path.write_text(json.dumps(metadata, indent=2))

    workspace_path = evaluation_dir / "evaluation_workspace.json"
    if workspace_path.exists():
        workspace = json.loads(workspace_path.read_text())
        workspace.setdefault("tasks", {})["task_2"] = "complete"
        workspace["treatment_evaluation"] = str(
            eval_metadata_path.relative_to(REPO_ROOT)
        )
        workspace_path.write_text(json.dumps(workspace, indent=2))

    print("-" * 72)
    print(f"Evaluated containers : {len(evaluation_df)}")
    print(f"Skipped containers   : {len(skipped)}")
    print(f"Failures             : {len(failures)}")
    print(f"Runtime              : {runtime_seconds / 60:.1f} min")
    print()
    print(summary.round(4))
    print()
    print("Saved outputs:")
    print(f"  evaluation_df.csv       : {eval_df_path}")
    print(f"  evaluation_summary.csv  : {eval_summary_path}")
    print(f"  evaluation_metadata.json: {eval_metadata_path}")
    print(f"  inference cache         : {inference_cache_path}")

    if failures:
        raise RuntimeError(f"Evaluation failures: {failures}")


if __name__ == "__main__":
    main()
