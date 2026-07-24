#!/usr/bin/env python3
"""Run Global GRU v1 multi-container Day 1 evaluation."""

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

from utils.global_artifacts import load_global_artifacts  # noqa: E402
from utils.global_config import (  # noqa: E402
    BASELINE_SPEC_REFERENCE,
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEMO_CONTAINER_ID,
    FORECAST_HORIZON,
    GLOBAL_GRU_REFERENCE_DIR,
    INPUT_WINDOW,
    MAPE_EPSILON,
    MODEL_VARIANT,
    STALE_SELECTED_CONTAINER_ID,
)
from utils.global_evaluation import (  # noqa: E402
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Global GRU v1 on the frozen 99-container Day 1 cohort."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained Global GRU Keras model",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to Global GRU metadata JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override evaluation experiment directory "
            "(default: experiments/global_gru_evaluation_<timestamp>)"
        ),
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _evaluation_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    return REPO_ROOT / f"experiments/global_gru_evaluation_{stamp}"


def _save_inference_cache(
    inference_results: dict,
    path: Path,
) -> None:
    """Save minimal Day-1 arrays for plotting and downstream reuse."""
    cache: dict[str, dict[str, np.ndarray]] = {}
    for container_id, result in inference_results.items():
        cache[container_id] = {
            "actual_day1_real": np.ravel(result.actual_day1_real),
            "day1_pred_real": np.ravel(result.day1_pred_real),
        }
    with open(path, "wb") as handle:
        pickle.dump(cache, handle)


def _assert_task2_checks(
    evaluation_df: pd.DataFrame,
    skipped: list[tuple[str, str]],
    failures: list[tuple[str, str]],
) -> None:
    """Raise if Task 2 cohort evaluation checks fail."""
    if len(evaluation_df) != 99:
        raise RuntimeError(
            f"Expected 99 evaluated containers, got {len(evaluation_df)}"
        )
    skipped_ids = {container_id for container_id, _ in skipped}
    if STALE_SELECTED_CONTAINER_ID not in skipped_ids:
        raise RuntimeError(
            f"Expected {STALE_SELECTED_CONTAINER_ID!r} in skipped list"
        )
    if len(skipped) != 1:
        raise RuntimeError(f"Expected 1 skipped container, got {len(skipped)}")
    if failures:
        raise RuntimeError(f"Evaluation failures: {failures}")

    required_columns = [
        "container_id",
        "day1_mae",
        "day1_rmse",
        "day1_mape",
        "train_steps",
        "validation_steps",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in evaluation_df.columns
    ]
    if missing_columns:
        raise RuntimeError(f"Missing evaluation_df columns: {missing_columns}")


def main() -> None:
    args = _parse_args()
    model_path = args.model_path.resolve()
    metadata_path = args.metadata_path.resolve()
    experiment_dir = _evaluation_dir(
        args.output_dir.resolve() if args.output_dir else None,
    )
    evaluation_dir = experiment_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    eval_df_path = evaluation_dir / "evaluation_df.csv"
    eval_summary_path = evaluation_dir / "evaluation_summary.csv"
    eval_metadata_path = evaluation_dir / "evaluation_metadata.json"
    inference_cache_path = evaluation_dir / "inference_cache.pkl"

    print("=" * 72)
    print("Global GRU v1 — Multi-Container Day 1 Evaluation")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print(f"Model                : {model_path}")
    print(f"Metadata             : {metadata_path}")
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

    global_gru_model, model_metadata = load_global_artifacts(
        model_path=model_path,
        metadata_path=metadata_path,
    )

    started_at = _iso_timestamp()
    t0 = time.perf_counter()

    evaluation_df, inference_results, failures, skipped = (
        evaluate_selected_containers(
            selected_containers=selected_containers,
            global_train=global_train,
            global_val=global_val,
            global_gru_model=global_gru_model,
            scalers=scalers,
            input_window=INPUT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )
    )

    runtime_seconds = time.perf_counter() - t0
    _assert_task2_checks(evaluation_df, skipped, failures)
    summary = summarize_evaluation_metrics(evaluation_df)

    evaluation_df.to_csv(eval_df_path, index=False)
    summary.round(6).to_csv(eval_summary_path)
    _save_inference_cache(inference_results, inference_cache_path)

    metadata: dict[str, object] = {
        "evaluation_type": "global_gru_baseline_primary",
        "phase": 5,
        "task": 2,
        "model_variant": MODEL_VARIANT,
        "baseline_spec_reference": BASELINE_SPEC_REFERENCE,
        "frozen_reference_dir": str(
            GLOBAL_GRU_REFERENCE_DIR.relative_to(REPO_ROOT),
        ),
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "metadata_path": str(metadata_path.relative_to(REPO_ROOT)),
        "input_window": INPUT_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
        "mape_epsilon": MAPE_EPSILON,
        "demo_container_id": DEMO_CONTAINER_ID,
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
        "evaluation_module": (
            "utils.global_evaluation.evaluate_selected_containers"
        ),
        "inference_module": "utils.global_inference.run_global_inference",
        "metrics_module": "utils.hybrid_evaluation",
        "training_metadata_snapshot": {
            "random_seed": model_metadata.get("random_seed"),
            "epochs_trained": model_metadata.get("epochs_trained"),
            "n_sequences": model_metadata.get("n_sequences"),
        },
        "started_at": started_at,
        "completed_at": _iso_timestamp(),
        "runtime_seconds": round(runtime_seconds, 2),
        "outputs": {
            "evaluation_df": str(eval_df_path.relative_to(experiment_dir)),
            "evaluation_summary": str(
                eval_summary_path.relative_to(experiment_dir),
            ),
            "inference_cache": str(
                inference_cache_path.relative_to(experiment_dir),
            ),
        },
        "aggregate_summary": summary.round(6).to_dict(),
    }
    eval_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("-" * 72)
    print(f"Selected containers  : {len(selected_containers)}")
    print(f"Evaluated containers : {len(evaluation_df)}")
    print(f"Skipped containers   : {len(skipped)}")
    for container_id, reason in skipped:
        print(f"  - {container_id}: {reason}")
    print(f"Failures             : {len(failures)}")
    print(f"Runtime              : {runtime_seconds / 60:.1f} min")
    print()
    print(summary.round(4))
    print()
    print("Saved outputs:")
    print(f"  evaluation_df.csv       : {eval_df_path}")
    print(f"  evaluation_summary.csv  : {eval_summary_path}")
    print(f"  evaluation_metadata.json: {eval_metadata_path}")
    print(f"  inference_cache.pkl     : {inference_cache_path}")
    print()
    print("Task 2 cohort checks: PASS")


if __name__ == "__main__":
    main()
