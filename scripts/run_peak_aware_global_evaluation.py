#!/usr/bin/env python3
"""Run Peak-Aware Global GRU Day 1 primary evaluation (treatment model)."""

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
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    FORECAST_HORIZON,
    INPUT_WINDOW,
    MAPE_EPSILON,
    STALE_SELECTED_CONTAINER_ID,
)
from utils.global_evaluation import (  # noqa: E402
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)
from utils.global_peak_aware_config import (  # noqa: E402
    CONTROL_REFERENCE_DIR,
    DESIGN_LOCK_REFERENCE,
    EVALUATION_SUBDIR,
    MODEL_VARIANT,
    MODELS_SUBDIR,
    PEAK_AWARE_METADATA_FILENAME,
    PEAK_AWARE_MODEL_FILENAME,
    PRIMARY_EXPERIMENT_DIR,
)
from utils.peak_evaluation import GLOBAL_INFERENCE_CACHE_PRED_KEY  # noqa: E402

EXPECTED_EVALUATED = 99
EXPECTED_SKIPPED = 1
REQUIRED_COLUMNS = [
    "container_id",
    "day1_mae",
    "day1_rmse",
    "day1_mape",
    "train_steps",
    "validation_steps",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Peak-Aware Global GRU on the frozen validation cohort."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PRIMARY_EXPERIMENT_DIR,
        help="Peak-aware Global experiment directory",
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
            GLOBAL_INFERENCE_CACHE_PRED_KEY: np.ravel(result.day1_pred_real),
        }
    with open(path, "wb") as handle:
        pickle.dump(cache, handle)


def _assert_task2_checks(
    evaluation_df: pd.DataFrame,
    skipped: list[tuple[str, str]],
    failures: list[tuple[str, str]],
) -> None:
    """Raise if Task 2 cohort evaluation checks fail."""
    if len(evaluation_df) != EXPECTED_EVALUATED:
        raise RuntimeError(
            f"Expected {EXPECTED_EVALUATED} evaluated containers, "
            f"got {len(evaluation_df)}"
        )
    skipped_ids = {container_id for container_id, _ in skipped}
    if STALE_SELECTED_CONTAINER_ID not in skipped_ids:
        raise RuntimeError(
            f"Expected {STALE_SELECTED_CONTAINER_ID!r} in skipped list"
        )
    if len(skipped) != EXPECTED_SKIPPED:
        raise RuntimeError(
            f"Expected {EXPECTED_SKIPPED} skipped container, got {len(skipped)}"
        )
    if failures:
        raise RuntimeError(f"Evaluation failures: {failures}")

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in evaluation_df.columns
    ]
    if missing_columns:
        raise RuntimeError(f"Missing evaluation_df columns: {missing_columns}")


def main() -> None:
    args = _parse_args()
    experiment_dir = args.experiment_dir.resolve()
    evaluation_dir = experiment_dir / EVALUATION_SUBDIR
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    model_path = experiment_dir / MODELS_SUBDIR / PEAK_AWARE_MODEL_FILENAME
    metadata_path = experiment_dir / MODELS_SUBDIR / PEAK_AWARE_METADATA_FILENAME
    eval_df_path = evaluation_dir / "evaluation_df.csv"
    eval_summary_path = evaluation_dir / "evaluation_summary.csv"
    eval_metadata_path = evaluation_dir / "evaluation_metadata.json"
    inference_cache_path = evaluation_dir / "treatment_inference_cache.pkl"

    print("=" * 72)
    print("Peak-Aware Global — Treatment Primary Evaluation (Day 1)")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print(f"Model                : {model_path}")
    print(f"Control reference    : {CONTROL_REFERENCE_DIR}")
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
        "evaluation_type": "peak_aware_global_treatment_primary",
        "stage": 3,
        "task": 2,
        "model_variant": model_metadata.get("model_variant", MODEL_VARIANT),
        "design_lock_reference": str(DESIGN_LOCK_REFERENCE.relative_to(REPO_ROOT)),
        "control_reference": str(CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "metadata_path": str(metadata_path.relative_to(REPO_ROOT)),
        "input_window": INPUT_WINDOW,
        "forecast_horizon": FORECAST_HORIZON,
        "mape_epsilon": MAPE_EPSILON,
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
        "fairness_contract": "docs/peak-aware-global/stage-03-evaluation-plan.md#14-evaluation-fairness-contract",
        "training_metadata_snapshot": {
            "random_seed": model_metadata.get("random_seed"),
            "epochs_run": model_metadata.get("epochs_run"),
            "n_sequences_total": model_metadata.get("n_sequences_total"),
            "loss": model_metadata.get("loss"),
        },
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
    eval_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    workspace_path = evaluation_dir / "evaluation_workspace.json"
    if workspace_path.exists():
        workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
        workspace.setdefault("tasks", {})["task_2"] = "complete"
        workspace["treatment_evaluation"] = str(
            eval_metadata_path.relative_to(REPO_ROOT)
        )
        workspace_path.write_text(
            json.dumps(workspace, indent=2) + "\n",
            encoding="utf-8",
        )

    print("-" * 72)
    print(f"Evaluated containers : {len(evaluation_df)}")
    print(f"Skipped containers   : {len(skipped)}")
    for container_id, reason in skipped:
        print(f"  - {container_id}: {reason}")
    print(f"Failures             : {len(failures)}")
    print(f"Runtime              : {runtime_seconds:.1f} s")
    print()
    print(summary.round(4))
    print()
    print("Official Global control reference (Day 1 mean):")
    ctrl_summary = pd.read_csv(
        CONTROL_REFERENCE_DIR / "evaluation" / "evaluation_summary.csv",
        index_col=0,
    )
    print(
        f"  MAE  {ctrl_summary.loc['mean', 'day1_mae']:.3f} | "
        f"RMSE {ctrl_summary.loc['mean', 'day1_rmse']:.3f} | "
        f"MAPE {ctrl_summary.loc['mean', 'day1_mape']:.2f}"
    )
    print()
    print("Saved outputs:")
    print(f"  evaluation_df.csv            : {eval_df_path}")
    print(f"  evaluation_summary.csv       : {eval_summary_path}")
    print(f"  evaluation_metadata.json     : {eval_metadata_path}")
    print(f"  treatment_inference_cache.pkl: {inference_cache_path}")
    print()
    print("Task 2 cohort checks: PASS")


if __name__ == "__main__":
    main()
