#!/usr/bin/env python3
"""Run Peak-Aware Global peak-evaluation sanity checks (Stage 3 Task 1)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_artifacts import load_global_artifacts  # noqa: E402
from utils.global_config import (  # noqa: E402
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    DEMO_CONTAINER_ID,
)
from utils.global_inference import run_global_inference  # noqa: E402
from utils.global_peak_aware_config import (  # noqa: E402
    CONTROL_MODEL_PATH,
    CONTROL_REFERENCE_DIR,
    DESIGN_LOCK_REFERENCE,
    EVALUATION_SUBDIR,
    PEAK_AWARE_METADATA_FILENAME,
    PEAK_AWARE_MODEL_FILENAME,
    PEAK_THRESHOLDS_FILENAME,
    PRIMARY_EXPERIMENT_DIR,
)
from utils.hybrid_evaluation import compute_day1_metrics  # noqa: E402
from utils.peak_detection import label_peak_timesteps, load_peak_thresholds  # noqa: E402
from utils.peak_evaluation import (  # noqa: E402
    compute_peak_subset_metrics_from_global_result,
    label_day1_peak_timesteps,
    summarize_peak_subset_metrics_from_global_cache,
)

METRIC_TOLERANCE = 1e-9


def _metadata_path_for_model(model_path: Path) -> Path:
    """Resolve the metadata JSON adjacent to a Global GRU model file."""
    if model_path.name == PEAK_AWARE_MODEL_FILENAME:
        return model_path.parent / PEAK_AWARE_METADATA_FILENAME
    return model_path.parent / "global_gru_metadata.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peak-Aware Global peak evaluation sanity checks.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PRIMARY_EXPERIMENT_DIR,
        help="Treatment experiment directory",
    )
    parser.add_argument(
        "--container-id",
        type=str,
        default=DEMO_CONTAINER_ID,
        help="Demo container for manual vs module metric check",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=CONTROL_MODEL_PATH,
        help="Model used for sanity inference (control or treatment weights)",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _metrics_close(a: float, b: float, tol: float = METRIC_TOLERANCE) -> bool:
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(float(a) - float(b)) <= tol


def _manual_peak_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Reference implementation for sanity comparison."""
    actual = np.ravel(actual)
    predicted = np.ravel(predicted)
    peak_mask = label_peak_timesteps(actual, threshold)
    non_peak_mask = ~peak_mask

    def _subset(mask: np.ndarray) -> tuple[float, float]:
        if not mask.any():
            return float("nan"), float("nan")
        actual_sub = actual[mask]
        predicted_sub = predicted[mask]
        mae = float(mean_absolute_error(actual_sub, predicted_sub))
        rmse = float(np.sqrt(mean_squared_error(actual_sub, predicted_sub)))
        return mae, rmse

    peak_mae, peak_rmse = _subset(peak_mask)
    non_peak_mae, non_peak_rmse = _subset(non_peak_mask)
    overall_mae, overall_rmse = compute_day1_metrics(actual, predicted)

    return {
        "peak_mae": peak_mae,
        "peak_rmse": peak_rmse,
        "non_peak_mae": non_peak_mae,
        "non_peak_rmse": non_peak_rmse,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "peak_steps": int(peak_mask.sum()),
        "non_peak_steps": int(non_peak_mask.sum()),
    }


def run_sanity_checks(
    experiment_dir: Path,
    container_id: str,
    model_path: Path,
) -> dict[str, Any]:
    """Run single-container Global peak-evaluation sanity checks."""
    peak_thresholds_path = experiment_dir / "peak" / PEAK_THRESHOLDS_FILENAME
    thresholds = load_peak_thresholds(peak_thresholds_path)
    if container_id not in thresholds:
        raise KeyError(f"No peak threshold for demo container {container_id!r}")

    train_df = pd.read_parquet(DATA_TRAIN)
    val_df = pd.read_parquet(DATA_VAL)
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    global_train = train_df[train_df["container_id"].isin(selected)].copy()
    global_val = val_df[val_df["container_id"].isin(selected)].copy()

    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    model, _metadata = load_global_artifacts(
        model_path=model_path,
        metadata_path=_metadata_path_for_model(model_path),
    )
    inference_result = run_global_inference(
        container_id=container_id,
        global_train=global_train,
        global_val=global_val,
        model=model,
        scalers=scalers,
    )

    threshold = float(thresholds[container_id])
    module_metrics = compute_peak_subset_metrics_from_global_result(
        container_id=container_id,
        inference_result=inference_result,
        threshold=threshold,
    )
    manual_metrics = _manual_peak_metrics(
        actual=inference_result.actual_day1_real,
        predicted=inference_result.day1_pred_real,
        threshold=threshold,
    )

    peak_mask = label_day1_peak_timesteps(
        inference_result.actual_day1_real,
        threshold,
    )
    label_rule_match = bool(
        np.array_equal(
            peak_mask,
            label_peak_timesteps(
                np.ravel(inference_result.actual_day1_real),
                threshold,
            ),
        )
    )

    inference_cache = {
        container_id: {
            "actual_day1_real": np.ravel(inference_result.actual_day1_real),
            "day1_pred_real": np.ravel(inference_result.day1_pred_real),
        }
    }
    pooled_summary = summarize_peak_subset_metrics_from_global_cache(
        inference_cache,
        {container_id: threshold},
    )

    checks = {
        "peak_mae_match": _metrics_close(
            manual_metrics["peak_mae"],
            module_metrics["peak_mae"],
        ),
        "peak_rmse_match": _metrics_close(
            manual_metrics["peak_rmse"],
            module_metrics["peak_rmse"],
        ),
        "non_peak_mae_match": _metrics_close(
            manual_metrics["non_peak_mae"],
            module_metrics["non_peak_mae"],
        ),
        "non_peak_rmse_match": _metrics_close(
            manual_metrics["non_peak_rmse"],
            module_metrics["non_peak_rmse"],
        ),
        "overall_mae_match": _metrics_close(
            manual_metrics["overall_mae"],
            module_metrics["overall_mae"],
        ),
        "overall_rmse_match": _metrics_close(
            manual_metrics["overall_rmse"],
            module_metrics["overall_rmse"],
        ),
        "peak_steps_match": (
            manual_metrics["peak_steps"] == module_metrics["peak_steps"]
        ),
        "label_rule_match": label_rule_match,
        "pooled_peak_mae_match": _metrics_close(
            manual_metrics["peak_mae"],
            pooled_summary["peak_mae"],
        ),
        "pooled_non_peak_mae_match": _metrics_close(
            manual_metrics["non_peak_mae"],
            pooled_summary["non_peak_mae"],
        ),
    }
    pipeline_status = "PASS" if all(checks.values()) else "FAIL"

    return {
        "sanity_test": True,
        "stage": 3,
        "task": "task_1_peak_evaluation_sanity",
        "generated_at": _iso_timestamp(),
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "demo_container": container_id,
        "threshold": threshold,
        "day1_steps": module_metrics["day1_steps"],
        "peak_steps": module_metrics["peak_steps"],
        "non_peak_steps": module_metrics["non_peak_steps"],
        "peak_fraction": module_metrics["peak_fraction"],
        "manual_metrics": manual_metrics,
        "module_metrics": module_metrics,
        "pooled_summary": pooled_summary,
        "checks": checks,
        "pipeline_status": pipeline_status,
    }


def _write_evaluation_workspace(
    experiment_dir: Path,
    sanity_path: Path,
) -> Path:
    """Update Stage 3 evaluation workspace metadata without clobbering a locked record."""
    evaluation_dir = experiment_dir / EVALUATION_SUBDIR
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    workspace_path = evaluation_dir / "evaluation_workspace.json"
    if workspace_path.exists():
        workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
        if workspace.get("status") == "complete":
            workspace["sanity_check"] = str(sanity_path.relative_to(REPO_ROOT))
            workspace["sanity_check_regenerated_at"] = _iso_timestamp()
            workspace["control_reference"] = str(
                CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)
            )
            workspace_path.write_text(
                json.dumps(workspace, indent=2) + "\n",
                encoding="utf-8",
            )
            return workspace_path

    workspace = {
        "stage": 3,
        "stage_name": "Evaluation + Verification",
        "status": "in_progress",
        "initialized_at": _iso_timestamp(),
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "control_reference": str(CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "design_reference": str(DESIGN_LOCK_REFERENCE.relative_to(REPO_ROOT)),
        "documentation": "docs/peak-aware-global/stage-03-evaluation-plan.md",
        "new_modules": {
            "peak_evaluation_global_adapters": "utils/peak_evaluation.py",
        },
        "peak_thresholds": str(
            (experiment_dir / "peak" / PEAK_THRESHOLDS_FILENAME).relative_to(
                REPO_ROOT
            )
        ),
        "sanity_check": str(sanity_path.relative_to(REPO_ROOT)),
        "tasks": {
            "task_1": "complete",
            "task_2": "pending",
            "task_3": "pending",
            "task_4": "pending",
            "task_5": "pending",
            "task_6": "pending",
        },
        "primary_stage4_artifact": "evaluation/per_container_comparison.csv",
    }
    workspace_path.write_text(
        json.dumps(workspace, indent=2) + "\n",
        encoding="utf-8",
    )
    return workspace_path


def main() -> None:
    args = _parse_args()
    experiment_dir = args.experiment_dir.resolve()
    model_path = args.model_path.resolve()
    evaluation_dir = experiment_dir / EVALUATION_SUBDIR
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    result = run_sanity_checks(
        experiment_dir=experiment_dir,
        container_id=args.container_id,
        model_path=model_path,
    )

    sanity_path = evaluation_dir / "peak_evaluation_sanity.json"
    sanity_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    workspace_path = _write_evaluation_workspace(experiment_dir, sanity_path)

    print("=" * 72)
    print("Peak-Aware Global — Peak Evaluation Sanity (Stage 3 Task 1)")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print(f"Demo container       : {args.container_id}")
    print(f"Peak steps           : {result['peak_steps']} / {result['day1_steps']}")
    print(f"Pipeline status      : {result['pipeline_status']}")
    print(f"Saved sanity         : {sanity_path}")
    print(f"Saved workspace      : {workspace_path}")

    if result["pipeline_status"] != "PASS":
        failed = [name for name, ok in result["checks"].items() if not ok]
        raise SystemExit(f"Sanity checks failed: {failed}")


if __name__ == "__main__":
    main()
