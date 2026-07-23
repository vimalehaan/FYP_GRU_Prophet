#!/usr/bin/env python3
"""Compute peak-subset metrics for Peak-Aware Global vs frozen Global baseline."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    STALE_SELECTED_CONTAINER_ID,
)
from utils.global_evaluation import (  # noqa: E402
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)
from utils.global_peak_aware_config import (  # noqa: E402
    CONTROL_MODEL_PATH,
    CONTROL_REFERENCE_DIR,
    DESIGN_LOCK_REFERENCE,
    EVALUATION_SUBDIR,
    PEAK_THRESHOLDS_FILENAME,
    PRIMARY_EXPERIMENT_DIR,
)
from utils.peak_detection import load_peak_thresholds  # noqa: E402
from utils.peak_evaluation import (  # noqa: E402
    GLOBAL_INFERENCE_CACHE_PRED_KEY,
    build_peak_subset_comparison_per_container,
    compute_peak_subset_metrics_batch_global,
    compute_peak_subset_metrics_from_global_cache,
    summarize_peak_subset_metrics_from_global_cache,
)

METRIC_TOLERANCE = 1e-5
EXPECTED_EVALUATED = 99
EXPECTED_SKIPPED = 1
PEAK_FRACTION_MIN = 0.17
PEAK_FRACTION_MAX = 0.31
METRIC_COLUMNS = ("day1_mae", "day1_rmse", "day1_mape")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3 Task 3 — control re-run, reproducibility gate, "
            "peak-subset metrics."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PRIMARY_EXPERIMENT_DIR,
        help="Official Peak-Aware Global experiment directory",
    )
    parser.add_argument(
        "--control-reference-dir",
        type=Path,
        default=CONTROL_REFERENCE_DIR,
        help="Frozen Global GRU reference directory (read-only)",
    )
    return parser.parse_args()


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save_inference_cache(
    inference_results: dict,
    path: Path,
) -> None:
    cache: dict[str, dict[str, np.ndarray]] = {}
    for container_id, result in inference_results.items():
        cache[container_id] = {
            "actual_day1_real": np.ravel(result.actual_day1_real),
            GLOBAL_INFERENCE_CACHE_PRED_KEY: np.ravel(result.day1_pred_real),
        }
    with open(path, "wb") as handle:
        pickle.dump(cache, handle)


def _compare_series(
    rerun: pd.Series,
    frozen: pd.Series,
    metric: str,
) -> dict[str, Any]:
    matches = bool(
        np.allclose(
            rerun.values,
            frozen.values,
            rtol=0.0,
            atol=METRIC_TOLERANCE,
        )
    )
    max_abs_delta = float(np.max(np.abs(rerun.values - frozen.values)))
    return {
        "metric": metric,
        "matches_within_tolerance": matches,
        "max_abs_delta": max_abs_delta,
        "tolerance": METRIC_TOLERANCE,
    }


def _run_reproducibility_gate(
    rerun_df: pd.DataFrame,
    rerun_summary: pd.DataFrame,
    frozen_df: pd.DataFrame,
    frozen_summary: pd.DataFrame,
) -> dict[str, Any]:
    merged = rerun_df.merge(
        frozen_df,
        on="container_id",
        suffixes=("_rerun", "_frozen"),
    )
    if len(merged) != len(frozen_df):
        return {
            "gate_status": "FAIL",
            "reason": "container_count_mismatch",
            "rerun_containers": int(len(rerun_df)),
            "frozen_containers": int(len(frozen_df)),
            "paired_containers": int(len(merged)),
        }

    per_container_checks = [
        _compare_series(
            merged[f"{metric}_rerun"],
            merged[f"{metric}_frozen"],
            metric,
        )
        for metric in METRIC_COLUMNS
    ]
    aggregate_checks: list[dict[str, Any]] = []
    for metric in METRIC_COLUMNS:
        rerun_mean = float(rerun_summary.loc["mean", metric])
        frozen_mean = float(frozen_summary.loc["mean", metric])
        delta = abs(rerun_mean - frozen_mean)
        aggregate_checks.append({
            "metric": metric,
            "stat": "mean",
            "rerun": rerun_mean,
            "frozen": frozen_mean,
            "abs_delta": delta,
            "matches_within_tolerance": delta <= METRIC_TOLERANCE,
        })
        for stat in ("std", "min", "max"):
            rerun_val = float(rerun_summary.loc[stat, metric])
            frozen_val = float(frozen_summary.loc[stat, metric])
            stat_delta = abs(rerun_val - frozen_val)
            aggregate_checks.append({
                "metric": metric,
                "stat": stat,
                "rerun": rerun_val,
                "frozen": frozen_val,
                "abs_delta": stat_delta,
                "matches_within_tolerance": stat_delta <= METRIC_TOLERANCE,
            })

    per_container_pass = all(
        check["matches_within_tolerance"] for check in per_container_checks
    )
    aggregate_pass = all(
        check["matches_within_tolerance"] for check in aggregate_checks
    )
    gate_status = "PASS" if per_container_pass and aggregate_pass else "FAIL"

    return {
        "gate_status": gate_status,
        "checked_at": _iso_timestamp(),
        "metric_tolerance": METRIC_TOLERANCE,
        "frozen_reference": str(
            (CONTROL_REFERENCE_DIR / "evaluation").relative_to(REPO_ROOT)
        ),
        "rerun_containers": int(len(rerun_df)),
        "frozen_containers": int(len(frozen_df)),
        "per_container_checks": per_container_checks,
        "aggregate_checks": aggregate_checks,
        "per_container_pass": per_container_pass,
        "aggregate_pass": aggregate_pass,
    }


def _assert_task3_checks(
    baseline_peak_df: pd.DataFrame,
    treatment_peak_df: pd.DataFrame,
    skipped: list[tuple[str, str]],
    peak_step_fraction: float,
) -> None:
    if len(baseline_peak_df) != EXPECTED_EVALUATED:
        raise RuntimeError(
            f"Expected {EXPECTED_EVALUATED} baseline containers, "
            f"got {len(baseline_peak_df)}"
        )
    if len(treatment_peak_df) != EXPECTED_EVALUATED:
        raise RuntimeError(
            f"Expected {EXPECTED_EVALUATED} treatment containers, "
            f"got {len(treatment_peak_df)}"
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
    if not (PEAK_FRACTION_MIN <= peak_step_fraction <= PEAK_FRACTION_MAX):
        raise RuntimeError(
            f"Pooled peak fraction {peak_step_fraction:.4f} outside "
            f"feasibility range [{PEAK_FRACTION_MIN}, {PEAK_FRACTION_MAX}]"
        )


def main() -> None:
    args = _parse_args()
    experiment_dir = args.experiment_dir.resolve()
    control_reference_dir = args.control_reference_dir.resolve()
    evaluation_dir = experiment_dir / EVALUATION_SUBDIR
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    treatment_cache_path = evaluation_dir / "treatment_inference_cache.pkl"
    baseline_cache_path = evaluation_dir / "baseline_inference_cache.pkl"
    treatment_peak_path = evaluation_dir / "treatment_peak_subset_per_container.csv"
    baseline_peak_path = evaluation_dir / "baseline_peak_subset_per_container.csv"
    peak_subset_metrics_path = evaluation_dir / "peak_subset_metrics.csv"
    comparison_path = evaluation_dir / "peak_subset_per_container_comparison.csv"
    baseline_eval_df_path = evaluation_dir / "baseline_evaluation_df.csv"
    baseline_eval_summary_path = evaluation_dir / "baseline_evaluation_summary.csv"
    reproducibility_path = evaluation_dir / "baseline_reproducibility_check.json"
    pooled_summary_path = evaluation_dir / "peak_subset_summary.csv"
    metadata_path = evaluation_dir / "peak_subset_metadata.json"

    frozen_eval_df_path = control_reference_dir / "evaluation" / "evaluation_df.csv"
    frozen_eval_summary_path = (
        control_reference_dir / "evaluation" / "evaluation_summary.csv"
    )

    thresholds = load_peak_thresholds(
        experiment_dir / "peak" / PEAK_THRESHOLDS_FILENAME
    )

    print("=" * 72)
    print("Peak-Aware Global — Task 3 Peak-Subset Evaluation")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print(f"Control reference    : {control_reference_dir}")
    print()

    train_df = pd.read_parquet(DATA_TRAIN)
    val_df = pd.read_parquet(DATA_VAL)
    selected_containers = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[
        train_df["container_id"].isin(selected_containers)
    ].copy()
    global_val = val_df[val_df["container_id"].isin(selected_containers)].copy()

    baseline_model, _baseline_metadata = load_global_artifacts(
        model_path=CONTROL_MODEL_PATH,
        metadata_path=control_reference_dir / "models/global_gru_metadata.json",
    )

    print("Step 1 — Control re-run (frozen Global baseline)...")
    t0 = time.perf_counter()
    baseline_eval_df, baseline_inference, failures, skipped = (
        evaluate_selected_containers(
            selected_containers=selected_containers,
            global_train=global_train,
            global_val=global_val,
            global_gru_model=baseline_model,
            scalers=scalers,
            input_window=INPUT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )
    )
    baseline_runtime = time.perf_counter() - t0

    if failures:
        raise RuntimeError(f"Baseline inference failures: {failures}")

    baseline_summary = summarize_evaluation_metrics(baseline_eval_df)
    baseline_eval_df.to_csv(baseline_eval_df_path, index=False)
    baseline_summary.round(6).to_csv(baseline_eval_summary_path)

    print("Step 2 — Control reproducibility gate...")
    frozen_eval_df = pd.read_csv(frozen_eval_df_path)
    frozen_summary = pd.read_csv(frozen_eval_summary_path, index_col=0)
    gate_result = _run_reproducibility_gate(
        rerun_df=baseline_eval_df,
        rerun_summary=baseline_summary,
        frozen_df=frozen_eval_df,
        frozen_summary=frozen_summary,
    )
    reproducibility_path.write_text(
        json.dumps(gate_result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Reproducibility gate: {gate_result['gate_status']}")
    if gate_result["gate_status"] != "PASS":
        raise SystemExit(
            "Control reproducibility gate FAILED — "
            "treatment peak-subset work aborted."
        )

    print("Step 3 — Treatment peak-subset metrics (from Task 2 cache)...")
    if not treatment_cache_path.exists():
        raise FileNotFoundError(
            f"Treatment cache not found: {treatment_cache_path}. "
            "Run scripts/run_peak_aware_global_evaluation.py first."
        )
    with open(treatment_cache_path, "rb") as handle:
        treatment_cache = pickle.load(handle)

    treatment_peak_df = compute_peak_subset_metrics_from_global_cache(
        treatment_cache,
        thresholds,
    )
    treatment_peak_df.to_csv(treatment_peak_path, index=False)

    print("Step 4 — Baseline peak-subset metrics...")
    _save_inference_cache(baseline_inference, baseline_cache_path)
    with open(baseline_cache_path, "rb") as handle:
        baseline_cache = pickle.load(handle)
    baseline_peak_df = compute_peak_subset_metrics_batch_global(
        baseline_inference,
        thresholds,
    )
    baseline_peak_df.to_csv(baseline_peak_path, index=False)

    comparison_df = build_peak_subset_comparison_per_container(
        treatment_peak_df,
        baseline_peak_df,
    )
    comparison_df.to_csv(comparison_path, index=False)

    treatment_long = treatment_peak_df.copy()
    treatment_long.insert(0, "model", "peak_aware_global")
    baseline_long = baseline_peak_df.copy()
    baseline_long.insert(0, "model", "baseline")
    peak_subset_metrics_df = pd.concat(
        [baseline_long, treatment_long],
        ignore_index=True,
    ).sort_values(["container_id", "model"]).reset_index(drop=True)
    peak_subset_metrics_df.to_csv(peak_subset_metrics_path, index=False)

    baseline_pooled = summarize_peak_subset_metrics_from_global_cache(
        baseline_cache,
        thresholds,
    )
    treatment_pooled = summarize_peak_subset_metrics_from_global_cache(
        treatment_cache,
        thresholds,
    )

    pooled_df = pd.DataFrame([
        {"model": "baseline", **baseline_pooled},
        {"model": "peak_aware_global", **treatment_pooled},
    ])
    pooled_df.to_csv(pooled_summary_path, index=False)

    peak_step_fraction = float(baseline_pooled["peak_step_fraction"])
    _assert_task3_checks(
        baseline_peak_df,
        treatment_peak_df,
        skipped,
        peak_step_fraction,
    )

    metadata = {
        "computed_at": _iso_timestamp(),
        "stage": 3,
        "task": 3,
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "control_reference": str(control_reference_dir.relative_to(REPO_ROOT)),
        "design_lock_reference": str(DESIGN_LOCK_REFERENCE.relative_to(REPO_ROOT)),
        "reproducibility_gate": gate_result["gate_status"],
        "reproducibility_check": str(
            reproducibility_path.relative_to(experiment_dir)
        ),
        "containers_treatment": int(len(treatment_peak_df)),
        "containers_baseline": int(len(baseline_peak_df)),
        "skipped_containers": [
            {"container_id": cid, "reason": reason}
            for cid, reason in skipped
        ],
        "baseline_runtime_seconds": round(baseline_runtime, 2),
        "pooled_peak_step_fraction": peak_step_fraction,
        "peak_fraction_feasibility_range": [PEAK_FRACTION_MIN, PEAK_FRACTION_MAX],
        "outputs": {
            "baseline_evaluation_df": str(
                baseline_eval_df_path.relative_to(experiment_dir)
            ),
            "baseline_evaluation_summary": str(
                baseline_eval_summary_path.relative_to(experiment_dir)
            ),
            "baseline_inference_cache": str(
                baseline_cache_path.relative_to(experiment_dir)
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
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    workspace_path = evaluation_dir / "evaluation_workspace.json"
    if workspace_path.exists():
        workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
        workspace.setdefault("tasks", {})["task_3"] = "complete"
        workspace["peak_subset_metadata"] = str(
            metadata_path.relative_to(REPO_ROOT)
        )
        workspace["baseline_reproducibility_check"] = str(
            reproducibility_path.relative_to(REPO_ROOT)
        )
        workspace_path.write_text(
            json.dumps(workspace, indent=2) + "\n",
            encoding="utf-8",
        )

    experiment_metadata_path = experiment_dir / "experiment_metadata.json"
    if experiment_metadata_path.exists():
        experiment_meta = json.loads(
            experiment_metadata_path.read_text(encoding="utf-8")
        )
        experiment_meta.setdefault("status", {})
        experiment_meta["status"]["evaluation_task_3"] = "complete"
        experiment_meta["peak_subset_metadata"] = str(
            metadata_path.relative_to(experiment_dir)
        )
        experiment_metadata_path.write_text(
            json.dumps(experiment_meta, indent=2) + "\n",
            encoding="utf-8",
        )

    print("-" * 72)
    print(f"Baseline containers     : {len(baseline_peak_df)}")
    print(f"Treatment containers    : {len(treatment_peak_df)}")
    print(f"Pooled peak fraction    : {peak_step_fraction:.2%}")
    print()
    print("Pooled peak-subset summary:")
    print(pooled_df.round(4).to_string(index=False))
    print()
    print("Saved:")
    print(f"  {reproducibility_path}")
    print(f"  {baseline_eval_df_path}")
    print(f"  {peak_subset_metrics_path}")
    print(f"  {comparison_path}")
    print(f"  {pooled_summary_path}")
    print()
    print("Task 3 checks: PASS")


if __name__ == "__main__":
    main()
