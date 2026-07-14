#!/usr/bin/env python3
"""Verify Peak-Aware Hybrid Phase 5 evaluation integrity."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.hybrid_evaluation import (  # noqa: E402
    compute_day1_mape,
    compute_day1_metrics,
)
from utils.peak_config import (  # noqa: E402
    BASELINE_REFERENCE_DIR,
    DATA_SCALERS,
    DATA_TRAIN,
    PEAK_AWARE_MODEL_FILENAME,
    RESIDUAL_STATS_FILENAME,
)
from utils.peak_detection import compute_peak_thresholds, load_peak_thresholds  # noqa: E402

FROZEN_EVAL_MODULES = (
    REPO_ROOT / "utils/hybrid_evaluation.py",
    REPO_ROOT / "utils/hybrid_inference.py",
)

METRIC_TOLERANCE = 1e-5
THRESHOLD_TOLERANCE = 1e-6
PEAK_FRACTION_MIN = 0.17
PEAK_FRACTION_MAX = 0.31


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Peak-Aware Hybrid evaluation verification checks.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=REPO_ROOT / "experiments/peak_aware_2026-07-14_164518",
        help="Peak-aware experiment directory.",
    )
    return parser.parse_args()


def _check_result(
    check_id: int,
    name: str,
    passed: bool,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "details": details,
    }


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_has_uncommitted_changes(path: Path) -> bool:
    rel_path = path.relative_to(REPO_ROOT)
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(rel_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode != 0


def _load_inference_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _recompute_metrics_from_cache(
    inference_cache: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for container_id, arrays in sorted(inference_cache.items()):
        actual = np.ravel(arrays["actual_day1_real"])
        predicted = np.ravel(arrays["day1_final_real"])
        mae, rmse = compute_day1_metrics(actual, predicted)
        mape = compute_day1_mape(actual, predicted)
        rows.append(
            {
                "container_id": container_id,
                "day1_mae": mae,
                "day1_rmse": rmse,
                "day1_mape": mape,
            }
        )
    return pd.DataFrame(rows)


def check_1_frozen_eval_modules() -> dict[str, Any]:
    """Frozen hybrid_evaluation.py / hybrid_inference.py unmodified."""
    missing_files = [str(path) for path in FROZEN_EVAL_MODULES if not path.exists()]
    import_errors: list[str] = []
    for module_name in ("utils.hybrid_evaluation", "utils.hybrid_inference"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"{module_name}: {exc}")

    uncommitted_changes = [
        str(path.relative_to(REPO_ROOT))
        for path in FROZEN_EVAL_MODULES
        if path.exists() and _git_has_uncommitted_changes(path)
    ]

    passed = not missing_files and not import_errors and not uncommitted_changes
    return _check_result(
        1,
        "frozen_eval_modules",
        passed,
        {
            "missing_files": missing_files,
            "import_errors": import_errors,
            "uncommitted_changes": uncommitted_changes,
        },
    )


def check_2_treatment_artifacts(experiment_dir: Path) -> dict[str, Any]:
    """Treatment uses peak-aware model and treatment residual stats."""
    model_path = experiment_dir / "models" / PEAK_AWARE_MODEL_FILENAME
    stats_path = experiment_dir / "models" / RESIDUAL_STATS_FILENAME
    metadata_path = experiment_dir / "evaluation" / "evaluation_metadata.json"

    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    expected_model = f"models/{PEAK_AWARE_MODEL_FILENAME}"
    expected_stats = f"models/{RESIDUAL_STATS_FILENAME}"

    with open(stats_path, "rb") as handle:
        stats = pickle.load(handle)

    baseline_metadata_path = BASELINE_REFERENCE_DIR / "config/baseline_metadata.json"
    baseline_metadata = json.loads(baseline_metadata_path.read_text())

    res_mean_match = np.isclose(
        float(stats["res_mean"]),
        float(baseline_metadata["res_mean"]),
        rtol=0.0,
        atol=THRESHOLD_TOLERANCE,
    )
    res_std_match = np.isclose(
        float(stats["res_std"]),
        float(baseline_metadata["res_std"]),
        rtol=0.0,
        atol=THRESHOLD_TOLERANCE,
    )

    passed = (
        model_path.exists()
        and stats_path.exists()
        and metadata.get("model_path", "").endswith(expected_model)
        and metadata.get("residual_stats_path", "").endswith(expected_stats)
        and PEAK_AWARE_MODEL_FILENAME in metadata.get("model_path", "")
        and res_mean_match
        and res_std_match
    )
    return _check_result(
        2,
        "treatment_artifacts",
        passed,
        {
            "model_path": str(model_path.relative_to(REPO_ROOT)),
            "model_exists": model_path.exists(),
            "stats_path": str(stats_path.relative_to(REPO_ROOT)),
            "stats_exists": stats_path.exists(),
            "metadata_model_path": metadata.get("model_path"),
            "metadata_stats_path": metadata.get("residual_stats_path"),
            "res_mean_match": bool(res_mean_match),
            "res_std_match": bool(res_std_match),
        },
    )


def check_3_control_baseline_model(experiment_dir: Path) -> dict[str, Any]:
    """Control evaluation uses baseline reference weights, not peak-aware."""
    baseline_model_path = BASELINE_REFERENCE_DIR / "models" / "hybrid_gru.keras"
    treatment_model_path = experiment_dir / "models" / PEAK_AWARE_MODEL_FILENAME
    baseline_eval_path = experiment_dir / "evaluation" / "baseline_evaluation_df.csv"
    frozen_eval_path = (
        BASELINE_REFERENCE_DIR / "evaluation" / "evaluation_df.csv"
    )

    baseline_hash = _file_sha256(baseline_model_path) if baseline_model_path.exists() else ""
    treatment_hash = (
        _file_sha256(treatment_model_path) if treatment_model_path.exists() else ""
    )
    models_differ = baseline_hash != treatment_hash and bool(baseline_hash)

    baseline_rerun = pd.read_csv(baseline_eval_path)
    frozen_baseline = pd.read_csv(frozen_eval_path)
    merged = baseline_rerun.merge(
        frozen_baseline,
        on="container_id",
        suffixes=("_rerun", "_frozen"),
    )
    mae_matches = np.allclose(
        merged["day1_mae_rerun"].values,
        merged["day1_mae_frozen"].values,
        rtol=0.0,
        atol=METRIC_TOLERANCE,
    )

    passed = (
        baseline_model_path.exists()
        and baseline_eval_path.exists()
        and models_differ
        and len(baseline_rerun) == len(frozen_baseline)
        and mae_matches
    )
    return _check_result(
        3,
        "control_baseline_model",
        passed,
        {
            "baseline_model_path": str(baseline_model_path.relative_to(REPO_ROOT)),
            "treatment_model_path": str(treatment_model_path.relative_to(REPO_ROOT)),
            "models_differ": models_differ,
            "baseline_rerun_containers": int(len(baseline_rerun)),
            "frozen_baseline_containers": int(len(frozen_baseline)),
            "paired_mae_matches_frozen_reference": bool(mae_matches),
        },
    )


def check_4_shared_cohort(experiment_dir: Path) -> dict[str, Any]:
    """Same 99-container cohort as baseline_reference."""
    treatment_df = pd.read_csv(
        experiment_dir / "evaluation" / "evaluation_df.csv"
    )
    baseline_rerun_df = pd.read_csv(
        experiment_dir / "evaluation" / "baseline_evaluation_df.csv"
    )
    frozen_baseline_df = pd.read_csv(
        BASELINE_REFERENCE_DIR / "evaluation" / "evaluation_df.csv"
    )
    baseline_metadata = json.loads(
        (BASELINE_REFERENCE_DIR / "config/baseline_metadata.json").read_text()
    )

    treatment_ids = set(treatment_df["container_id"])
    baseline_rerun_ids = set(baseline_rerun_df["container_id"])
    frozen_ids = set(frozen_baseline_df["container_id"])
    expected_evaluated = int(baseline_metadata["evaluated_containers"])
    skipped = baseline_metadata.get("skipped_containers", [])

    passed = (
        treatment_ids == baseline_rerun_ids == frozen_ids
        and len(treatment_ids) == expected_evaluated
        and len(skipped) == 1
        and skipped[0]["container_id"] == "c_14674"
    )
    return _check_result(
        4,
        "shared_cohort",
        passed,
        {
            "containers_evaluated": len(treatment_ids),
            "expected_containers": expected_evaluated,
            "cohort_matches_baseline_rerun": treatment_ids == baseline_rerun_ids,
            "cohort_matches_frozen_reference": treatment_ids == frozen_ids,
            "skipped_containers": skipped,
        },
    )


def check_5_metrics_reproducible(experiment_dir: Path) -> dict[str, Any]:
    """Overall metrics reproducible from inference caches."""
    treatment_cache_path = (
        experiment_dir / "evaluation" / "treatment_inference_cache.pkl"
    )
    baseline_cache_path = (
        experiment_dir / "evaluation" / "baseline_inference_cache.pkl"
    )
    treatment_df = pd.read_csv(
        experiment_dir / "evaluation" / "evaluation_df.csv"
    )
    baseline_df = pd.read_csv(
        experiment_dir / "evaluation" / "baseline_evaluation_df.csv"
    )

    treatment_recomputed = _recompute_metrics_from_cache(
        _load_inference_cache(treatment_cache_path)
    )
    baseline_recomputed = _recompute_metrics_from_cache(
        _load_inference_cache(baseline_cache_path)
    )

    treatment_merged = treatment_df.merge(
        treatment_recomputed,
        on="container_id",
        suffixes=("_saved", "_recomputed"),
    )
    baseline_merged = baseline_df.merge(
        baseline_recomputed,
        on="container_id",
        suffixes=("_saved", "_recomputed"),
    )

    treatment_mae_ok = np.allclose(
        treatment_merged["day1_mae_saved"],
        treatment_merged["day1_mae_recomputed"],
        rtol=0.0,
        atol=METRIC_TOLERANCE,
    )
    treatment_rmse_ok = np.allclose(
        treatment_merged["day1_rmse_saved"],
        treatment_merged["day1_rmse_recomputed"],
        rtol=0.0,
        atol=METRIC_TOLERANCE,
    )
    baseline_mae_ok = np.allclose(
        baseline_merged["day1_mae_saved"],
        baseline_merged["day1_mae_recomputed"],
        rtol=0.0,
        atol=METRIC_TOLERANCE,
    )
    baseline_rmse_ok = np.allclose(
        baseline_merged["day1_rmse_saved"],
        baseline_merged["day1_rmse_recomputed"],
        rtol=0.0,
        atol=METRIC_TOLERANCE,
    )

    passed = (
        treatment_mae_ok
        and treatment_rmse_ok
        and baseline_mae_ok
        and baseline_rmse_ok
    )
    return _check_result(
        5,
        "metrics_reproducible",
        passed,
        {
            "treatment_mae_match": bool(treatment_mae_ok),
            "treatment_rmse_match": bool(treatment_rmse_ok),
            "baseline_mae_match": bool(baseline_mae_ok),
            "baseline_rmse_match": bool(baseline_rmse_ok),
            "metric_tolerance": METRIC_TOLERANCE,
        },
    )


def check_6_peak_thresholds_stable(experiment_dir: Path) -> dict[str, Any]:
    """Peak thresholds match saved artifact (no recompute drift)."""
    thresholds_path = experiment_dir / "peak" / "peak_thresholds.pkl"
    saved_thresholds = load_peak_thresholds(thresholds_path)

    train_df = pd.read_parquet(DATA_TRAIN)
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)
    recomputed = compute_peak_thresholds(train_df, scalers)

    evaluated_ids = pd.read_csv(
        experiment_dir / "evaluation" / "evaluation_df.csv"
    )["container_id"].tolist()

    mismatches: list[dict[str, float | str]] = []
    for container_id in evaluated_ids:
        saved_value = float(saved_thresholds[container_id])
        expected_value = float(recomputed[container_id])
        if not np.isclose(saved_value, expected_value, rtol=0.0, atol=THRESHOLD_TOLERANCE):
            mismatches.append(
                {
                    "container_id": container_id,
                    "saved": saved_value,
                    "recomputed": expected_value,
                }
            )

    peak_subset_df = pd.read_csv(
        experiment_dir / "evaluation" / "treatment_peak_subset_per_container.csv"
    )
    threshold_column_ok = np.allclose(
        peak_subset_df["threshold"].values,
        [saved_thresholds[cid] for cid in peak_subset_df["container_id"]],
        rtol=0.0,
        atol=THRESHOLD_TOLERANCE,
    )

    passed = not mismatches and threshold_column_ok
    return _check_result(
        6,
        "peak_thresholds_stable",
        passed,
        {
            "thresholds_path": str(thresholds_path.relative_to(REPO_ROOT)),
            "containers_checked": len(evaluated_ids),
            "mismatch_count": len(mismatches),
            "mismatches_sample": mismatches[:5],
            "saved_thresholds_used_in_peak_subset_csv": bool(threshold_column_ok),
            "threshold_tolerance": THRESHOLD_TOLERANCE,
        },
    )


def check_7_peak_subset_feasibility(experiment_dir: Path) -> dict[str, Any]:
    """Pooled peak count > 0 and within Phase 2 feasibility range."""
    peak_subset_summary = pd.read_csv(
        experiment_dir / "evaluation" / "peak_subset_summary.csv"
    )
    baseline_row = peak_subset_summary.loc[
        peak_subset_summary["model"] == "baseline"
    ].iloc[0]

    peak_steps = int(baseline_row["peak_steps"])
    peak_fraction = float(baseline_row["peak_step_fraction"])
    within_range = PEAK_FRACTION_MIN <= peak_fraction <= PEAK_FRACTION_MAX

    passed = peak_steps > 0 and within_range
    return _check_result(
        7,
        "peak_subset_feasibility",
        passed,
        {
            "peak_steps": peak_steps,
            "total_day1_steps": int(baseline_row["total_day1_steps"]),
            "peak_step_fraction": peak_fraction,
            "feasibility_range": [PEAK_FRACTION_MIN, PEAK_FRACTION_MAX],
            "within_phase2_range": within_range,
        },
    )


def check_8_protected_paths_unmodified(experiment_dir: Path) -> dict[str, Any]:
    """No writes to production models/ or baseline_reference/."""
    baseline_model = BASELINE_REFERENCE_DIR / "models" / "hybrid_gru.keras"
    baseline_stats = BASELINE_REFERENCE_DIR / "models" / RESIDUAL_STATS_FILENAME
    freeze_statement = BASELINE_REFERENCE_DIR / "FREEZE_STATEMENT.md"
    treatment_model = experiment_dir / "models" / PEAK_AWARE_MODEL_FILENAME

    protected_exist = all(
        path.exists()
        for path in (baseline_model, baseline_stats, freeze_statement, treatment_model)
    )

    baseline_reference_changes = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "--", "experiments/baseline_reference_2026-07-14"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    production_model_path = REPO_ROOT / "models" / "hybrid_gru.keras"
    treatment_not_in_production = (
        not production_model_path.exists()
        or not treatment_model.samefile(production_model_path)
    )

    evaluation_outputs_under_experiment = True
    for metadata_name in (
        "evaluation_metadata.json",
        "peak_subset_metadata.json",
        "evaluation_summary.json",
    ):
        metadata_path = experiment_dir / "evaluation" / metadata_name
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        for value in metadata.get("outputs", {}).values():
            if isinstance(value, str) and value.startswith("../"):
                evaluation_outputs_under_experiment = False

    passed = (
        protected_exist
        and not baseline_reference_changes
        and treatment_not_in_production
        and evaluation_outputs_under_experiment
        and str(experiment_dir).startswith(str(REPO_ROOT / "experiments"))
    )
    return _check_result(
        8,
        "protected_paths_unmodified",
        passed,
        {
            "baseline_reference_git_changes": baseline_reference_changes.splitlines(),
            "treatment_not_in_production_models": treatment_not_in_production,
            "evaluation_outputs_under_experiment": evaluation_outputs_under_experiment,
            "baseline_model_exists": baseline_model.exists(),
            "treatment_model_under_experiment_dir": treatment_model.exists(),
        },
    )


def _update_workspace(experiment_dir: Path, all_passed: bool) -> None:
    workspace_path = experiment_dir / "evaluation" / "evaluation_workspace.json"
    if not workspace_path.exists():
        return
    workspace = json.loads(workspace_path.read_text())
    workspace["tasks"]["task_5"] = "complete" if all_passed else "failed"
    workspace["verification_check"] = "evaluation/verification_check.json"
    workspace["verification_completed_at"] = _iso_timestamp()
    workspace_path.write_text(json.dumps(workspace, indent=2))


def main() -> None:
    args = _parse_args()
    experiment_dir = args.experiment_dir.resolve()
    evaluation_dir = experiment_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    output_path = evaluation_dir / "verification_check.json"

    print("=" * 72)
    print("Peak-Aware Hybrid — Evaluation Verification (Phase 5 Task 5)")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print()

    checks = [
        check_1_frozen_eval_modules,
        lambda: check_2_treatment_artifacts(experiment_dir),
        lambda: check_3_control_baseline_model(experiment_dir),
        lambda: check_4_shared_cohort(experiment_dir),
        lambda: check_5_metrics_reproducible(experiment_dir),
        lambda: check_6_peak_thresholds_stable(experiment_dir),
        lambda: check_7_peak_subset_feasibility(experiment_dir),
        lambda: check_8_protected_paths_unmodified(experiment_dir),
    ]

    results = [runner() for runner in checks]
    for result in results:
        print(f"[{result['status']}] Check {result['check_id']}: {result['name']}")

    all_passed = all(result["status"] == "PASS" for result in results)
    summary = {
        "verified_at": _iso_timestamp(),
        "phase": 5,
        "task": 5,
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "baseline_reference": str(BASELINE_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "overall_status": "PASS" if all_passed else "FAIL",
        "checks_passed": sum(1 for result in results if result["status"] == "PASS"),
        "checks_total": len(results),
        "checks": results,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    _update_workspace(experiment_dir, all_passed)

    print()
    print(f"Overall status: {summary['overall_status']}")
    print(f"Checks passed : {summary['checks_passed']}/{summary['checks_total']}")
    print(f"Saved report  : {output_path}")

    if not all_passed:
        failed = [
            f"{result['check_id']} ({result['name']})"
            for result in results
            if result["status"] == "FAIL"
        ]
        raise SystemExit(f"Evaluation verification failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
