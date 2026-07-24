#!/usr/bin/env python3
"""Verify Peak-Aware Global GRU Stage 3 evaluation integrity."""

from __future__ import annotations

import argparse
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

from utils.global_config import DATA_VAL  # noqa: E402
from utils.global_peak_aware_config import (  # noqa: E402
    ARCHIVED_EXPERIMENT_DIR,
    CONTROL_REFERENCE_DIR,
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    PEAK_AWARE_MODEL_FILENAME,
    PRIMARY_EXPERIMENT_DIR,
)
from utils.peak_detection import compute_peak_thresholds, load_peak_thresholds  # noqa: E402
from utils.peak_evaluation import summarize_peak_subset_metrics_from_global_cache  # noqa: E402

FROZEN_EVAL_MODULES = (
    REPO_ROOT / "utils/global_evaluation.py",
    REPO_ROOT / "utils/global_inference.py",
)

ARCHIVED_EXPERIMENT_NAME = ARCHIVED_EXPERIMENT_DIR.name
OFFICIAL_EXPERIMENT_NAME = PRIMARY_EXPERIMENT_DIR.name

METRIC_TOLERANCE = 1e-5
THRESHOLD_TOLERANCE = 1e-6
PEAK_FRACTION_MIN = 0.17
PEAK_FRACTION_MAX = 0.31

PER_CONTAINER_COLUMNS: tuple[str, ...] = (
    "container_id",
    "baseline_day1_mae",
    "baseline_day1_rmse",
    "baseline_day1_mape",
    "treatment_day1_mae",
    "treatment_day1_rmse",
    "treatment_day1_mape",
    "delta_day1_mae",
    "delta_day1_rmse",
    "delta_day1_mape",
    "baseline_peak_mae",
    "baseline_peak_rmse",
    "treatment_peak_mae",
    "treatment_peak_rmse",
    "delta_peak_mae",
    "delta_peak_rmse",
    "baseline_non_peak_mae",
    "baseline_non_peak_rmse",
    "treatment_non_peak_mae",
    "treatment_non_peak_rmse",
    "delta_non_peak_mae",
    "delta_non_peak_rmse",
)

EXPECTED_PLOTS: tuple[str, ...] = (
    "actual_vs_predicted_samples.png",
    "actual_vs_predicted_samples.pdf",
    "error_distribution.png",
    "error_distribution.pdf",
    "peak_subset_comparison.png",
    "peak_subset_comparison.pdf",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Peak-Aware Global Stage 3 evaluation verification checks.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=PRIMARY_EXPERIMENT_DIR,
        help="Official promoted treatment experiment directory.",
    )
    return parser.parse_args()


def _check_result(
    check_id: int,
    name: str,
    passed: bool,
    explanation: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "explanation": explanation,
        "details": details,
    }


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _read_summary_mean(path: Path) -> dict[str, float]:
    summary = pd.read_csv(path, index_col=0)
    return {
        "day1_mae": float(summary.loc["mean", "day1_mae"]),
        "day1_rmse": float(summary.loc["mean", "day1_rmse"]),
        "day1_mape": float(summary.loc["mean", "day1_mape"]),
    }



def check_1_official_treatment_path(experiment_dir: Path) -> dict[str, Any]:
    """Verify evaluation uses the promoted VS-1 experiment, not the archived run."""
    promotion_path = experiment_dir / "official_treatment_promotion.json"
    eval_metadata_path = experiment_dir / "evaluation" / "evaluation_metadata.json"
    workspace_path = experiment_dir / "evaluation" / "evaluation_workspace.json"
    eval_summary_path = experiment_dir / "evaluation" / "evaluation_summary.json"
    peak_subset_meta_path = experiment_dir / "evaluation" / "peak_subset_metadata.json"

    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    eval_metadata = json.loads(eval_metadata_path.read_text(encoding="utf-8"))
    workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
    eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
    peak_subset_meta = json.loads(peak_subset_meta_path.read_text(encoding="utf-8"))

    expected_rel = str(experiment_dir.relative_to(REPO_ROOT))
    promotion_ok = (
        promotion.get("promotion_status") == "official_peak_aware_global_treatment"
        and promotion.get("official_experiment_dir") == expected_rel
        and promotion.get("archived_experiment_dir", "").endswith(ARCHIVED_EXPERIMENT_NAME)
    )

    active_paths = [
        eval_metadata.get("experiment_dir"),
        eval_metadata.get("model_path"),
        workspace.get("experiment_dir"),
        eval_summary.get("experiment_dir"),
        peak_subset_meta.get("experiment_dir"),
    ]
    active_paths_ok = all(
        isinstance(path, str)
        and (
            path == expected_rel
            or path.startswith(f"{expected_rel}/")
            or path.startswith(f"{expected_rel}\\")
        )
        for path in active_paths
    )

    archived_as_active_source: list[str] = []
    for label, path in (
        ("evaluation_metadata.experiment_dir", eval_metadata.get("experiment_dir")),
        ("evaluation_metadata.model_path", eval_metadata.get("model_path")),
        ("evaluation_summary.experiment_dir", eval_summary.get("experiment_dir")),
        ("peak_subset_metadata.experiment_dir", peak_subset_meta.get("experiment_dir")),
    ):
        if isinstance(path, str) and ARCHIVED_EXPERIMENT_NAME in path:
            archived_as_active_source.append(f"{label}={path}")

    passed = promotion_ok and active_paths_ok and not archived_as_active_source
    explanation = (
        "Evaluation artifacts use the promoted VS-1 experiment as the active source; "
        "archived run is documented as predecessor only."
        if passed
        else "Official treatment path or active evaluation source is inconsistent."
    )
    return _check_result(
        1,
        "official_treatment_path",
        passed,
        explanation,
        {
            "experiment_dir": expected_rel,
            "promotion_status": promotion.get("promotion_status"),
            "official_experiment_dir": promotion.get("official_experiment_dir"),
            "archived_experiment_dir": promotion.get("archived_experiment_dir"),
            "archived_predecessor_documented": workspace.get("archived_predecessor"),
            "active_paths_ok": active_paths_ok,
            "archived_used_as_active_source": archived_as_active_source,
        },
    )


def check_2_control_reproducibility(experiment_dir: Path) -> dict[str, Any]:
    """Confirm baseline re-run matches frozen Global GRU reference."""
    repro_path = experiment_dir / "evaluation" / "baseline_reproducibility_check.json"
    repro = json.loads(repro_path.read_text(encoding="utf-8"))

    gate_pass = repro.get("gate_status") == "PASS"
    per_container_pass = repro.get("per_container_pass") is True
    aggregate_pass = repro.get("aggregate_pass") is True

    passed = gate_pass and per_container_pass and aggregate_pass
    explanation = (
        "Control reproducibility gate PASS; re-run matches frozen reference within tolerance."
        if passed
        else "Control reproducibility gate failed or report missing."
    )
    return _check_result(
        2,
        "control_reproducibility",
        passed,
        explanation,
        {
            "gate_status": repro.get("gate_status"),
            "per_container_pass": per_container_pass,
            "aggregate_pass": aggregate_pass,
            "metric_tolerance": repro.get("metric_tolerance"),
            "frozen_reference": repro.get("frozen_reference"),
            "report_path": str(repro_path.relative_to(REPO_ROOT)),
        },
    )


def check_3_cohort_consistency(experiment_dir: Path) -> dict[str, Any]:
    """Verify 99/100 cohort with c_14674 skipped and shared selected_containers.npy."""
    eval_metadata = json.loads(
        (experiment_dir / "evaluation" / "evaluation_metadata.json").read_text()
    )
    treatment_df = pd.read_csv(experiment_dir / "evaluation" / "evaluation_df.csv")
    baseline_df = pd.read_csv(
        experiment_dir / "evaluation" / "baseline_evaluation_df.csv"
    )
    frozen_df = pd.read_csv(
        CONTROL_REFERENCE_DIR / "evaluation" / "evaluation_df.csv"
    )
    baseline_metadata = json.loads(
        (CONTROL_REFERENCE_DIR / "config" / "baseline_metadata.json").read_text()
    )

    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    treatment_ids = set(treatment_df["container_id"])
    baseline_ids = set(baseline_df["container_id"])
    frozen_ids = set(frozen_df["container_id"])

    skipped = eval_metadata.get("skipped_containers", [])
    failures = eval_metadata.get("failures", [])

    passed = (
        eval_metadata.get("evaluated_containers") == 99
        and eval_metadata.get("selected_containers") == 100
        and len(failures) == 0
        and len(skipped) == 1
        and skipped[0]["container_id"] == "c_14674"
        and treatment_ids == baseline_ids == frozen_ids
        and len(treatment_ids) == baseline_metadata["evaluated_containers"]
        and len(selected) == 100
        and "c_14674" in selected
        and "c_14674" not in treatment_ids
    )
    explanation = (
        "99 evaluable containers, 1 skipped (c_14674), 0 failures; cohort matches baseline."
        if passed
        else "Cohort counts or container IDs do not match the locked baseline."
    )
    return _check_result(
        3,
        "cohort_consistency",
        passed,
        explanation,
        {
            "evaluated_containers": len(treatment_ids),
            "selected_containers_npy_count": int(len(selected)),
            "skipped_containers": skipped,
            "failures": failures,
            "treatment_matches_baseline_rerun": treatment_ids == baseline_ids,
            "treatment_matches_frozen_reference": treatment_ids == frozen_ids,
        },
    )


def check_4_metric_consistency(experiment_dir: Path) -> dict[str, Any]:
    """Verify cohort metrics agree across summary CSV, comparison table, and JSON."""
    evaluation_dir = experiment_dir / "evaluation"
    treatment_summary = _read_summary_mean(evaluation_dir / "evaluation_summary.csv")
    frozen_baseline_summary = _read_summary_mean(
        CONTROL_REFERENCE_DIR / "evaluation" / "evaluation_summary.csv"
    )
    comparison_table = pd.read_csv(evaluation_dir / "comparison_table.csv")
    evaluation_summary = json.loads(
        (evaluation_dir / "evaluation_summary.json").read_text()
    )

    mismatches: list[dict[str, Any]] = []
    for metric in ("day1_mae", "day1_rmse", "day1_mape"):
        csv_value = treatment_summary[metric]
        table_row = comparison_table[
            (comparison_table["scope"] == "overall")
            & (comparison_table["metric"] == metric)
        ]
        json_value = float(
            evaluation_summary["overall_metrics"][metric]["peak_aware_global"]
        )
        table_value = float(table_row.iloc[0]["peak_aware_global"])

        if not np.isclose(csv_value, table_value, rtol=0.0, atol=METRIC_TOLERANCE):
            mismatches.append(
                {
                    "metric": metric,
                    "source_a": "evaluation_summary.csv",
                    "source_b": "comparison_table.csv",
                    "value_a": csv_value,
                    "value_b": table_value,
                }
            )
        if not np.isclose(csv_value, json_value, rtol=0.0, atol=METRIC_TOLERANCE):
            mismatches.append(
                {
                    "metric": metric,
                    "source_a": "evaluation_summary.csv",
                    "source_b": "evaluation_summary.json",
                    "value_a": csv_value,
                    "value_b": json_value,
                }
            )

        baseline_table = float(table_row.iloc[0]["baseline"])
        baseline_json = float(
            evaluation_summary["overall_metrics"][metric]["baseline"]
        )
        baseline_frozen = frozen_baseline_summary[metric]
        if not np.isclose(
            baseline_frozen, baseline_table, rtol=0.0, atol=METRIC_TOLERANCE
        ):
            mismatches.append(
                {
                    "metric": metric,
                    "source_a": "frozen_baseline_summary.csv",
                    "source_b": "comparison_table.csv",
                    "value_a": baseline_frozen,
                    "value_b": baseline_table,
                }
            )
        if not np.isclose(
            baseline_frozen, baseline_json, rtol=0.0, atol=METRIC_TOLERANCE
        ):
            mismatches.append(
                {
                    "metric": metric,
                    "source_a": "frozen_baseline_summary.csv",
                    "source_b": "evaluation_summary.json",
                    "value_a": baseline_frozen,
                    "value_b": baseline_json,
                }
            )

    peak_subset_summary = pd.read_csv(evaluation_dir / "peak_subset_summary.csv")
    for metric in ("peak_mae", "peak_rmse", "non_peak_mae", "non_peak_rmse"):
        saved = float(
            peak_subset_summary.loc[
                peak_subset_summary["model"] == "peak_aware_global", metric
            ].iloc[0]
        )
        table_value = float(
            comparison_table.loc[
                comparison_table["metric"] == metric, "peak_aware_global"
            ].iloc[0]
        )
        json_value = float(
            evaluation_summary["peak_subset_metrics"][metric]["peak_aware_global"]
        )
        if not np.isclose(saved, table_value, rtol=0.0, atol=METRIC_TOLERANCE):
            mismatches.append(
                {
                    "metric": metric,
                    "source_a": "peak_subset_summary.csv",
                    "source_b": "comparison_table.csv",
                    "value_a": saved,
                    "value_b": table_value,
                }
            )
        if not np.isclose(saved, json_value, rtol=0.0, atol=METRIC_TOLERANCE):
            mismatches.append(
                {
                    "metric": metric,
                    "source_a": "peak_subset_summary.csv",
                    "source_b": "evaluation_summary.json",
                    "value_a": saved,
                    "value_b": json_value,
                }
            )

    passed = not mismatches
    explanation = (
        "Cohort metrics are consistent across evaluation_summary.csv, "
        "comparison_table.csv, and evaluation_summary.json."
        if passed
        else "Metric mismatches found between saved summary artifacts."
    )
    return _check_result(
        4,
        "metric_consistency",
        passed,
        explanation,
        {
            "treatment_summary": treatment_summary,
            "frozen_baseline_summary": frozen_baseline_summary,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:10],
            "metric_tolerance": METRIC_TOLERANCE,
        },
    )


def check_5_peak_threshold_consistency(experiment_dir: Path) -> dict[str, Any]:
    """Confirm train-fitted P90 thresholds with no validation-derived drift."""
    thresholds_path = experiment_dir / "peak" / "peak_thresholds.pkl"
    peak_config_path = experiment_dir / "peak" / "peak_config.json"
    peak_config = json.loads(peak_config_path.read_text(encoding="utf-8"))
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
                    "recomputed_from_train": expected_value,
                }
            )

    val_df = pd.read_parquet(DATA_VAL)
    val_only_ids = set(val_df["container_id"]) - set(train_df["container_id"])
    val_derived_thresholds = [
        container_id
        for container_id in saved_thresholds
        if container_id in val_only_ids
    ]

    config_ok = (
        peak_config.get("peak_percentile") == 90
        and peak_config.get("threshold_fit_period") == "train_only"
        and peak_config.get("threshold_scope") == "per_container"
    )

    passed = (
        config_ok
        and len(saved_thresholds) == 99
        and len(evaluated_ids) == 99
        and not mismatches
        and not val_derived_thresholds
    )
    explanation = (
        "99 train-fitted P90 per-container thresholds; no validation-derived thresholds."
        if passed
        else "Peak threshold artifact does not match train-only P90 specification."
    )
    return _check_result(
        5,
        "peak_threshold_consistency",
        passed,
        explanation,
        {
            "threshold_count": len(saved_thresholds),
            "evaluated_container_count": len(evaluated_ids),
            "peak_percentile": peak_config.get("peak_percentile"),
            "threshold_fit_period": peak_config.get("threshold_fit_period"),
            "mismatch_count": len(mismatches),
            "mismatches_sample": mismatches[:5],
            "validation_only_threshold_ids": val_derived_thresholds,
        },
    )


def check_6_peak_subset_consistency(experiment_dir: Path) -> dict[str, Any]:
    """Verify pooled peak fraction and metrics reproduce from stored inference caches."""
    evaluation_dir = experiment_dir / "evaluation"
    peak_subset_summary = pd.read_csv(evaluation_dir / "peak_subset_summary.csv")
    peak_metadata = json.loads(
        (evaluation_dir / "peak_subset_metadata.json").read_text()
    )
    thresholds = load_peak_thresholds(experiment_dir / "peak" / "peak_thresholds.pkl")

    treatment_cache = _load_inference_cache(
        evaluation_dir / "treatment_inference_cache.pkl"
    )
    baseline_cache = _load_inference_cache(
        evaluation_dir / "baseline_inference_cache.pkl"
    )

    treatment_recomputed = summarize_peak_subset_metrics_from_global_cache(
        treatment_cache,
        thresholds,
    )
    baseline_recomputed = summarize_peak_subset_metrics_from_global_cache(
        baseline_cache,
        thresholds,
    )

    mismatches: list[dict[str, Any]] = []
    for model_label, recomputed in (
        ("peak_aware_global", treatment_recomputed),
        ("baseline", baseline_recomputed),
    ):
        saved_row = peak_subset_summary.loc[
            peak_subset_summary["model"] == model_label
        ].iloc[0]
        for field in (
            "peak_step_fraction",
            "peak_mae",
            "peak_rmse",
            "non_peak_mae",
            "non_peak_rmse",
            "peak_steps",
            "non_peak_steps",
        ):
            saved_value = float(saved_row[field])
            recomputed_value = float(recomputed[field])
            if not np.isclose(
                saved_value, recomputed_value, rtol=0.0, atol=METRIC_TOLERANCE
            ):
                mismatches.append(
                    {
                        "model": model_label,
                        "field": field,
                        "saved": saved_value,
                        "recomputed_from_cache": recomputed_value,
                    }
                )

    peak_fraction = float(peak_metadata["pooled_peak_step_fraction"])
    within_range = PEAK_FRACTION_MIN <= peak_fraction <= PEAK_FRACTION_MAX

    passed = not mismatches and within_range
    explanation = (
        f"Pooled peak fraction {peak_fraction:.2%} within 17–31%; "
        "peak metrics reproduce from stored inference caches."
        if passed
        else "Peak-subset metrics or fraction do not match stored arrays."
    )
    return _check_result(
        6,
        "peak_subset_consistency",
        passed,
        explanation,
        {
            "pooled_peak_step_fraction": peak_fraction,
            "feasibility_range": [PEAK_FRACTION_MIN, PEAK_FRACTION_MAX],
            "within_range": within_range,
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:10],
        },
    )


def check_7_per_container_comparison_integrity(experiment_dir: Path) -> dict[str, Any]:
    """Validate per_container_comparison.csv schema, row count, and delta arithmetic."""
    path = experiment_dir / "evaluation" / "per_container_comparison.csv"
    df = pd.read_csv(path)

    schema_ok = list(df.columns) == list(PER_CONTAINER_COLUMNS)
    row_count_ok = len(df) == 99
    unique_ids_ok = df["container_id"].nunique() == 99

    delta_checks: dict[str, bool] = {}
    delta_mismatches = 0

    day1_pairs = (
        ("delta_day1_mae", "baseline_day1_mae", "treatment_day1_mae"),
        ("delta_day1_rmse", "baseline_day1_rmse", "treatment_day1_rmse"),
        ("delta_day1_mape", "baseline_day1_mape", "treatment_day1_mape"),
    )
    peak_pairs = (
        ("delta_peak_mae", "baseline_peak_mae", "treatment_peak_mae"),
        ("delta_peak_rmse", "baseline_peak_rmse", "treatment_peak_rmse"),
        ("delta_non_peak_mae", "baseline_non_peak_mae", "treatment_non_peak_mae"),
        ("delta_non_peak_rmse", "baseline_non_peak_rmse", "treatment_non_peak_rmse"),
    )
    for delta_col, baseline_col, treatment_col in day1_pairs + peak_pairs:
        expected = df[treatment_col] - df[baseline_col]
        actual = df[delta_col]
        both_nan = expected.isna() & actual.isna()
        numeric_ok = np.isclose(
            expected[~both_nan],
            actual[~both_nan],
            rtol=0.0,
            atol=METRIC_TOLERANCE,
            equal_nan=True,
        )
        ok = bool(np.all(numeric_ok))
        delta_checks[delta_col] = ok
        if not ok:
            delta_mismatches += int((~numeric_ok).sum())

    passed = (
        schema_ok
        and row_count_ok
        and unique_ids_ok
        and all(delta_checks.values())
    )
    explanation = (
        "per_container_comparison.csv has 99 rows, full schema, and correct deltas."
        if passed
        else "Per-container comparison file failed schema or delta validation."
    )
    return _check_result(
        7,
        "per_container_comparison_integrity",
        passed,
        explanation,
        {
            "row_count": len(df),
            "schema_ok": schema_ok,
            "columns_expected": list(PER_CONTAINER_COLUMNS),
            "columns_actual": list(df.columns),
            "delta_checks": delta_checks,
            "delta_mismatch_cells": delta_mismatches,
        },
    )


def check_8_plot_integrity(experiment_dir: Path) -> dict[str, Any]:
    """Verify expected plot artifacts exist without regenerating them."""
    plots_dir = experiment_dir / "evaluation" / "plots"
    evaluation_summary = json.loads(
        (experiment_dir / "evaluation" / "evaluation_summary.json").read_text()
    )

    missing: list[str] = []
    empty: list[str] = []
    for plot_name in EXPECTED_PLOTS:
        plot_path = plots_dir / plot_name
        if not plot_path.exists():
            missing.append(plot_name)
        elif plot_path.stat().st_size == 0:
            empty.append(plot_name)

    sample_containers = evaluation_summary.get("sample_plot_containers", [])
    treatment_cache = _load_inference_cache(
        experiment_dir / "evaluation" / "treatment_inference_cache.pkl"
    )
    sample_containers_ok = (
        len(sample_containers) >= 1
        and all(container_id in treatment_cache for container_id in sample_containers)
    )

    plot_outputs = evaluation_summary.get("plot_outputs", [])
    recorded_outputs_ok = all(
        (REPO_ROOT / rel_path).exists() for rel_path in plot_outputs
    )

    passed = not missing and not empty and sample_containers_ok and recorded_outputs_ok
    explanation = (
        "All expected PNG/PDF plots exist and sample containers are present in cache."
        if passed
        else "One or more expected plot artifacts are missing or invalid."
    )
    return _check_result(
        8,
        "plot_integrity",
        passed,
        explanation,
        {
            "plots_dir": str(plots_dir.relative_to(REPO_ROOT)),
            "missing_plots": missing,
            "empty_plots": empty,
            "sample_plot_containers": sample_containers,
            "recorded_plot_outputs": plot_outputs,
            "recorded_outputs_exist": recorded_outputs_ok,
        },
    )


def check_9_frozen_evaluation_path() -> dict[str, Any]:
    """Confirm frozen Global evaluation and inference modules are unchanged."""
    missing_files = [str(path) for path in FROZEN_EVAL_MODULES if not path.exists()]
    import_errors: list[str] = []
    for module_name in ("utils.global_evaluation", "utils.global_inference"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"{module_name}: {exc}")

    uncommitted_changes = [
        str(path.relative_to(REPO_ROOT))
        for path in FROZEN_EVAL_MODULES
        if path.exists() and _git_has_uncommitted_changes(path)
    ]

    from utils.global_evaluation import evaluate_selected_containers  # noqa: WPS433
    from utils.global_inference import run_global_inference  # noqa: WPS433

    eval_fn_ok = callable(evaluate_selected_containers)
    infer_fn_ok = callable(run_global_inference)

    passed = (
        not missing_files
        and not import_errors
        and not uncommitted_changes
        and eval_fn_ok
        and infer_fn_ok
    )
    explanation = (
        "Frozen evaluate_selected_containers() and run_global_inference() "
        "modules are present and unmodified."
        if passed
        else "Frozen Global evaluation/inference path shows modifications or import errors."
    )
    return _check_result(
        9,
        "frozen_evaluation_path",
        passed,
        explanation,
        {
            "missing_files": missing_files,
            "import_errors": import_errors,
            "uncommitted_changes": uncommitted_changes,
            "evaluate_selected_containers_callable": eval_fn_ok,
            "run_global_inference_callable": infer_fn_ok,
        },
    )


def check_10_methodology_audit(experiment_dir: Path) -> dict[str, Any]:
    """Confirm only allowed training differences vs frozen Global baseline."""
    baseline_metadata = json.loads(
        (CONTROL_REFERENCE_DIR / "config" / "baseline_metadata.json").read_text()
    )
    treatment_metadata = json.loads(
        (
            experiment_dir / "models" / "global_gru_peak_aware_metadata.json"
        ).read_text()
    )
    eval_metadata = json.loads(
        (experiment_dir / "evaluation" / "evaluation_metadata.json").read_text()
    )
    promotion = json.loads(
        (experiment_dir / "official_treatment_promotion.json").read_text()
    )

    baseline_training = baseline_metadata["training"]
    shared_fields = {
        "input_window": baseline_metadata["input_window"],
        "forecast_horizon": baseline_metadata["forecast_horizon"],
        "features": baseline_metadata["features"],
        "target": baseline_metadata["target"],
        "optimizer": baseline_training["optimizer"],
        "epochs_max": baseline_training["epochs_max"],
        "batch_size": baseline_training["batch_size"],
        "shuffle": baseline_training["shuffle"],
        "random_seed": baseline_training["random_seed"],
        "sequence_val_split": baseline_training["sequence_val_split"],
        "n_sequences": baseline_training["n_sequences"],
    }

    treatment_shared_ok = (
        treatment_metadata["input_window"] == shared_fields["input_window"]
        and treatment_metadata["forecast_horizon"] == shared_fields["forecast_horizon"]
        and treatment_metadata["features"] == shared_fields["features"]
        and treatment_metadata["target"] == shared_fields["target"]
        and treatment_metadata["optimizer"] == shared_fields["optimizer"]
        and treatment_metadata["epochs_max"] == shared_fields["epochs_max"]
        and treatment_metadata["batch_size"] == shared_fields["batch_size"]
        and treatment_metadata["shuffle"] == shared_fields["shuffle"]
        and treatment_metadata["random_seed"] == shared_fields["random_seed"]
        and treatment_metadata["sequence_val_split"] == shared_fields["sequence_val_split"]
        and treatment_metadata["n_sequences_total"] == shared_fields["n_sequences"]
    )

    allowed_diffs_ok = (
        treatment_metadata["loss"] == "timestep_weighted_mse"
        and treatment_metadata["base_loss"] == "mse"
        and float(treatment_metadata["peak_weight"]) == 5.0
        and treatment_metadata["early_stopping_patience"] == 10
        and baseline_training["loss"] == "mse"
        and baseline_training["early_stopping_patience"] == 10
    )

    eval_protocol_ok = (
        eval_metadata.get("evaluation_module")
        == "utils.global_evaluation.evaluate_selected_containers"
        and eval_metadata.get("inference_module")
        == "utils.global_inference.run_global_inference"
        and eval_metadata.get("input_window") == shared_fields["input_window"]
        and eval_metadata.get("forecast_horizon") == shared_fields["forecast_horizon"]
    )

    promotion_amendment_ok = (
        promotion["training_protocol_amendment"]["official_treatment_value"] == 10
        and promotion["training_protocol_amendment"]["stage1_design_lock_value"] == 3
        and "lambda=5"
        in promotion["training_protocol_amendment"]["methodology_unchanged"]
    )

    model_path = experiment_dir / "models" / PEAK_AWARE_MODEL_FILENAME
    control_reference_untouched = not _git_has_uncommitted_changes(
        CONTROL_REFERENCE_DIR / "models" / "global_gru.keras"
    )

    passed = (
        treatment_shared_ok
        and allowed_diffs_ok
        and eval_protocol_ok
        and promotion_amendment_ok
        and model_path.exists()
        and control_reference_untouched
    )
    explanation = (
        "Only timestep-weighted loss (λ=5) differs from the official Global baseline; "
        "architecture, data, EarlyStopping patience, inference, and evaluation "
        "protocol are unchanged."
        if passed
        else "Unexpected methodology differences detected relative to frozen baseline."
    )
    return _check_result(
        10,
        "final_methodology_audit",
        passed,
        explanation,
        {
            "shared_fields_match": treatment_shared_ok,
            "allowed_training_differences": {
                "baseline_loss": baseline_training["loss"],
                "treatment_loss": treatment_metadata["loss"],
                "baseline_early_stopping_patience": baseline_training[
                    "early_stopping_patience"
                ],
                "treatment_early_stopping_patience": treatment_metadata[
                    "early_stopping_patience"
                ],
                "peak_weight_lambda": treatment_metadata["peak_weight"],
            },
            "evaluation_protocol": {
                "evaluation_module": eval_metadata.get("evaluation_module"),
                "inference_module": eval_metadata.get("inference_module"),
            },
            "promotion_amendment_ok": promotion_amendment_ok,
            "treatment_model_exists": model_path.exists(),
            "control_reference_model_unmodified": control_reference_untouched,
        },
    )


def _collect_warnings(experiment_dir: Path) -> list[str]:
    """Non-fatal notes that do not fail verification."""
    warnings: list[str] = []

    model_metadata_path = experiment_dir / "models" / "global_gru_peak_aware_metadata.json"
    if model_metadata_path.exists():
        model_metadata = json.loads(model_metadata_path.read_text())
        if ARCHIVED_EXPERIMENT_NAME in model_metadata.get(
            "primary_experiment_reference", ""
        ):
            warnings.append(
                "Training metadata retains archived patience=3 run as VS-1 lineage "
                "reference; evaluation artifacts use the promoted VS-1 experiment only."
            )

    comparison = pd.read_csv(experiment_dir / "evaluation" / "comparison_table.csv")
    overall_mae_delta = float(
        comparison.loc[
            (comparison["scope"] == "overall") & (comparison["metric"] == "day1_mae"),
            "delta_treatment_minus_baseline",
        ].iloc[0]
    )
    peak_mae_delta = float(
        comparison.loc[
            (comparison["scope"] == "peak_subset")
            & (comparison["metric"] == "peak_mae"),
            "delta_treatment_minus_baseline",
        ].iloc[0]
    )
    if overall_mae_delta > 0 and peak_mae_delta < 0:
        warnings.append(
            "Treatment improves pooled peak MAE but worsens overall MAE due to "
            "non-peak degradation; interpret Tier 1 and Tier 2 jointly in Stage 4."
        )

    return warnings


def _update_workspace(experiment_dir: Path, all_passed: bool) -> None:
    workspace_path = experiment_dir / "evaluation" / "evaluation_workspace.json"
    if not workspace_path.exists():
        return
    workspace = json.loads(workspace_path.read_text())
    workspace["tasks"]["task_5"] = "complete" if all_passed else "failed"
    workspace["evaluation_verification"] = "verification/evaluation_verification.json"
    workspace["verification_completed_at"] = _iso_timestamp()
    workspace_path.write_text(json.dumps(workspace, indent=2))


def main() -> None:
    args = _parse_args()
    experiment_dir = args.experiment_dir.resolve()
    verification_dir = experiment_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    output_path = verification_dir / "evaluation_verification.json"

    if experiment_dir.name != OFFICIAL_EXPERIMENT_NAME:
        raise SystemExit(
            f"Refusing to verify non-official experiment: {experiment_dir.name}. "
            f"Expected {OFFICIAL_EXPERIMENT_NAME}."
        )

    print("=" * 72)
    print("Peak-Aware Global — Evaluation Verification (Stage 3 Task 5)")
    print("=" * 72)
    print(f"Experiment directory : {experiment_dir}")
    print()

    checks = [
        lambda: check_1_official_treatment_path(experiment_dir),
        lambda: check_2_control_reproducibility(experiment_dir),
        lambda: check_3_cohort_consistency(experiment_dir),
        lambda: check_4_metric_consistency(experiment_dir),
        lambda: check_5_peak_threshold_consistency(experiment_dir),
        lambda: check_6_peak_subset_consistency(experiment_dir),
        lambda: check_7_per_container_comparison_integrity(experiment_dir),
        lambda: check_8_plot_integrity(experiment_dir),
        lambda: check_9_frozen_evaluation_path(),
        lambda: check_10_methodology_audit(experiment_dir),
    ]

    results = [runner() for runner in checks]
    for result in results:
        print(
            f"[{result['status']}] Check {result['check_id']:2d}: "
            f"{result['name']} — {result['explanation']}"
        )

    warnings = _collect_warnings(experiment_dir)
    all_passed = all(result["status"] == "PASS" for result in results)
    summary = {
        "verified_at": _iso_timestamp(),
        "stage": 3,
        "task": 5,
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "baseline_reference": str(CONTROL_REFERENCE_DIR.relative_to(REPO_ROOT)),
        "overall_status": "PASS" if all_passed else "FAIL",
        "checks_passed": sum(1 for result in results if result["status"] == "PASS"),
        "checks_total": len(results),
        "checks": results,
        "warnings": warnings,
        "stage_3_lock_recommendation": (
            "READY — Stage 3 evaluation is internally consistent and may proceed to "
            "Task 6 lock record."
            if all_passed
            else "NOT READY — resolve failing checks before Stage 3 lock."
        ),
    }
    output_path.write_text(json.dumps(summary, indent=2))
    _update_workspace(experiment_dir, all_passed)

    print()
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
        print()
    print(f"Overall status: {summary['overall_status']}")
    print(f"Checks passed : {summary['checks_passed']}/{summary['checks_total']}")
    print(f"Recommendation: {summary['stage_3_lock_recommendation']}")
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
