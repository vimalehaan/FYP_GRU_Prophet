#!/usr/bin/env python3
"""Verify Peak-Aware Hybrid implementation satisfies the Phase 3 fairness contract."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.hybrid_config import DEFAULT_INPUT_WINDOW, DAY1_HORIZON  # noqa: E402
from utils.hybrid_evaluation import evaluate_selected_containers  # noqa: E402
from utils.hybrid_inference import run_hybrid_inference  # noqa: E402
from utils.hybrid_training import (  # noqa: E402
    build_hybrid_gru_model,
    generate_prophet_residuals,
)
from utils.peak_config import (  # noqa: E402
    BASELINE_REFERENCE_DIR,
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    EARLY_STOPPING_MONITOR,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
    FORECAST_HORIZON,
    INPUT_WINDOW,
    PHASE4_WORKSPACE_DIR,
    PEAK_THRESHOLDS_FILENAME,
    RESIDUAL_STATS_FILENAME,
    TRAINING_METADATA_FILENAME,
)
from utils.peak_detection import compute_peak_thresholds, load_peak_thresholds  # noqa: E402
from utils.sequence_utils import create_residual_sequences  # noqa: E402

DEMO_CID = "c_11461"
RESIDUAL_TOLERANCE = 1e-6

FROZEN_MODULES = (
    "utils.hybrid_training",
    "utils.hybrid_inference",
    "utils.hybrid_evaluation",
    "utils.sequence_utils",
)

FROZEN_FILES = (
    REPO_ROOT / "utils/hybrid_training.py",
    REPO_ROOT / "utils/hybrid_inference.py",
    REPO_ROOT / "utils/hybrid_evaluation.py",
    REPO_ROOT / "utils/sequence_utils.py",
)

PEAK_AWARE_SOURCES = (
    REPO_ROOT / "utils/peak_config.py",
    REPO_ROOT / "utils/peak_detection.py",
    REPO_ROOT / "utils/hybrid_training_peak_aware.py",
    REPO_ROOT / "scripts/run_peak_aware_hybrid_experiment.py",
)

FORBIDDEN_WRITE_SNIPPETS: tuple[tuple[str, str], ...] = (
    ("save_hybrid_artifacts(", "must not save via baseline production helper"),
    ("DEFAULT_MODEL_PATH", "must not reference production model path"),
    ("DEFAULT_RESIDUAL_STATS_PATH", "must not reference production stats path"),
    ('REPO_ROOT / "models"', "must not write to production models directory"),
    ("REPO_ROOT / 'models'", "must not write to production models directory"),
    ('open(BASELINE_REFERENCE_DIR', "must not open baseline reference for writing"),
    (".write_text(BASELINE_REFERENCE_DIR", "must not write to baseline reference"),
    (".write_bytes(BASELINE_REFERENCE_DIR", "must not write to baseline reference"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Peak-Aware Hybrid fairness verification checks.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Peak-aware experiment directory (default: latest from workspace metadata)",
    )
    return parser.parse_args()


def _resolve_experiment_dir(cli_dir: Path | None) -> Path:
    if cli_dir is not None:
        return cli_dir.resolve()

    workspace_meta_path = PHASE4_WORKSPACE_DIR / "experiment_metadata.json"
    if workspace_meta_path.exists():
        workspace_meta = json.loads(workspace_meta_path.read_text())
        training_run_dir = workspace_meta.get("training_run_dir")
        if training_run_dir:
            return (REPO_ROOT / training_run_dir).resolve()

    raise FileNotFoundError(
        "No experiment directory provided and training_run_dir not set in workspace metadata."
    )


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


def _layer_signature(model: Any) -> list[dict[str, Any]]:
    signature: list[dict[str, Any]] = []
    for layer in model.layers:
        config = layer.get_config()
        signature.append({
            "class_name": layer.__class__.__name__,
            "units": config.get("units"),
            "activation": config.get("activation"),
            "rate": config.get("rate"),
            "return_sequences": config.get("return_sequences"),
        })
    return signature


def _load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray]:
    train_df = pd.read_parquet(DATA_TRAIN)
    val_df = pd.read_parquet(REPO_ROOT / "data/val_df.parquet")
    selected = np.load(DATA_SELECTED_CONTAINERS, allow_pickle=True)
    with open(DATA_SCALERS, "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[train_df["container_id"].isin(selected)].copy()
    global_val = val_df[val_df["container_id"].isin(selected)].copy()
    return global_train, global_val, scalers, selected


def check_1_frozen_file_integrity() -> dict[str, Any]:
    missing_files = [str(path) for path in FROZEN_FILES if not path.exists()]
    import_errors: list[str] = []
    for module_name in FROZEN_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            import_errors.append(f"{module_name}: {exc}")

    forbidden_hits: list[dict[str, str]] = []
    for source_path in PEAK_AWARE_SOURCES:
        if not source_path.exists():
            continue
        text = source_path.read_text()
        for snippet, reason in FORBIDDEN_WRITE_SNIPPETS:
            if snippet in text:
                forbidden_hits.append({
                    "file": str(source_path.relative_to(REPO_ROOT)),
                    "matched_snippet": snippet,
                    "reason": reason,
                })

    baseline_reference_exists = BASELINE_REFERENCE_DIR.exists()
    production_model_exists = (REPO_ROOT / "models/hybrid_gru.keras").exists()

    passed = (
        not missing_files
        and not import_errors
        and not forbidden_hits
        and baseline_reference_exists
        and production_model_exists
    )
    return _check_result(
        1,
        "frozen_file_integrity",
        passed,
        {
            "frozen_files_missing": missing_files,
            "import_errors": import_errors,
            "forbidden_write_snippet_hits": forbidden_hits,
            "baseline_reference_exists": baseline_reference_exists,
            "production_model_exists": production_model_exists,
        },
    )


def check_2_identical_residual_statistics(
    experiment_dir: Path,
    global_train: pd.DataFrame,
) -> dict[str, Any]:
    stats_path = experiment_dir / "models" / RESIDUAL_STATS_FILENAME
    with open(stats_path, "rb") as handle:
        saved_stats = pickle.load(handle)

    train_residual_df = generate_prophet_residuals(global_train)
    expected_res_mean = float(train_residual_df["residual"].mean())
    expected_res_std = float(train_residual_df["residual"].std())

    saved_res_mean = float(saved_stats["res_mean"])
    saved_res_std = float(saved_stats["res_std"])

    mean_match = np.isclose(
        saved_res_mean,
        expected_res_mean,
        rtol=0.0,
        atol=RESIDUAL_TOLERANCE,
    )
    std_match = np.isclose(
        saved_res_std,
        expected_res_std,
        rtol=0.0,
        atol=RESIDUAL_TOLERANCE,
    )

    passed = mean_match and std_match
    return _check_result(
        2,
        "identical_residual_statistics",
        passed,
        {
            "saved_res_mean": saved_res_mean,
            "expected_res_mean": expected_res_mean,
            "saved_res_std": saved_res_std,
            "expected_res_std": expected_res_std,
            "tolerance": RESIDUAL_TOLERANCE,
            "res_mean_match": bool(mean_match),
            "res_std_match": bool(std_match),
        },
    )


def check_3_identical_sequences(global_train: pd.DataFrame) -> dict[str, Any]:
    train_residual_df = generate_prophet_residuals(global_train)
    res_mean = float(train_residual_df["residual"].mean())
    res_std = float(train_residual_df["residual"].std())
    train_residual_df = train_residual_df.copy()
    train_residual_df["residual_scaled"] = (
        train_residual_df["residual"] - res_mean
    ) / res_std

    features = ["residual_scaled"]
    X_all, y_all, sample_cids = create_residual_sequences(
        train_residual_df,
        features=features,
        target="residual_scaled",
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )

    X_all_second, y_all_second, sample_cids_second = create_residual_sequences(
        train_residual_df,
        features=features,
        target="residual_scaled",
        input_window=INPUT_WINDOW,
        forecast_horizon=FORECAST_HORIZON,
    )

    x_match = np.array_equal(X_all, X_all_second)
    y_match = np.array_equal(y_all, y_all_second)
    cid_match = np.array_equal(sample_cids, sample_cids_second)

    passed = x_match and y_match and cid_match
    return _check_result(
        3,
        "identical_sequences",
        passed,
        {
            "x_shape": list(X_all.shape),
            "y_shape": list(y_all.shape),
            "n_sequences": int(len(sample_cids)),
            "x_all_reproducible": bool(x_match),
            "y_all_reproducible": bool(y_match),
            "sample_cids_reproducible": bool(cid_match),
        },
    )


def check_4_identical_architecture(experiment_dir: Path) -> dict[str, Any]:
    import tensorflow as tf

    model_path = experiment_dir / "models" / "hybrid_gru_peak_aware.keras"
    peak_aware_model = tf.keras.models.load_model(str(model_path))
    reference_model = build_hybrid_gru_model(
        input_window=INPUT_WINDOW,
        n_features=1,
        forecast_horizon=FORECAST_HORIZON,
    )

    peak_signature = _layer_signature(peak_aware_model)
    reference_signature = _layer_signature(reference_model)
    passed = peak_signature == reference_signature

    return _check_result(
        4,
        "identical_architecture",
        passed,
        {
            "peak_aware_layers": peak_signature,
            "reference_layers": reference_signature,
            "layer_count_peak_aware": len(peak_signature),
            "layer_count_reference": len(reference_signature),
        },
    )


def check_5_early_stopping_config(experiment_dir: Path) -> dict[str, Any]:
    training_metadata_path = (
        experiment_dir / "training" / TRAINING_METADATA_FILENAME
    )
    training_metadata = json.loads(training_metadata_path.read_text())

    monitor_match = (
        training_metadata.get("early_stopping_monitor") == EARLY_STOPPING_MONITOR
    )
    patience_match = (
        training_metadata.get("early_stopping_patience")
        == EARLY_STOPPING_PATIENCE
    )
    restore_match = (
        training_metadata.get("early_stopping_restore_best_weights")
        == EARLY_STOPPING_RESTORE_BEST_WEIGHTS
    )

    passed = monitor_match and patience_match and restore_match
    return _check_result(
        5,
        "early_stopping_config",
        passed,
        {
            "expected_monitor": EARLY_STOPPING_MONITOR,
            "saved_monitor": training_metadata.get("early_stopping_monitor"),
            "expected_patience": EARLY_STOPPING_PATIENCE,
            "saved_patience": training_metadata.get("early_stopping_patience"),
            "expected_restore_best_weights": EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
            "saved_restore_best_weights": training_metadata.get(
                "early_stopping_restore_best_weights"
            ),
            "monitor_match": bool(monitor_match),
            "patience_match": bool(patience_match),
            "restore_match": bool(restore_match),
        },
    )


def check_6_unweighted_validation() -> dict[str, Any]:
    from utils import hybrid_training_peak_aware

    source = inspect.getsource(hybrid_training_peak_aware.train_hybrid_gru_peak_aware)
    validation_line_present = "validation_data=(X_val, y_val)" in source
    val_sample_weight_absent = "validation_data=(X_val, y_val, W_val)" not in source
    val_weight_arg_absent = "sample_weight=W_val" not in source

    passed = (
        validation_line_present
        and val_sample_weight_absent
        and val_weight_arg_absent
    )
    return _check_result(
        6,
        "unweighted_validation",
        passed,
        {
            "validation_data_unweighted_tuple": bool(validation_line_present),
            "no_weighted_validation_tuple": bool(val_sample_weight_absent),
            "no_validation_sample_weight_arg": bool(val_weight_arg_absent),
        },
    )


def check_7_inference_path(
    experiment_dir: Path,
    global_train: pd.DataFrame,
    global_val: pd.DataFrame,
    scalers: dict[str, Any],
) -> dict[str, Any]:
    import tensorflow as tf

    model_path = experiment_dir / "models" / "hybrid_gru_peak_aware.keras"
    stats_path = experiment_dir / "models" / RESIDUAL_STATS_FILENAME

    peak_aware_model = tf.keras.models.load_model(str(model_path))
    with open(stats_path, "rb") as handle:
        residual_stats = pickle.load(handle)

    inference_callable = getattr(run_hybrid_inference, "__call__", None)
    evaluation_callable = getattr(evaluate_selected_containers, "__call__", None)

    inference_result = run_hybrid_inference(
        container_id=DEMO_CID,
        global_train=global_train,
        global_val=global_val,
        hybrid_gru_model=peak_aware_model,
        scalers=scalers,
        res_mean=float(residual_stats["res_mean"]),
        res_std=float(residual_stats["res_std"]),
        input_window=int(residual_stats.get("input_window", INPUT_WINDOW)),
    )

    eval_df, _, failures, skipped = evaluate_selected_containers(
        selected_containers=np.array([DEMO_CID]),
        global_train=global_train,
        global_val=global_val,
        hybrid_gru_model=peak_aware_model,
        scalers=scalers,
        res_mean=float(residual_stats["res_mean"]),
        res_std=float(residual_stats["res_std"]),
        input_window=int(residual_stats.get("input_window", INPUT_WINDOW)),
    )

    passed = (
        inference_callable is not None
        and evaluation_callable is not None
        and len(inference_result.day1_final_real) == DAY1_HORIZON
        and len(eval_df) == 1
        and not failures
    )
    return _check_result(
        7,
        "inference_path",
        passed,
        {
            "demo_container": DEMO_CID,
            "run_hybrid_inference_importable": inference_callable is not None,
            "evaluate_selected_containers_importable": evaluation_callable is not None,
            "day1_prediction_steps": len(inference_result.day1_final_real),
            "expected_day1_steps": DAY1_HORIZON,
            "evaluated_containers": int(len(eval_df)),
            "failures": failures,
            "skipped": [
                {"container_id": cid, "reason": reason}
                for cid, reason in skipped
            ],
        },
    )


def check_8_threshold_artifact(
    experiment_dir: Path,
    global_train: pd.DataFrame,
    scalers: dict[str, Any],
) -> dict[str, Any]:
    thresholds_path = experiment_dir / "peak" / PEAK_THRESHOLDS_FILENAME
    saved_thresholds = load_peak_thresholds(thresholds_path)
    expected_thresholds = compute_peak_thresholds(global_train, scalers)

    train_container_count = int(global_train["container_id"].nunique())
    count_match = len(saved_thresholds) == train_container_count == len(
        expected_thresholds
    )
    keys_match = set(saved_thresholds) == set(expected_thresholds)

    value_mismatches: dict[str, dict[str, float]] = {}
    if keys_match:
        for container_id, expected_value in expected_thresholds.items():
            saved_value = saved_thresholds[container_id]
            if not np.isclose(saved_value, expected_value, rtol=0.0, atol=1e-9):
                value_mismatches[container_id] = {
                    "saved": float(saved_value),
                    "expected": float(expected_value),
                }

    passed = count_match and keys_match and not value_mismatches
    return _check_result(
        8,
        "threshold_artifact",
        passed,
        {
            "train_container_count": train_container_count,
            "saved_threshold_count": len(saved_thresholds),
            "expected_threshold_count": len(expected_thresholds),
            "count_match": bool(count_match),
            "keys_match": bool(keys_match),
            "value_mismatch_count": len(value_mismatches),
            "value_mismatches": value_mismatches,
        },
    )


def _update_workspace(task_5_complete: bool) -> None:
    workspace_meta_path = PHASE4_WORKSPACE_DIR / "experiment_metadata.json"
    if not workspace_meta_path.exists():
        return
    workspace_meta = json.loads(workspace_meta_path.read_text())
    if task_5_complete:
        workspace_meta["tasks"]["task_5"] = "complete"
    workspace_meta_path.write_text(json.dumps(workspace_meta, indent=2))


def main() -> None:
    args = _parse_args()
    experiment_dir = _resolve_experiment_dir(args.experiment_dir)
    verification_dir = experiment_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    output_path = verification_dir / "fairness_check.json"

    print("=" * 72)
    print("Peak-Aware Hybrid — Fairness Verification")
    print("=" * 72)
    print(f"Experiment directory: {experiment_dir}")
    print()

    global_train, global_val, scalers, _ = _load_training_data()

    checks: list[Callable[[], dict[str, Any]]] = [
        check_1_frozen_file_integrity,
        lambda: check_2_identical_residual_statistics(experiment_dir, global_train),
        lambda: check_3_identical_sequences(global_train),
        lambda: check_4_identical_architecture(experiment_dir),
        lambda: check_5_early_stopping_config(experiment_dir),
        check_6_unweighted_validation,
        lambda: check_7_inference_path(
            experiment_dir,
            global_train,
            global_val,
            scalers,
        ),
        lambda: check_8_threshold_artifact(experiment_dir, global_train, scalers),
    ]

    results: list[dict[str, Any]] = []
    for runner in checks:
        result = runner()
        results.append(result)
        status = result["status"]
        print(f"[{status}] Check {result['check_id']}: {result['name']}")

    all_passed = all(result["status"] == "PASS" for result in results)
    summary = {
        "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
        "overall_status": "PASS" if all_passed else "FAIL",
        "checks_passed": sum(1 for result in results if result["status"] == "PASS"),
        "checks_total": len(results),
        "checks": results,
    }
    output_path.write_text(json.dumps(summary, indent=2))

    experiment_metadata_path = experiment_dir / "experiment_metadata.json"
    if experiment_metadata_path.exists():
        experiment_metadata = json.loads(experiment_metadata_path.read_text())
        experiment_metadata.setdefault("status", {})
        experiment_metadata["status"]["fairness_verification"] = (
            "complete" if all_passed else "failed"
        )
        experiment_metadata["fairness_check"] = str(
            output_path.relative_to(experiment_dir)
        )
        experiment_metadata_path.write_text(
            json.dumps(experiment_metadata, indent=2)
        )

    _update_workspace(all_passed)

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
        raise SystemExit(f"Fairness verification failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
