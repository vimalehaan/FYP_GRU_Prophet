#!/usr/bin/env python3
"""Verify Global GRU multi-container evaluation outputs and integrity."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.global_config import (  # noqa: E402
    DATA_SCALERS,
    DATA_SELECTED_CONTAINERS,
    DATA_TRAIN,
    DATA_VAL,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEMO_CONTAINER_ID,
    FORECAST_HORIZON,
    INPUT_WINDOW,
    STALE_SELECTED_CONTAINER_ID,
)
from utils.global_artifacts import load_global_artifacts  # noqa: E402
from utils.global_evaluation import (  # noqa: E402
    _evaluable_container_ids,
    compute_day1_mape,
    compute_day1_metrics,
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)

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
METRIC_COLUMNS = ["day1_mae", "day1_rmse", "day1_mape"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Global GRU evaluation cohort and output integrity.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help=(
            "Evaluation experiment directory containing evaluation/ artifacts "
            "(default: latest experiments/global_gru_evaluation_*)"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained Global GRU model",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to Global GRU metadata JSON",
    )
    return parser.parse_args()


def _latest_evaluation_dir() -> Path:
    candidates = sorted(REPO_ROOT.glob("experiments/global_gru_evaluation_*"))
    if not candidates:
        raise FileNotFoundError(
            "No experiments/global_gru_evaluation_* directory found"
        )
    return candidates[-1]


def _load_cohort() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
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
    return global_train, global_val, selected_containers, scalers


def _verify_saved_evaluation_df(evaluation_df: pd.DataFrame) -> None:
    """Run integrity checks on saved evaluation outputs."""
    print("Saved evaluation_df integrity checks:")

    assert len(evaluation_df) == EXPECTED_EVALUATED, (
        f"Expected {EXPECTED_EVALUATED} evaluated containers, "
        f"got {len(evaluation_df)}"
    )
    print(f"  OK  evaluated count = {EXPECTED_EVALUATED}")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in evaluation_df.columns
    ]
    assert not missing_columns, f"Missing columns: {missing_columns}"
    print("  OK  required schema present")

    duplicate_ids = evaluation_df["container_id"].duplicated().sum()
    assert duplicate_ids == 0, f"Found {duplicate_ids} duplicate container IDs"
    print("  OK  no duplicate container IDs")

    for column in METRIC_COLUMNS:
        assert not evaluation_df[column].isna().any(), (
            f"NaN values found in {column}"
        )
    print("  OK  no NaN metric values")

    for column in METRIC_COLUMNS:
        assert (evaluation_df[column] >= 0).all(), (
            f"Negative values found in {column}"
        )
    print("  OK  MAE, RMSE, MAPE are non-negative")

    rmse_below_mae = evaluation_df["day1_rmse"] < evaluation_df["day1_mae"]
    assert not rmse_below_mae.any(), (
        "RMSE < MAE found for containers: "
        f"{evaluation_df.loc[rmse_below_mae, 'container_id'].tolist()}"
    )
    print("  OK  RMSE >= MAE for every container")

    assert STALE_SELECTED_CONTAINER_ID not in set(
        evaluation_df["container_id"]
    ), (
        f"{STALE_SELECTED_CONTAINER_ID} must not appear in evaluation_df"
    )
    print(f"  OK  {STALE_SELECTED_CONTAINER_ID} excluded from evaluation_df")


def main() -> None:
    args = _parse_args()
    experiment_dir = (
        args.experiment_dir.resolve()
        if args.experiment_dir is not None
        else _latest_evaluation_dir()
    )
    evaluation_dir = experiment_dir / "evaluation"
    eval_df_path = evaluation_dir / "evaluation_df.csv"
    eval_summary_path = evaluation_dir / "evaluation_summary.csv"
    eval_metadata_path = evaluation_dir / "evaluation_metadata.json"

    if not eval_df_path.exists():
        raise FileNotFoundError(f"Missing evaluation artifact: {eval_df_path}")

    print("=== Global GRU Evaluation Verification ===")
    print(f"Experiment directory: {experiment_dir}")
    print()

    saved_evaluation_df = pd.read_csv(eval_df_path)
    _verify_saved_evaluation_df(saved_evaluation_df)

    if eval_summary_path.exists():
        saved_summary = pd.read_csv(eval_summary_path, index_col=0)
        recomputed_summary = summarize_evaluation_metrics(saved_evaluation_df)
        for metric in METRIC_COLUMNS:
            assert np.isclose(
                saved_summary.loc["mean", metric],
                recomputed_summary.loc["mean", metric],
                rtol=0.0,
                atol=1e-6,
            ), f"Saved summary mean mismatch for {metric}"
        print("  OK  saved evaluation_summary.csv matches recomputed summary")

    if eval_metadata_path.exists():
        metadata = json.loads(eval_metadata_path.read_text())
        assert metadata.get("evaluated_containers") == EXPECTED_EVALUATED
        skipped_entries = metadata.get("skipped_containers", [])
        skipped_ids = {
            entry["container_id"] for entry in skipped_entries
        }
        assert STALE_SELECTED_CONTAINER_ID in skipped_ids
        assert len(skipped_entries) == EXPECTED_SKIPPED
        print("  OK  evaluation_metadata.json cohort counts")

    print()
    print("Live evaluation rerun checks:")
    global_train, global_val, selected_containers, scalers = _load_cohort()
    model, _ = load_global_artifacts(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
    )

    evaluable_ids, pre_skipped = _evaluable_container_ids(
        selected_containers=selected_containers,
        global_train=global_train,
        global_val=global_val,
        scalers=scalers,
    )

    evaluation_df, inference_results, failures, skipped = (
        evaluate_selected_containers(
            selected_containers=selected_containers,
            global_train=global_train,
            global_val=global_val,
            global_gru_model=model,
            scalers=scalers,
            input_window=INPUT_WINDOW,
            forecast_horizon=FORECAST_HORIZON,
        )
    )

    print(f"Selected containers : {len(selected_containers)}")
    print(f"Evaluable containers: {len(evaluable_ids)}")
    print(f"Evaluated containers: {len(evaluation_df)}")
    print(f"Skipped containers  : {len(skipped)}")
    print(f"Failures            : {len(failures)}")

    assert len(failures) == 0, failures
    assert len(evaluation_df) == EXPECTED_EVALUATED
    assert len(skipped) == EXPECTED_SKIPPED
    assert len(evaluation_df) == len(evaluable_ids)
    assert len(skipped) == len(pre_skipped)
    assert STALE_SELECTED_CONTAINER_ID in dict(skipped)
    print("  OK  live cohort counts")

    _verify_saved_evaluation_df(evaluation_df)

    demo_row = evaluation_df.loc[
        evaluation_df["container_id"] == DEMO_CONTAINER_ID
    ].iloc[0]
    demo_result = inference_results[DEMO_CONTAINER_ID]
    recomputed_mae, recomputed_rmse = compute_day1_metrics(
        demo_result.actual_day1_real,
        demo_result.day1_pred_real,
    )
    recomputed_mape = compute_day1_mape(
        demo_result.actual_day1_real,
        demo_result.day1_pred_real,
    )
    assert np.isclose(recomputed_mae, demo_row["day1_mae"], atol=1e-12)
    assert np.isclose(recomputed_rmse, demo_row["day1_rmse"], atol=1e-12)
    assert np.isclose(recomputed_mape, demo_row["day1_mape"], atol=1e-12)
    print(f"  OK  demo container {DEMO_CONTAINER_ID} metrics recompute")

    merged = saved_evaluation_df.merge(
        evaluation_df,
        on="container_id",
        suffixes=("_saved", "_live"),
    )
    assert len(merged) == EXPECTED_EVALUATED
    for metric in METRIC_COLUMNS:
        if not np.allclose(
            merged[f"{metric}_saved"],
            merged[f"{metric}_live"],
            rtol=0.0,
            atol=1e-9,
        ):
            diff = float(
                np.max(np.abs(merged[f"{metric}_saved"] - merged[f"{metric}_live"]))
            )
            raise AssertionError(
                f"Saved vs live {metric} mismatch (max diff = {diff})"
            )
    print("  OK  saved evaluation_df matches live rerun")

    print()
    print("Skipped detail:")
    for container_id, reason in skipped:
        print(f"  {container_id}: {reason}")

    print()
    print("Aggregate summary (live rerun):")
    print(summarize_evaluation_metrics(evaluation_df).round(4))
    print()
    print("Evaluation integrity: PASS")
    print("All Global GRU evaluation checks passed.")


if __name__ == "__main__":
    main()
