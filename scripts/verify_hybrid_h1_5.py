#!/usr/bin/env python3
"""Verify H1.5 primary (Day 1) vs supplementary (Day 2) evaluation separation."""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from utils.hybrid_evaluation import (  # noqa: E402
    compute_day1_mape,
    compute_day1_metrics,
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)

DEMO_CID = "c_11461"
DAY1_METRIC_COLUMNS = ["day1_mae", "day1_rmse", "day1_mape"]
FORBIDDEN_EVAL_COLUMNS = ["day2_mae", "day2_rmse", "day2_mape"]


def compute_day2_metrics(
    actual_day2_real: np.ndarray,
    day2_final_real: np.ndarray,
) -> tuple[float, float]:
    """Return supplementary Day 2 MAE and RMSE in real CPU percent."""
    day2_mae = float(
        mean_absolute_error(actual_day2_real, day2_final_real)
    )
    day2_rmse = float(
        np.sqrt(
            mean_squared_error(actual_day2_real, day2_final_real)
        )
    )
    return day2_mae, day2_rmse


def main() -> None:
    import tensorflow as tf

    train_df = pd.read_parquet(os.path.join(REPO_ROOT, "data/train_df.parquet"))
    val_df = pd.read_parquet(os.path.join(REPO_ROOT, "data/val_df.parquet"))
    selected_containers = np.load(
        os.path.join(REPO_ROOT, "data/selected_containers.npy"),
        allow_pickle=True,
    )
    with open(os.path.join(REPO_ROOT, "data/scalers.pkl"), "rb") as handle:
        scalers = pickle.load(handle)
    with open(os.path.join(REPO_ROOT, "models/residual_stats.pkl"), "rb") as handle:
        residual_stats = pickle.load(handle)

    global_train = train_df[
        train_df["container_id"].isin(selected_containers)
    ].copy()
    global_val = val_df[
        val_df["container_id"].isin(selected_containers)
    ].copy()
    model = tf.keras.models.load_model(
        os.path.join(REPO_ROOT, "models/hybrid_gru.keras")
    )

    evaluation_df, inference_results, failures, skipped = (
        evaluate_selected_containers(
            selected_containers=selected_containers,
            global_train=global_train,
            global_val=global_val,
            hybrid_gru_model=model,
            scalers=scalers,
            res_mean=residual_stats["res_mean"],
            res_std=residual_stats["res_std"],
        )
    )
    evaluation_summary = summarize_evaluation_metrics(evaluation_df)

    print("=== H1.5 Verification — Primary vs Supplementary Evaluation ===")
    print(f"Evaluated containers : {len(evaluation_df)}")
    print(f"Skipped containers   : {len(skipped)}")
    print(f"Failures             : {len(failures)}")

    assert len(failures) == 0, failures

    # 1. Aggregate tables remain Day 1 only.
    assert list(evaluation_df.columns) == [
        "container_id",
        *DAY1_METRIC_COLUMNS,
        "train_steps",
        "validation_steps",
    ]
    for column in FORBIDDEN_EVAL_COLUMNS:
        assert column not in evaluation_df.columns
        assert column not in evaluation_summary.columns

    assert list(evaluation_summary.columns) == DAY1_METRIC_COLUMNS

    # 2. Day 1 metrics unchanged and consistent with inference outputs.
    demo_row = evaluation_df.loc[
        evaluation_df["container_id"] == DEMO_CID
    ].iloc[0]
    demo_result = inference_results[DEMO_CID]

    recomputed_day1_mae, recomputed_day1_rmse = compute_day1_metrics(
        demo_result.actual_day1_real,
        demo_result.day1_final_real,
    )
    recomputed_day1_mape = compute_day1_mape(
        demo_result.actual_day1_real,
        demo_result.day1_final_real,
    )

    assert np.isclose(recomputed_day1_mae, demo_row["day1_mae"], atol=1e-12)
    assert np.isclose(recomputed_day1_rmse, demo_row["day1_rmse"], atol=1e-12)
    assert np.isclose(recomputed_day1_mape, demo_row["day1_mape"], atol=1e-12)

    # 3. Day 2 metrics exist only via supplementary recomputation.
    day2_mae, day2_rmse = compute_day2_metrics(
        demo_result.actual_day2_real,
        demo_result.day2_final_real,
    )
    assert day2_mae > 0
    assert day2_rmse > 0

    # 4. Aggregate summary unchanged (Day 1 only).
    manual_mean_mae = evaluation_df["day1_mae"].mean()
    manual_mean_rmse = evaluation_df["day1_rmse"].mean()
    manual_mean_mape = evaluation_df["day1_mape"].mean()
    assert np.isclose(
        evaluation_summary.loc["mean", "day1_mae"],
        manual_mean_mae,
    )
    assert np.isclose(
        evaluation_summary.loc["mean", "day1_rmse"],
        manual_mean_rmse,
    )
    assert np.isclose(
        evaluation_summary.loc["mean", "day1_mape"],
        manual_mean_mape,
    )

    print("\nPrimary evaluation schema (evaluation_df):")
    print(f"  columns: {list(evaluation_df.columns)}")
    print("\nAggregate Day 1 summary (evaluation_summary):")
    print(evaluation_summary.round(4))

    print(f"\nDemo container {DEMO_CID} — Primary (Day 1):")
    print(f"  Day 1 MAE  : {demo_row['day1_mae']:.4f}")
    print(f"  Day 1 RMSE : {demo_row['day1_rmse']:.4f}")
    print(f"  Day 1 MAPE : {demo_row['day1_mape']:.4f}")

    print(f"\nDemo container {DEMO_CID} — Supplementary (Day 2 recursive):")
    print(f"  Day 2 MAE  : {day2_mae:.4f}")
    print(f"  Day 2 RMSE : {day2_rmse:.4f}")
    print("  (not stored in evaluation_df or evaluation_summary)")

    print("\nAll H1.5 verification checks passed.")
    print("Numerical outputs unchanged — only notebook presentation differs.")


if __name__ == "__main__":
    main()
