#!/usr/bin/env python3
"""Verify H1.1 multi-container Hybrid evaluation."""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from utils.hybrid_evaluation import (  # noqa: E402
    _evaluable_container_ids,
    compute_day1_mape,
    compute_day1_metrics,
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)


DEMO_CID = "c_11461"
STALE_SELECTED_CID = "c_14674"


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
            hybrid_gru_model=model,
            scalers=scalers,
            res_mean=residual_stats["res_mean"],
            res_std=residual_stats["res_std"],
            input_window=residual_stats.get("input_window", 288),
        )
    )
    summary = summarize_evaluation_metrics(evaluation_df)

    print("=== H1.1 + H1.4 Verification ===")
    print(f"Selected containers : {len(selected_containers)}")
    print(f"Evaluable containers: {len(evaluable_ids)}")
    print(f"Evaluated containers: {len(evaluation_df)}")
    print(f"Skipped containers  : {len(skipped)}")
    print(f"Failures            : {len(failures)}")

    assert len(failures) == 0, failures
    assert len(evaluation_df) == len(evaluable_ids)
    assert len(skipped) == len(pre_skipped)
    assert STALE_SELECTED_CID in dict(skipped)
    assert len(evaluation_df) == len(selected_containers) - len(skipped)

    demo_row = evaluation_df.loc[
        evaluation_df["container_id"] == DEMO_CID
    ].iloc[0]
    demo_result = inference_results[DEMO_CID]
    recomputed_mae, recomputed_rmse = compute_day1_metrics(
        demo_result.actual_day1_real,
        demo_result.day1_final_real,
    )
    recomputed_mape = compute_day1_mape(
        demo_result.actual_day1_real,
        demo_result.day1_final_real,
    )
    assert np.isclose(recomputed_mae, demo_row["day1_mae"], atol=1e-12)
    assert np.isclose(recomputed_rmse, demo_row["day1_rmse"], atol=1e-12)
    assert np.isclose(recomputed_mape, demo_row["day1_mape"], atol=1e-12)
    assert "day1_mape" in evaluation_df.columns

    manual_mean_mae = evaluation_df["day1_mae"].mean()
    manual_std_mae = evaluation_df["day1_mae"].std()
    assert np.isclose(summary.loc["mean", "day1_mae"], manual_mean_mae)
    assert np.isclose(summary.loc["std", "day1_mae"], manual_std_mae)

    manual_mean_mape = evaluation_df["day1_mape"].mean()
    assert np.isclose(summary.loc["mean", "day1_mape"], manual_mean_mape)

    print("\nSkipped detail:")
    for container_id, reason in skipped:
        print(f"  {container_id}: {reason}")

    print("\nAggregate summary:")
    print(summary.round(4))
    print(f"\nDemo container {DEMO_CID}:")
    print(f"  Day 1 MAE  : {demo_row['day1_mae']:.4f}")
    print(f"  Day 1 RMSE : {demo_row['day1_rmse']:.4f}")
    print(f"  Day 1 MAPE : {demo_row['day1_mape']:.4f}")
    print("\nEvaluation DataFrame sample:")
    print(evaluation_df.head(5).to_string(index=False))
    print("\nAll H1.1 + H1.4 verification checks passed.")


if __name__ == "__main__":
    main()
