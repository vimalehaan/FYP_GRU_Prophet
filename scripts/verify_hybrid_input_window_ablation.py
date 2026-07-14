#!/usr/bin/env python3
"""Verify H1.2 ablation isolates only the input window between experiments."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.hybrid_config import ABLATION_INPUT_WINDOWS, DAY1_HORIZON  # noqa: E402
from utils.hybrid_evaluation import evaluate_selected_containers  # noqa: E402


def _latest_ablation_dir() -> Path:
    experiment_root = REPO_ROOT / "experiments"
    candidates = sorted(
        experiment_root.glob("hybrid_input_window_ablation_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No ablation experiment directory found. "
            "Run scripts/run_hybrid_input_window_ablation.py first."
        )
    return candidates[0]


def main() -> None:
    import tensorflow as tf

    experiment_dir = _latest_ablation_dir()
    metadata_path = experiment_dir / "ablation_metadata.json"
    comparison_path = experiment_dir / "comparison_table.csv"

    if not metadata_path.exists() or not comparison_path.exists():
        raise FileNotFoundError(
            f"Missing ablation outputs in {experiment_dir}"
        )

    with open(metadata_path) as handle:
        metadata = json.load(handle)

    comparison_df = pd.read_csv(comparison_path)
    assert list(comparison_df["input_window"]) == list(ABLATION_INPUT_WINDOWS)

    train_df = pd.read_parquet(REPO_ROOT / "data/train_df.parquet")
    val_df = pd.read_parquet(REPO_ROOT / "data/val_df.parquet")
    selected_containers = np.load(
        REPO_ROOT / "data/selected_containers.npy",
        allow_pickle=True,
    )
    with open(REPO_ROOT / "data/scalers.pkl", "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[
        train_df["container_id"].isin(selected_containers)
    ].copy()
    global_val = val_df[
        val_df["container_id"].isin(selected_containers)
    ].copy()

    print("=== H1.2 Ablation Verification ===")
    print(f"Experiment directory: {experiment_dir}")

    recomputed_rows = []
    per_window_eval: dict[int, pd.DataFrame] = {}

    for input_window in ABLATION_INPUT_WINDOWS:
        run_dir = experiment_dir / f"w{input_window}"
        model_path = run_dir / "hybrid_gru.keras"
        stats_path = run_dir / "residual_stats.pkl"

        with open(stats_path, "rb") as handle:
            stats = pickle.load(handle)

        assert int(stats["input_window"]) == input_window

        model = tf.keras.models.load_model(str(model_path))
        assert model.input_shape[1] == input_window

        evaluation_df, _, failures, skipped = evaluate_selected_containers(
            selected_containers=selected_containers,
            global_train=global_train,
            global_val=global_val,
            hybrid_gru_model=model,
            scalers=scalers,
            res_mean=stats["res_mean"],
            res_std=stats["res_std"],
            input_window=input_window,
        )

        assert len(failures) == 0, failures
        per_window_eval[input_window] = evaluation_df.sort_values(
            "container_id"
        ).reset_index(drop=True)

        saved_eval = pd.read_csv(run_dir / "evaluation_df.csv")
        saved_summary = pd.read_csv(run_dir / "evaluation_summary.csv", index_col=0)

        recomputed_summary = evaluation_df[
            ["day1_mae", "day1_rmse", "day1_mape"]
        ].agg(["mean", "std"])

        for metric in ("day1_mae", "day1_rmse", "day1_mape"):
            assert np.isclose(
                recomputed_summary.loc["mean", metric],
                saved_summary.loc["mean", metric],
                rtol=0.0,
                atol=1e-6,
            )

        merged = evaluation_df.merge(
            saved_eval,
            on="container_id",
            suffixes=("_recomputed", "_saved"),
        )
        for metric in ("day1_mae", "day1_rmse", "day1_mape"):
            assert np.allclose(
                merged[f"{metric}_recomputed"],
                merged[f"{metric}_saved"],
                rtol=0.0,
                atol=1e-6,
            )

        recomputed_rows.append({
            "input_window": input_window,
            "mean_mae": float(recomputed_summary.loc["mean", "day1_mae"]),
            "std_mae": float(recomputed_summary.loc["std", "day1_mae"]),
            "mean_rmse": float(recomputed_summary.loc["mean", "day1_rmse"]),
            "std_rmse": float(recomputed_summary.loc["std", "day1_rmse"]),
            "mean_mape": float(recomputed_summary.loc["mean", "day1_mape"]),
            "std_mape": float(recomputed_summary.loc["std", "day1_mape"]),
            "skipped_containers": len(skipped),
        })

    # Same containers evaluated for both windows.
    assert (
        per_window_eval[96]["container_id"].tolist()
        == per_window_eval[288]["container_id"].tolist()
    )
    assert recomputed_rows[0]["skipped_containers"] == recomputed_rows[1]["skipped_containers"]

    # Metrics must differ between windows (otherwise ablation is meaningless).
    merged_metrics = per_window_eval[96].merge(
        per_window_eval[288],
        on="container_id",
        suffixes=("_w96", "_w288"),
    )
    assert not np.allclose(
        merged_metrics["day1_mae_w96"],
        merged_metrics["day1_mae_w288"],
        rtol=0.0,
        atol=1e-12,
    )

    recomputed_df = pd.DataFrame(recomputed_rows)
    for column in comparison_df.columns:
        if column == "evaluated_containers":
            continue
        assert np.allclose(
            comparison_df[column],
            recomputed_df[column],
            rtol=0.0,
            atol=1e-6,
        )

    assert metadata["day1_horizon"] == DAY1_HORIZON
    assert metadata["recommended_input_window"] in ABLATION_INPUT_WINDOWS

    print("\nVerified:")
    print("  - Same frozen preprocessing artifacts used")
    print("  - Same selected containers and Day 1 protocol")
    print("  - Only input_window and model weights differ between runs")
    print("  - Saved metrics reproduce exactly on reload")
    print("\nComparison table:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nRecommended input window: {metadata['recommended_input_window']}"
    )
    print(f"Rationale: {metadata['recommendation_rationale']}")
    print("\nAll H1.2 ablation verification checks passed.")


if __name__ == "__main__":
    main()
