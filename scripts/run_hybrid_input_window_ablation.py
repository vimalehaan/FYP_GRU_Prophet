#!/usr/bin/env python3
"""Run H1.2 Hybrid input-window ablation (96 vs 288)."""

from __future__ import annotations

import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT))

from utils.hybrid_artifacts import save_hybrid_artifacts  # noqa: E402
from utils.hybrid_config import ABLATION_INPUT_WINDOWS, DAY1_HORIZON  # noqa: E402
from utils.hybrid_evaluation import (  # noqa: E402
    evaluate_selected_containers,
    summarize_evaluation_metrics,
)
from utils.hybrid_training import train_hybrid_gru  # noqa: E402


def _set_random_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def _recommend_window(comparison_df: pd.DataFrame) -> tuple[int, str]:
    """Select the better input window using mean MAE, then mean RMSE."""
    ranked = comparison_df.sort_values(
        by=["mean_mae", "mean_rmse", "mean_mape"],
        ascending=[True, True, True],
    )
    best_window = int(ranked.iloc[0]["input_window"])

    row_96 = comparison_df.loc[
        comparison_df["input_window"] == 96
    ].iloc[0]
    row_288 = comparison_df.loc[
        comparison_df["input_window"] == 288
    ].iloc[0]

    if best_window == 96:
        rationale = (
            "96-step input window achieves lower mean Day 1 MAE and RMSE, "
            "aligns with the research specification (24-hour input context), "
            "and uses a simpler model input shape."
        )
    else:
        rationale = (
            "288-step input window achieves lower mean Day 1 MAE and RMSE in "
            "this ablation, suggesting the extended residual context helps "
            "the GRU capture longer nonlinear patterns."
        )

    if row_96["mean_mae"] == row_288["mean_mae"]:
        rationale += (
            " Mean MAE is tied; RMSE and MAPE were used as tie-breakers."
        )

    return best_window, rationale


def main() -> None:
    _set_random_seeds()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    experiment_dir = (
        REPO_ROOT / "experiments" / f"hybrid_input_window_ablation_{timestamp}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

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

    comparison_rows: list[dict[str, float | int]] = []
    run_metadata: dict[str, object] = {
        "timestamp": timestamp,
        "selected_containers": int(len(selected_containers)),
        "day1_horizon": DAY1_HORIZON,
        "input_windows": list(ABLATION_INPUT_WINDOWS),
        "runs": {},
    }

    print("=" * 72)
    print("H1.2 — Hybrid Input Window Ablation (96 vs 288)")
    print("=" * 72)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Containers         : {len(selected_containers)}")
    print(f"Day 1 horizon      : {DAY1_HORIZON}")
    print()

    for input_window in ABLATION_INPUT_WINDOWS:
        print("-" * 72)
        print(f"Training and evaluating input_window = {input_window}")
        print("-" * 72)

        run_dir = experiment_dir / f"w{input_window}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "hybrid_gru.keras"
        stats_path = run_dir / "residual_stats.pkl"

        model, res_mean, res_std, _ = train_hybrid_gru(
            global_train=global_train,
            input_window=input_window,
            verbose=1,
        )
        save_hybrid_artifacts(
            hybrid_gru_model=model,
            res_mean=res_mean,
            res_std=res_std,
            model_path=model_path,
            residual_stats_path=stats_path,
            input_window=input_window,
        )

        evaluation_df, _, failures, skipped = evaluate_selected_containers(
            selected_containers=selected_containers,
            global_train=global_train,
            global_val=global_val,
            hybrid_gru_model=model,
            scalers=scalers,
            res_mean=res_mean,
            res_std=res_std,
            input_window=input_window,
        )
        summary = summarize_evaluation_metrics(evaluation_df)

        evaluation_df.to_csv(
            run_dir / "evaluation_df.csv",
            index=False,
        )
        summary.round(6).to_csv(run_dir / "evaluation_summary.csv")

        if failures:
            raise RuntimeError(
                f"Failures for input_window={input_window}: {failures}"
            )

        comparison_rows.append({
            "input_window": input_window,
            "evaluated_containers": len(evaluation_df),
            "skipped_containers": len(skipped),
            "mean_mae": float(summary.loc["mean", "day1_mae"]),
            "std_mae": float(summary.loc["std", "day1_mae"]),
            "mean_rmse": float(summary.loc["mean", "day1_rmse"]),
            "std_rmse": float(summary.loc["std", "day1_rmse"]),
            "mean_mape": float(summary.loc["mean", "day1_mape"]),
            "std_mape": float(summary.loc["std", "day1_mape"]),
        })

        run_metadata["runs"][str(input_window)] = {
            "model_path": str(model_path.relative_to(REPO_ROOT)),
            "residual_stats_path": str(stats_path.relative_to(REPO_ROOT)),
            "res_mean": res_mean,
            "res_std": res_std,
            "evaluated_containers": len(evaluation_df),
            "skipped_containers": [
                {"container_id": cid, "reason": reason}
                for cid, reason in skipped
            ],
            "aggregate_summary": summary.round(6).to_dict(),
        }

        print(f"Evaluated containers: {len(evaluation_df)}")
        print(f"Skipped containers  : {len(skipped)}")
        print(summary.round(4))
        print()

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        experiment_dir / "comparison_table.csv",
        index=False,
    )

    recommended_window, rationale = _recommend_window(comparison_df)
    run_metadata["comparison_table"] = comparison_df.to_dict(orient="records")
    run_metadata["recommended_input_window"] = recommended_window
    run_metadata["recommendation_rationale"] = rationale

    with open(experiment_dir / "ablation_metadata.json", "w") as handle:
        json.dump(run_metadata, handle, indent=2)

    print("=" * 72)
    print("Comparison Table — Day 1 Primary Evaluation")
    print("=" * 72)
    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print()
    print(f"Recommended input window: {recommended_window}")
    print(f"Rationale: {rationale}")
    print()
    print(f"Saved comparison table: {experiment_dir / 'comparison_table.csv'}")
    print(f"Saved run metadata    : {experiment_dir / 'ablation_metadata.json'}")


if __name__ == "__main__":
    main()
