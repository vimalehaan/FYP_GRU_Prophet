#!/usr/bin/env python3
"""Verify saved Hybrid artifacts produce identical predictions to in-memory model."""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.hybrid_artifacts import load_hybrid_artifacts, save_hybrid_artifacts  # noqa: E402
from utils.hybrid_evaluation import evaluate_selected_containers  # noqa: E402
from utils.hybrid_inference import run_hybrid_inference  # noqa: E402
from utils.hybrid_training import generate_prophet_residuals  # noqa: E402
from utils.sequence_utils import create_residual_sequences  # noqa: E402


DEMO_CID = "c_11461"


def _train_model(global_train: pd.DataFrame):
    """Replicate notebook Section 1 training without modifying utils."""
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.models import Sequential

    train_residual_df = generate_prophet_residuals(global_train)
    res_mean = train_residual_df["residual"].mean()
    res_std = train_residual_df["residual"].std()
    train_residual_df["residual_scaled"] = (
        train_residual_df["residual"] - res_mean
    ) / res_std

    features = ["residual_scaled"]
    X_all, y_all, _ = create_residual_sequences(
        train_residual_df,
        features=features,
        target="residual_scaled",
        input_window=288,
        forecast_horizon=96,
    )
    split_idx = int(len(X_all) * 0.8)
    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]

    model = Sequential([
        GRU(256, return_sequences=True, input_shape=(288, len(features))),
        Dropout(0.2),
        GRU(128, return_sequences=True),
        Dropout(0.2),
        GRU(64),
        Dense(128, activation="relu"),
        Dense(96),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0,
    )
    return model, res_mean, res_std


def _compare_inference(a, b, label: str) -> None:
    arrays = [
        ("day1_final_scaled", a.day1_final_scaled, b.day1_final_scaled),
        ("day1_final_real", a.day1_final_real, b.day1_final_real),
        ("actual_day1_real", a.actual_day1_real, b.actual_day1_real),
        ("prophet_day1_real", a.prophet_day1_real, b.prophet_day1_real),
        ("day1_residual_scaled", a.day1_residual_scaled, b.day1_residual_scaled),
    ]
    for name, left, right in arrays:
        if left.shape != right.shape:
            raise AssertionError(f"{label} {name}: shape {left.shape} vs {right.shape}")
        if not np.allclose(left, right, rtol=0.0, atol=0.0):
            diff = np.max(np.abs(left - right))
            raise AssertionError(f"{label} {name}: max diff {diff}")
    print(f"  OK  {label}")


def main() -> None:
    train_df = pd.read_parquet(REPO_ROOT / "data/train_df.parquet")
    val_df = pd.read_parquet(REPO_ROOT / "data/val_df.parquet")
    selected = np.load(
        REPO_ROOT / "data/selected_containers.npy",
        allow_pickle=True,
    )
    with open(REPO_ROOT / "data/scalers.pkl", "rb") as handle:
        scalers = pickle.load(handle)

    global_train = train_df[train_df["container_id"].isin(selected)].copy()
    global_val = val_df[val_df["container_id"].isin(selected)].copy()

    print("Training in-memory model (notebook Section 1 equivalent)...")
    trained_model, res_mean, res_std = _train_model(global_train)

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "hybrid_gru.keras"
        stats_path = Path(tmp) / "residual_stats.pkl"
        save_hybrid_artifacts(
            trained_model,
            res_mean,
            res_std,
            model_path=model_path,
            residual_stats_path=stats_path,
        )
        loaded_model, loaded_res_mean, loaded_res_std = load_hybrid_artifacts(
            model_path=model_path,
            residual_stats_path=stats_path,
        )

    assert np.isclose(res_mean, loaded_res_mean)
    assert np.isclose(res_std, loaded_res_std)

    print("Comparing demo container inference (in-memory vs loaded)...")
    mem_result = run_hybrid_inference(
        DEMO_CID,
        global_train,
        global_val,
        trained_model,
        scalers,
        res_mean,
        res_std,
    )
    loaded_result = run_hybrid_inference(
        DEMO_CID,
        global_train,
        global_val,
        loaded_model,
        scalers,
        loaded_res_mean,
        loaded_res_std,
    )
    _compare_inference(mem_result, loaded_result, DEMO_CID)

    print("Comparing full multi-container evaluation (in-memory vs loaded)...")
    mem_eval, _, mem_failures, mem_skipped = evaluate_selected_containers(
        selected,
        global_train,
        global_val,
        trained_model,
        scalers,
        res_mean,
        res_std,
    )
    loaded_eval, _, loaded_failures, loaded_skipped = evaluate_selected_containers(
        selected,
        global_train,
        global_val,
        loaded_model,
        scalers,
        loaded_res_mean,
        loaded_res_std,
    )

    assert len(mem_failures) == 0 == len(loaded_failures)
    assert mem_skipped == loaded_skipped
    assert len(mem_eval) == len(loaded_eval)

    merged = mem_eval.merge(
        loaded_eval,
        on="container_id",
        suffixes=("_mem", "_loaded"),
    )
    for metric in ("day1_mae", "day1_rmse"):
        if not np.allclose(
            merged[f"{metric}_mem"],
            merged[f"{metric}_loaded"],
            rtol=0.0,
            atol=0.0,
        ):
            diff = np.max(
                np.abs(merged[f"{metric}_mem"] - merged[f"{metric}_loaded"])
            )
            raise AssertionError(f"{metric} mismatch max diff {diff}")

    print(f"  OK  {len(merged)} containers — evaluation metrics identical")
    print("\nAll train/load equivalence checks passed.")


if __name__ == "__main__":
    main()
