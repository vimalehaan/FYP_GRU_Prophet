"""CSRLE Condition A Hybrid methodology reproduction helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW
from utils.hybrid_inference import HybridInferenceResult
from utils.hybrid_training import build_hybrid_gru_model, generate_prophet_residuals
from utils.peak_evaluation import compute_peak_subset_metrics_from_arrays
from utils.sequence_utils import create_residual_sequences


def build_methodology_record() -> dict[str, Any]:
    """Authoritative frozen Hybrid methodology sources for CSRLE Stage 1."""
    return {
        "sources": {
            "baseline_metadata": (
                "experiments/baseline_reference_2026-07-14/config/baseline_metadata.json"
            ),
            "hybrid_config": "utils/hybrid_config.py",
            "hybrid_training": "utils/hybrid_training.py",
            "hybrid_inference": "utils/hybrid_inference.py",
            "hybrid_evaluation": "utils/hybrid_evaluation.py",
            "sequence_utils": "utils/sequence_utils.py",
        },
        "cohort": {
            "selected_containers": "data/selected_containers.npy",
            "evaluable_containers": 99,
            "skipped": "c_14674 (missing from frozen train/val)",
        },
        "data": {
            "train": "data/train_df.parquet",
            "val": "data/val_df.parquet",
            "scalers": "data/scalers.pkl",
            "feature": "cpu_scaled (per-container MinMax on train CPU %)",
        },
        "prophet": {
            "daily_seasonality": True,
            "weekly_seasonality": False,
            "fit_scope": "per-container train period only",
            "target": "cpu_scaled",
            "persisted": False,
            "source": "utils/hybrid_training.generate_prophet_residuals",
        },
        "residual": {
            "definition_scaled": "cpu_scaled - prophet_pred",
            "normalization": "global z-score on train residuals",
            "res_mean": "mean(train residual)",
            "res_std": "std(train residual)",
            "residual_scaled": "(residual - res_mean) / res_std",
        },
        "sequences": {
            "input_window": DEFAULT_INPUT_WINDOW,
            "forecast_horizon": DAY1_HORIZON,
            "features": ["residual_scaled"],
            "target": "residual_scaled",
            "builder": "utils.sequence_utils.create_residual_sequences",
            "internal_val_split": "chronological first 80% train / last 20% val",
            "shuffle": False,
        },
        "gru": {
            "architecture": "GRU(256)->Dropout(0.2)->GRU(128)->Dropout(0.2)->GRU(64)->Dense(128,relu)->Dense(96)",
            "optimizer": "adam",
            "loss": "mse",
            "metrics": ["mae"],
            "epochs": 100,
            "batch_size": 64,
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 10,
                "restore_best_weights": True,
            },
            "random_seed_in_train_hybrid_gru": None,
            "random_seed_in_baseline_metadata": 42,
            "seed_note": (
                "train_hybrid_gru does not set TF/Python seeds; baseline notebook "
                "used this path. Exact weight reproduction is not expected."
            ),
        },
        "inference": {
            "day1_horizon": DAY1_HORIZON,
            "residual_input": "last input_window train residual_scaled values",
            "final_day1_scaled": "prophet_day1 + gru_residual",
            "inverse_transform": "per-container MinMaxScaler from scalers.pkl",
            "evaluation_scope": "Day-1 only (96 steps)",
        },
    }


def train_control_hybrid(
    global_train: pd.DataFrame,
    input_window: int = DEFAULT_INPUT_WINDOW,
    forecast_horizon: int = DAY1_HORIZON,
    epochs: int = 100,
    batch_size: int = 64,
    verbose: int = 0,
) -> tuple[Any, float, float, pd.DataFrame, dict[str, Any]]:
    """
    Train Hybrid GRU mirroring ``train_hybrid_gru`` with training metadata.

    Does not modify ``utils/hybrid_training.py``.
    """
    from tensorflow.keras.callbacks import EarlyStopping

    train_residual_df = generate_prophet_residuals(global_train)
    res_mean = float(train_residual_df["residual"].mean())
    res_std = float(train_residual_df["residual"].std())
    train_residual_df["residual_scaled"] = (
        train_residual_df["residual"] - res_mean
    ) / res_std

    features = ["residual_scaled"]
    X_all, y_all, sample_cids = create_residual_sequences(
        train_residual_df,
        features=features,
        target="residual_scaled",
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )

    split_idx = int(len(X_all) * 0.8)
    X_train = X_all[:split_idx]
    y_train = y_all[:split_idx]
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]

    model = build_hybrid_gru_model(
        input_window=input_window,
        n_features=len(features),
        forecast_horizon=forecast_horizon,
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stop],
        verbose=verbose,
    )

    hist = history.history
    best_epoch = int(np.argmin(hist["val_loss"]) + 1)
    meta = {
        "total_sequences": int(len(X_all)),
        "train_sequences": int(len(X_train)),
        "val_sequences": int(len(X_val)),
        "epochs_run": int(len(hist["loss"])),
        "best_epoch": best_epoch,
        "final_train_loss": float(hist["loss"][-1]),
        "final_val_loss": float(hist["val_loss"][-1]),
        "best_train_loss": float(hist["loss"][best_epoch - 1]),
        "best_val_loss": float(hist["val_loss"][best_epoch - 1]),
        "res_mean": res_mean,
        "res_std": res_std,
        "history": hist,
    }
    return model, res_mean, res_std, train_residual_df, meta


def compute_container_extended_metrics(
    result: HybridInferenceResult,
    peak_threshold: float,
) -> dict[str, float]:
    """Day-1 CPU, peak-subset, and residual forecast metrics for one container."""
    actual = np.ravel(result.actual_day1_real)
    predicted = np.ravel(result.day1_final_real)
    prophet = np.ravel(result.prophet_day1_real)

    day1_mae, day1_rmse = (
        float(mean_absolute_error(actual, predicted)),
        float(np.sqrt(mean_squared_error(actual, predicted))),
    )

    peak = compute_peak_subset_metrics_from_arrays(
        result.container_id, actual, predicted, peak_threshold,
    )

    actual_res = np.ravel(result.actual_day1_scaled - result.day1_prophet)
    pred_res = np.ravel(result.day1_residual)
    res_mae = float(mean_absolute_error(actual_res, pred_res))
    res_rmse = float(np.sqrt(mean_squared_error(actual_res, pred_res)))

    if np.std(actual_res) < 1e-12 or np.std(pred_res) < 1e-12:
        pearson_r = float("nan")
        r2 = float("nan")
    else:
        pearson_r = float(pearsonr(actual_res, pred_res)[0])
        r2 = float(r2_score(actual_res, pred_res))

    std_actual = float(np.std(actual_res))
    std_pred = float(np.std(pred_res))
    std_ratio = std_pred / std_actual if std_actual > 1e-12 else float("nan")

    return {
        "container_id": result.container_id,
        "day1_mae": day1_mae,
        "day1_rmse": day1_rmse,
        "peak_mae": peak["peak_mae"],
        "peak_rmse": peak["peak_rmse"],
        "non_peak_mae": peak["non_peak_mae"],
        "non_peak_rmse": peak["non_peak_rmse"],
        "residual_mae_scaled": res_mae,
        "residual_rmse_scaled": res_rmse,
        "residual_pearson_r": pearson_r,
        "residual_r2": r2,
        "residual_std_actual": std_actual,
        "residual_std_predicted": std_pred,
        "residual_std_ratio": std_ratio,
    }


def compare_to_frozen_baseline(
    new_df: pd.DataFrame,
    frozen_df: pd.DataFrame,
    gates: dict[str, float],
) -> dict[str, Any]:
    """Compare CSRLE Condition A metrics to frozen baseline evaluation."""
    merged = new_df.merge(
        frozen_df,
        on="container_id",
        suffixes=("_csrle", "_frozen"),
        how="inner",
    )
    for metric in ("day1_mae", "day1_rmse"):
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_csrle"] - merged[f"{metric}_frozen"]
        )

    mae_mean_csrle = float(new_df["day1_mae"].mean())
    rmse_mean_csrle = float(new_df["day1_rmse"].mean())
    mae_mean_frozen = gates["frozen_mae_mean"]
    rmse_mean_frozen = gates["frozen_rmse_mean"]

    rank_corr = float(
        merged["day1_mae_csrle"].corr(merged["day1_mae_frozen"], method="pearson")
    )

    checks = {
        "mae_mean_abs_delta": {
            "value": abs(mae_mean_csrle - mae_mean_frozen),
            "threshold": gates["mae_abs_tolerance"],
            "pass": abs(mae_mean_csrle - mae_mean_frozen)
            <= gates["mae_abs_tolerance"],
        },
        "rmse_mean_abs_delta": {
            "value": abs(rmse_mean_csrle - rmse_mean_frozen),
            "threshold": gates["rmse_abs_tolerance"],
            "pass": abs(rmse_mean_csrle - rmse_mean_frozen)
            <= gates["rmse_abs_tolerance"],
        },
        "rank_corr_day1_mae": {
            "value": rank_corr,
            "threshold": gates["rank_corr_min"],
            "pass": rank_corr >= gates["rank_corr_min"],
        },
    }
    passed = all(c["pass"] for c in checks.values())

    return {
        "merged_per_container": merged,
        "summary": {
            "csrle_mean_mae": mae_mean_csrle,
            "csrle_mean_rmse": rmse_mean_csrle,
            "frozen_mean_mae": mae_mean_frozen,
            "frozen_mean_rmse": rmse_mean_frozen,
            "delta_mae_mean": mae_mean_csrle - mae_mean_frozen,
            "delta_rmse_mean": rmse_mean_csrle - rmse_mean_frozen,
            "delta_mae_median": float(merged["delta_day1_mae"].median()),
            "delta_mae_std": float(merged["delta_day1_mae"].std()),
            "delta_mae_max_abs": float(merged["delta_day1_mae"].abs().max()),
            "delta_rmse_max_abs": float(merged["delta_day1_rmse"].abs().max()),
            "n_improved_mae": int((merged["delta_day1_mae"] < 0).sum()),
            "n_worsened_mae": int((merged["delta_day1_mae"] > 0).sum()),
            "rank_corr_day1_mae": rank_corr,
        },
        "checks": checks,
        "pass": passed,
    }


def frozen_eval_path_sanity(
    csrle_eval_df: pd.DataFrame,
    frozen_eval_df: pd.DataFrame,
    atol: float = 1e-9,
) -> dict[str, Any]:
    """Compare frozen model evaluated via CSRLE path to authoritative frozen CSV."""
    merged = csrle_eval_df.merge(
        frozen_eval_df,
        on="container_id",
        suffixes=("_csrle_path", "_frozen_csv"),
    )
    max_diffs: dict[str, float] = {}
    for metric in ("day1_mae", "day1_rmse", "day1_mape"):
        diff = np.abs(
            merged[f"{metric}_csrle_path"] - merged[f"{metric}_frozen_csv"]
        )
        max_diffs[metric] = float(diff.max())

    passed = all(v <= atol for v in max_diffs.values())
    return {
        "max_abs_diff": max_diffs,
        "tolerance": atol,
        "pass": passed,
        "n_containers": len(merged),
    }
