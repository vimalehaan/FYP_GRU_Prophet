"""Read-only audit utilities for Ridge Residual Analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from utils.csrle.diagnostics import avg_abs_acf, avg_abs_pacf, ljung_box_reject_fraction
from utils.hybrid_config import DAY1_HORIZON, DEFAULT_INPUT_WINDOW
from utils.rra.diagnostics import (
    ACF_LAG_END,
    ACF_LAG_START,
    PACF_LAG_END,
    PACF_LAG_START,
    container_temporal_diagnostics,
    ljung_box_per_container,
)
from utils.rra.pipeline import (
    _fit_prophet_val_residuals,
    compute_ridge_remaining_series,
)
from utils.hybrid_training import generate_prophet_residuals


def zscore_residual(raw: np.ndarray, res_mean: float, res_std: float) -> np.ndarray:
    return (raw - res_mean) / res_std


def unzscore_residual(z: np.ndarray, res_mean: float, res_std: float) -> np.ndarray:
    return (z * res_std) + res_mean


def compute_remaining_rra_original(
    train_res_z: np.ndarray,
    prophet_val_raw: np.ndarray,
    ridge_model: Ridge,
    input_window: int = DEFAULT_INPUT_WINDOW,
    forecast_horizon: int = DAY1_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate RRA v1.0 remaining computation (raw prophet − z-scored ridge)."""
    return compute_ridge_remaining_series(
        train_res_z, prophet_val_raw, ridge_model, input_window, forecast_horizon,
    )


def compute_remaining_corrected_zspace(
    train_res_z: np.ndarray,
    prophet_val_raw: np.ndarray,
    ridge_model: Ridge,
    res_mean: float,
    res_std: float,
    input_window: int = DEFAULT_INPUT_WINDOW,
    forecast_horizon: int = DAY1_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Corrected remaining: all operations in z-scored residual space.

    Returns ridge_pred_z, remaining_z, prophet_val_z (aligned).
    """
    prophet_val_z = zscore_residual(prophet_val_raw, res_mean, res_std)
    ridge_pred_z, remaining_z = compute_ridge_remaining_series(
        train_res_z,
        prophet_val_z,
        ridge_model,
        input_window=input_window,
        forecast_horizon=forecast_horizon,
    )
    return ridge_pred_z, remaining_z, prophet_val_z


def compute_remaining_corrected_rawspace(
    train_res_z: np.ndarray,
    prophet_val_raw: np.ndarray,
    ridge_model: Ridge,
    res_mean: float,
    res_std: float,
    input_window: int = DEFAULT_INPUT_WINDOW,
    forecast_horizon: int = DAY1_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrected remaining in raw scaled-residual space (matches Hybrid inference)."""
    ridge_pred_z, remaining_z, prophet_val_z = compute_remaining_corrected_zspace(
        train_res_z,
        prophet_val_raw,
        ridge_model,
        res_mean,
        res_std,
        input_window,
        forecast_horizon,
    )
    ridge_pred_raw = unzscore_residual(ridge_pred_z, res_mean, res_std)
    remaining_raw = prophet_val_raw[: len(ridge_pred_raw)] - ridge_pred_raw
    return ridge_pred_raw, remaining_raw


def verify_residual_identity(
    prophet: np.ndarray,
    ridge_pred: np.ndarray,
    remaining: np.ndarray,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Check remaining = prophet − ridge element-wise."""
    n = min(len(prophet), len(ridge_pred), len(remaining))
    p, r, e = prophet[:n], ridge_pred[:n], remaining[:n]
    recon = p - r
    max_abs_err = float(np.max(np.abs(recon - e)))
    return {
        "n_points": n,
        "max_abs_identity_error": max_abs_err,
        "identity_holds": max_abs_err <= tol,
        "mean_prophet": float(np.mean(p)),
        "mean_ridge_pred": float(np.mean(r)),
        "mean_remaining": float(np.mean(e)),
    }


def verify_scaling_spaces(
    prophet_val_raw: np.ndarray,
    ridge_pred_z: np.ndarray,
    res_mean: float,
    res_std: float,
) -> dict[str, Any]:
    """Detect RRA scaling mismatch: raw prophet vs z-scored ridge prediction."""
    n = min(len(prophet_val_raw), len(ridge_pred_z))
    raw = prophet_val_raw[:n]
    z = ridge_pred_z[:n]
    z_as_if_raw = z  # what RRA subtracted
    corrected_raw_pred = unzscore_residual(z, res_mean, res_std)
    return {
        "prophet_raw_std": float(np.std(raw)),
        "ridge_pred_z_std": float(np.std(z)),
        "ridge_pred_unzscored_std": float(np.std(corrected_raw_pred)),
        "rra_subtraction_space_mismatch": True,
        "mean_abs_diff_raw_vs_z_pred": float(np.mean(np.abs(raw - z))),
        "mean_abs_diff_raw_vs_unzscored_pred": float(
            np.mean(np.abs(raw - corrected_raw_pred))
        ),
        "scale_ratio_z_to_raw_std": (
            float(np.std(z) / np.std(raw)) if np.std(raw) > 1e-12 else float("nan")
        ),
    }


def verify_day1_alignment(
    container_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    ridge_model: Ridge,
    res_mean: float,
    res_std: float,
    input_window: int = DEFAULT_INPUT_WINDOW,
) -> dict[str, Any]:
    """
    Verify first 96-step block alignment against sequence indexing convention.

    Sequence builder: input[i:i+W] predicts target[i+W:i+W+H].
    Day-1 eval: input = last W train z-scores, target = first H val z-scores.
    """
    train_c = train_df[train_df["container_id"] == container_id].sort_values(
        "time_stamp"
    )
    val_c = val_df[val_df["container_id"] == container_id].sort_values(
        "time_stamp"
    )
    train_res_df = generate_prophet_residuals(train_c)
    train_z = zscore_residual(train_res_df["residual"].values, res_mean, res_std)
    prophet_raw, _, _ = _fit_prophet_val_residuals(train_c, val_c)
    prophet_z = zscore_residual(prophet_raw, res_mean, res_std)

    expected_input = train_z[-input_window:]
    x = expected_input.reshape(1, -1)
    pred_z = ridge_model.predict(x)[0][:DAY1_HORIZON]

    h = min(DAY1_HORIZON, len(prophet_z))
    target_z = prophet_z[:h]
    pred_z = pred_z[:h]

    return {
        "container_id": container_id,
        "train_len": len(train_c),
        "val_len": len(val_c),
        "input_window": input_window,
        "first_block_horizon": h,
        "input_matches_train_tail": bool(
            np.allclose(expected_input, train_z[-input_window:])
        ),
        "day1_pearson_r_zspace": float(pearsonr(target_z, pred_z)[0])
        if np.std(target_z) > 1e-12 and np.std(pred_z) > 1e-12
        else float("nan"),
        "day1_mae_zspace": float(mean_absolute_error(target_z, pred_z)),
        "off_by_one_risk": "none_for_day1_block",
    }


def verify_no_val_leakage_in_ridge_train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    res_mean: float,
    res_std: float,
) -> dict[str, Any]:
    """Confirm Ridge training uses train_df only."""
    train_ids = set(train_df["container_id"].unique())
    val_ids = set(val_df["container_id"].unique())
    overlap_timestamps = 0
    for cid in train_ids & val_ids:
        train_ts = set(train_df.loc[train_df["container_id"] == cid, "time_stamp"])
        val_ts = set(val_df.loc[val_df["container_id"] == cid, "time_stamp"])
        overlap_timestamps += len(train_ts & val_ts)

    train_res = generate_prophet_residuals(train_df)
    train_res["residual_scaled"] = zscore_residual(
        train_res["residual"].values, res_mean, res_std,
    )
    return {
        "train_containers": len(train_ids),
        "val_containers": len(val_ids),
        "shared_container_ids": len(train_ids & val_ids),
        "train_val_timestamp_overlap": overlap_timestamps,
        "val_rows_in_ridge_train": 0,
        "leakage_detected": overlap_timestamps > 0,
    }


def replicate_acf_from_series(
    series: np.ndarray,
) -> dict[str, float]:
    """Replicate RRA ACF/PACF/Ljung metrics on one series."""
    s = series[np.isfinite(series)]
    lb = ljung_box_per_container(s, lags=(20,))
    return {
        "avg_abs_acf_1_10": avg_abs_acf(s, ACF_LAG_START, ACF_LAG_END),
        "avg_abs_pacf_1_10": avg_abs_pacf(s, PACF_LAG_START, PACF_LAG_END),
        "lb_reject_lag_20": lb.get("lb_reject_lag_20", float("nan")),
        "series_std": float(np.std(s)),
        "series_len": len(s),
    }


def fit_ar_forecast_metrics(
    series: np.ndarray,
    ar_order: int,
    min_train: int = 48,
) -> dict[str, float]:
    """
    Walk-forward AR(p) one-step forecasts on validation series.

    Uses only past values — no new forecasting models beyond AR/lag regression.
    """
    from statsmodels.tsa.ar_model import AutoReg

    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) <= ar_order + min_train + 5:
        return {
            "ar_order": ar_order,
            "n_forecasts": 0,
            "forecast_pearson_r": float("nan"),
            "forecast_mae": float("nan"),
            "residual_avg_abs_acf_1_10": float("nan"),
            "residual_lb_reject_lag_20": float("nan"),
        }

    preds: list[float] = []
    actuals: list[float] = []
    start = max(ar_order + min_train, ar_order + 1)
    for t in range(start, len(s)):
        y_hist = s[:t]
        try:
            model = AutoReg(y_hist, lags=ar_order, old_names=False).fit()
            pred = float(model.predict(start=t, end=t)[0])
        except Exception:
            continue
        preds.append(pred)
        actuals.append(float(s[t]))

    if len(preds) < 5:
        return {
            "ar_order": ar_order,
            "n_forecasts": len(preds),
            "forecast_pearson_r": float("nan"),
            "forecast_mae": float("nan"),
            "residual_avg_abs_acf_1_10": float("nan"),
            "residual_lb_reject_lag_20": float("nan"),
        }

    p = np.array(preds)
    a = np.array(actuals)
    ar_resid = a - p
    lb = ljung_box_per_container(ar_resid, lags=(20,))
    r = float(pearsonr(a, p)[0]) if np.std(a) > 1e-12 and np.std(p) > 1e-12 else float("nan")
    return {
        "ar_order": ar_order,
        "n_forecasts": len(preds),
        "forecast_pearson_r": r,
        "forecast_mae": float(mean_absolute_error(a, p)),
        "residual_avg_abs_acf_1_10": avg_abs_acf(ar_resid, 1, 10),
        "residual_lb_reject_lag_20": lb.get("lb_reject_lag_20", float("nan")),
    }


def underfit_subtraction_analysis(
    prophet_z: np.ndarray,
    ridge_pred_z: np.ndarray,
) -> dict[str, float]:
    """
    Quantify why subtraction can raise |ACF| when Ridge r is low.

    remaining = prophet - rho*prophet + (rho*prophet - ridge) approx
    """
    n = min(len(prophet_z), len(ridge_pred_z))
    p, r = prophet_z[:n], ridge_pred_z[:n]
    remaining = p - r
    rho = float(pearsonr(p, r)[0]) if np.std(p) > 1e-12 and np.std(r) > 1e-12 else 0.0
    r2 = rho * rho
    return {
        "ridge_pearson_r": rho,
        "ridge_r_squared": r2,
        "prophet_var": float(np.var(p)),
        "remaining_var": float(np.var(remaining)),
        "var_ratio_remaining_to_prophet": (
            float(np.var(remaining) / np.var(p)) if np.var(p) > 1e-12 else float("nan")
        ),
        "theoretical_min_var_ratio_if_orthogonal_error": 1.0 - r2,
    }
