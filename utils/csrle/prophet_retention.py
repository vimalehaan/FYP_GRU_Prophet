"""Prophet-only cross-condition retention analysis (P_A vs P_B)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

from utils.csrle.config import CSRLEConfig
from utils.csrle.dataset import inverse_cpu_real, transform_condition_scaler
from utils.hybrid_config import DAY1_HORIZON


def fit_prophet_forecast_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    container_id: str,
    train_scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Prophet on train cpu_scaled; return val Day-1 forecasts in real CPU %.

    Returns yhat_val_real, y_val_real (Day-1 length).
    """
    train_c = train_df[train_df["container_id"] == container_id].sort_values(
        "time_stamp"
    )
    val_c = val_df[val_df["container_id"] == container_id].sort_values(
        "time_stamp"
    )
    prophet_train = pd.DataFrame({
        "ds": train_c["time_stamp"],
        "y": train_c["cpu_scaled"],
    })
    model = Prophet(daily_seasonality=True, weekly_seasonality=False)
    model.fit(prophet_train)

    future = pd.DataFrame({
        "ds": val_c["time_stamp"].values[:DAY1_HORIZON],
    })
    forecast = model.predict(future)
    yhat_scaled = forecast["yhat"].values

    y_val_scaled = val_c["cpu_scaled"].values[:DAY1_HORIZON]
    yhat_real = inverse_cpu_real(yhat_scaled, train_scaler)
    y_val_real = inverse_cpu_real(y_val_scaled, train_scaler)
    return yhat_real, y_val_real


def compute_retention_metrics(
    n_vec: np.ndarray,
    delta_p: np.ndarray,
    i_vec: np.ndarray,
    r_a: np.ndarray,
    r_b: np.ndarray,
) -> dict[str, float]:
    """Per-container retention metrics on val Day-1."""
    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def safe_var(x: np.ndarray) -> float:
        return float(np.var(x))

    var_n = safe_var(n_vec)
    identity_err = float(np.max(np.abs(i_vec - (r_b - r_a))))

    return {
        "rho_n_delta_p": safe_corr(n_vec, delta_p),
        "rho_n_i": safe_corr(n_vec, i_vec),
        "vs": safe_var(delta_p) / var_n if var_n > 1e-12 else float("nan"),
        "vr": safe_var(i_vec) / var_n if var_n > 1e-12 else float("nan"),
        "var_n": var_n,
        "var_delta_p": safe_var(delta_p),
        "var_i": safe_var(i_vec),
        "identity_max_abs_err": identity_err,
        "mean_abs_n": float(np.mean(np.abs(n_vec))),
        "mean_abs_i": float(np.mean(np.abs(i_vec))),
    }


def run_prophet_retention_for_pair(
    container_id: str,
    control_train: pd.DataFrame,
    control_val: pd.DataFrame,
    synth_train: pd.DataFrame,
    synth_val: pd.DataFrame,
    frozen_scaler: MinMaxScaler,
    synth_train_scaler: MinMaxScaler,
    manifest_val: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compare Condition A (control) vs synthetic condition B* on val Day-1.

    manifest_val must contain y_base, N, y_syn for the synthetic condition val split.
    """
    yhat_a, y_a = fit_prophet_forecast_val(
        control_train, control_val, container_id, frozen_scaler
    )
    yhat_b, y_b = fit_prophet_forecast_val(
        synth_train, synth_val, container_id, synth_train_scaler
    )

    m = manifest_val[manifest_val["container_id"] == container_id].sort_values(
        "time_stamp"
    )
    n_vec = m["N"].values[:DAY1_HORIZON]
    y_base = m["y_base"].values[:DAY1_HORIZON]

    delta_p = yhat_b - yhat_a
    i_vec = n_vec - delta_p
    r_a = y_a - yhat_a
    r_b = y_b - yhat_b

    metrics = compute_retention_metrics(n_vec, delta_p, i_vec, r_a, r_b)
    metrics["container_id"] = container_id
    return metrics
