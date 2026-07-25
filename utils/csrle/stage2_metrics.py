"""CSRLE Stage 2 evaluation metrics and per-container analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.csrle.control_reproduction import compute_container_extended_metrics
from utils.csrle.stage2_baselines import (
    predict_baseline_residuals,
    scaled_to_unscaled_residual,
)
from utils.hybrid_config import DAY1_HORIZON
from utils.hybrid_inference import HybridInferenceResult, run_hybrid_inference
from utils.hybrid_evaluation import compute_day1_mape, compute_day1_metrics
from utils.peak_evaluation import compute_peak_subset_metrics_from_arrays


HORIZONS = (1, 4, 12, 24, 48, 96)
HORIZON_BANDS = (
    (1, 4),
    (5, 12),
    (13, 24),
    (25, 48),
    (49, 96),
)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ravel(a)
    b = np.ravel(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


def _safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.ravel(a)
    b = np.ravel(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.std(a) < 1e-12:
        return float("nan")
    return float(r2_score(a, b))


def actual_predicted_residual_arrays(
    result: HybridInferenceResult,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.ravel(result.actual_day1_scaled - result.day1_prophet)
    pred = np.ravel(result.day1_residual)
    return actual, pred


def horizon_metrics_from_arrays(
    actual: np.ndarray,
    pred: np.ndarray,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h in horizons:
        sl = slice(0, h)
        a, p = actual[sl], pred[sl]
        std_a, std_p = float(np.std(a)), float(np.std(p))
        rows.append({
            "horizon": h,
            "mae": float(mean_absolute_error(a, p)),
            "rmse": float(np.sqrt(mean_squared_error(a, p))),
            "r2": _safe_r2(a, p),
            "std_actual": std_a,
            "std_predicted": std_p,
            "std_ratio": std_p / std_a if std_a > 1e-12 else float("nan"),
        })
    return pd.DataFrame(rows)


def horizon_band_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    bands: tuple[tuple[int, int], ...] = HORIZON_BANDS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lo, hi in bands:
        a = actual[lo - 1 : hi]
        p = pred[lo - 1 : hi]
        std_a = float(np.std(a))
        std_p = float(np.std(p))
        rows.append({
            "band": f"{lo}-{hi}",
            "mae": float(mean_absolute_error(a, p)),
            "rmse": float(np.sqrt(mean_squared_error(a, p))),
            "r2": _safe_r2(a, p),
            "pearson_r_within_trajectory": _safe_corr(a, p),
            "std_ratio": std_p / std_a if std_a > 1e-12 else float("nan"),
        })
    return pd.DataFrame(rows)


def cross_container_horizon_correlation(
    per_container: dict[str, tuple[np.ndarray, np.ndarray]],
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Cross-container Pearson r at fixed horizon h (not temporal correlation)."""
    rows: list[dict[str, Any]] = []
    for h in horizons:
        actuals, preds = [], []
        for actual, pred in per_container.values():
            if len(actual) >= h:
                actuals.append(actual[h - 1])
                preds.append(pred[h - 1])
        a = np.array(actuals)
        p = np.array(preds)
        std_a, std_p = float(np.std(a)), float(np.std(p))
        rows.append({
            "horizon": h,
            "cross_container_pearson_r": _safe_corr(a, p),
            "mae": float(mean_absolute_error(a, p)),
            "rmse": float(np.sqrt(mean_squared_error(a, p))),
            "std_ratio": std_p / std_a if std_a > 1e-12 else float("nan"),
        })
    return pd.DataFrame(rows)


def evaluate_container_baselines(
    result: HybridInferenceResult,
    res_mean: float,
    res_std: float,
    ridge_model: Any,
    peak_threshold: float,
) -> dict[str, dict[str, float]]:
    """Evaluate ZERO, PERSISTENCE, RIDGE, GRU on residual and CPU metrics."""
    actual = np.ravel(result.actual_day1_real)
    prophet = np.ravel(result.prophet_day1_real)
    actual_res, gru_res = actual_predicted_residual_arrays(result)

    out: dict[str, dict[str, float]] = {}
    predictors = {
        "gru": gru_res,
        "zero": scaled_to_unscaled_residual(
            predict_baseline_residuals(result.residual_input, "zero"),
            res_mean, res_std,
        ),
        "persistence": scaled_to_unscaled_residual(
            predict_baseline_residuals(result.residual_input, "persistence"),
            res_mean, res_std,
        ),
        "ridge": scaled_to_unscaled_residual(
            predict_baseline_residuals(
                result.residual_input, "ridge", ridge_model=ridge_model,
            ),
            res_mean, res_std,
        ),
    }

    for name, pred_res in predictors.items():
        pred_cpu = prophet + pred_res
        mae, rmse = compute_day1_metrics(actual, pred_cpu)
        peak = compute_peak_subset_metrics_from_arrays(
            result.container_id, actual, pred_cpu, peak_threshold,
        )
        out[name] = {
            "residual_mae": float(mean_absolute_error(actual_res, pred_res)),
            "residual_rmse": float(np.sqrt(mean_squared_error(actual_res, pred_res))),
            "residual_pearson_r": _safe_corr(actual_res, pred_res),
            "residual_r2": _safe_r2(actual_res, pred_res),
            "residual_std_ratio": (
                float(np.std(pred_res)) / float(np.std(actual_res))
                if np.std(actual_res) > 1e-12 else float("nan")
            ),
            "day1_mae": mae,
            "day1_rmse": rmse,
            "day1_mape": compute_day1_mape(actual, pred_cpu),
            "peak_mae": peak["peak_mae"],
            "non_peak_mae": peak["non_peak_mae"],
        }
    return out


def synthetic_recovery_metrics(
    result: HybridInferenceResult,
    manifest_val: pd.DataFrame,
    control_val: pd.DataFrame | None = None,
    control_train: pd.DataFrame | None = None,
    b_train: pd.DataFrame | None = None,
) -> dict[str, float]:
    """
    G_hat vs R_B (primary); G_hat vs N_effective; optional G_hat vs I.
    """
    cid = result.container_id
    man = manifest_val[manifest_val["container_id"] == cid].sort_values("time_stamp")
    man = man.iloc[:DAY1_HORIZON]
    if man.empty:
        return {}

    g_hat, r_b = actual_predicted_residual_arrays(result)
    g_hat = g_hat[: len(man)]
    r_b = r_b[: len(man)]

    n_eff = man["N"].values.astype(float)
    out = {
        "recovery_pearson_r_g_vs_rb": _safe_corr(g_hat, r_b),
        "recovery_r2_g_vs_rb": _safe_r2(r_b, g_hat),
        "recovery_mae_g_vs_rb": float(mean_absolute_error(r_b, g_hat)),
        "recovery_rmse_g_vs_rb": float(np.sqrt(mean_squared_error(r_b, g_hat))),
        "recovery_pearson_r_g_vs_n": _safe_corr(g_hat, n_eff),
        "recovery_r2_g_vs_n": _safe_r2(n_eff, g_hat),
        "recovery_std_n": float(np.std(n_eff)),
        "recovery_std_g_hat": float(np.std(g_hat)),
        "recovery_std_rb": float(np.std(r_b)),
    }

    if (
        control_val is not None
        and control_train is not None
        and b_train is not None
        and "I" in man.columns
    ):
        i_vec = man["I"].values.astype(float)
        out["recovery_pearson_r_g_vs_i"] = _safe_corr(g_hat, i_vec)
        out["recovery_r2_g_vs_i"] = _safe_r2(i_vec, g_hat)
    elif control_val is not None and b_train is not None:
        from prophet import Prophet

        ctrl_tr = control_train[control_train["container_id"] == cid].sort_values(
            "time_stamp",
        )
        ctrl_va = control_val[control_val["container_id"] == cid].sort_values(
            "time_stamp",
        )
        b_tr = b_train[b_train["container_id"] == cid].sort_values("time_stamp")
        if len(ctrl_tr) > 0 and len(ctrl_va) >= DAY1_HORIZON:
            pa = Prophet(daily_seasonality=True, weekly_seasonality=False)
            pa.fit(pd.DataFrame({"ds": ctrl_tr["time_stamp"], "y": ctrl_tr["cpu_scaled"]}))
            pb = Prophet(daily_seasonality=True, weekly_seasonality=False)
            pb.fit(pd.DataFrame({"ds": b_tr["time_stamp"], "y": b_tr["cpu_scaled"]}))
            future = pd.DataFrame({"ds": ctrl_va["time_stamp"].values[:DAY1_HORIZON]})
            yhat_a = pa.predict(future)["yhat"].values
            yhat_b = pb.predict(future)["yhat"].values
            delta_p = yhat_b - yhat_a
            i_vec = n_eff - delta_p
            out["recovery_pearson_r_g_vs_i"] = _safe_corr(g_hat, i_vec)
            out["recovery_r2_g_vs_i"] = _safe_r2(i_vec, g_hat)

    return out


def prophet_cpu_metrics(result: HybridInferenceResult) -> dict[str, float]:
    actual = np.ravel(result.actual_day1_real)
    prophet = np.ravel(result.prophet_day1_real)
    hybrid = np.ravel(result.day1_final_real)
    p_mae, p_rmse = compute_day1_metrics(actual, prophet)
    h_mae, h_rmse = compute_day1_metrics(actual, hybrid)
    return {
        "prophet_day1_mae": p_mae,
        "prophet_day1_rmse": p_rmse,
        "prophet_day1_mape": compute_day1_mape(actual, prophet),
        "hybrid_day1_mae": h_mae,
        "hybrid_day1_rmse": h_rmse,
        "hybrid_day1_mape": compute_day1_mape(actual, hybrid),
        "hybrid_minus_prophet_mae": h_mae - p_mae,
        "hybrid_minus_prophet_rmse": h_rmse - p_rmse,
    }


def run_condition_evaluation(
    container_ids: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scalers: dict[str, Any],
    model: Any,
    res_mean: float,
    res_std: float,
    ridge_model: Any,
    peak_thresholds: dict[str, float],
    manifest_val: pd.DataFrame | None = None,
    condition_id: str | None = None,
    control_train: pd.DataFrame | None = None,
    control_val: pd.DataFrame | None = None,
    compute_recovery: bool = False,
) -> dict[str, Any]:
    """Full per-container evaluation for one synthetic condition."""
    inference: dict[str, HybridInferenceResult] = {}
    ext_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    cpu_rows: list[dict[str, Any]] = []
    per_container_res: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cid in container_ids:
        try:
            result = run_hybrid_inference(
                cid, train_df, val_df, model, scalers, res_mean, res_std,
            )
        except Exception as exc:  # noqa: BLE001
            continue

        inference[cid] = result
        peak_th = peak_thresholds.get(cid, float("inf"))
        ext = compute_container_extended_metrics(result, peak_th)
        ext_rows.append(ext)

        actual_res, pred_res = actual_predicted_residual_arrays(result)
        per_container_res[cid] = (actual_res, pred_res)

        hm = horizon_metrics_from_arrays(actual_res, pred_res)
        hm["container_id"] = cid
        horizon_rows.append(hm)

        bm = horizon_band_metrics(actual_res, pred_res)
        bm["container_id"] = cid
        band_rows.append(bm)

        baselines = evaluate_container_baselines(
            result, res_mean, res_std, ridge_model, peak_th,
        )
        for bname, metrics in baselines.items():
            baseline_rows.append({"container_id": cid, "baseline": bname, **metrics})

        cpu = prophet_cpu_metrics(result)
        cpu_rows.append({"container_id": cid, **cpu})

        if compute_recovery and manifest_val is not None:
            man_c = manifest_val[
                (manifest_val["container_id"] == cid)
                & (manifest_val["split"] == "val")
            ]
            if condition_id is not None and "condition_id" in manifest_val.columns:
                man_c = man_c[man_c["condition_id"] == condition_id]
            rec = synthetic_recovery_metrics(
                result,
                man_c,
                control_val=control_val,
                control_train=control_train,
                b_train=train_df,
            )
            if rec:
                recovery_rows.append({"container_id": cid, **rec})

    ext_df = pd.DataFrame(ext_rows)
    cross_horizon = cross_container_horizon_correlation(per_container_res)

    return {
        "inference": inference,
        "extended_metrics": ext_df,
        "horizon_per_container": pd.concat(horizon_rows, ignore_index=True)
        if horizon_rows else pd.DataFrame(),
        "horizon_bands_per_container": pd.concat(band_rows, ignore_index=True)
        if band_rows else pd.DataFrame(),
        "cross_horizon": cross_horizon,
        "baselines": pd.DataFrame(baseline_rows),
        "recovery": pd.DataFrame(recovery_rows),
        "cpu_metrics": pd.DataFrame(cpu_rows),
    }
