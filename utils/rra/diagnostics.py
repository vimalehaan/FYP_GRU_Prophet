"""Temporal-structure diagnostics for Ridge Residual Analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import periodogram

from utils.csrle.diagnostics import avg_abs_acf, avg_abs_pacf, ljung_box_reject_fraction


ACF_LAG_START = 1
ACF_LAG_END = 10
PACF_LAG_START = 1
PACF_LAG_END = 10
LJUNG_LAGS = (10, 20, 30)
LJUNG_PRIMARY_LAG = 20
LJUNG_ALPHA = 0.05
FFT_FS = 1.0  # 15-minute steps; frequency in cycles per step


def _finite_series(series: np.ndarray) -> np.ndarray:
    return series[np.isfinite(series)]


def ljung_box_per_container(
    series: np.ndarray,
    lags: tuple[int, ...] = LJUNG_LAGS,
    alpha: float = LJUNG_ALPHA,
) -> dict[str, float]:
    """Ljung–Box statistics and rejection flags for one residual series."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    series = _finite_series(series)
    out: dict[str, float] = {}
    for lag in lags:
        if len(series) <= lag + 1:
            out[f"lb_stat_lag_{lag}"] = float("nan")
            out[f"lb_p_lag_{lag}"] = float("nan")
            out[f"lb_reject_lag_{lag}"] = float("nan")
            continue
        lb = acorr_ljungbox(series, lags=[lag], return_df=True)
        pval = float(lb["lb_pvalue"].iloc[0])
        out[f"lb_stat_lag_{lag}"] = float(lb["lb_stat"].iloc[0])
        out[f"lb_p_lag_{lag}"] = pval
        out[f"lb_reject_lag_{lag}"] = float(pval < alpha)
    return out


def fft_dominant_frequency(
    series: np.ndarray,
    fs: float = FFT_FS,
) -> dict[str, float]:
    """Power spectrum summary for one residual series."""
    series = _finite_series(series)
    if len(series) < 8:
        return {
            "dominant_freq": float("nan"),
            "dominant_period_steps": float("nan"),
            "total_power": float("nan"),
            "peak_power": float("nan"),
        }
    freqs, power = periodogram(series, fs=fs)
    if len(power) <= 1:
        return {
            "dominant_freq": float("nan"),
            "dominant_period_steps": float("nan"),
            "total_power": float(np.sum(power)),
            "peak_power": float("nan"),
        }
    # Skip DC component at index 0
    idx = int(np.argmax(power[1:]) + 1)
    dom_freq = float(freqs[idx])
    period = 1.0 / dom_freq if dom_freq > 1e-12 else float("nan")
    return {
        "dominant_freq": dom_freq,
        "dominant_period_steps": period,
        "total_power": float(np.sum(power)),
        "peak_power": float(power[idx]),
    }


def container_temporal_diagnostics(
    container_id: str,
    prophet_series: np.ndarray,
    remaining_series: np.ndarray,
) -> dict[str, Any]:
    """Full temporal diagnostic row for Prophet vs Ridge-remaining residuals."""
    prophet = _finite_series(prophet_series)
    remaining = _finite_series(remaining_series)

    row: dict[str, Any] = {"container_id": container_id}
    row["prophet_avg_abs_acf_1_10"] = avg_abs_acf(prophet, ACF_LAG_START, ACF_LAG_END)
    row["remaining_avg_abs_acf_1_10"] = avg_abs_acf(
        remaining, ACF_LAG_START, ACF_LAG_END
    )
    row["prophet_avg_abs_pacf_1_10"] = avg_abs_pacf(
        prophet, PACF_LAG_START, PACF_LAG_END
    )
    row["remaining_avg_abs_pacf_1_10"] = avg_abs_pacf(
        remaining, PACF_LAG_START, PACF_LAG_END
    )

    row.update({f"prophet_{k}": v for k, v in ljung_box_per_container(prophet).items()})
    row.update({f"remaining_{k}": v for k, v in ljung_box_per_container(remaining).items()})

    prophet_fft = fft_dominant_frequency(prophet)
    remaining_fft = fft_dominant_frequency(remaining)
    for key, val in prophet_fft.items():
        row[f"prophet_{key}"] = val
    for key, val in remaining_fft.items():
        row[f"remaining_{key}"] = val

    row["acf_reduction"] = (
        row["prophet_avg_abs_acf_1_10"] - row["remaining_avg_abs_acf_1_10"]
    )
    row["pacf_reduction"] = (
        row["prophet_avg_abs_pacf_1_10"] - row["remaining_avg_abs_pacf_1_10"]
    )
    return row


def cohort_temporal_summary(diag_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate temporal metrics across containers."""
    def _mean(col: str) -> float:
        return float(diag_df[col].mean()) if col in diag_df.columns else float("nan")

    def _reject_frac(prefix: str, lag: int) -> float:
        col = f"{prefix}_lb_reject_lag_{lag}"
        if col not in diag_df.columns:
            return float("nan")
        return float(diag_df[col].mean())

    return {
        "n_containers": int(len(diag_df)),
        "prophet_mean_avg_abs_acf_1_10": _mean("prophet_avg_abs_acf_1_10"),
        "remaining_mean_avg_abs_acf_1_10": _mean("remaining_avg_abs_acf_1_10"),
        "prophet_mean_avg_abs_pacf_1_10": _mean("prophet_avg_abs_pacf_1_10"),
        "remaining_mean_avg_abs_pacf_1_10": _mean("remaining_avg_abs_pacf_1_10"),
        "mean_acf_reduction": _mean("acf_reduction"),
        "mean_pacf_reduction": _mean("pacf_reduction"),
        "prophet_ljung_reject_fraction_lag_20": _reject_frac("prophet", 20),
        "remaining_ljung_reject_fraction_lag_20": _reject_frac("remaining", 20),
        "ljung_reject_reduction_lag_20": (
            _reject_frac("prophet", 20) - _reject_frac("remaining", 20)
        ),
        "prophet_dominant_freq_mean": _mean("prophet_dominant_freq"),
        "remaining_dominant_freq_mean": _mean("remaining_dominant_freq"),
    }


def bootstrap_paired_mean_ci(
    prophet_values: np.ndarray,
    remaining_values: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 12345,
    ci: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI for mean difference (prophet − remaining) per container."""
    rng = np.random.default_rng(seed)
    diffs = prophet_values - remaining_values
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    n = len(diffs)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = diffs[rng.integers(0, n, size=n)]
        boot_means[i] = float(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.quantile(boot_means, alpha)),
        "ci_high": float(np.quantile(boot_means, 1.0 - alpha)),
    }


def white_noise_comparison(
    container_results: dict[str, dict[str, Any]],
    diag_df: pd.DataFrame,
    bootstrap_seed: int = 12345,
) -> dict[str, Any]:
    """Compare Prophet vs Ridge-remaining white-noise proximity."""
    prophet_acf = diag_df["prophet_avg_abs_acf_1_10"].values
    remaining_acf = diag_df["remaining_avg_abs_acf_1_10"].values
    prophet_pacf = diag_df["prophet_avg_abs_pacf_1_10"].values
    remaining_pacf = diag_df["remaining_avg_abs_pacf_1_10"].values

    prophet_by_cid = {
        cid: res["prophet_residual_scaled"]
        for cid, res in container_results.items()
    }
    remaining_by_cid = {
        cid: res["ridge_remaining_scaled"]
        for cid, res in container_results.items()
    }

    cohort = cohort_temporal_summary(diag_df)
    return {
        **cohort,
        "acf_bootstrap": bootstrap_paired_mean_ci(
            prophet_acf, remaining_acf, seed=bootstrap_seed
        ),
        "pacf_bootstrap": bootstrap_paired_mean_ci(
            prophet_pacf, remaining_pacf, seed=bootstrap_seed
        ),
        "prophet_ljung_reject_fraction": ljung_box_reject_fraction(
            prophet_by_cid, lag=LJUNG_PRIMARY_LAG, alpha=LJUNG_ALPHA
        ),
        "remaining_ljung_reject_fraction": ljung_box_reject_fraction(
            remaining_by_cid, lag=LJUNG_PRIMARY_LAG, alpha=LJUNG_ALPHA
        ),
    }


def select_case_study_containers(eval_df: pd.DataFrame) -> dict[str, str]:
    """Best / median / worst Ridge containers by validation Pearson r."""
    metric = "ridge_pearson_r"
    valid = eval_df.dropna(subset=[metric])
    best = str(valid.loc[valid[metric].idxmax(), "container_id"])
    worst = str(valid.loc[valid[metric].idxmin(), "container_id"])
    median = float(valid[metric].median())
    med_idx = (valid[metric] - median).abs().idxmin()
    med_cid = str(valid.loc[med_idx, "container_id"])
    return {"best": best, "median": med_cid, "worst": worst}
