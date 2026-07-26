"""FFT / PSD analysis for Temporal Memory Analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import periodogram, welch


def fft_summary(
    series: np.ndarray,
    fs: float = 1.0,
) -> dict[str, float]:
    """Dominant frequency and daily-cycle power for one series."""
    series = series[np.isfinite(series)]
    if len(series) < 8:
        return {
            "dominant_freq": float("nan"),
            "dominant_period_steps": float("nan"),
            "daily_freq_power_fraction": float("nan"),
            "peak_power": float("nan"),
            "total_power": float("nan"),
        }

    freqs, power = periodogram(series, fs=fs)
    if len(power) <= 1:
        return {
            "dominant_freq": float("nan"),
            "dominant_period_steps": float("nan"),
            "daily_freq_power_fraction": float("nan"),
            "peak_power": float("nan"),
            "total_power": float(np.sum(power)),
        }

    idx = int(np.argmax(power[1:]) + 1)
    dom_freq = float(freqs[idx])
    period = 1.0 / dom_freq if dom_freq > 1e-12 else float("nan")

    # Power near daily cycle (period ≈ 96 steps → f ≈ 1/96)
    daily_f = 1.0 / 96.0
    band = (freqs >= daily_f * 0.9) & (freqs <= daily_f * 1.1)
    daily_power = float(np.sum(power[band]))
    total = float(np.sum(power[1:])) if len(power) > 1 else float(np.sum(power))
    daily_frac = daily_power / total if total > 1e-12 else float("nan")

    return {
        "dominant_freq": dom_freq,
        "dominant_period_steps": period,
        "daily_freq_power_fraction": daily_frac,
        "peak_power": float(power[idx]),
        "total_power": float(np.sum(power)),
    }


def cohort_fft_summary(
    series_by_container: dict[str, np.ndarray],
    fs: float = 1.0,
) -> dict[str, Any]:
    """Aggregate FFT metrics across containers."""
    rows = []
    for cid, series in series_by_container.items():
        row = fft_summary(series, fs=fs)
        row["container_id"] = cid
        rows.append(row)
    return {"per_container": rows}


def mean_psd(
    series_by_container: dict[str, np.ndarray],
    fs: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cohort-mean Welch PSD on a common frequency grid."""
    psds: list[np.ndarray] = []
    ref_freqs: np.ndarray | None = None
    for series in series_by_container.values():
        series = series[np.isfinite(series)]
        if len(series) < 16:
            continue
        freqs, psd = welch(series, fs=fs, nperseg=min(256, len(series) // 2))
        if ref_freqs is None:
            ref_freqs = freqs
        elif len(freqs) != len(ref_freqs):
            continue
        psds.append(psd)
    if ref_freqs is None or not psds:
        return np.array([]), np.array([])
    return ref_freqs, np.nanmean(np.vstack(psds), axis=0)
