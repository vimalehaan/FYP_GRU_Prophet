"""Frequency-domain diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import periodogram, welch

from utils.tma.fft_analysis import cohort_fft_summary, mean_psd


def frequency_profile(
    series_by_container: dict[str, np.ndarray],
    fs: float = 1.0,
) -> dict[str, Any]:
    """FFT/PSD summary for one representation."""
    fft_summary = cohort_fft_summary(series_by_container, fs=fs)
    freqs, psd_mean = mean_psd(series_by_container, fs=fs)
    rows = fft_summary.get("per_container", [])

    def _mean_key(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "dominant_freq_mean": _mean_key("dominant_freq"),
        "dominant_period_steps_mean": _mean_key("dominant_period_steps"),
        "daily_freq_power_fraction_mean": _mean_key("daily_freq_power_fraction"),
        "psd_freqs": freqs.tolist() if len(freqs) else [],
        "psd_mean": psd_mean.tolist() if len(psd_mean) else [],
        "per_container_fft": rows,
    }
