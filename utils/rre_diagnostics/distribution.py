"""Distribution diagnostics for residual representations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def pooled_finite_values(series_by_container: dict[str, np.ndarray]) -> np.ndarray:
    parts = [
        s[np.isfinite(s)]
        for s in series_by_container.values()
        if len(s[np.isfinite(s)]) > 0
    ]
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def shannon_entropy(values: np.ndarray, bins: int = 50) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    counts, _ = np.histogram(values, bins=bins, density=False)
    total = counts.sum()
    if total == 0:
        return float("nan")
    probs = counts[counts > 0] / total
    return float(stats.entropy(probs, base=2))


def distribution_profile(
    series_by_container: dict[str, np.ndarray],
    bins: int = 50,
) -> dict[str, Any]:
    """Cohort distribution statistics."""
    values = pooled_finite_values(series_by_container)
    if len(values) == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "variance": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "entropy_bits": float("nan"),
            "sparsity_frac_abs_lt_0_05": float("nan"),
            "sparsity_frac_abs_lt_0_01": float("nan"),
            "histogram_bins": [],
            "histogram_counts": [],
            "qq_theoretical": [],
            "qq_sample": [],
        }

    hist_counts, bin_edges = np.histogram(values, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_sample = min(5000, len(values))
    rng = np.random.default_rng(42)
    sample = rng.choice(values, size=n_sample, replace=False)
    sample.sort()
    theoretical = stats.norm.ppf(
        np.linspace(0.01, 0.99, n_sample),
        loc=float(np.mean(values)),
        scale=float(np.std(values)),
    )

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "variance": float(np.var(values)),
        "skewness": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
        "entropy_bits": shannon_entropy(values, bins=bins),
        "sparsity_frac_abs_lt_0_05": float(np.mean(np.abs(values) < 0.05)),
        "sparsity_frac_abs_lt_0_01": float(np.mean(np.abs(values) < 0.01)),
        "histogram_bins": bin_centers.tolist(),
        "histogram_counts": hist_counts.astype(int).tolist(),
        "qq_theoretical": theoretical.tolist(),
        "qq_sample": sample.tolist(),
    }
