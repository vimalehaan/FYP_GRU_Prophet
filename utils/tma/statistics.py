"""Paired statistical tests for Temporal Memory Analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from utils.tma.periodicity import bootstrap_ci


def paired_day1_vs_beyond_test(
    within: list[float],
    beyond: list[float],
    bootstrap_n: int,
    bootstrap_seed: int,
    ci_level: float,
) -> dict[str, Any]:
    """
    Paired comparison of |ACF| memory within first 96 lags vs beyond.

    Uses Wilcoxon signed-rank on paired differences (beyond - within).
    """
    pairs = [
        (w, b)
        for w, b in zip(within, beyond, strict=False)
        if np.isfinite(w) and np.isfinite(b)
    ]
    if len(pairs) < 5:
        return {
            "n_pairs": len(pairs),
            "mean_within": float("nan"),
            "mean_beyond": float("nan"),
            "mean_difference_beyond_minus_within": float("nan"),
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_pvalue": float("nan"),
            "cohens_dz": float("nan"),
            "bootstrap_ci_difference_lo": float("nan"),
            "bootstrap_ci_difference_hi": float("nan"),
        }

    w_arr = np.array([p[0] for p in pairs])
    b_arr = np.array([p[1] for p in pairs])
    diff = b_arr - w_arr

    try:
        stat, pval = stats.wilcoxon(diff, alternative="two-sided")
    except ValueError:
        stat, pval = float("nan"), float("nan")

    dz = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff) > 0 else 0.0
    ci_lo, ci_hi = bootstrap_ci(diff, bootstrap_n, bootstrap_seed, ci_level)

    return {
        "n_pairs": len(pairs),
        "mean_within": float(np.mean(w_arr)),
        "mean_beyond": float(np.mean(b_arr)),
        "mean_difference_beyond_minus_within": float(np.mean(diff)),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_pvalue": float(pval),
        "cohens_dz": dz,
        "bootstrap_ci_difference_lo": ci_lo,
        "bootstrap_ci_difference_hi": ci_hi,
    }
