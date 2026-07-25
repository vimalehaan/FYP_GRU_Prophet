"""CSRLE Stage 2 cohort summaries and paired bootstrap statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def cohort_summary(series: pd.Series) -> dict[str, float]:
    """Summarize a per-container metric across the cohort."""
    s = series.dropna()
    if s.empty:
        return {
            "n": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "iqr": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
        }
    q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "iqr": q75 - q25,
        "p10": float(s.quantile(0.10)),
        "p25": q25,
        "p75": q75,
        "p90": float(s.quantile(0.90)),
    }


def fraction_above(series: pd.Series, threshold: float) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float((s > threshold).mean())


def residual_cohort_summary(ext_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate residual-learning metrics with threshold fractions."""
    out: dict[str, Any] = {}
    for col in (
        "residual_mae_scaled",
        "residual_rmse_scaled",
        "residual_pearson_r",
        "residual_r2",
        "residual_std_ratio",
    ):
        out[col] = cohort_summary(ext_df[col])
    out["fraction_pearson_r_gt_0"] = fraction_above(ext_df["residual_pearson_r"], 0.0)
    out["fraction_pearson_r_gt_0_30"] = fraction_above(
        ext_df["residual_pearson_r"], 0.30,
    )
    out["fraction_r2_gt_0"] = fraction_above(ext_df["residual_r2"], 0.0)
    out["fraction_r2_gt_0_30"] = fraction_above(ext_df["residual_r2"], 0.30)
    return out


def paired_bootstrap_diff(
    left: pd.Series,
    right: pd.Series,
    n_bootstrap: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """
    Paired container-level bootstrap for left - right.

    Containers aligned on index (container_id).
    """
    aligned = pd.concat([left, right], axis=1, join="inner").dropna()
    if aligned.empty:
        return {
            "n_pairs": 0,
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "fraction_left_better": float("nan"),
        }

    diffs = aligned.iloc[:, 0].values - aligned.iloc[:, 1].values
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    fraction_left_better = float(np.mean(diffs > 0))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.mean(diffs[idx])

    alpha = (1.0 - ci) / 2.0
    ci_low, ci_high = np.quantile(boot, [alpha, 1.0 - alpha])

    return {
        "n_pairs": int(n),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "fraction_left_better": fraction_left_better,
    }
