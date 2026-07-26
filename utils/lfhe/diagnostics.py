"""LFHE diagnostic metrics — horizon variance and energy recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.csrle.stage2_stats import cohort_summary, paired_bootstrap_diff

# Frozen LFHE v1.1 horizon bands (1-indexed, inclusive)
HORIZON_VARIANCE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("h01_12", 1, 12),
    ("h13_24", 13, 24),
    ("h25_48", 25, 48),
    ("h49_72", 49, 72),
    ("h73_96", 73, 96),
)

SIGMA_FLOOR = 1e-8


def _pop_std(arr: np.ndarray) -> float:
    arr = np.ravel(arr)
    if len(arr) == 0:
        return float("nan")
    return float(np.std(arr, ddof=0))


def horizon_variance_per_container(
    per_container: dict[str, tuple[np.ndarray, np.ndarray]],
    arm: str,
) -> pd.DataFrame:
    """Per-container std ratio in each frozen horizon band."""
    rows: list[dict[str, Any]] = []
    for cid, (actual, pred) in per_container.items():
        for band_id, lo, hi in HORIZON_VARIANCE_BANDS:
            a = actual[lo - 1 : hi]
            p = pred[lo - 1 : hi]
            sigma_a = _pop_std(a)
            sigma_p = _pop_std(p)
            if sigma_a < SIGMA_FLOOR:
                ratio = float("nan")
                flagged = True
            else:
                ratio = sigma_p / sigma_a
                flagged = False
            rows.append(
                {
                    "container_id": cid,
                    "band": band_id,
                    "band_range": f"{lo}-{hi}",
                    "sigma_actual": sigma_a,
                    "sigma_predicted": sigma_p,
                    "std_ratio": ratio,
                    "flagged_low_sigma": flagged,
                    "arm": arm,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: pd.Series,
    n_bootstrap: int = 10_000,
    seed: int = 12_345,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for cohort mean."""
    v = values.dropna().values
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(v)
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = np.mean(v[idx])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def horizon_variance_cohort(pc_df: pd.DataFrame) -> pd.DataFrame:
    """Cohort mean, median, bootstrap CI per arm and band."""
    rows: list[dict[str, Any]] = []
    for (arm, band), grp in pc_df.groupby(["arm", "band"]):
        s = grp["std_ratio"]
        ci_lo, ci_hi = bootstrap_ci(s)
        rows.append(
            {
                "arm": arm,
                "band": band,
                "band_range": grp["band_range"].iloc[0],
                "n": int(s.notna().sum()),
                "mean_std_ratio": float(s.mean()),
                "median_std_ratio": float(s.median()),
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "n_flagged": int(grp["flagged_low_sigma"].sum()),
            }
        )
    return pd.DataFrame(rows)


def horizon_variance_paired_bootstrap(
    mse_pc: pd.DataFrame,
    da_pc: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 12_345,
) -> dict[str, Any]:
    """Paired DA-MSE minus MSE std ratio by band (descriptive)."""
    out: dict[str, Any] = {}
    for band_id, _, _ in HORIZON_VARIANCE_BANDS:
        left = mse_pc[mse_pc["band"] == band_id].set_index("container_id")["std_ratio"]
        right = da_pc[da_pc["band"] == band_id].set_index("container_id")["std_ratio"]
        # DA - MSE: left_better means DA > MSE when we use da - mse with left=da
        stats = paired_bootstrap_diff(right, left, n_bootstrap=n_bootstrap, seed=seed)
        out[band_id] = stats
    return out


def energy_recovery_per_container(
    per_container: dict[str, tuple[np.ndarray, np.ndarray]],
    arm: str,
) -> pd.DataFrame:
    """ERR = sum(pred²) / sum(actual²) per container."""
    rows: list[dict[str, Any]] = []
    for cid, (actual, pred) in per_container.items():
        e_a = float(np.sum(np.square(actual)))
        e_p = float(np.sum(np.square(pred)))
        err = e_p / e_a if e_a > SIGMA_FLOOR else float("nan")
        rows.append(
            {
                "container_id": cid,
                "energy_actual": e_a,
                "energy_predicted": e_p,
                "energy_recovery_ratio": err,
                "arm": arm,
            }
        )
    return pd.DataFrame(rows)


def energy_recovery_cohort(ec_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, grp in ec_df.groupby("arm"):
        s = grp["energy_recovery_ratio"]
        ci_lo, ci_hi = bootstrap_ci(s)
        rows.append(
            {
                "arm": arm,
                "n": int(s.notna().sum()),
                "mean_err": float(s.mean()),
                "median_err": float(s.median()),
                "ci_low": ci_lo,
                "ci_high": ci_hi,
            }
        )
    return pd.DataFrame(rows)


def energy_recovery_paired_bootstrap(
    mse_ec: pd.DataFrame,
    da_ec: pd.DataFrame,
    n_bootstrap: int = 10_000,
    seed: int = 12_345,
) -> dict[str, Any]:
    left = mse_ec.set_index("container_id")["energy_recovery_ratio"]
    right = da_ec.set_index("container_id")["energy_recovery_ratio"]
    return paired_bootstrap_diff(right, left, n_bootstrap=n_bootstrap, seed=seed)


def variance_collapse_interpretation(
    mse_ext: pd.DataFrame,
    da_ext: pd.DataFrame,
    mse_hc: pd.DataFrame,
    da_hc: pd.DataFrame,
    ridge_std_ratio: float = 0.220,
) -> dict[str, Any]:
    """Absolute / relative improvement and remaining gap (descriptive)."""
    mse_r = float(mse_ext["residual_std_ratio"].mean())
    da_r = float(da_ext["residual_std_ratio"].mean())
    mse_err = float(mse_ext.get("energy_recovery_ratio", pd.Series(dtype=float)).mean())
    # energy from separate table if merged
    abs_improvement = da_r - mse_r
    rel_improvement = abs_improvement / mse_r if mse_r > 1e-12 else float("nan")
    gap_to_ridge = ridge_std_ratio - da_r

    # Horizon monotonicity check on MSE arm
    mse_band_means = (
        mse_hc.sort_values("band")[["band", "mean_std_ratio"]]
        .set_index("band")["mean_std_ratio"]
        .tolist()
    )
    monotonic_decrease = all(
        mse_band_means[i] >= mse_band_means[i + 1]
        for i in range(len(mse_band_means) - 1)
        if not (np.isnan(mse_band_means[i]) or np.isnan(mse_band_means[i + 1]))
    )
    flat_bands = float(np.std(mse_band_means)) < 0.02 if mse_band_means else float("nan")

    return {
        "full_trajectory_std_ratio_mse": mse_r,
        "full_trajectory_std_ratio_da_mse": da_r,
        "absolute_std_ratio_improvement": abs_improvement,
        "relative_std_ratio_improvement_pct": rel_improvement * 100.0,
        "remaining_gap_to_ridge_reference": gap_to_ridge,
        "ridge_reference_std_ratio": ridge_std_ratio,
        "mse_horizon_band_means": dict(zip([b[0] for b in HORIZON_VARIANCE_BANDS], mse_band_means)),
        "da_horizon_band_means": dict(
            zip(
                [b[0] for b in HORIZON_VARIANCE_BANDS],
                da_hc.sort_values("band")["mean_std_ratio"].tolist(),
            )
        ),
        "mse_monotonic_decrease_across_bands": monotonic_decrease,
        "mse_approximately_flat_across_bands": flat_bands,
        "interpretation_note": (
            "Progressive horizon collapse if band means decrease toward longer horizons; "
            "approximately flat if band std < 0.02."
        ),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
