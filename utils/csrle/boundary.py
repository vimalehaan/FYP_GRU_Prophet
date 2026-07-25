"""Train-derived boundary-safety scaling for CSRLE (Protocol Amendment v2)."""

from __future__ import annotations

import numpy as np


def per_timestep_alpha_upper(
    y: np.ndarray,
    n_target: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per-timestep maximum α ≥ 0 such that y_t + α n_t stays within [0, 100].

    When y_t = 0 and n_t < 0, the upper bound is 0 (no positive scaling is feasible).
    Timesteps with |n_t| < eps are unconstrained (inf).
    """
    margins: list[float] = []
    for yt, nt in zip(y, n_target):
        if abs(nt) < eps:
            margins.append(np.inf)
            continue
        if nt > 0:
            margins.append((100.0 - yt) / nt)
        else:
            margins.append(yt / (-nt))
    return np.array(margins, dtype=float)


def compute_alpha_train(
    y_train: np.ndarray,
    n_target_train: np.ndarray,
) -> float:
    """
    Strict train-only boundary-safety factor α_c.

    α_c = min(1, min_t α_t^max) over training timesteps, including zeros.

    This is the largest uniform scale such that all training injections
    y_train + α_c n_target_train remain inside [0, 100] without clipping.
    """
    margins = per_timestep_alpha_upper(y_train, n_target_train)
    finite = margins[np.isfinite(margins)]
    if len(finite) == 0:
        return 1.0
    return float(min(1.0, np.min(finite)))


def apply_boundary_safe_injection(
    y_base: np.ndarray,
    n_target: np.ndarray,
    alpha: float,
    cpu_min: float = 0.0,
    cpu_max: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply frozen α scaling then physical clip safeguard.

    Returns n_effective (pre-clip), y_syn, clipped mask.
    """
    n_effective = alpha * n_target
    raw = y_base + n_effective
    y_syn = np.clip(raw, cpu_min, cpu_max)
    clipped = raw != y_syn
    return n_effective, y_syn, clipped


def kappa_effective(alpha: float, kappa_target: float) -> float:
    """Effective κ after boundary scaling."""
    return alpha * kappa_target


def alpha_distribution_summary(alphas: dict[str, float]) -> dict[str, float]:
    """Cohort summary statistics for α_c."""
    vals = np.array(list(alphas.values()), dtype=float)
    q1, q3 = np.quantile(vals, [0.25, 0.75])
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "q1": float(q1),
        "q3": float(q3),
        "fraction_alpha_eq_1": float(np.mean(vals >= 1.0 - 1e-9)),
        "fraction_alpha_lt_1": float(np.mean(vals < 1.0 - 1e-9)),
        "fraction_alpha_lt_0_75": float(np.mean(vals < 0.75)),
        "fraction_alpha_lt_0_50": float(np.mean(vals < 0.50)),
        "fraction_alpha_lt_0_25": float(np.mean(vals < 0.25)),
        "fraction_alpha_eq_0": float(np.mean(vals <= 1e-12)),
        "n_containers": int(len(vals)),
    }
