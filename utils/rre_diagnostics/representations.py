"""Causal residual representation transforms (train-fit, apply-only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils.hybrid_training import generate_prophet_residuals


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def compute_train_level_residuals(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prophet residuals on train period — identical to Hybrid training."""
    return generate_prophet_residuals(train_df)


def fit_representation_stats(
    residual_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fit global and per-container statistics on train-period level residuals."""
    r = residual_df["residual"].astype(float)
    global_mean = float(r.mean())
    global_std = float(r.std())
    if global_std < 1e-12:
        global_std = 1.0

    global_median = float(np.median(r))
    global_mad = _mad(r.values)
    if global_mad < 1e-12:
        global_mad = 1.0

    per_container: dict[str, dict[str, float]] = {}
    for cid, group in residual_df.groupby("container_id"):
        rv = group["residual"].astype(float)
        mu = float(rv.mean())
        sigma = float(rv.std())
        if sigma < 1e-12:
            sigma = 1.0
        per_container[str(cid)] = {"mean": mu, "std": sigma}

    delta_parts: list[np.ndarray] = []
    for _, group in residual_df.groupby("container_id"):
        rv = group.sort_values("time_stamp")["residual"].astype(float).values
        if len(rv) > 1:
            delta_parts.append(np.diff(rv))
    delta_all = np.concatenate(delta_parts) if delta_parts else np.array([0.0])
    delta_mean = float(np.mean(delta_all))
    delta_std = float(np.std(delta_all))
    if delta_std < 1e-12:
        delta_std = 1.0

    return {
        "global_mean": global_mean,
        "global_std": global_std,
        "global_median": global_median,
        "global_mad": global_mad,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "per_container": per_container,
    }


def transform_container_series(
    level_residuals: np.ndarray,
    representation_id: str,
    stats: dict[str, Any],
    container_id: str,
) -> np.ndarray:
    """Apply a representation transform to one container's level residual series."""
    r = np.asarray(level_residuals, dtype=float)
    if representation_id == "R0":
        return (r - stats["global_mean"]) / stats["global_std"]

    if representation_id == "R1":
        out = np.full_like(r, np.nan, dtype=float)
        if len(r) > 1:
            delta = np.diff(r)
            out[1:] = (delta - stats["delta_mean"]) / stats["delta_std"]
        return out

    if representation_id == "R2":
        cstats = stats["per_container"][container_id]
        return (r - cstats["mean"]) / cstats["std"]

    if representation_id == "R3":
        return (r - stats["global_median"]) / stats["global_mad"]

    raise ValueError(f"Unknown representation: {representation_id}")


def reconstruction_notes(representation_id: str) -> dict[str, str]:
    """Document inference/reconstruction complexity and leakage risk."""
    notes = {
        "R0": {
            "inference_complexity": "Low — direct level forecast",
            "reconstruction_complexity": "None — output is final scaled residual",
            "leakage_risk": "Low — global μ/σ from train only",
        },
        "R1": {
            "inference_complexity": "Medium — integrate Δ with train anchor r_T",
            "reconstruction_complexity": "Cumulative sum over 96-step horizon",
            "leakage_risk": "Low — Δ stats from train; anchor from train tail",
        },
        "R2": {
            "inference_complexity": "Low — per-container scaler required",
            "reconstruction_complexity": "Inverse per-container scaling",
            "leakage_risk": "Low — container μ/σ from train only",
        },
        "R3": {
            "inference_complexity": "Low — global robust stats",
            "reconstruction_complexity": "Inverse median/MAD transform",
            "leakage_risk": "Low — median/MAD from train only",
        },
    }
    return notes[representation_id]
