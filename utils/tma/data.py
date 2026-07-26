"""Read-only data loading for Temporal Memory Analysis."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.csrle.dataset import (
    build_container_timeline,
    evaluable_container_ids,
    load_frozen_base_data,
)
from utils.tma.config import BASELINE_INFERENCE_CACHE, FOURIER_HARMONICS


def _fourier_design(
    step_indices: np.ndarray,
    period: int,
    n_harmonics: int,
) -> np.ndarray:
    """Fourier basis for daily seasonality (period in steps)."""
    t = step_indices.astype(float)
    cols = [np.ones(len(t)), t]
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * t / period
        cols.append(np.sin(angle))
        cols.append(np.cos(angle))
    return np.column_stack(cols)


def prophet_equivalent_residuals(
    y_train: np.ndarray,
    y_val: np.ndarray,
    period: int = 96,
    n_harmonics: int = FOURIER_HARMONICS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read-only Prophet-equivalent residuals without Prophet.fit().

    Fits additive daily Fourier seasonality + linear trend on train CPU only
    (OLS), then forms residuals on train and validation segments.
    """
    train_idx = np.arange(len(y_train))
    x_train = _fourier_design(train_idx, period, n_harmonics)
    coef, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    train_fit = x_train @ coef

    val_idx = np.arange(len(y_train), len(y_train) + len(y_val))
    x_val = _fourier_design(val_idx, period, n_harmonics)
    val_fit = x_val @ coef

    train_res = y_train - train_fit
    val_res = y_val - val_fit
    full_res = np.concatenate([train_res, val_res])
    return train_res, val_res, full_res


def load_hybrid_post_forecast_residuals(
    repo_root: Path,
    container_ids: list[str],
    cache_rel_path: str = BASELINE_INFERENCE_CACHE,
) -> dict[str, np.ndarray]:
    """Day-1 hybrid post-forecast residuals from frozen inference cache."""
    cache_path = repo_root / cache_rel_path
    with cache_path.open("rb") as handle:
        cache: dict[str, dict[str, np.ndarray]] = pickle.load(handle)

    out: dict[str, np.ndarray] = {}
    for cid in container_ids:
        entry = cache[cid]
        actual = np.ravel(entry["actual_day1_real"])
        pred = np.ravel(entry["day1_final_real"])
        out[cid] = actual - pred
    return out


def load_container_series_bundle(
    repo_root: Path,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """
    Load CPU, Prophet-equivalent, and hybrid post-forecast series per container.

    Returns evaluable container IDs and a dict keyed by container_id with:
    - y_full, y_train, y_val (real CPU %)
    - prophet_residual_full, prophet_residual_train, prophet_residual_val
    - hybrid_post_forecast_residual_day1 (96 steps, validation day-1)
    - train_len, val_len, full_len
    """
    train_df, val_df, scalers, selected = load_frozen_base_data(repo_root)
    container_ids, _ = evaluable_container_ids(
        selected, train_df, val_df, scalers
    )
    hybrid_day1 = load_hybrid_post_forecast_residuals(
        repo_root, container_ids
    )

    bundles: dict[str, dict[str, Any]] = {}
    for cid in container_ids:
        timeline = build_container_timeline(
            cid, train_df, val_df, scalers[cid]
        )
        y_train = timeline["y_train"]
        y_val = timeline["y_val"]
        y_full = timeline["y_base"]

        _, _, prophet_full = prophet_equivalent_residuals(y_train, y_val)
        train_res, val_res, _ = prophet_equivalent_residuals(y_train, y_val)

        bundles[cid] = {
            "container_id": cid,
            "y_full": y_full,
            "y_train": y_train,
            "y_val": y_val,
            "prophet_residual_full": prophet_full,
            "prophet_residual_train": train_res,
            "prophet_residual_val": val_res,
            "hybrid_post_forecast_residual_day1": hybrid_day1[cid],
            "train_len": timeline["train_len"],
            "val_len": timeline["val_len"],
            "full_len": len(y_full),
        }
    return container_ids, bundles


def series_by_type(
    bundle: dict[str, Any],
    series_type: str,
) -> np.ndarray:
    """Return analysis series for a container bundle."""
    mapping = {
        "cpu_original": "y_full",
        "prophet_residual": "prophet_residual_full",
        "hybrid_post_forecast_residual": "hybrid_post_forecast_residual_day1",
    }
    key = mapping[series_type]
    return np.asarray(bundle[key], dtype=float)
