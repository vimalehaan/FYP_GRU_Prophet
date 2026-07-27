"""Load frozen cohort and build representation series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.csrle.dataset import evaluable_container_ids, load_frozen_base_data
from utils.rre_diagnostics.representations import (
    compute_train_level_residuals,
    fit_representation_stats,
    transform_container_series,
)


def load_cohort_train_residuals(
    repo_root: Path,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    """Load 99-container train-period Prophet level residuals."""
    train_df, val_df, scalers, selected = load_frozen_base_data(repo_root)
    container_ids, skipped = evaluable_container_ids(
        selected, train_df, val_df, scalers
    )
    cohort_train = train_df[train_df["container_id"].isin(container_ids)].copy()
    residual_df = compute_train_level_residuals(cohort_train)
    stats = fit_representation_stats(residual_df)
    meta = {
        "container_ids": container_ids,
        "skipped": skipped,
        "n_timesteps": int(len(residual_df)),
        "stats": stats,
    }
    return container_ids, residual_df, meta


def build_representation_series(
    residual_df: pd.DataFrame,
    container_ids: list[str],
    stats: dict[str, Any],
    representation_id: str,
) -> dict[str, np.ndarray]:
    """Per-container transformed series on train period."""
    out: dict[str, np.ndarray] = {}
    for cid in container_ids:
        group = residual_df[residual_df["container_id"] == cid].sort_values(
            "time_stamp"
        )
        level = group["residual"].astype(float).values
        out[cid] = transform_container_series(
            level, representation_id, stats, cid
        )
    return out
