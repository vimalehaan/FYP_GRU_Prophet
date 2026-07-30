"""Rolling operational metrics from walk-forward origins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class OriginMetrics:
    container_id: str
    origin_index: int
    origin_timestamp: str
    hybrid_mae: float
    hybrid_rmse: float
    prophet_mae: float
    prophet_rmse: float
    n_history_steps: int


def compute_day1_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    return mae, rmse


def append_rolling_metrics(
    origin_df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Add rolling MAE/RMSE columns per container."""
    out = origin_df.sort_values(["container_id", "origin_index"]).copy()
    for col in ("hybrid_mae", "hybrid_rmse", "prophet_mae", "prophet_rmse"):
        roll_col = f"rolling_{col}"
        out[roll_col] = (
            out.groupby("container_id")[col]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
    return out
