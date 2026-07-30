"""Per-container drift detection against frozen baselines."""

from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import pandas as pd

from utils.adaptive_lifecycle.config import AFMLFConfig


@dataclass
class DriftDetectionResult:
    container_id: str
    origin_index: int
    rolling_hybrid_mae: float
    baseline_hybrid_mae: float
    threshold_value: float
    threshold_breach: bool


def drift_threshold(
    baseline_mae: float,
    relative_threshold: float,
    absolute_threshold: float,
) -> float:
    return max(
        baseline_mae * (1.0 + relative_threshold),
        baseline_mae + absolute_threshold,
    )


def detect_container_drift(
    container_id: str,
    origin_index: int,
    rolling_hybrid_mae: float,
    baseline_hybrid_mae: float,
    config: AFMLFConfig,
) -> DriftDetectionResult:
    threshold = drift_threshold(
        baseline_hybrid_mae,
        config.relative_threshold,
        config.absolute_threshold,
    )
    return DriftDetectionResult(
        container_id=container_id,
        origin_index=origin_index,
        rolling_hybrid_mae=rolling_hybrid_mae,
        baseline_hybrid_mae=baseline_hybrid_mae,
        threshold_value=threshold,
        threshold_breach=bool(rolling_hybrid_mae > threshold),
    )


def load_hybrid_baselines(path: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        df = pd.read_csv(path)
    return df[["container_id", "day1_mae", "day1_rmse"]].rename(
        columns={"day1_mae": "baseline_hybrid_mae", "day1_rmse": "baseline_hybrid_rmse"}
    )
