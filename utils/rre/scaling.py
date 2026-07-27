"""Train-fit residual scaling for R0 and R3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ScalingStats:
    """Causal scaling statistics fit on train-period level residuals only."""

    variant_id: str
    method: str
    center: float
    divisor: float

    def transform(self, values: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return (arr - self.center) / self.divisor

    def inverse(self, values: np.ndarray | pd.Series) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr * self.divisor + self.center

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScalingStats:
        return cls(
            variant_id=str(payload["variant_id"]),
            method=str(payload["method"]),
            center=float(payload["center"]),
            divisor=float(payload.get("divisor", payload.get("scale", 1.0))),
        )


def _mad(values: np.ndarray) -> float:
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def fit_scaling_stats(
    residual_df: pd.DataFrame,
    variant_id: str,
) -> ScalingStats:
    """Fit global scaling on pooled train residuals."""
    r = residual_df["residual"].astype(float).values
    if variant_id == "R0":
        center = float(np.mean(r))
        scale = float(np.std(r))
        if scale < 1e-12:
            scale = 1.0
        return ScalingStats(
            variant_id="R0",
            method="global_zscore",
            center=center,
            divisor=scale,
        )
    if variant_id == "R3":
        center = float(np.median(r))
        scale = _mad(r)
        if scale < 1e-12:
            scale = 1.0
        return ScalingStats(
            variant_id="R3",
            method="median_mad",
            center=center,
            divisor=scale,
        )
    raise ValueError(f"Unknown variant: {variant_id}")


def apply_scaling(
    residual_df: pd.DataFrame,
    stats: ScalingStats,
) -> pd.DataFrame:
    """Append ``residual_scaled`` using fitted stats."""
    out = residual_df.copy()
    out["residual_scaled"] = stats.transform(out["residual"])
    return out
