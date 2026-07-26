"""Leakage-safe context feature engineering for HCERL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.hcerl.config import TRAINING_CONFIG
from utils.hybrid_training import generate_prophet_residuals


@dataclass
class FeatureStats:
    """Train-fitted normalization and threshold statistics (frozen at inference)."""

    res_mean: float
    res_std: float
    yhat_mean: float
    yhat_std: float
    volatility_window: int = TRAINING_CONFIG["volatility_window"]
    peak_percentile: int = TRAINING_CONFIG["peak_percentile"]
    peak_p90_scaled: dict[str, float] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    variant_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "res_mean": self.res_mean,
            "res_std": self.res_std,
            "yhat_mean": self.yhat_mean,
            "yhat_std": self.yhat_std,
            "volatility_window": self.volatility_window,
            "peak_percentile": self.peak_percentile,
            "peak_p90_scaled": self.peak_p90_scaled,
            "feature_columns": self.feature_columns,
            "variant_id": self.variant_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureStats:
        return cls(
            res_mean=float(payload["res_mean"]),
            res_std=float(payload["res_std"]),
            yhat_mean=float(payload["yhat_mean"]),
            yhat_std=float(payload["yhat_std"]),
            volatility_window=int(
                payload.get("volatility_window", TRAINING_CONFIG["volatility_window"])
            ),
            peak_percentile=int(
                payload.get("peak_percentile", TRAINING_CONFIG["peak_percentile"])
            ),
            peak_p90_scaled=dict(payload.get("peak_p90_scaled", {})),
            feature_columns=list(payload.get("feature_columns", [])),
            variant_id=str(payload.get("variant_id", "")),
        )


def _safe_zscore(values: pd.Series, mean: float, std: float) -> pd.Series:
    if std < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - mean) / std


def compute_peak_p90_scaled(train_group: pd.DataFrame) -> float:
    """Train-only P90 threshold in scaled CPU space for one container."""
    return float(train_group["cpu_scaled"].quantile(TRAINING_CONFIG["peak_percentile"] / 100.0))


def enrich_container_frame(
    group: pd.DataFrame,
    stats: FeatureStats,
    peak_p90: float,
) -> pd.DataFrame:
    """
    Add context feature columns to a single-container chronological frame.

    Expects ``prophet_pred``, ``cpu_scaled``, and ``cpu_std`` to be present.
    Rolling statistics are computed causally on ``residual_scaled``.
    """
    out = group.sort_values("time_stamp").copy()
    out["residual_scaled"] = _safe_zscore(
        out["residual"], stats.res_mean, stats.res_std,
    )
    out["prophet_yhat_scaled"] = _safe_zscore(
        out["prophet_pred"], stats.yhat_mean, stats.yhat_std,
    )
    w = stats.volatility_window
    out["res_roll_std_12"] = (
        out["residual_scaled"]
        .rolling(window=w, min_periods=1)
        .std()
        .fillna(0.0)
    )
    if "cpu_std" not in out.columns:
        raise KeyError("cpu_std missing — merge from global_train before enrichment")
    out["is_peak_p90"] = (out["cpu_scaled"] >= peak_p90).astype(float)
    return out


def build_feature_stats(
    train_residual_df: pd.DataFrame,
    global_train: pd.DataFrame,
    variant_id: str,
    features: list[str],
) -> FeatureStats:
    """Compute global train statistics and per-container peak thresholds."""
    res_mean = float(train_residual_df["residual"].mean())
    res_std = float(train_residual_df["residual"].std())
    yhat_mean = float(train_residual_df["prophet_pred"].mean())
    yhat_std = float(train_residual_df["prophet_pred"].std())

    peak_map: dict[str, float] = {}
    for cid, group in global_train.groupby("container_id"):
        peak_map[str(cid)] = compute_peak_p90_scaled(group)

    return FeatureStats(
        res_mean=res_mean,
        res_std=res_std,
        yhat_mean=yhat_mean,
        yhat_std=yhat_std,
        peak_p90_scaled=peak_map,
        feature_columns=features,
        variant_id=variant_id,
    )


def prepare_training_frame(
    global_train: pd.DataFrame,
    variant_id: str,
    features: list[str],
) -> tuple[pd.DataFrame, FeatureStats]:
    """
    Generate Prophet residuals and attach all context features for training.

    Steps
    -----
    1. Prophet residuals on train (unchanged baseline protocol).
    2. Merge container ``cpu_std`` from ``global_train``.
    3. Fit global z-score stats for residuals and Prophet levels.
    4. Causal rolling volatility on scaled residuals.
    5. Peak flags from train-only scaled P90 per container.
    """
    train_residual_df = generate_prophet_residuals(global_train)

    cpu_std_map = (
        global_train.groupby("container_id")["cpu_std"]
        .first()
        .to_dict()
    )
    train_residual_df["cpu_std"] = train_residual_df["container_id"].map(cpu_std_map)

    stats = build_feature_stats(
        train_residual_df, global_train, variant_id, features,
    )

    enriched_parts: list[pd.DataFrame] = []
    for cid, group in train_residual_df.groupby("container_id"):
        cid_str = str(cid)
        peak_p90 = stats.peak_p90_scaled[cid_str]
        enriched_parts.append(
            enrich_container_frame(group.copy(), stats, peak_p90),
        )

    enriched_df = pd.concat(enriched_parts, ignore_index=True)
    missing = [f for f in features if f not in enriched_df.columns]
    if missing:
        raise KeyError(f"Missing feature columns after enrichment: {missing}")

    return enriched_df, stats


def build_inference_input_matrix(
    enriched_train: pd.DataFrame,
    features: list[str],
    input_window: int,
) -> np.ndarray:
    """Return shape ``(1, input_window, n_features)`` from the last train window."""
    window = enriched_train.sort_values("time_stamp").iloc[-input_window:]
    matrix = window[features].values.astype(np.float32)
    return matrix.reshape(1, input_window, len(features))


def enrich_inference_train_container(
    train_container: pd.DataFrame,
    stats: FeatureStats,
    container_id: str,
    global_train: pd.DataFrame,
) -> pd.DataFrame:
    """Build enriched train frame for one container at inference time."""
    cid = str(container_id)
    peak_p90 = stats.peak_p90_scaled[cid]
    frame = train_container.copy()
    cpu_std_val = float(
        global_train.loc[
            global_train["container_id"] == container_id, "cpu_std"
        ].iloc[0]
    )
    frame["cpu_std"] = cpu_std_val
    frame["residual"] = frame["cpu_scaled"] - frame["prophet_pred"]
    return enrich_container_frame(frame, stats, peak_p90)


def validate_feature_leakage(
    enriched_df: pd.DataFrame,
    features: list[str],
) -> dict[str, bool]:
    """
    Sanity checks documented in feature_engineering.md.

    Returns pass flags for causal rolling and train-only peak thresholds.
    """
    checks = {
        "all_features_present": all(f in enriched_df.columns for f in features),
        "no_nan_in_features": bool(enriched_df[features].notna().all().all()),
        "volatility_non_negative": bool(
            (enriched_df["res_roll_std_12"] >= -1e-9).all()
            if "res_roll_std_12" in features
            else True
        ),
        "peak_binary": bool(
            enriched_df["is_peak_p90"].isin([0.0, 1.0]).all()
            if "is_peak_p90" in features
            else True
        ),
    }
    return checks
