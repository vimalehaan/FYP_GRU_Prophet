"""Global retraining trigger policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.adaptive_lifecycle.config import AFMLFConfig
from utils.adaptive_lifecycle.drift_detection import DriftDetectionResult


@dataclass
class TriggerPolicyResult:
    origin_index: int
    triggered: bool
    drifted_fraction: float
    cohort_median_rolling_mae: float
    cohort_median_baseline_mae: float
    cooldown_active: bool
    reason: str


def evaluate_global_trigger(
    origin_index: int,
    detections: list[DriftDetectionResult],
    config: AFMLFConfig,
    cooldown_remaining: int,
) -> TriggerPolicyResult:
    if cooldown_remaining > 0:
        return TriggerPolicyResult(
            origin_index=origin_index,
            triggered=False,
            drifted_fraction=0.0,
            cohort_median_rolling_mae=0.0,
            cohort_median_baseline_mae=0.0,
            cooldown_active=True,
            reason="cooldown_active",
        )

    if not detections:
        return TriggerPolicyResult(
            origin_index=origin_index,
            triggered=False,
            drifted_fraction=0.0,
            cohort_median_rolling_mae=0.0,
            cohort_median_baseline_mae=0.0,
            cooldown_active=False,
            reason="no_detections",
        )

    drifted = [d for d in detections if d.threshold_breach]
    fraction = len(drifted) / len(detections)
    rolling = np.array([d.rolling_hybrid_mae for d in detections], dtype=float)
    baseline = np.array([d.baseline_hybrid_mae for d in detections], dtype=float)
    med_roll = float(np.median(rolling))
    med_base = float(np.median(baseline))
    med_deg = (med_roll - med_base) / med_base if med_base > 0 else 0.0

    by_fraction = fraction >= config.drifted_container_fraction
    by_median = med_deg >= config.cohort_median_degradation
    triggered = bool(by_fraction or by_median)
    reason = "fraction" if by_fraction else ("median" if by_median else "not_triggered")

    return TriggerPolicyResult(
        origin_index=origin_index,
        triggered=triggered,
        drifted_fraction=float(fraction),
        cohort_median_rolling_mae=med_roll,
        cohort_median_baseline_mae=med_base,
        cooldown_active=False,
        reason=reason,
    )
