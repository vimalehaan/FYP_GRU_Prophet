"""Prophet vs Hybrid diagnostic classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from utils.adaptive_lifecycle.config import AFMLFConfig
from utils.adaptive_lifecycle.drift_detection import drift_threshold


class DiagnosticClass(str, Enum):
    GRU_STALE = "GRU_STALE"
    REGIME_CHANGE = "REGIME_CHANGE"
    INVESTIGATE = "INVESTIGATE"
    ANOMALY = "ANOMALY"
    STABLE = "STABLE"


@dataclass
class DiagnosticResult:
    container_id: str
    origin_index: int
    diagnostic_class: DiagnosticClass
    confidence: float
    recommended_action: str
    interpretation: str
    delta_hybrid: float
    delta_prophet: float


def _is_significant(
    delta: float,
    baseline: float,
    config: AFMLFConfig,
) -> bool:
    threshold = drift_threshold(
        baseline,
        config.diagnostic_relative_threshold,
        config.diagnostic_absolute_threshold,
    )
    rolling_proxy = baseline + delta
    return rolling_proxy > threshold


def classify_container(
    container_id: str,
    origin_index: int,
    rolling_hybrid_mae: float,
    rolling_prophet_mae: float,
    baseline_hybrid_mae: float,
    baseline_prophet_mae: float,
    config: AFMLFConfig,
) -> DiagnosticResult:
    delta_h = rolling_hybrid_mae - baseline_hybrid_mae
    delta_p = rolling_prophet_mae - baseline_prophet_mae
    h_up = _is_significant(delta_h, baseline_hybrid_mae, config)
    p_up = _is_significant(delta_p, baseline_prophet_mae, config)
    h_down = delta_h < -config.diagnostic_absolute_threshold
    p_up_only = p_up and not h_up

    if h_up and not p_up:
        return DiagnosticResult(
            container_id=container_id,
            origin_index=origin_index,
            diagnostic_class=DiagnosticClass.GRU_STALE,
            confidence=0.85,
            recommended_action="recommend_retraining",
            interpretation="Hybrid error increased while Prophet remained stable; GRU/residual path likely stale.",
            delta_hybrid=delta_h,
            delta_prophet=delta_p,
        )
    if h_up and p_up:
        return DiagnosticResult(
            container_id=container_id,
            origin_index=origin_index,
            diagnostic_class=DiagnosticClass.REGIME_CHANGE,
            confidence=0.75,
            recommended_action="recommend_retraining",
            interpretation="Both Hybrid and Prophet errors increased; underlying workload regime likely changed.",
            delta_hybrid=delta_h,
            delta_prophet=delta_p,
        )
    if p_up_only:
        return DiagnosticResult(
            container_id=container_id,
            origin_index=origin_index,
            diagnostic_class=DiagnosticClass.INVESTIGATE,
            confidence=0.35,
            recommended_action="needs_investigation",
            interpretation="Prophet degraded while Hybrid remained stable; investigate data or decomposition before GRU retrain.",
            delta_hybrid=delta_h,
            delta_prophet=delta_p,
        )
    if h_down and p_up:
        return DiagnosticResult(
            container_id=container_id,
            origin_index=origin_index,
            diagnostic_class=DiagnosticClass.ANOMALY,
            confidence=0.25,
            recommended_action="needs_investigation",
            interpretation="Unusual compensation pattern between Prophet and Hybrid components.",
            delta_hybrid=delta_h,
            delta_prophet=delta_p,
        )
    return DiagnosticResult(
        container_id=container_id,
        origin_index=origin_index,
        diagnostic_class=DiagnosticClass.STABLE,
        confidence=0.90,
        recommended_action="no_action",
        interpretation="No significant diagnostic degradation detected.",
        delta_hybrid=delta_h,
        delta_prophet=delta_p,
    )


def cohort_diagnostic_confidence(results: list[DiagnosticResult]) -> float:
    if not results:
        return 0.0
    weights = {
        DiagnosticClass.GRU_STALE: 0.5,
        DiagnosticClass.REGIME_CHANGE: 0.4,
        DiagnosticClass.INVESTIGATE: -0.3,
        DiagnosticClass.ANOMALY: -0.2,
        DiagnosticClass.STABLE: 0.0,
    }
    n = len(results)
    return float(
        sum(weights.get(r.diagnostic_class, 0.0) for r in results) / max(n, 1)
    )
