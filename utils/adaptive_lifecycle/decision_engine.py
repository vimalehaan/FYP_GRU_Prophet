"""AFMLF decision engine — maps diagnostics to retraining decisions."""

from __future__ import annotations

from dataclasses import dataclass

from utils.adaptive_lifecycle.config import DecisionOutput
from utils.adaptive_lifecycle.diagnostics import DiagnosticClass, DiagnosticResult


@dataclass
class DecisionResult:
    origin_index: int
    decision: DecisionOutput
    rationale: str
    cohort_diagnostic_confidence: float
    n_recommend: int
    n_investigate: int
    n_stable: int


def make_cohort_decision(
    origin_index: int,
    diagnostics: list[DiagnosticResult],
    cohort_confidence: float,
    global_trigger_fired: bool,
) -> DecisionResult:
    n_recommend = sum(
        1
        for d in diagnostics
        if d.diagnostic_class
        in {DiagnosticClass.GRU_STALE, DiagnosticClass.REGIME_CHANGE}
    )
    n_investigate = sum(
        1
        for d in diagnostics
        if d.diagnostic_class in {DiagnosticClass.INVESTIGATE, DiagnosticClass.ANOMALY}
    )
    n_stable = sum(
        1 for d in diagnostics if d.diagnostic_class == DiagnosticClass.STABLE
    )

    if not global_trigger_fired:
        return DecisionResult(
            origin_index=origin_index,
            decision=DecisionOutput.DO_NOT_RETRAIN,
            rationale="Global trigger policy not satisfied.",
            cohort_diagnostic_confidence=cohort_confidence,
            n_recommend=n_recommend,
            n_investigate=n_investigate,
            n_stable=n_stable,
        )

    if n_investigate > n_recommend and cohort_confidence < 0.2:
        return DecisionResult(
            origin_index=origin_index,
            decision=DecisionOutput.NEEDS_INVESTIGATION,
            rationale="Investigation-class diagnostics dominate cohort signal.",
            cohort_diagnostic_confidence=cohort_confidence,
            n_recommend=n_recommend,
            n_investigate=n_investigate,
            n_stable=n_stable,
        )

    if n_recommend > 0 and cohort_confidence >= 0.1:
        return DecisionResult(
            origin_index=origin_index,
            decision=DecisionOutput.RECOMMEND_RETRAINING,
            rationale="Diagnostic and trigger policies support offline candidate retraining.",
            cohort_diagnostic_confidence=cohort_confidence,
            n_recommend=n_recommend,
            n_investigate=n_investigate,
            n_stable=n_stable,
        )

    return DecisionResult(
        origin_index=origin_index,
        decision=DecisionOutput.DO_NOT_RETRAIN,
        rationale="Insufficient diagnostic confidence for retraining recommendation.",
        cohort_diagnostic_confidence=cohort_confidence,
        n_recommend=n_recommend,
        n_investigate=n_investigate,
        n_stable=n_stable,
    )
