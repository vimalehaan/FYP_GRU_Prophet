"""AFMLF lifecycle state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LifecycleState(str, Enum):
    STABLE = "stable"
    WARNING = "warning"
    DRIFT_SUSPECTED = "drift_suspected"
    DRIFT_CONFIRMED = "drift_confirmed"
    DIAGNOSTIC_ANALYSIS = "diagnostic_analysis"
    DECISION = "decision"
    CANDIDATE_RETRAINING = "candidate_retraining"
    CANDIDATE_EVALUATION = "candidate_evaluation"
    DEPLOYMENT_RECOMMENDATION = "deployment_recommendation"
    COOLDOWN = "cooldown"


@dataclass
class LifecycleTransition:
    from_state: LifecycleState
    to_state: LifecycleState
    origin_index: int
    container_id: str | None
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerLifecycleState:
    container_id: str
    state: LifecycleState = LifecycleState.STABLE
    consecutive_breaches: int = 0
    ph_alarm: bool = False
    last_diagnostic_class: str = "STABLE"
    history: list[LifecycleTransition] = field(default_factory=list)

    def transition(
        self,
        new_state: LifecycleState,
        origin_index: int,
        reason: str,
        **metadata: Any,
    ) -> None:
        if new_state == self.state:
            return
        self.history.append(
            LifecycleTransition(
                from_state=self.state,
                to_state=new_state,
                origin_index=origin_index,
                container_id=self.container_id,
                reason=reason,
                metadata=dict(metadata),
            )
        )
        self.state = new_state


@dataclass
class CohortLifecycleState:
    state: LifecycleState = LifecycleState.STABLE
    cooldown_remaining: int = 0
    retrain_count: int = 0
    history: list[LifecycleTransition] = field(default_factory=list)

    def transition(
        self,
        new_state: LifecycleState,
        origin_index: int,
        reason: str,
        **metadata: Any,
    ) -> None:
        if new_state == self.state:
            return
        self.history.append(
            LifecycleTransition(
                from_state=self.state,
                to_state=new_state,
                origin_index=origin_index,
                container_id=None,
                reason=reason,
                metadata=dict(metadata),
            )
        )
        self.state = new_state


def apply_container_drift_progression(
    container: ContainerLifecycleState,
    origin_index: int,
    threshold_breach: bool,
    ph_confirm: bool,
    page_hinkley_mode: str,
) -> None:
    """Advance container lifecycle based on detection signals."""
    if container.state in {
        LifecycleState.CANDIDATE_RETRAINING,
        LifecycleState.CANDIDATE_EVALUATION,
        LifecycleState.DEPLOYMENT_RECOMMENDATION,
        LifecycleState.COOLDOWN,
    }:
        return

    if not threshold_breach:
        container.consecutive_breaches = 0
        if container.state in {
            LifecycleState.WARNING,
            LifecycleState.DRIFT_SUSPECTED,
        }:
            container.transition(
                LifecycleState.STABLE,
                origin_index,
                "metrics_normalized",
            )
        return

    container.consecutive_breaches += 1

    if container.state == LifecycleState.STABLE:
        container.transition(
            LifecycleState.WARNING,
            origin_index,
            "threshold_breach",
        )

    if container.consecutive_breaches < 2:
        return

    if container.state in {LifecycleState.STABLE, LifecycleState.WARNING}:
        container.transition(
            LifecycleState.DRIFT_SUSPECTED,
            origin_index,
            "rolling_persistence",
        )

    if container.state == LifecycleState.DRIFT_SUSPECTED:
        if page_hinkley_mode == "off":
            confirm = True
        elif page_hinkley_mode == "mandatory":
            confirm = ph_confirm
        else:
            confirm = ph_confirm
        if confirm:
            container.transition(
                LifecycleState.DRIFT_CONFIRMED,
                origin_index,
                "drift_confirmed",
            )
            container.transition(
                LifecycleState.DIAGNOSTIC_ANALYSIS,
                origin_index,
                "enter_diagnostics",
            )
