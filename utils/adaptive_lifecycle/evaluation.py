"""AFMLF framework evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from utils.adaptive_lifecycle.config import DecisionOutput, DeploymentRecommendation


@dataclass
class FrameworkEvaluationSummary:
    confirmed_drift_events: int = 0
    false_alarm_events: int = 0
    detection_delay_origins: list[int] = field(default_factory=list)
    recommendation_precision: float | None = None
    deployment_success_rate: float | None = None
    prevented_bad_deployments: int = 0
    diagnostic_distribution: dict[str, int] = field(default_factory=dict)
    retraining_frequency: int = 0
    decision_latency_origins: list[int] = field(default_factory=list)
    recommendation_quality: list[dict[str, Any]] = field(default_factory=list)
    secondary_candidate_delta_mae: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_framework_evaluation(
    lifecycle_log: pd.DataFrame,
    decision_log: pd.DataFrame,
    diagnostic_log: pd.DataFrame,
    recommendation_log: pd.DataFrame,
) -> FrameworkEvaluationSummary:
    summary = FrameworkEvaluationSummary()

    if not diagnostic_log.empty:
        summary.diagnostic_distribution = (
            diagnostic_log["diagnostic_class"].value_counts().to_dict()
        )

    if not lifecycle_log.empty:
        confirmed = lifecycle_log[
            lifecycle_log["to_state"] == "drift_confirmed"
        ]
        summary.confirmed_drift_events = int(len(confirmed))

    if not decision_log.empty:
        retrains = decision_log[
            decision_log["decision"] == DecisionOutput.RECOMMEND_RETRAINING.value
        ]
        summary.retraining_frequency = int(len(retrains))
        deferred = decision_log[
            decision_log["decision"] == DecisionOutput.DO_NOT_RETRAIN.value
        ]
        summary.false_alarm_events = int(len(deferred))

    if not recommendation_log.empty:
        quality_rows = []
        correct = 0
        total = 0
        deploy_success = 0
        prevented = 0
        for _, row in recommendation_log.iterrows():
            rec = row.get("deployment_recommendation")
            correct_flag = row.get("recommendation_correct")
            if correct_flag is True:
                correct += 1
            if correct_flag is not None:
                total += 1
            if rec == DeploymentRecommendation.DEPLOY_CANDIDATE.value:
                deploy_success += 1
            if (
                rec == DeploymentRecommendation.KEEP_CURRENT_MODEL.value
                and row.get("delta_mae", 0) > 0
            ):
                prevented += 1
            quality_rows.append(row.to_dict())
            if pd.notna(row.get("delta_mae")):
                summary.secondary_candidate_delta_mae.append(float(row["delta_mae"]))
        summary.recommendation_quality = quality_rows
        summary.recommendation_precision = correct / total if total else None
        summary.deployment_success_rate = (
            deploy_success / len(recommendation_log) if len(recommendation_log) else None
        )
        summary.prevented_bad_deployments = prevented

    return summary
