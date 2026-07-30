"""Candidate vs incumbent evaluation on future unseen origins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.adaptive_lifecycle.candidate_retraining import load_model_bundle
from utils.adaptive_lifecycle.config import AFMLFConfig, DeploymentRecommendation
from utils.adaptive_lifecycle.model_versioning import ModelVersionRecord
from utils.adaptive_lifecycle.monitoring import forecast_at_origin


@dataclass
class CandidateEvaluationResult:
    trigger_origin_index: int
    n_eval_origins: int
    incumbent_cohort_mae: float
    candidate_cohort_mae: float
    delta_mae: float
    deployment_recommendation: DeploymentRecommendation
    recommendation_correct: bool | None
    details: dict[str, Any]


def select_future_origins(
    origin_df: pd.DataFrame,
    trigger_origin_index: int,
    config: AFMLFConfig,
) -> pd.DataFrame:
    warmup = config.eval_warmup_origins * config.walk_forward_stride
    min_origin = trigger_origin_index + warmup
    eval_origins = origin_df[origin_df["origin_index"] >= min_origin].copy()
    if not eval_origins.empty:
        assert (eval_origins["origin_index"] > trigger_origin_index).all(), (
            "Candidate evaluation must use origins strictly after the retraining trigger"
        )
        assert (eval_origins["origin_index"] >= min_origin).all(), (
            "Candidate evaluation must respect eval warmup after trigger"
        )
    return eval_origins


def evaluate_candidate_vs_incumbent(
    candidate_record: ModelVersionRecord,
    incumbent_dir: Path,
    origin_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    container_ids: list[str],
    trigger_origin_index: int,
    config: AFMLFConfig,
) -> CandidateEvaluationResult:
    eval_origins = select_future_origins(origin_df, trigger_origin_index, config)
    if eval_origins.empty:
        return CandidateEvaluationResult(
            trigger_origin_index=trigger_origin_index,
            n_eval_origins=0,
            incumbent_cohort_mae=float("nan"),
            candidate_cohort_mae=float("nan"),
            delta_mae=float("nan"),
            deployment_recommendation=DeploymentRecommendation.NEEDS_INVESTIGATION,
            recommendation_correct=None,
            details={"reason": "no_future_origins"},
        )

    incumbent_model, inc_mean, inc_std = load_model_bundle(incumbent_dir)
    candidate_model, cand_mean, cand_std = load_model_bundle(Path(candidate_record.artifact_dir))

    inc_maes: list[float] = []
    cand_maes: list[float] = []
    from utils.adaptive_lifecycle.monitoring import build_container_series

    for _, row in eval_origins.iterrows():
        cid = str(row["container_id"])
        origin = int(row["origin_index"])
        series = build_container_series(train_df, val_df, cid)
        cutoff = origin - config.train_cutoff_buffer_steps
        assert cutoff >= config.min_history_steps
        inc = forecast_at_origin(series, origin, incumbent_model, inc_mean, inc_std, config)
        cand = forecast_at_origin(series, origin, candidate_model, cand_mean, cand_std, config)
        inc_maes.append(inc.hybrid_mae)
        cand_maes.append(cand.hybrid_mae)

    incumbent_mae = float(np.mean(inc_maes))
    candidate_mae = float(np.mean(cand_maes))
    delta = candidate_mae - incumbent_mae

    if candidate_mae + config.candidate_improvement_margin < incumbent_mae:
        rec = DeploymentRecommendation.DEPLOY_CANDIDATE
        correct = True
    elif candidate_mae > incumbent_mae:
        rec = DeploymentRecommendation.KEEP_CURRENT_MODEL
        correct = True
    else:
        rec = DeploymentRecommendation.KEEP_CURRENT_MODEL
        correct = None

    candidate_record.evaluation_summary = {
        "incumbent_cohort_mae": incumbent_mae,
        "candidate_cohort_mae": candidate_mae,
        "delta_mae": delta,
        "n_eval_origins": len(eval_origins),
    }
    candidate_record.deployment_recommendation = rec.value

    return CandidateEvaluationResult(
        trigger_origin_index=trigger_origin_index,
        n_eval_origins=len(eval_origins),
        incumbent_cohort_mae=incumbent_mae,
        candidate_cohort_mae=candidate_mae,
        delta_mae=delta,
        deployment_recommendation=rec,
        recommendation_correct=correct,
        details=dict(candidate_record.evaluation_summary),
    )
