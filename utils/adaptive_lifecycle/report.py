"""AFMLF final report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.adaptive_lifecycle.config import DecisionOutput, DeploymentRecommendation
from utils.adaptive_lifecycle.evaluation import FrameworkEvaluationSummary


def write_final_report(
    exp_dir: Path,
    config: dict[str, Any],
    framework_eval: FrameworkEvaluationSummary,
    answers: dict[str, str],
) -> None:
    reports_dir = exp_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment_dir": str(exp_dir),
        "config": config,
        "framework_evaluation": framework_eval.to_dict(),
        "research_answers": answers,
    }
    (reports_dir / "final_report.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# AFMLF Final Report",
        "",
        "## Framework evaluation (primary)",
        "",
    ]
    for key, value in framework_eval.to_dict().items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Research answers", ""])
    for question, answer in answers.items():
        lines.append(f"### {question}")
        lines.append(answer)
        lines.append("")
    (reports_dir / "final_report.md").write_text("\n".join(lines) + "\n")


def write_executive_summary(
    exp_dir: Path,
    framework_eval: FrameworkEvaluationSummary,
    origin_df: pd.DataFrame,
    lifecycle_df: pd.DataFrame,
    ph_df: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    registry_versions: list[dict[str, Any]],
    container_ids: list[str],
    failures: list[dict[str, Any]],
    official: bool,
) -> None:
    """Write AFMLF_FINAL_RESULTS.md — official thesis executive summary."""
    n_containers = len(container_ids)
    n_origins = int(len(origin_df))
    n_unique_origins = int(origin_df["origin_index"].nunique()) if not origin_df.empty else 0

    drift_suspected = int(
        len(lifecycle_df[lifecycle_df["to_state"] == "drift_suspected"])
    ) if not lifecycle_df.empty else 0
    drift_confirmed = int(framework_eval.confirmed_drift_events)
    ph_confirmed = 0
    if not lifecycle_df.empty and not ph_df.empty:
        confirmed = lifecycle_df[lifecycle_df["to_state"] == "drift_confirmed"]
        for _, tr in confirmed.iterrows():
            cid = tr.get("container_id")
            oidx = tr.get("origin_index")
            if cid and pd.notna(oidx):
                match = ph_df[
                    (ph_df["container_id"] == cid)
                    & (ph_df["origin_index"] == oidx)
                    & (ph_df["ph_alarm"] == True)  # noqa: E712
                ]
                if len(match):
                    ph_confirmed += 1

    retrain_recs = int(
        len(decision_df[decision_df["decision"] == DecisionOutput.RECOMMEND_RETRAINING.value])
    ) if not decision_df.empty else 0
    n_candidates = max(0, len(registry_versions) - 1)

    outperformed = 0
    if not recommendation_df.empty and "delta_mae" in recommendation_df.columns:
        outperformed = int((recommendation_df["delta_mae"] < 0).sum())

    n_deploy_recs = int(len(recommendation_df))
    prevented = int(framework_eval.prevented_bad_deployments)
    n_correct = sum(
        1 for r in framework_eval.recommendation_quality
        if r.get("recommendation_correct") is True
    )
    n_scored = sum(
        1 for r in framework_eval.recommendation_quality
        if r.get("recommendation_correct") is not None
    )

    diag_dist = framework_eval.diagnostic_distribution
    top_diag = (
        sorted(diag_dist.items(), key=lambda x: -x[1])[:5] if diag_dist else []
    )

    lines = [
        "# AFMLF Final Results — Executive Summary",
        "",
        f"**Experiment directory:** `{exp_dir.name}`  ",
        f"**Official thesis evaluation:** {'Yes' if official else 'No'}  ",
        f"**Monitoring failures skipped:** {len(failures)}  ",
        "",
        "---",
        "",
        "## Key metrics",
        "",
        f"1. **Containers evaluated:** {n_containers} (cohort); "
        f"{origin_df['container_id'].nunique() if not origin_df.empty else 0} with successful monitoring rows",
        f"2. **Walk-forward origins processed:** {n_origins} origin-container pairs "
        f"({n_unique_origins} unique origin indices)",
        f"3. **Drift events detected (suspected):** {drift_suspected} container-level transitions",
        f"4. **Drift events confirmed (incl. Page-Hinkley confirmatory path):** {drift_confirmed} "
        f"({ph_confirmed} with concurrent PH alarm at confirmation)",
        f"5. **Retraining recommendations issued:** {retrain_recs}",
        f"6. **Candidate models trained:** {n_candidates}",
        f"7. **Candidates outperforming incumbent (ΔMAE < 0):** {outperformed} of {n_deploy_recs}",
        f"8. **Deployment recommendations issued:** {n_deploy_recs}",
        f"9. **Bad deployments prevented:** {prevented}",
        f"10. **Recommendations marked correct:** {n_correct} of {n_scored} scored",
        f"11. **Most common diagnostic classifications:** "
        + (", ".join(f"{k} ({v})" for k, v in top_diag) if top_diag else "N/A"),
        "",
        "## Overall conclusions",
        "",
        _conclusions_paragraph(framework_eval, retrain_recs, n_candidates, prevented, drift_confirmed),
        "",
        "---",
        "",
        "## Research discussion",
        "",
        _research_discussion(framework_eval, drift_confirmed, retrain_recs, prevented, diag_dist),
        "",
        "---",
        "",
        "## Framework evaluation detail",
        "",
    ]
    for key, value in framework_eval.to_dict().items():
        if key != "recommendation_quality":
            lines.append(f"- **{key}:** {value}")
    lines.append("")
    if failures:
        lines.extend(["## Monitoring failures", ""])
        for f in failures[:30]:
            lines.append(f"- `{f['container_id']}` origin {f['origin_index']}: {f['error'][:120]}")
        if len(failures) > 30:
            lines.append(f"- … and {len(failures) - 30} more")
        lines.append("")

    (exp_dir / "reports" / "AFMLF_FINAL_RESULTS.md").write_text("\n".join(lines) + "\n")


def _conclusions_paragraph(
    fw: FrameworkEvaluationSummary,
    retrain_recs: int,
    n_candidates: int,
    prevented: int,
    drift_confirmed: int,
) -> str:
    prec = fw.recommendation_precision
    prec_s = f"{prec:.1%}" if prec is not None else "N/A"
    return (
        f"AFMLF completed a full walk-forward lifecycle simulation over the research cohort. "
        f"The framework recorded {drift_confirmed} confirmed drift transitions, issued {retrain_recs} "
        f"retraining recommendations, trained {n_candidates} offline candidate model(s), and produced "
        f"deployment guidance with recommendation precision {prec_s}. "
        f"{prevented} potentially harmful deployment(s) were avoided by recommending to keep the incumbent. "
        f"Primary evaluation targets recommendation quality and lifecycle governance — not raw forecast MAE improvement."
    )


def _research_discussion(
    fw: FrameworkEvaluationSummary,
    drift_confirmed: int,
    retrain_recs: int,
    prevented: int,
    diag_dist: dict[str, int],
) -> str:
    sections = [
        "### Monitoring\n"
        "AFMLF successfully simulated operational monitoring via leakage-free walk-forward Day-1 "
        "Hybrid and Prophet error tracking per container. Rolling metrics fed drift detection without "
        "modifying the frozen Hybrid architecture.",
        "### Concept drift detection\n"
        + (
            f"Per-container baseline thresholds produced {drift_confirmed} confirmed drift events. "
            "Detection reliability should be interpreted alongside false deferral counts and "
            "Page-Hinkley confirmatory signals in the full lifecycle log."
            if drift_confirmed
            else "Few or no drift confirmations occurred in this run; the framework still demonstrated "
            "stable monitoring and deferral behaviour — see diagnostic distribution for degradation patterns."
        ),
        "### Diagnostic layer\n"
        + (
            f"Prophet vs Hybrid diagnostics yielded: {diag_dist}. "
            "GRU_STALE and REGIME_CHANGE classes support explainable retraining rationale; "
            "INVESTIGATE/ANOMALY classes defer action when signals are ambiguous."
        ),
        "### Recommendation quality\n"
        + (
            f"Recommendation precision: {fw.recommendation_precision}. "
            f"Prevented bad deployments: {prevented}. "
            "This is the primary thesis contribution — whether lifecycle recommendations are trustworthy."
        ),
        "### Offline candidate retraining\n"
        f"{retrain_recs} offline retraining event(s) used the corrected methodology "
        "(full original train + accumulated val before trigger). This is appropriate for a global GRU "
        "that cannot be updated online; candidates were evaluated on future origins only.",
        "### Limitations\n"
        "- Historical simulation on Alibaba trace — not live production telemetry\n"
        "- Prophet refit per origin is computationally expensive\n"
        "- Cohort-global retrain is coarse when only a subset of containers drift\n"
        "- Human gate required for actual deployment promotion",
        "### Future extensions\n"
        "- Live forecast-service metric ingestion\n"
        "- Alerting on drift_confirmed / needs_investigation\n"
        "- Canary promotion workflow with human approval\n"
        "- Online learning explicitly out of scope — separate research track",
    ]
    return "\n\n".join(sections)


def default_research_answers(framework_eval: FrameworkEvaluationSummary) -> dict[str, str]:
    return {
        "Can concept drift be reliably detected?": (
            f"The framework recorded {framework_eval.confirmed_drift_events} confirmed drift events "
            f"with {framework_eval.false_alarm_events} deferred/no-retrain decisions for comparison."
        ),
        "Are per-container baselines superior to global thresholds?": (
            "AFMLF uses per-container frozen Hybrid baselines because cohort MAE spans orders "
            "of magnitude; a single global threshold would mis-classify low- and high-error containers."
        ),
        "Does the diagnostic layer distinguish GRU drift from workload drift?": (
            f"Diagnostic distribution: {framework_eval.diagnostic_distribution}. "
            "GRU_STALE vs REGIME_CHANGE classes drive retraining confidence separately from INVESTIGATE."
        ),
        "Is offline retraining appropriate?": (
            "Yes. The frozen global GRU and residual statistics require batch retraining; "
            "online weight updates were intentionally excluded."
        ),
        "How does AFMLF extend the production forecasting pipeline?": (
            "AFMLF adds monitoring, drift detection, diagnostic reasoning, candidate versioning, "
            "and human-in-the-loop deployment recommendations without modifying Hybrid architecture."
        ),
    }
