"""Final report generation for Ridge Residual Analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_justification(white_noise: dict[str, Any]) -> dict[str, Any]:
    """
    Answer the four justification questions and overall Ridge→GRU recommendation.

    Uses pre-registered thresholds aligned with residual_pattern_analysis baselines.
    """
    prophet_acf = white_noise.get("prophet_mean_avg_abs_acf_1_10", float("nan"))
    remaining_acf = white_noise.get("remaining_mean_avg_abs_acf_1_10", float("nan"))
    acf_reduction = white_noise.get("mean_acf_reduction", float("nan"))
    prophet_lb = white_noise.get("prophet_ljung_reject_fraction_lag_20", float("nan"))
    remaining_lb = white_noise.get("remaining_ljung_reject_fraction_lag_20", float("nan"))
    lb_reduction = white_noise.get("ljung_reject_reduction_lag_20", float("nan"))

    q1_ridge_removes_structure = (
        acf_reduction > 0.01 or lb_reduction > 0.05
    )
    q2_closer_to_white_noise = (
        remaining_acf < prophet_acf and remaining_lb < prophet_lb
    )
    q3_still_rejects_white_noise = remaining_lb > 0.30
    q4_still_significant_acf = remaining_acf > 0.05

    # Nonlinear learner justified if structure remains after Ridge
    q5_nonlinear_justified = q3_still_rejects_white_noise or q4_still_significant_acf

    # Future Ridge→GRU architecture warranted only if nonlinear stage still needed
    q6_ridge_gru_future = q5_nonlinear_justified

    if q5_nonlinear_justified and q1_ridge_removes_structure:
        verdict = "partial_structure_remains"
    elif q5_nonlinear_justified:
        verdict = "structure_persists_minimal_ridge_gain"
    elif q1_ridge_removes_structure and not q3_still_rejects_white_noise:
        verdict = "ridge_sufficient_no_nonlinear_needed"
    else:
        verdict = "inconclusive"

    return {
        "q1_ridge_removes_most_temporal_dependence": q1_ridge_removes_structure,
        "q2_ridge_residual_closer_to_white_noise": q2_closer_to_white_noise,
        "q3_remaining_still_rejects_white_noise": q3_still_rejects_white_noise,
        "q4_remaining_still_significant_acf": q4_still_significant_acf,
        "q5_nonlinear_learner_justified": q5_nonlinear_justified,
        "q6_ridge_gru_architecture_future_recommended": q6_ridge_gru_future,
        "verdict": verdict,
        "prophet_mean_avg_abs_acf_1_10": prophet_acf,
        "remaining_mean_avg_abs_acf_1_10": remaining_acf,
        "prophet_ljung_reject_fraction_lag_20": prophet_lb,
        "remaining_ljung_reject_fraction_lag_20": remaining_lb,
    }


def build_final_report(
    config: dict[str, Any],
    eval_df: dict[str, Any],
    white_noise: dict[str, Any],
    justification: dict[str, Any],
    case_studies: dict[str, str],
) -> dict[str, Any]:
    """Assemble machine-readable final report."""
    return {
        "experiment_name": "ridge_residual_analysis",
        "protocol_version": config.get("protocol_version"),
        "timestamp": config.get("timestamp"),
        "condition": config.get("primary_condition"),
        "n_containers": eval_df.get("n_containers"),
        "ridge_training": config.get("ridge_configuration"),
        "cohort_ridge_performance": eval_df.get("ridge_performance"),
        "temporal_structure": {
            "prophet_mean_avg_abs_acf_1_10": white_noise.get(
                "prophet_mean_avg_abs_acf_1_10"
            ),
            "remaining_mean_avg_abs_acf_1_10": white_noise.get(
                "remaining_mean_avg_abs_acf_1_10"
            ),
            "mean_acf_reduction": white_noise.get("mean_acf_reduction"),
            "mean_pacf_reduction": white_noise.get("mean_pacf_reduction"),
            "prophet_ljung_reject_fraction_lag_20": white_noise.get(
                "prophet_ljung_reject_fraction_lag_20"
            ),
            "remaining_ljung_reject_fraction_lag_20": white_noise.get(
                "remaining_ljung_reject_fraction_lag_20"
            ),
            "acf_bootstrap_ci": white_noise.get("acf_bootstrap"),
            "pacf_bootstrap_ci": white_noise.get("pacf_bootstrap"),
        },
        "explicit_answers": {
            "1_temporal_structure_removed_by_ridge": {
                "acf_reduction": white_noise.get("mean_acf_reduction"),
                "pacf_reduction": white_noise.get("mean_pacf_reduction"),
                "ljung_reject_reduction_lag_20": white_noise.get(
                    "ljung_reject_reduction_lag_20"
                ),
            },
            "2_ridge_residual_closer_to_white_noise": justification[
                "q2_ridge_residual_closer_to_white_noise"
            ],
            "3_pct_containers_still_reject_white_noise": (
                white_noise.get("remaining_ljung_reject_fraction_lag_20", 0) * 100
            ),
            "4_remaining_has_significant_temporal_dependence": justification[
                "q4_remaining_still_significant_acf"
            ],
            "5_nonlinear_learner_justified": justification[
                "q5_nonlinear_learner_justified"
            ],
            "6_ridge_gru_future_recommended": justification[
                "q6_ridge_gru_architecture_future_recommended"
            ],
        },
        "justification": justification,
        "case_studies": case_studies,
        "verdict": justification["verdict"],
    }


def write_final_report_md(
    report: dict[str, Any],
    path: Path,
) -> None:
    """Write human-readable final report."""
    j = report["justification"]
    w = report["temporal_structure"]
    lines = [
        "# Ridge Residual Analysis — Final Report",
        "",
        f"**Timestamp:** `{report['timestamp']}`  ",
        f"**Condition:** {report['condition']}  ",
        f"**Containers:** {report['n_containers']}  ",
        f"**Verdict:** `{report['verdict']}`",
        "",
        "## Research question",
        "",
        "Does Ridge remove all predictable temporal structure from Prophet residuals,",
        "or does the remaining Ridge residual still justify a nonlinear learner (GRU)?",
        "",
        "## Ridge training",
        "",
        f"- Alpha: {report['ridge_training'].get('alpha')}",
        f"- Sequence: {report['ridge_training'].get('input_window')}→"
        f"{report['ridge_training'].get('forecast_horizon')}",
        f"- Internal split: {report['ridge_training'].get('internal_val_split')}",
        "",
        "## Cohort Ridge performance (validation)",
        "",
        f"- Mean Pearson r: {report['cohort_ridge_performance'].get('mean_pearson_r'):.4f}",
        f"- Mean MAE (scaled): {report['cohort_ridge_performance'].get('mean_mae'):.4f}",
        f"- Mean remaining std ratio: "
        f"{report['cohort_ridge_performance'].get('mean_remaining_std_ratio'):.4f}",
        "",
        "## Temporal structure comparison",
        "",
        "| Metric | Prophet residual | Ridge remaining | Reduction |",
        "|--------|------------------|-----------------|-----------|",
        f"| Mean \\|ACF\\| lags 1–10 | {w['prophet_mean_avg_abs_acf_1_10']:.4f} | "
        f"{w['remaining_mean_avg_abs_acf_1_10']:.4f} | {w['mean_acf_reduction']:.4f} |",
        f"| Ljung–Box reject fraction (lag 20) | "
        f"{w['prophet_ljung_reject_fraction_lag_20']*100:.1f}% | "
        f"{w['remaining_ljung_reject_fraction_lag_20']*100:.1f}% | "
        f"{(w['prophet_ljung_reject_fraction_lag_20'] - w['remaining_ljung_reject_fraction_lag_20'])*100:.1f} pp |",
        "",
        "## Explicit answers",
        "",
        "1. **How much temporal structure did Ridge remove?**  ",
        f"   Mean |ACF| reduction = {w['mean_acf_reduction']:.4f}; "
        f"mean |PACF| reduction = {w['mean_pacf_reduction']:.4f}.",
        "",
        "2. **Is the Ridge residual closer to white noise?**  ",
        f"   {'Yes' if j['q2_ridge_residual_closer_to_white_noise'] else 'No'} "
        f"(remaining |ACF| = {w['remaining_mean_avg_abs_acf_1_10']:.4f} vs "
        f"Prophet {w['prophet_mean_avg_abs_acf_1_10']:.4f}).",
        "",
        "3. **What percentage of containers still reject white noise?**  ",
        f"   {report['explicit_answers']['3_pct_containers_still_reject_white_noise']:.1f}% "
        f"(Ljung–Box lag 20, p<0.05).",
        "",
        "4. **Does the Ridge residual still contain statistically significant "
        "temporal dependence?**  ",
        f"   {'Yes' if j['q4_remaining_still_significant_acf'] else 'No'} "
        f"(cohort mean |ACF| lags 1–10 = {w['remaining_mean_avg_abs_acf_1_10']:.4f}).",
        "",
        "5. **Is there sufficient evidence to justify a nonlinear learner after Ridge?**  ",
        f"   {'Yes' if j['q5_nonlinear_learner_justified'] else 'No'}.",
        "",
        "6. **Should a Ridge→GRU architecture be implemented in a future experiment?**  ",
        f"   {'Yes — pending dedicated architecture experiment' if j['q6_ridge_gru_architecture_future_recommended'] else 'Not yet — Ridge may absorb most learnable structure'}.",
        "",
        "## Case studies",
        "",
        f"- Best Ridge: `{report['case_studies']['best']}`",
        f"- Median: `{report['case_studies']['median']}`",
        f"- Worst Ridge: `{report['case_studies']['worst']}`",
        "",
        "## Conclusion",
        "",
        _verdict_paragraph(report["verdict"], j),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _verdict_paragraph(verdict: str, j: dict[str, Any]) -> str:
    if verdict == "partial_structure_remains":
        return (
            "Ridge removes part of the temporal dependence in Prophet residuals, "
            "but a substantial fraction of containers still reject white-noise "
            "hypotheses and exhibit non-negligible |ACF|. A dedicated Ridge→GRU "
            "architecture experiment is scientifically justified as a follow-up."
        )
    if verdict == "ridge_sufficient_no_nonlinear_needed":
        return (
            "Ridge absorbs most detectable temporal structure; remaining residuals "
            "are closer to white noise. A Ridge→GRU stack is not strongly justified "
            "without new evidence of nonlinear residual structure."
        )
    if verdict == "structure_persists_minimal_ridge_gain":
        return (
            "Temporal structure persists after Ridge with limited ACF/Ljung reduction. "
            "A nonlinear learner may still be justified, but Ridge adds little beyond "
            "Prophet in this linear 96→96 configuration."
        )
    return (
        "Results are mixed under the pre-registered thresholds. Review per-container "
        "diagnostics and case studies before committing to a Ridge→GRU architecture."
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
