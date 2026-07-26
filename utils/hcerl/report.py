"""HCERL final report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.hcerl.config import VARIANT_LABELS, VARIANT_ORDER


def _answer_research_questions(
    cohort_table: pd.DataFrame,
    ablation: dict[str, Any],
    best_variant: str,
) -> dict[str, str]:
    v0 = cohort_table.loc[
        cohort_table["variant_id"] == "v0_baseline"
    ].iloc[0]
    best = cohort_table.loc[
        cohort_table["variant_id"] == best_variant
    ].iloc[0]

    mae_improved = float(best["mean_day1_mae"]) < float(v0["mean_day1_mae"])
    r_improved = float(best["mean_residual_pearson_r"]) > float(
        v0["mean_residual_pearson_r"]
    )

    ranking = ablation["feature_contribution_ranking"]
    most = ranking[0]["added_feature"] if ranking else "none"
    least = ranking[-1]["added_feature"] if ranking else "none"

    v1_delta_r = next(
        (
            r["delta_pearson_r_mean"]
            for r in ranking
            if r["added_feature"] == "prophet_yhat_scaled"
        ),
        None,
    )
    vol_delta_r = next(
        (
            r["delta_pearson_r_mean"]
            for r in ranking
            if r["added_feature"] == "res_roll_std_12"
        ),
        None,
    )
    prophet_vs_stat = (
        "Prophet level context"
        if (v1_delta_r or 0) >= (vol_delta_r or 0)
        else "Statistical (volatility) context"
    )

    peak_row = next(
        (r for r in ranking if r["added_feature"] == "is_peak_p90"), None,
    )
    peak_helps = bool(
        peak_row and (peak_row.get("delta_pearson_r_mean") or 0) > 0
    )

    mae_delta_pct = (
        (float(best["mean_day1_mae"]) - float(v0["mean_day1_mae"]))
        / float(v0["mean_day1_mae"])
        * 100
    )
    practically_meaningful = abs(mae_delta_pct) >= 1.0 and mae_improved

    replace_baseline = (
        best_variant != "v0_baseline"
        and mae_improved
        and r_improved
        and practically_meaningful
    )

    return {
        "1_context_improves_forecasting": (
            "Yes — modest improvement"
            if mae_improved and r_improved
            else "Partial / inconclusive"
            if mae_improved or r_improved
            else "No clear improvement"
        ),
        "2_most_contributing_feature": most,
        "3_least_contributing_feature": least,
        "4_prophet_vs_statistical": prophet_vs_stat,
        "5_peak_features_help": "Yes" if peak_helps else "No / marginal",
        "6_practically_meaningful": (
            f"{'Yes' if practically_meaningful else 'No'} "
            f"(MAE Δ={mae_delta_pct:.2f}%)"
        ),
        "7_replace_baseline": (
            f"Recommend {best_variant}"
            if replace_baseline
            else "Keep baseline — simpler model preferred"
        ),
    }


def recommend_best_variant(
    cohort_table: pd.DataFrame,
    ablation: dict[str, Any],
) -> tuple[str, str]:
    """
    Select simplest variant that is not statistically worse on MAE.

    Prefer lower feature count if cohort MAE within tolerance of best.
    """
    best_mae = float(cohort_table["mean_day1_mae"].min())
    tolerance = 0.01  # pp MAE — minimal practical threshold

    for vid in VARIANT_ORDER:
        row = cohort_table.loc[cohort_table["variant_id"] == vid].iloc[0]
        if float(row["mean_day1_mae"]) <= best_mae + tolerance:
            rationale = (
                f"{VARIANT_LABELS[vid]} achieves near-best cohort MAE "
                f"({row['mean_day1_mae']:.3f}) with the fewest context channels "
                f"among variants within {tolerance} pp of the best."
            )
            return vid, rationale

    vid = str(cohort_table.loc[cohort_table["mean_day1_mae"].idxmin(), "variant_id"])
    return vid, f"Lowest cohort MAE at {VARIANT_LABELS[vid]}."


def write_final_report(
    exp_dir: Path,
    cohort_table: pd.DataFrame,
    ablation: dict[str, Any],
    case_studies: dict[str, Any],
    config: dict[str, Any],
) -> None:
    best_vid, rationale = recommend_best_variant(cohort_table, ablation)
    answers = _answer_research_questions(cohort_table, ablation, best_vid)

    report_json: dict[str, Any] = {
        "experiment": config.get("experiment_name"),
        "protocol_version": config.get("protocol_version"),
        "timestamp": config.get("experiment_timestamp"),
        "recommended_variant": best_vid,
        "recommendation_rationale": rationale,
        "explicit_answers": answers,
        "cohort_comparison": cohort_table.to_dict(orient="records"),
        "feature_contribution_ranking": ablation["feature_contribution_ranking"],
        "case_studies": case_studies.get("cases", {}),
    }

    reports_dir = exp_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "final_report.json", "w") as fh:
        json.dump(report_json, fh, indent=2)

    md_lines = [
        "# HCERL Final Report",
        "",
        f"**Protocol:** {config.get('protocol_version')}",
        f"**Timestamp:** {config.get('experiment_timestamp')}",
        "",
        "## Recommendation",
        "",
        f"**{best_vid}** — {rationale}",
        "",
        "## Explicit Answers",
        "",
    ]
    q_text = {
        "1_context_improves_forecasting": "Does contextual information improve Hybrid forecasting?",
        "2_most_contributing_feature": "Which contextual feature contributes the most?",
        "3_least_contributing_feature": "Which contextual feature contributes the least?",
        "4_prophet_vs_statistical": "Is Prophet forecast context more useful than statistical context?",
        "5_peak_features_help": "Do peak-aware features improve residual learning?",
        "6_practically_meaningful": "Are gains practically meaningful?",
        "7_replace_baseline": "Should Hybrid v2 replace the baseline model?",
    }
    for key, question in q_text.items():
        md_lines.append(f"### {question}")
        md_lines.append("")
        md_lines.append(f"**{answers[key]}**")
        md_lines.append("")

    md_lines.extend([
        "## Cohort Comparison",
        "",
        "```",
        cohort_table.to_string(index=False),
        "```",
        "",
        "## Feature Contribution Ranking",
        "",
        "```",
        pd.DataFrame(ablation["feature_contribution_ranking"]).to_string(index=False),
        "```",
    ])

    (reports_dir / "final_report.md").write_text("\n".join(md_lines))
