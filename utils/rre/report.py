"""Final RRE report and research-question answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def answer_research_questions(
    comparison: dict[str, Any],
    cohort_r0: dict[str, Any],
    cohort_r3: dict[str, Any],
    opt_r0: dict[str, Any],
    opt_r3: dict[str, Any],
) -> dict[str, Any]:
    """Explicit answers to the eight final research questions."""
    fm = comparison.get("forecast_metrics", {})
    om = comparison.get("optimization_metrics", {})
    mae = fm.get("day1_mae", {})
    pearson = fm.get("residual_pearson_r", {})

    mae_sig = mae.get("significant_wilcoxon_005", False)
    mae_improved = (mae.get("mean_delta", 0) or 0) < 0
    practical = comparison.get("practical_significance", {})

    opt_improved = sum(
        1 for v in om.values() if isinstance(v, dict) and v.get("R3_better")
    )
    opt_total = len(om)

    return {
        "q1_accuracy_improved": {
            "answer": bool(mae_improved),
            "evidence": (
                f"Mean day1 MAE delta (R3-R0)={mae.get('mean_delta', float('nan')):.4f}; "
                f"fraction improved={mae.get('fraction_improved', float('nan')):.2%}"
            ),
        },
        "q2_optimisation_improved": {
            "answer": opt_improved > opt_total / 2,
            "evidence": (
                f"R3 better on {opt_improved}/{opt_total} optimisation metrics; "
                f"best_epoch R0={opt_r0.get('best_epoch')} R3={opt_r3.get('best_epoch')}"
            ),
        },
        "q3_convergence_speed": {
            "answer": bool(om.get("convergence_speed", {}).get("R3_better", False)),
            "evidence": (
                f"Best epoch R0={opt_r0.get('best_epoch')}, R3={opt_r3.get('best_epoch')}"
            ),
        },
        "q4_generalisation": {
            "answer": bool(om.get("generalisation_gap", {}).get("R3_better", False)),
            "evidence": (
                f"Generalisation gap R0={opt_r0.get('generalisation_gap'):.4f}, "
                f"R3={opt_r3.get('generalisation_gap'):.4f}"
            ),
        },
        "q5_statistically_significant": {
            "answer": bool(mae_sig),
            "evidence": (
                f"Wilcoxon p={mae.get('wilcoxon', {}).get('pvalue', float('nan'))}; "
                f"Pearson Wilcoxon sig={pearson.get('significant_wilcoxon_005', False)}"
            ),
        },
        "q6_practically_meaningful": {
            "answer": bool(practical.get("practically_meaningful", False)),
            "evidence": (
                f"Mean MAE improvement={practical.get('mean_mae_improvement', float('nan')):.4f} "
                f"(threshold {practical.get('mae_threshold_pct')})"
            ),
        },
        "q7_replace_baseline_scaling": {
            "answer": bool(mae_improved and mae_sig and practical.get("practically_meaningful")),
            "evidence": (
                "Requires accuracy improvement that is both statistically and practically "
                "significant; otherwise retain R0 global z-score as Hybrid standard."
            ),
        },
        "q8_optimisation_vs_temporal_information": {
            "answer": (
                "Stage 0 showed R3 preserves identical temporal structure to R0. "
                "Any difference in this experiment reflects optimisation/numerical "
                "conditioning, not new temporal information."
            ),
            "evidence": (
                f"Residual Pearson R0 mean={cohort_r0.get('residual_pearson_r', {}).get('mean', 'NA')}, "
                f"R3 mean={cohort_r3.get('residual_pearson_r', {}).get('mean', 'NA')}"
            ),
        },
    }


def build_final_report_md(
    config: dict[str, Any],
    comparison: dict[str, Any],
    answers: dict[str, Any],
    cohort_r0: dict[str, Any],
    cohort_r3: dict[str, Any],
) -> str:
    lines = [
        "# Residual Scaling Experiment (RRE v1.0)",
        "",
        f"**Protocol:** {config.get('protocol_version')}",
        f"**Timestamp:** {config.get('experiment_timestamp')}",
        "",
        "## Research question",
        "",
        config.get("research_question", ""),
        "",
        "## Stage 0 context",
        "",
        "Stage 0 rejected velocity (R1) because it removed temporal structure. "
        "R3 preserves the same ACF/PACF as R0 while improving numerical robustness "
        "(lower sparsity in scaled units). This experiment tests whether that "
        "improves GRU optimisation — not whether a better temporal representation exists.",
        "",
        "## Cohort summaries",
        "",
        f"- R0 mean day1 MAE: {cohort_r0.get('day1_mae', {}).get('mean', 'NA')}",
        f"- R3 mean day1 MAE: {cohort_r3.get('day1_mae', {}).get('mean', 'NA')}",
        "",
        "## Research question answers",
        "",
    ]
    for key, block in answers.items():
        lines.append(f"### {key}")
        lines.append(f"- **Answer:** {block.get('answer')}")
        lines.append(f"- **Evidence:** {block.get('evidence')}")
        lines.append("")

    mae = comparison.get("forecast_metrics", {}).get("day1_mae", {})
    lines.extend([
        "## Statistical test (day1 MAE, R3 vs R0)",
        "",
        f"- Mean delta: {mae.get('mean_delta', 'NA')}",
        f"- Wilcoxon p: {mae.get('wilcoxon', {}).get('pvalue', 'NA')}",
        f"- Bootstrap 95% CI: {mae.get('bootstrap', {})}",
        "",
    ])
    return "\n".join(lines)
