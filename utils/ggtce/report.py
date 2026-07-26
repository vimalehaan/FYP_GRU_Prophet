"""GGTCE final report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.ggtce.config import VARIANT_LABELS, VARIANT_ORDER


def _pick_best_variant(cohort_table: pd.DataFrame) -> str:
    return str(
        cohort_table.loc[cohort_table["mean_day1_mae"].idxmin(), "variant_id"],
    )


def _transition_metrics(
    window_comparison: dict[str, Any],
    left: str,
    right: str,
) -> dict[str, Any] | None:
    for tr in window_comparison.get("transitions", []):
        if tr["left_variant"] == left and tr["right_variant"] == right:
            return tr["metrics"]
    return None


def answer_research_questions(
    cohort_table: pd.DataFrame,
    window_comparison: dict[str, Any],
    memory_analysis: dict[str, Any],
    seq_table: pd.DataFrame,
    g96_replication: dict[str, Any],
) -> dict[str, str]:
    """Explicit answers to the 10 final GGTCE questions."""
    g96 = cohort_table.loc[cohort_table["variant_id"] == "g96"].iloc[0]
    best_vid = _pick_best_variant(cohort_table)
    best = cohort_table.loc[cohort_table["variant_id"] == best_vid].iloc[0]

    mae_improved = float(best["mean_day1_mae"]) < float(g96["mean_day1_mae"])
    mae_delta_pct = (
        (float(best["mean_day1_mae"]) - float(g96["mean_day1_mae"]))
        / float(g96["mean_day1_mae"])
        * 100
    )

    g96_g288 = _transition_metrics(window_comparison, "g96", "g288") or {}
    mae_stats = g96_g288.get("day1_mae", {})
    stat_sig = bool(mae_stats.get("significant_bootstrap_95")) or bool(
        mae_stats.get("significant_wilcoxon_005"),
    )
    practically_meaningful = abs(mae_delta_pct) >= 1.0 and mae_improved

    g96_g192 = _transition_metrics(window_comparison, "g96", "g192") or {}
    g192_g288 = _transition_metrics(window_comparison, "g192", "g288") or {}
    d96_192 = abs(g96_g192.get("day1_mae", {}).get("mean_delta", 0))
    d192_288 = abs(g192_g288.get("day1_mae", {}).get("mean_delta", 0))
    diminishing = d192_288 < d96_192 if d96_192 > 0 else False

    seq_g96 = int(seq_table.loc[seq_table["variant_id"] == "g96", "n_sequences"].iloc[0])
    seq_best = int(
        seq_table.loc[seq_table["variant_id"] == best_vid, "n_sequences"].iloc[0],
    )
    seq_reduction_pct = (seq_g96 - seq_best) / seq_g96 * 100 if seq_g96 else 0

    mem_ans = memory_analysis.get("answers", {})
    corr = memory_analysis.get("correlation_mae", {})
    high_more = mem_ans.get("high_memory_benefits_more", False)

    replace_baseline = (
        best_vid != "g96"
        and mae_improved
        and practically_meaningful
        and stat_sig
    )

    return {
        "1_does_longer_window_improve": (
        f"{'Yes' if mae_improved else 'No'} — {best_vid.upper()} vs G96 "
        f"MAE {float(g96['mean_day1_mae']):.3f}→{float(best['mean_day1_mae']):.3f} "
            f"({mae_delta_pct:+.2f}%)"
        ),
        "2_best_input_window": (
            f"{best_vid.upper()} ({VARIANT_LABELS[best_vid]}) — "
            f"mean Day-1 MAE={float(best['mean_day1_mae']):.3f}"
        ),
        "3_statistically_significant": (
            f"{'Yes' if stat_sig else 'No'} (G96→G288 bootstrap/Wilcoxon on MAE; "
            f"mean Δ={mae_stats.get('mean_delta', float('nan')):.4f})"
        ),
        "4_practically_meaningful": (
            f"{'Yes' if practically_meaningful else 'No'} "
            f"(|MAE change|={abs(mae_delta_pct):.2f}%, threshold 1%)"
        ),
        "5_high_memory_benefit_more": (
            f"{'Yes' if high_more else 'No / inconclusive'} — "
            f"memory–ΔMAE Pearson r={corr.get('pearson_r', float('nan')):.3f}; "
            f"low-memory gain negligible: {mem_ans.get('low_memory_gain_negligible')}"
        ),
        "6_diminishing_returns": (
            f"{'Evidence of diminishing returns after 192' if diminishing else 'No clear diminishing returns'} "
            f"(|Δ| G96→G192={d96_192:.4f}, G192→G288={d192_288:.4f})"
        ),
        "7_context_vs_sequence_tradeoff": (
            f"G96 sequences={seq_g96}, {best_vid}={seq_best} "
            f"({seq_reduction_pct:.1f}% reduction). "
            f"Accuracy {'improved' if mae_improved else 'did not improve'} despite fewer samples."
        ),
        "8_replace_global_baseline": (
            f"{'Recommend replacing G96 with ' + best_vid.upper() if replace_baseline else 'Keep G96 baseline'}"
        ),
        "9_production_recommendation": (
            f"Deploy {best_vid.upper()} Global GRU"
            if replace_baseline
            else "Keep frozen G96 Global GRU — longer windows did not justify complexity/cost"
        ),
        "10_relation_to_tma": (
            "TMA showed 40–73% squared-ACF capture at 96–288 lags; GGTCE tests whether "
            f"exploiting that memory improves forecasts. High-memory subgroup "
            f"{'gained more' if high_more else 'did not consistently gain more'} than low-memory workloads."
        ),
        "g96_replication_passed": str(g96_replication.get("passed", False)),
    }


def _df_to_md_table(df: pd.DataFrame) -> str:
    """Simple markdown table without tabulate dependency."""
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, sep, *rows])


def write_final_report(
    exp_dir: Path,
    cohort_table: pd.DataFrame,
    window_comparison: dict[str, Any],
    memory_analysis: dict[str, Any],
    seq_table: pd.DataFrame,
    g96_replication: dict[str, Any],
    training_meta: dict[str, Any],
) -> None:
    """Write reports/final_report.md and .json."""
    answers = answer_research_questions(
        cohort_table, window_comparison, memory_analysis, seq_table, g96_replication,
    )
    best_vid = _pick_best_variant(cohort_table)

    report_json: dict[str, Any] = {
        "experiment": "GGTCE",
        "best_variant": best_vid,
        "cohort_summary": cohort_table.to_dict(orient="records"),
        "sequence_reduction": seq_table.to_dict(orient="records"),
        "g96_replication": g96_replication,
        "training_summary": {
            vid: {
                k: v for k, v in meta.items() if k != "history"
            }
            for vid, meta in training_meta.items()
        },
        "window_transitions": [
            {
                "transition": t["transition"],
                "metrics": {
                    m: {
                        "mean_delta": s.get("mean_delta"),
                        "significant_bootstrap_95": s.get("significant_bootstrap_95"),
                        "significant_wilcoxon_005": s.get("significant_wilcoxon_005"),
                    }
                    for m, s in t.get("metrics", {}).items()
                },
            }
            for t in window_comparison.get("transitions", [])
        ],
        "memory_analysis": {
            "correlation_mae": memory_analysis.get("correlation_mae"),
            "subgroup_comparison": memory_analysis.get("subgroup_comparison"),
            "answers": memory_analysis.get("answers"),
        },
        "final_answers": answers,
    }

    reports_dir = exp_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with (reports_dir / "final_report.json").open("w") as fh:
        json.dump(report_json, fh, indent=2, default=str)

    lines = [
        "# GGTCE — Final Report",
        "",
        "## Global GRU Temporal Context Experiment",
        "",
        "### Final Answers",
        "",
    ]
    for key in sorted(answers.keys(), key=lambda k: int(k.split("_")[0]) if k[0].isdigit() else 99):
        lines.append(f"**{key}:** {answers[key]}")
        lines.append("")

    lines.extend([
        "## Cohort Comparison",
        "",
        _df_to_md_table(cohort_table),
        "",
        "## Sequence Reduction",
        "",
        _df_to_md_table(seq_table),
        "",
    ])

    with (reports_dir / "final_report.md").open("w") as fh:
        fh.write("\n".join(lines))
