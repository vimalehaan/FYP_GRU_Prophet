"""Report writers for RRE Stage 0 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def build_final_report_md(
    config: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    recommendation: dict[str, Any],
) -> str:
    """Markdown summary suitable for thesis appendix."""
    lines = [
        "# Residual Representation Diagnostic Study (Stage 0)",
        "",
        f"**Protocol:** {config.get('protocol_version')}",
        f"**Timestamp:** {config.get('timestamp')}",
        f"**Containers:** {config.get('n_containers')}",
        "",
        "## Why this stage exists",
        "",
        "Prior work established that Prophet explains ~78% of CPU variance and that "
        "Hybrid improves over Prophet, but changes to loss, context, window size, and "
        "peak weighting did not yield significant gains. The remaining limitation may "
        "lie in **how** the residual is represented for GRU learning — not **how** the "
        "GRU is trained. Representation selection must therefore precede RRE v1.0.",
        "",
        "Velocity (first-order difference) was **not** assumed optimal. This study "
        "compares R0–R3 using read-only statistical diagnostics only.",
        "",
        "## Comparison summary",
        "",
        "| ID | Name | mean |ACF| | ADF reject frac | Skew | Sparsity (<0.05) |",
        "|----|------|----------|-----------------|------|------------------|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['representation_id']} | {row['name']} | "
            f"{row['temporal_dependence']:.4f} | "
            f"{row['stationarity_adf_reject_frac']:.2%} | "
            f"{row['skewness']:.2f} | "
            f"{row['sparsity_frac_abs_lt_0_05']:.2%} |"
        )

    ans = recommendation["answers"]
    lines.extend(
        [
            "",
            "## Evidence-based answers",
            "",
            f"1. **Most useful temporal structure:** {ans['q1_most_temporal_structure']}",
            f"2. **Most learnable (diagnostic proxy):** {ans['q2_most_learnable_proxy']}",
            f"3. **Best temporal/stability balance:** {ans['q3_best_balance']}",
            f"4. **Primary RRE candidate:** {ans['q4_primary_candidate']} — "
            f"{recommendation['primary_recommendation_name']}",
            f"5. **Level baseline remains primary?** "
            f"{'Yes' if ans['q5_level_baseline_remains'] else 'No — an alternative representation is recommended'}",
            "",
            "## Recommendation rationale",
            "",
            recommendation["rationale"],
            "",
        ]
    )
    return "\n".join(lines)
