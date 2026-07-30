"""Presentation-quality AFMLF figures for notebooks (read-only artifact visualization)."""

from __future__ import annotations

from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_candidate_mae_comparison(
    recommendation_log: pd.DataFrame,
    deploy_margin: float,
) -> tuple[Any, Any, dict[str, float]]:
    """Side-by-side incumbent vs candidate Day-1 MAE with deployment rationale."""
    if recommendation_log.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Candidate vs incumbent (no recommendations)")
        return fig, ax, {}

    row = recommendation_log.iloc[0]
    inc_mae = float(row["incumbent_cohort_mae"])
    cand_mae = float(row["candidate_cohort_mae"])
    delta = float(row["delta_mae"])
    improvement = abs(delta)
    rec = str(row["deployment_recommendation"])

    labels = ["Hybrid v1\n(Incumbent)", "Hybrid v2\n(Candidate)"]
    values = [inc_mae, cand_mae]
    colors = ["#64748b", "#2563eb"]

    fig, ax = plt.subplots(figsize=(7, 6))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="#1e293b", linewidth=0.8)
    ax.set_ylabel("Day-1 MAE (%)", fontsize=11)
    ax.set_title("Official cohort — incumbent vs candidate Day-1 MAE", fontsize=12, fontweight="bold")
    ymax = max(values) * 1.18
    ax.set_ylim(0, ymax)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{val:.4f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    reason = (
        f"Incumbent MAE = {inc_mae:.4f}%\n"
        f"Candidate MAE = {cand_mae:.4f}%\n"
        f"Improvement = {improvement:.4f} pp (ΔMAE {delta:+.4f}%)\n"
        f"Deployment margin = {deploy_margin:.4f} pp\n"
        f"Recommendation = {rec.replace('_', ' ').title()}\n\n"
        f"Reason: Candidate improvement ({improvement:.4f} pp) did not exceed the deployment "
        f"margin ({deploy_margin:.4f} pp); AFMLF correctly retained the incumbent model."
        if rec == "keep_current_model" and delta < 0
        else (
            f"Incumbent MAE = {inc_mae:.4f}%\n"
            f"Candidate MAE = {cand_mae:.4f}%\n"
            f"ΔMAE = {delta:+.4f}%\n"
            f"Deployment margin = {deploy_margin:.4f} pp\n"
            f"Recommendation = {rec.replace('_', ' ').title()}"
        )
    )
    fig.text(0.5, 0.02, reason, ha="center", va="bottom", fontsize=9, wrap=True, bbox=dict(boxstyle="round", fc="#f8fafc", ec="#cbd5e1"))
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()

    meta = {
        "incumbent_mae": inc_mae,
        "candidate_mae": cand_mae,
        "delta_mae": delta,
        "deploy_margin": deploy_margin,
        "recommendation": rec,
    }
    return fig, ax, meta


def _origin_drift_count(
    origin: int,
    transitions: pd.DataFrame,
    decision_row: pd.Series,
) -> int:
    t = transitions[transitions["origin_index"] == origin]
    n_breach = t.loc[t["reason"] == "threshold_breach", "container_id"].dropna().nunique()
    n_suspected = t.loc[t["to_state"] == "drift_suspected", "container_id"].dropna().nunique()
    if n_breach > 0:
        return int(n_breach)
    if n_suspected > 0:
        return int(n_suspected)
    return int(decision_row["n_recommend"])


def plot_decision_timeline_annotated(
    decision_log: pd.DataFrame,
    transitions: pd.DataFrame,
    n_containers: int,
) -> tuple[Any, Any, pd.DataFrame]:
    """Decision timeline with per-origin drift counts and trigger status."""
    colors = {
        "recommend_retraining": "#dc2626",
        "do_not_retrain": "#16a34a",
        "needs_investigation": "#eab308",
    }
    label_map = {
        "recommend_retraining": "Retraining Recommended",
        "do_not_retrain": "No Retraining",
        "needs_investigation": "Needs Investigation",
    }

    fig, ax = plt.subplots(figsize=(12, 5.5))
    if decision_log.empty:
        ax.set_title("Cohort decision timeline (empty)")
        return fig, ax, pd.DataFrame()

    annotations = []
    y_positions = {"recommend_retraining": 2, "do_not_retrain": 1, "needs_investigation": 0}

    for _, row in decision_log.sort_values("origin_index").iterrows():
        origin = int(row["origin_index"])
        decision = str(row["decision"])
        n_drift = _origin_drift_count(origin, transitions, row)
        pct = 100.0 * n_drift / n_containers
        trigger_ok = decision == "recommend_retraining"
        y = y_positions.get(decision, 1)

        ax.scatter(origin, y, c=colors.get(decision, "#64748b"), s=120, zorder=3, edgecolors="white", linewidths=1.2)
        trigger_text = "Global trigger satisfied" if trigger_ok else f"Trigger: {row['trigger_reason']}"
        ann = (
            f"Origin {origin}\n"
            f"{n_drift} / {n_containers} drifting\n"
            f"{pct:.1f}%\n"
            f"{label_map.get(decision, decision)}"
        )
        ax.annotate(
            ann,
            (origin, y),
            textcoords="offset points",
            xytext=(0, 22 if decision == "recommend_retraining" else -55),
            ha="center",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=colors.get(decision, "#64748b"), alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.8),
        )
        annotations.append(
            {
                "origin_index": origin,
                "n_drifting": n_drift,
                "drift_pct": round(pct, 1),
                "decision": decision,
                "trigger_reason": row["trigger_reason"],
                "global_trigger_satisfied": trigger_ok,
            }
        )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(["Needs Investigation", "No Retraining", "Retraining Recommended"])
    ax.set_xlabel("Walk-forward origin index", fontsize=11)
    ax.set_title("Cohort decision timeline — drift evidence and global trigger", fontsize=12, fontweight="bold")
    ax.set_xlim(decision_log["origin_index"].min() - 40, decision_log["origin_index"].max() + 40)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax, pd.DataFrame(annotations)


def plot_lifecycle_cohort_flow(
    lifecycle_log: pd.DataFrame,
    decision_log: pd.DataFrame,
) -> tuple[Any, Any, list[dict]]:
    """Cohort lifecycle as a connected process across walk-forward origins."""
    state_order = ["stable", "warning", "drift_suspected", "candidate_retraining"]
    state_colors = {
        "stable": "#86efac",
        "warning": "#fde047",
        "drift_suspected": "#fb923c",
        "candidate_retraining": "#6366f1",
    }
    origins = sorted(lifecycle_log["origin_index"].unique())

    milestones: list[dict] = []
    for origin in origins:
        t = lifecycle_log[lifecycle_log["origin_index"] == origin]
        n_warning = t.loc[t["to_state"] == "warning", "container_id"].dropna().nunique()
        n_suspected = t.loc[t["to_state"] == "drift_suspected", "container_id"].dropna().nunique()
        n_stable = t.loc[t["to_state"] == "stable", "container_id"].dropna().nunique()
        cohort_retrain = ((t["to_state"] == "candidate_retraining") & t["container_id"].isna()).any()
        if cohort_retrain:
            primary = "candidate_retraining"
            label = "Candidate Retraining\n(global)"
            count = 1
        elif n_suspected > 0:
            primary = "drift_suspected"
            label = f"Drift Suspected\n({n_suspected} containers)"
            count = n_suspected
        elif n_warning > 0:
            primary = "warning"
            label = f"Warning\n({n_warning} containers)"
            count = n_warning
        elif n_stable > 0:
            primary = "stable"
            label = f"Stable\n({n_stable} recovered)"
            count = n_stable
        else:
            continue
        dec = decision_log[decision_log["origin_index"] == origin]
        decision = dec.iloc[0]["decision"] if len(dec) else "n/a"
        milestones.append(
            {
                "origin_index": int(origin),
                "primary_state": primary,
                "label": label,
                "count": count,
                "decision": decision,
            }
        )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    n = len(milestones)
    if n == 0:
        ax.set_title("Lifecycle cohort flow (empty)")
        return fig, ax, milestones

    xs = np.linspace(0.8, 9.2, n)
    y = 1.5
    box_w, box_h = 1.5, 1.1

    for i, (m, x) in enumerate(zip(milestones, xs)):
        color = state_colors.get(m["primary_state"], "#cbd5e1")
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.03",
            fc=color,
            ec="#1e293b",
            lw=1.2,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.15, f"Origin {m['origin_index']}", ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x, y - 0.15, m["label"], ha="center", va="center", fontsize=8)
        if i < n - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - box_w / 2, y),
                xytext=(x + box_w / 2, y),
                arrowprops=dict(arrowstyle="->", lw=2, color="#475569"),
            )

    ax.set_title("Cohort lifecycle flow across walk-forward origins", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig, ax, milestones


def plot_diagnostic_distribution_enhanced(diagnostic_log: pd.DataFrame) -> tuple[Any, Any, pd.DataFrame]:
    """Sorted diagnostic bar chart with count and percentage labels."""
    counts = diagnostic_log["diagnostic_class"].value_counts().sort_values(ascending=False)
    total = int(counts.sum())
    pct = (counts / total * 100).round(1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(range(len(counts)), counts.values, color="#2563eb", edgecolor="#1e3a5f", linewidth=0.8)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=15, ha="right")
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Diagnostic class distribution (official cohort)", fontsize=12, fontweight="bold")

    for i, (bar, cnt, p) in enumerate(zip(bars, counts.values, pct.values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{cnt}\n({p}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()

    summary = pd.DataFrame({"count": counts, "pct": pct})
    return fig, ax, summary


def plot_official_lifecycle_summary(
    diagnostic_log: pd.DataFrame,
    lifecycle_log: pd.DataFrame,
    decision_log: pd.DataFrame,
    recommendation_log: pd.DataFrame,
) -> tuple[Any, Any, pd.DataFrame]:
    """End-to-end official AFMLF funnel with experiment-derived counts."""
    n_containers = diagnostic_log["container_id"].nunique()
    n_monitoring = len(diagnostic_log)
    n_suspected = int((lifecycle_log["to_state"] == "drift_suspected").sum())
    n_retrain_rec = int((decision_log["decision"] == "recommend_retraining").sum())
    n_candidates = int((lifecycle_log["to_state"] == "candidate_retraining").sum())
    deploy_rec = (
        recommendation_log.iloc[0]["deployment_recommendation"].replace("_", " ").title()
        if len(recommendation_log)
        else "N/A"
    )
    candidate_version = recommendation_log.iloc[0]["candidate_version"] if len(recommendation_log) else "hybrid_v2"

    stages = [
        ("Containers Monitored", n_containers),
        ("Monitoring Events", n_monitoring),
        ("Drift Suspected\nTransitions", n_suspected),
        ("Global Retraining\nRecommendations", n_retrain_rec),
        (f"{candidate_version}\nCandidate", n_candidates if n_candidates else n_retrain_rec),
        ("Candidate\nEvaluation", len(recommendation_log)),
        (deploy_rec, 1 if len(recommendation_log) else 0),
    ]

    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(0, 10)
    ax.axis("off")
    y = len(stages) * 1.35
    colors = plt.cm.Blues(np.linspace(0.25, 0.85, len(stages)))

    for i, ((label, count), color) in enumerate(zip(stages, colors)):
        rect = mpatches.FancyBboxPatch(
            (2, y), 6, 0.95, boxstyle="round,pad=0.02", fc=color, ec="#1e3a5f", lw=1.2
        )
        ax.add_patch(rect)
        text = f"{label}\n(n={count})" if i > 0 else f"{label}\n(n={count})"
        ax.text(5, y + 0.48, text, ha="center", va="center", fontsize=10, fontweight="bold" if i == 0 else "normal")
        if i < len(stages) - 1:
            ax.annotate("", xy=(5, y), xytext=(5, y + 0.35), arrowprops=dict(arrowstyle="->", lw=2, color="#334155"))
        y -= 1.35

    ax.text(5, len(stages) * 1.35 + 0.5, "Official AFMLF Lifecycle Summary", ha="center", fontsize=13, fontweight="bold")
    fig.tight_layout()

    funnel_df = pd.DataFrame(stages, columns=["stage", "count"])
    return fig, ax, funnel_df
