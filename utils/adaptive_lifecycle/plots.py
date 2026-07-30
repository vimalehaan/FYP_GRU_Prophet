"""Publication-quality AFMLF plots (return matplotlib figures)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_rolling_mae(origin_df: pd.DataFrame, container_id: str | None = None) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(10, 4))
    data = origin_df if container_id is None else origin_df[origin_df["container_id"] == container_id]
    for cid, grp in data.groupby("container_id"):
        ax.plot(grp["origin_index"], grp["rolling_hybrid_mae"], label=cid if container_id else None, alpha=0.7)
    ax.set_title("Rolling Day-1 MAE")
    ax.set_xlabel("Origin index")
    ax.set_ylabel("MAE (%)")
    if container_id:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_rolling_rmse(origin_df: pd.DataFrame, container_id: str | None = None) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(10, 4))
    data = origin_df if container_id is None else origin_df[origin_df["container_id"] == container_id]
    for cid, grp in data.groupby("container_id"):
        ax.plot(grp["origin_index"], grp["rolling_hybrid_rmse"], label=cid if container_id else None, alpha=0.7)
    ax.set_title("Rolling Day-1 RMSE")
    ax.set_xlabel("Origin index")
    ax.set_ylabel("RMSE (%)")
    if container_id:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_drift_threshold(
    origin_df: pd.DataFrame,
    container_id: str,
    baseline_mae: float,
    threshold: float,
) -> tuple[Any, Any]:
    """Per-container rolling MAE vs frozen drift threshold."""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = origin_df[origin_df["container_id"] == container_id].sort_values("origin_index")
    ax.plot(data["origin_index"], data["rolling_hybrid_mae"], "o-", label="Rolling Hybrid MAE", color="#2563eb")
    ax.axhline(baseline_mae, color="#64748b", linestyle=":", label=f"Deployment baseline ({baseline_mae:.2f}%)")
    ax.axhline(threshold, color="#dc2626", linestyle="--", label=f"Drift threshold ({threshold:.2f}%)")
    ax.fill_between(
        data["origin_index"],
        threshold,
        data["rolling_hybrid_mae"].max() * 1.05,
        where=data["rolling_hybrid_mae"] > threshold,
        alpha=0.15,
        color="#dc2626",
        label="Breach",
    )
    ax.set_xlabel("Origin index")
    ax.set_ylabel("MAE (%)")
    ax.set_title(f"Drift detection — {container_id}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_forecast_origin(
    timestamps: pd.Series,
    actual: pd.Series,
    hybrid: pd.Series,
    prophet: pd.Series | None = None,
    title: str = "Day-1 forecast",
) -> tuple[Any, Any]:
    """Actual vs predicted CPU for one forecast origin (presentation helper)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, actual, label="Actual", color="#111827", linewidth=2)
    ax.plot(timestamps, hybrid, label="Hybrid forecast", color="#2563eb", linestyle="--")
    if prophet is not None:
        ax.plot(timestamps, prophet, label="Prophet only", color="#94a3b8", linestyle=":")
    ax.set_xlabel("Time")
    ax.set_ylabel("CPU util (%)")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_cpu_history(
    series: pd.DataFrame,
    title: str = "Container CPU history",
) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(series["time_stamp"], series["cpu"], color="#6366f1", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("CPU util (%)")
    ax.set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_decision_timeline(decision_log: pd.DataFrame) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(10, 3))
    if decision_log.empty:
        ax.set_title("Decision timeline (empty)")
        return fig, ax
    colors = {
        "recommend_retraining": "#dc2626",
        "do_not_retrain": "#16a34a",
        "needs_investigation": "#eab308",
    }
    for decision, grp in decision_log.groupby("decision"):
        ax.scatter(
            grp["origin_index"],
            [decision] * len(grp),
            c=colors.get(decision, "#64748b"),
            s=60,
            label=decision,
        )
    ax.set_xlabel("Origin index")
    ax.set_title("Cohort decision timeline")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax


def plot_version_history(registry: dict) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(8, 3))
    versions = registry.get("versions", [])
    if not versions:
        ax.set_title("Model version history (empty)")
        return fig, ax
    labels = [v["version_id"] for v in versions]
    y = list(range(len(labels)))
    ax.barh(y, [1] * len(labels), color="#6366f1")
    ax.set_yticks(y, labels)
    for i, v in enumerate(versions):
        rec = v.get("deployment_recommendation", "n/a")
        ax.text(0.5, i, rec, va="center", ha="center", color="white", fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([])
    ax.set_title("Hybrid version lineage")
    fig.tight_layout()
    return fig, ax


def plot_diagnostic_distribution(diagnostic_log: pd.DataFrame) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = diagnostic_log["diagnostic_class"].value_counts()
    counts.plot(kind="bar", ax=ax, color="#2563eb")
    ax.set_title("Diagnostic class distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig, ax


def plot_state_timeline(lifecycle_log: pd.DataFrame) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(10, 3))
    if lifecycle_log.empty:
        ax.set_title("Lifecycle transitions (empty)")
        return fig, ax
    ymap = {s: i for i, s in enumerate(sorted(lifecycle_log["to_state"].unique()))}
    ax.scatter(
        lifecycle_log["origin_index"],
        [ymap[s] for s in lifecycle_log["to_state"]],
        c="#f97316",
        s=20,
    )
    ax.set_yticks(list(ymap.values()), list(ymap.keys()))
    ax.set_xlabel("Origin index")
    ax.set_title("Lifecycle state transitions")
    fig.tight_layout()
    return fig, ax


def plot_recommendation_quality(recommendation_log: pd.DataFrame) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(6, 4))
    if recommendation_log.empty:
        ax.set_title("Recommendation quality (empty)")
        return fig, ax
    labels = recommendation_log["deployment_recommendation"].value_counts()
    labels.plot(kind="bar", ax=ax, color="#16a34a")
    ax.set_title("Deployment recommendations")
    fig.tight_layout()
    return fig, ax


def plot_page_hinkley(ph_log: pd.DataFrame, container_id: str) -> tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=(8, 3))
    data = ph_log[ph_log["container_id"] == container_id]
    ax.plot(data["origin_index"], data["ph_statistic"], label="PH statistic")
    ax.axhline(data["ph_lambda"].iloc[0] if len(data) else 50, color="red", linestyle="--", label="lambda")
    ax.set_title(f"Page-Hinkley — {container_id}")
    ax.legend()
    fig.tight_layout()
    return fig, ax
