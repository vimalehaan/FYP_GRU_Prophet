#!/usr/bin/env python3
"""Scan unseen containers for AFMLF case-study scenario selection (read-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.adaptive_lifecycle.candidate_retraining import load_model_bundle
from utils.adaptive_lifecycle.config import AFMLFConfig
from utils.adaptive_lifecycle.diagnostics import DiagnosticClass, classify_container
from utils.adaptive_lifecycle.drift_detection import detect_container_drift, drift_threshold
from utils.adaptive_lifecycle.monitoring import (
    build_container_series,
    forecast_at_origin,
    generate_origin_indices,
)
from utils.adaptive_lifecycle.rolling_metrics import append_rolling_metrics


def analyze_container(
    container_id: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    incumbent,
    res_mean: float,
    res_std: float,
    cfg: AFMLFConfig,
) -> dict:
    series = build_container_series(train_df, val_df, container_id)
    origins = generate_origin_indices(len(series), cfg)
    if len(origins) < 4:
        return {"container_id": container_id, "skip": True, "reason": "too_few_origins"}

    rows = []
    for origin in origins:
        m = forecast_at_origin(series, origin, incumbent, res_mean, res_std, cfg)
        rows.append(m.__dict__)

    monitor_df = append_rolling_metrics(pd.DataFrame(rows), cfg.rolling_window_origins)
    deploy_baseline_mae = float(monitor_df.iloc[0]["hybrid_mae"])
    deploy_baseline_prophet = float(monitor_df.iloc[0]["prophet_mae"])
    threshold = drift_threshold(deploy_baseline_mae, cfg.relative_threshold, cfg.absolute_threshold)

    drift_rows = []
    diag_rows = []
    for _, r in monitor_df.iterrows():
        det = detect_container_drift(
            container_id,
            int(r.origin_index),
            r.rolling_hybrid_mae,
            deploy_baseline_mae,
            cfg,
        )
        drift_rows.append(det.__dict__)
        d = classify_container(
            container_id,
            int(r.origin_index),
            r.rolling_hybrid_mae,
            r.rolling_prophet_mae,
            deploy_baseline_mae,
            deploy_baseline_prophet,
            cfg,
        )
        diag_rows.append(d)

    drift_df = pd.DataFrame(drift_rows)
    breaches = drift_df[drift_df["threshold_breach"]]
    n_breaches = len(breaches)
    first_breach = int(breaches.iloc[0]["origin_index"]) if n_breaches else None

    consecutive = 0
    max_consecutive = 0
    for breach in drift_df["threshold_breach"]:
        if breach:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    diag_classes = [d.diagnostic_class for d in diag_rows]
    n_stable = sum(1 for c in diag_classes if c == DiagnosticClass.STABLE)
    n_recommend = sum(
        1 for c in diag_classes if c in {DiagnosticClass.GRU_STALE, DiagnosticClass.REGIME_CHANGE}
    )
    trigger_diag = None
    if first_breach is not None:
        trigger_diag = next(
            d for d in diag_rows if d.origin_index == first_breach
        ).diagnostic_class.value

    max_rolling = float(monitor_df["rolling_hybrid_mae"].max())
    headroom = threshold - max_rolling
    future_origins = sum(1 for o in origins if first_breach and o > first_breach + cfg.eval_warmup_origins * cfg.walk_forward_stride)

    return {
        "container_id": container_id,
        "skip": False,
        "n_origins": len(origins),
        "deploy_baseline_mae": round(deploy_baseline_mae, 4),
        "threshold": round(threshold, 4),
        "max_rolling_mae": round(max_rolling, 4),
        "headroom": round(headroom, 4),
        "n_breaches": n_breaches,
        "max_consecutive_breaches": max_consecutive,
        "first_breach_origin": first_breach,
        "trigger_diagnostic": trigger_diag,
        "n_stable_diag": n_stable,
        "n_recommend_diag": n_recommend,
        "future_eval_origins": future_origins,
        "timeline_len": len(series),
    }


def main() -> None:
    cfg = AFMLFConfig.from_json(
        sorted(REPO.glob("experiments/adaptive_forecast_model_lifecycle_*"))[-1]
        / "config"
        / "afmlf_config.json"
    )
    official = [p for p in REPO.glob("experiments/adaptive_forecast_model_lifecycle_*") if (p / "OFFICIAL_RUN").exists()][-1]
    cfg = AFMLFConfig.from_json(official / "config" / "afmlf_config.json")

    cohort = set(pd.read_csv(REPO / cfg.baseline_evaluation_csv)["container_id"])
    with (REPO / "production/hybrid/metadata/unseen_container_ids.json").open() as fh:
        unseen_ids = json.load(fh)["container_ids"]

    unseen_ids = [cid for cid in unseen_ids if cid not in cohort]
    train_df = pd.read_parquet(REPO / cfg.train_df_path)
    val_df = pd.read_parquet(REPO / cfg.val_df_path)
    available = [cid for cid in unseen_ids if cid in set(train_df["container_id"]) | set(val_df["container_id"])]

    incumbent, res_mean, res_std = load_model_bundle(official / "versions" / "hybrid_v1")

    results = []
    for i, cid in enumerate(available, 1):
        print(f"[{i}/{len(available)}] {cid}", flush=True)
        results.append(analyze_container(cid, train_df, val_df, incumbent, res_mean, res_std, cfg))

    df = pd.DataFrame([r for r in results if not r.get("skip")])
    out = REPO / "notebooks" / "outputs" / "afmlf_case_study" / "unseen_scan.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nSaved", out)

    stable = df[df["n_breaches"] == 0].sort_values("headroom", ascending=False)
    drifted = df[df["n_breaches"] > 0].copy()
    drifted["score"] = (
        drifted["n_breaches"] * 2
        + drifted["max_consecutive_breaches"]
        + drifted["n_recommend_diag"]
        + drifted["future_eval_origins"]
        + drifted["trigger_diagnostic"].map({"REGIME_CHANGE": 3, "GRU_STALE": 2, "INVESTIGATE": 1}).fillna(0)
    )
    drifted = drifted.sort_values("score", ascending=False)

    print("\n=== TOP STABLE CANDIDATES (no breach) ===")
    print(stable.head(10).to_string(index=False))
    print("\n=== TOP DRIFTED CANDIDATES ===")
    print(drifted.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
