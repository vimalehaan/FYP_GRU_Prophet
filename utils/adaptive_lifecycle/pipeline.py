"""AFMLF end-to-end experiment pipeline."""

from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf

from utils.adaptive_lifecycle.candidate_evaluation import evaluate_candidate_vs_incumbent
from utils.adaptive_lifecycle.candidate_retraining import (
    build_training_frame_to_cutoff,
    run_candidate_retraining,
    summarize_retraining_frame,
)
from utils.adaptive_lifecycle.config import AFMLFConfig, DecisionOutput
from utils.adaptive_lifecycle.decision_engine import make_cohort_decision
from utils.adaptive_lifecycle.diagnostics import classify_container, cohort_diagnostic_confidence
from utils.adaptive_lifecycle.drift_detection import detect_container_drift, load_hybrid_baselines
from utils.adaptive_lifecycle.evaluation import compute_framework_evaluation
from utils.adaptive_lifecycle.lifecycle import (
    CohortLifecycleState,
    ContainerLifecycleState,
    LifecycleState,
    apply_container_drift_progression,
)
from utils.adaptive_lifecycle.model_versioning import ModelVersionRegistry, bootstrap_incumbent_version
from utils.adaptive_lifecycle.monitoring import run_walk_forward_monitoring
from utils.adaptive_lifecycle.page_hinkley import page_hinkley_test
from utils.adaptive_lifecycle.progress import AFMLFProgress
from utils.adaptive_lifecycle.report import (
    default_research_answers,
    write_executive_summary,
    write_final_report,
)
from utils.adaptive_lifecycle.rolling_metrics import append_rolling_metrics
from utils.adaptive_lifecycle.trigger_policy import evaluate_global_trigger


def _timestamp_dir_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def create_experiment_dir(repo_root: Path, config: AFMLFConfig, official: bool = False) -> Path:
    exp_dir = repo_root / "experiments" / f"adaptive_forecast_model_lifecycle_{_timestamp_dir_name()}"
    for sub in (
        "config",
        "monitoring",
        "lifecycle",
        "diagnostics",
        "decisions",
        "recommendations",
        "versions",
        "plots",
        "reports",
        "tables",
    ):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    config.save_json(exp_dir / "config" / "afmlf_config.json")
    if official:
        (exp_dir / "OFFICIAL_RUN").write_text(
            f"official_thesis_evaluation\n{datetime.now(timezone.utc).isoformat()}\n"
        )
    return exp_dir


def run_afmlf_experiment(
    repo_root: Path,
    config: AFMLFConfig,
    official: bool = False,
) -> Path:
    t0 = time.perf_counter()
    exp_dir = create_experiment_dir(repo_root, config, official=official)
    progress = AFMLFProgress(official=official)
    failures: list[dict[str, Any]] = []

    baselines = load_hybrid_baselines(repo_root / config.baseline_evaluation_csv)
    container_ids = baselines["container_id"].tolist()
    if config.max_containers is not None:
        container_ids = container_ids[: config.max_containers]

    progress.banner(len(container_ids), str(exp_dir))

    train_df = pd.read_parquet(repo_root / config.train_df_path)
    val_df = pd.read_parquet(repo_root / config.val_df_path)

    registry = ModelVersionRegistry()
    incumbent = bootstrap_incumbent_version(
        exp_dir,
        config.to_dict(),
        repo_root / config.baseline_model_path,
        repo_root / config.baseline_residual_stats_path,
    )
    registry.register(incumbent)

    with (repo_root / config.baseline_residual_stats_path).open("rb") as fh:
        stats = pickle.load(fh)
    res_mean = float(stats["res_mean"])
    res_std = float(stats["res_std"])

    model = tf.keras.models.load_model(repo_root / config.baseline_model_path)

    progress.phase("Walk-forward monitoring")

    def _on_monitor(event: str, *args: Any) -> None:
        if event == "failure":
            cid, origin, err = args
            progress.monitoring_failure(str(cid), int(origin), str(err))
        elif event == "step":
            ci, nc, cid, origin, done, total = args
            progress.monitoring(int(ci), int(nc), str(cid), int(origin), int(done), int(total))

    origin_df = run_walk_forward_monitoring(
        container_ids,
        train_df,
        val_df,
        model,
        res_mean,
        res_std,
        config,
        failures=failures,
        on_progress=_on_monitor,
    )

    if failures:
        pd.DataFrame(failures).to_csv(exp_dir / "tables" / "monitoring_failures.csv", index=False)

    origin_df = append_rolling_metrics(origin_df, config.rolling_window_origins)
    origin_df = origin_df.merge(baselines, on="container_id", how="left")

    prophet_baselines = (
        origin_df.sort_values(["container_id", "origin_index"])
        .groupby("container_id", as_index=False)
        .first()[["container_id", "prophet_mae"]]
        .rename(columns={"prophet_mae": "baseline_prophet_mae"})
    )
    origin_df = origin_df.merge(prophet_baselines, on="container_id", how="left")
    origin_df.to_csv(exp_dir / "monitoring" / "origin_metrics.csv", index=False)

    container_states = {
        cid: ContainerLifecycleState(container_id=cid) for cid in container_ids
    }
    cohort = CohortLifecycleState()

    lifecycle_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    ph_rows: list[dict[str, Any]] = []
    recommendation_rows: list[dict[str, Any]] = []

    container_errors: dict[str, list[float]] = {cid: [] for cid in container_ids}
    current_incumbent_dir = Path(incumbent.artifact_dir)
    retrain_counter = 1

    unique_origins = sorted(origin_df["origin_index"].unique())
    progress.phase(f"Lifecycle simulation ({len(unique_origins)} cohort origins)")

    for oi, origin_index in enumerate(unique_origins, start=1):
        step = origin_df[origin_df["origin_index"] == origin_index]
        detections = []
        diagnostics = []
        n_breaches = 0
        n_ph_alarms = 0

        for _, row in step.iterrows():
            cid = str(row["container_id"])
            container_errors[cid].append(float(row["hybrid_mae"]))
            ph = page_hinkley_test(
                container_errors[cid],
                delta=config.page_hinkley_delta,
                lambd=config.page_hinkley_lambda,
            )
            if ph.alarm:
                n_ph_alarms += 1
            ph_rows.append({
                "container_id": cid,
                "origin_index": origin_index,
                "ph_statistic": ph.statistic,
                "ph_alarm": ph.alarm,
                "ph_lambda": config.page_hinkley_lambda,
            })

            det = detect_container_drift(
                cid,
                int(origin_index),
                float(row["rolling_hybrid_mae"]),
                float(row["baseline_hybrid_mae"]),
                config,
            )
            if det.threshold_breach:
                n_breaches += 1
            detections.append(det)

            ph_confirm = ph.alarm if config.page_hinkley_mode.value != "off" else True
            if config.page_hinkley_mode.value == "mandatory":
                ph_confirm = ph.alarm

            apply_container_drift_progression(
                container_states[cid],
                int(origin_index),
                det.threshold_breach,
                ph_confirm,
                config.page_hinkley_mode.value,
            )
            for tr in container_states[cid].history[-2:]:
                lifecycle_rows.append(asdict(tr))

            diag = classify_container(
                cid,
                int(origin_index),
                float(row["rolling_hybrid_mae"]),
                float(row["rolling_prophet_mae"]),
                float(row["baseline_hybrid_mae"]),
                float(row["baseline_prophet_mae"]),
                config,
            )
            diagnostics.append(diag)
            diagnostic_rows.append({**asdict(diag), "diagnostic_class": diag.diagnostic_class.value})

        top_diag = Counter(d.diagnostic_class.value for d in diagnostics).most_common(1)
        top_diag_s = top_diag[0][0] if top_diag else "—"
        progress.lifecycle_origin(
            int(origin_index), oi, len(unique_origins), n_breaches, n_ph_alarms, top_diag_s
        )

        trigger = evaluate_global_trigger(
            int(origin_index),
            detections,
            config,
            cohort.cooldown_remaining,
        )
        diag_conf = cohort_diagnostic_confidence(diagnostics)
        decision = make_cohort_decision(
            int(origin_index),
            diagnostics,
            diag_conf,
            trigger.triggered,
        )
        decision_rows.append({
            **asdict(decision),
            "decision": decision.decision.value,
            "trigger_reason": trigger.reason,
        })
        progress.decision(int(origin_index), decision.decision.value, decision.rationale[:80])

        if decision.decision == DecisionOutput.RECOMMEND_RETRAINING and not config.skip_candidate_retraining:
            if cohort.retrain_count >= config.max_retrain_events:
                continue
            if cohort.cooldown_remaining > 0:
                continue

            cohort.transition(
                LifecycleState.CANDIDATE_RETRAINING,
                int(origin_index),
                "retrain_triggered",
            )
            lifecycle_rows.append({
                "from_state": cohort.state.value,
                "to_state": LifecycleState.CANDIDATE_RETRAINING.value,
                "origin_index": origin_index,
                "container_id": None,
                "reason": "global_retrain",
                "metadata": {},
            })

            version_id = f"hybrid_v{retrain_counter + 1}"
            origin_map = {str(r["container_id"]): int(origin_index) for _, r in step.iterrows()}
            train_frame = build_training_frame_to_cutoff(
                train_df,
                val_df,
                container_ids,
                origin_map,
                config,
            )
            dataset_summary = summarize_retraining_frame(
                train_frame, origin_map, train_df, val_df
            )
            progress.retraining_start(
                version_id, int(origin_index), dataset_summary["n_rows_total"]
            )

            candidate = run_candidate_retraining(
                version_id,
                registry.latest().version_id if registry.latest() else "hybrid_v1",
                train_frame,
                exp_dir,
                config,
                int(origin_index),
                dataset_summary=dataset_summary,
            )
            meta = candidate.training_dataset_summary.get("training_metadata", {})
            progress.retraining_done(
                version_id,
                float(candidate.training_dataset_summary.get("training_duration_seconds", 0)),
                int(meta.get("n_sequences_total", 0)),
            )

            eval_result = evaluate_candidate_vs_incumbent(
                candidate,
                current_incumbent_dir,
                origin_df,
                train_df,
                val_df,
                container_ids,
                int(origin_index),
                config,
            )
            registry.register(candidate)
            recommendation_rows.append({
                **asdict(eval_result),
                "deployment_recommendation": eval_result.deployment_recommendation.value,
                "candidate_version": version_id,
            })
            progress.evaluation(
                version_id,
                eval_result.n_eval_origins,
                eval_result.incumbent_cohort_mae,
                eval_result.candidate_cohort_mae,
                eval_result.deployment_recommendation.value,
            )

            if eval_result.deployment_recommendation.value == "deploy_candidate":
                current_incumbent_dir = Path(candidate.artifact_dir)

            cohort.retrain_count += 1
            cohort.cooldown_remaining = config.cooldown_origins
            retrain_counter += 1

        if cohort.cooldown_remaining > 0:
            cohort.cooldown_remaining -= 1

    progress.phase("Writing artifacts and reports")

    pd.DataFrame(lifecycle_rows).to_csv(exp_dir / "lifecycle" / "transitions.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(exp_dir / "diagnostics" / "diagnostics.csv", index=False)
    pd.DataFrame(decision_rows).to_csv(exp_dir / "decisions" / "decisions.csv", index=False)
    pd.DataFrame(ph_rows).to_csv(exp_dir / "lifecycle" / "page_hinkley.csv", index=False)
    pd.DataFrame(recommendation_rows).to_csv(exp_dir / "recommendations" / "recommendations.csv", index=False)
    registry.save(exp_dir / "versions" / "registry.json")

    lifecycle_df = pd.DataFrame(lifecycle_rows)
    diagnostic_df = pd.DataFrame(diagnostic_rows)
    decision_df = pd.DataFrame(decision_rows)
    ph_df = pd.DataFrame(ph_rows)
    recommendation_df = pd.DataFrame(recommendation_rows)

    framework_eval = compute_framework_evaluation(
        lifecycle_df,
        decision_df,
        diagnostic_df,
        recommendation_df,
    )
    write_final_report(
        exp_dir,
        config.to_dict(),
        framework_eval,
        default_research_answers(framework_eval),
    )

    registry_payload = json.loads((exp_dir / "versions" / "registry.json").read_text())
    write_executive_summary(
        exp_dir,
        framework_eval,
        origin_df,
        lifecycle_df,
        ph_df,
        diagnostic_df,
        decision_df,
        recommendation_df,
        registry_payload.get("versions", []),
        container_ids,
        failures,
        official,
    )

    from utils.adaptive_lifecycle.plots import (
        plot_decision_timeline,
        plot_diagnostic_distribution,
        plot_recommendation_quality,
        plot_rolling_mae,
        plot_rolling_rmse,
        plot_state_timeline,
        plot_version_history,
    )

    plot_dir = exp_dir / "plots"
    fig, _ = plot_rolling_mae(origin_df)
    fig.savefig(plot_dir / "rolling_mae_cohort.png", dpi=150)
    plt_close(fig)
    fig, _ = plot_rolling_rmse(origin_df)
    fig.savefig(plot_dir / "rolling_rmse_cohort.png", dpi=150)
    plt_close(fig)
    fig, _ = plot_diagnostic_distribution(diagnostic_df)
    fig.savefig(plot_dir / "diagnostic_distribution.png", dpi=150)
    plt_close(fig)
    fig, _ = plot_state_timeline(lifecycle_df)
    fig.savefig(plot_dir / "lifecycle_timeline.png", dpi=150)
    plt_close(fig)
    fig, _ = plot_decision_timeline(decision_df)
    fig.savefig(plot_dir / "decision_timeline.png", dpi=150)
    plt_close(fig)
    fig, _ = plot_version_history(registry_payload)
    fig.savefig(plot_dir / "version_history.png", dpi=150)
    plt_close(fig)
    fig, _ = plot_recommendation_quality(recommendation_df)
    fig.savefig(plot_dir / "recommendation_quality.png", dpi=150)
    plt_close(fig)

    if failures:
        progress.failures_summary(failures)

    progress.complete(str(exp_dir), time.perf_counter() - t0)
    return exp_dir


def plt_close(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)
