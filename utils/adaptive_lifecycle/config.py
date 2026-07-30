"""AFMLF configuration — all thresholds externalised."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PageHinkleyMode(str, Enum):
    OFF = "off"
    CONFIRMATORY = "confirmatory"
    MANDATORY = "mandatory"


class DeploymentRecommendation(str, Enum):
    DEPLOY_CANDIDATE = "deploy_candidate"
    KEEP_CURRENT_MODEL = "keep_current_model"
    NEEDS_INVESTIGATION = "needs_investigation"


class DecisionOutput(str, Enum):
    RECOMMEND_RETRAINING = "recommend_retraining"
    DO_NOT_RETRAIN = "do_not_retrain"
    NEEDS_INVESTIGATION = "needs_investigation"


@dataclass
class AFMLFConfig:
    """Runtime configuration for an isolated AFMLF experiment."""

    experiment_name: str = "adaptive_forecast_model_lifecycle"
    random_seed: int = 42

    # Frozen read-only references (never written by AFMLF)
    baseline_reference_dir: Path = field(
        default_factory=lambda: Path("experiments/baseline_reference_2026-07-14")
    )
    baseline_evaluation_csv: Path = field(
        default_factory=lambda: Path(
            "experiments/baseline_reference_2026-07-14/evaluation/evaluation_df.csv"
        )
    )
    baseline_model_path: Path = field(
        default_factory=lambda: Path(
            "experiments/baseline_reference_2026-07-14/models/hybrid_gru.keras"
        )
    )
    baseline_residual_stats_path: Path = field(
        default_factory=lambda: Path(
            "experiments/baseline_reference_2026-07-14/models/residual_stats.pkl"
        )
    )
    train_df_path: Path = field(default_factory=lambda: Path("data/train_df.parquet"))
    val_df_path: Path = field(default_factory=lambda: Path("data/val_df.parquet"))

    # Walk-forward simulation
    min_history_steps: int = 200
    forecast_horizon: int = 96
    input_window: int = 96
    walk_forward_stride: int = 96
    monitoring_lag_steps: int = 96
    train_cutoff_buffer_steps: int = 96
    eval_warmup_origins: int = 1

    # Layer 2 — per-container drift thresholds
    relative_threshold: float = 0.25
    absolute_threshold: float = 0.5
    diagnostic_relative_threshold: float = 0.15
    diagnostic_absolute_threshold: float = 0.3
    rolling_window_origins: int = 3
    persistence_required: int = 2

    # Layer 3 — Page-Hinkley
    page_hinkley_mode: PageHinkleyMode = PageHinkleyMode.CONFIRMATORY
    page_hinkley_delta: float = 0.005
    page_hinkley_lambda: float = 50.0

    # Layer 4 — global trigger
    drifted_container_fraction: float = 0.20
    cohort_median_degradation: float = 0.15
    cooldown_origins: int = 2
    max_retrain_events: int = 3

    # Candidate evaluation
    candidate_improvement_margin: float = 0.01

    # Pilot / debug limits (None = full cohort)
    max_containers: int | None = None
    max_origins_per_container: int | None = None
    skip_candidate_retraining: bool = False
    candidate_training_verbose: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, Enum):
                data[key] = value.value
        return data

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_json(cls, path: Path) -> AFMLFConfig:
        raw = json.loads(path.read_text())
        raw["baseline_reference_dir"] = Path(raw["baseline_reference_dir"])
        raw["baseline_evaluation_csv"] = Path(raw["baseline_evaluation_csv"])
        raw["baseline_model_path"] = Path(raw["baseline_model_path"])
        raw["baseline_residual_stats_path"] = Path(raw["baseline_residual_stats_path"])
        raw["train_df_path"] = Path(raw["train_df_path"])
        raw["val_df_path"] = Path(raw["val_df_path"])
        raw["page_hinkley_mode"] = PageHinkleyMode(raw["page_hinkley_mode"])
        return cls(**raw)
