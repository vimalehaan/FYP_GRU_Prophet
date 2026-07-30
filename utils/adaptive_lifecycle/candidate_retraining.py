"""Offline candidate retraining pipeline."""

from __future__ import annotations

import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from production.utils.training import train_production_hybrid
from utils.adaptive_lifecycle.config import AFMLFConfig
from utils.adaptive_lifecycle.model_versioning import ModelVersionRecord, utc_now_iso


@dataclass(frozen=True)
class ContainerRetrainingSlice:
    """Per-container composition of the candidate retraining dataset."""

    container_id: str
    trigger_origin_index: int
    n_original_train: int
    n_accumulated_val: int
    n_total: int


def accumulated_val_row_count(
    trigger_origin: int,
    n_original_train: int,
    n_val_available: int,
) -> int:
    """
    Number of validation-period rows included in the retraining dataset.

    Unified timeline: [0, n_original_train) = original offline train,
    [n_original_train, ...) = post-training observations.
    """
    if trigger_origin <= n_original_train:
        return 0
    return min(n_val_available, trigger_origin - n_original_train)


def assert_retraining_temporal_integrity(
    slice_info: ContainerRetrainingSlice,
    n_val_available: int,
) -> None:
    """Ensure candidate training never includes rows at or beyond the drift trigger."""
    expected_val = accumulated_val_row_count(
        slice_info.trigger_origin_index,
        slice_info.n_original_train,
        n_val_available,
    )
    assert slice_info.n_accumulated_val == expected_val, (
        f"{slice_info.container_id}: expected {expected_val} accumulated val rows, "
        f"got {slice_info.n_accumulated_val}"
    )
    assert slice_info.n_total == slice_info.n_original_train + slice_info.n_accumulated_val

    if slice_info.n_accumulated_val > 0:
        assert (
            slice_info.n_original_train + slice_info.n_accumulated_val
            <= slice_info.trigger_origin_index
        ), (
            f"{slice_info.container_id}: retraining dataset extends to "
            f"{slice_info.n_original_train + slice_info.n_accumulated_val} "
            f"beyond trigger {slice_info.trigger_origin_index}"
        )


def build_training_frame_to_cutoff(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    container_ids: list[str],
    origin_index_by_container: dict[str, int],
    config: AFMLFConfig,
) -> pd.DataFrame:
    """
    Build cohort candidate training data:

    FULL original offline ``train_df`` per container
    PLUS validation-period rows accumulated strictly before the drift trigger.

    Uses frozen ``cpu_scaled`` from preprocessing (production scalers fit on
    original train only) — scalers are not refit here.
    """
    frames: list[pd.DataFrame] = []
    for cid in container_ids:
        train_part = (
            train_df[train_df["container_id"] == cid]
            .sort_values("time_stamp")
            .copy()
        )
        val_part = (
            val_df[val_df["container_id"] == cid]
            .sort_values("time_stamp")
            .copy()
        )
        trigger = int(origin_index_by_container.get(cid, len(train_part) + len(val_part)))
        n_train = len(train_part)
        n_val_take = accumulated_val_row_count(trigger, n_train, len(val_part))
        val_accumulated = val_part.iloc[:n_val_take].copy()

        slice_info = ContainerRetrainingSlice(
            container_id=cid,
            trigger_origin_index=trigger,
            n_original_train=n_train,
            n_accumulated_val=n_val_take,
            n_total=n_train + n_val_take,
        )
        assert_retraining_temporal_integrity(slice_info, len(val_part))

        part = pd.concat([train_part, val_accumulated], ignore_index=True)
        if "cpu_scaled" not in part.columns:
            raise ValueError(
                f"{cid}: missing cpu_scaled — frozen preprocessing artifacts required"
            )
        frames.append(part)

    return pd.concat(frames, ignore_index=True)


def summarize_retraining_frame(
    train_frame: pd.DataFrame,
    origin_index_by_container: dict[str, int],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize dataset composition for experiment metadata."""
    per_container: list[dict[str, Any]] = []
    for cid in train_frame["container_id"].unique():
        cid = str(cid)
        n_train = int(len(train_df[train_df["container_id"] == cid]))
        n_val_avail = int(len(val_df[val_df["container_id"] == cid]))
        trigger = int(origin_index_by_container.get(cid, 0))
        n_in_frame = int(len(train_frame[train_frame["container_id"] == cid]))
        n_acc = accumulated_val_row_count(trigger, n_train, n_val_avail)
        per_container.append(
            {
                "container_id": cid,
                "trigger_origin_index": trigger,
                "n_original_train": n_train,
                "n_accumulated_val": n_acc,
                "n_total": n_in_frame,
            }
        )
    return {
        "methodology": "original_offline_train_plus_accumulated_val_before_trigger",
        "scaler_policy": "frozen_production_cpu_scaled_from_parquet",
        "n_rows_total": int(len(train_frame)),
        "n_containers": int(train_frame["container_id"].nunique()),
        "per_container": per_container,
    }


def run_candidate_retraining(
    version_id: str,
    parent_version: str,
    train_frame: pd.DataFrame,
    exp_dir: Path,
    config: AFMLFConfig,
    trigger_origin_index: int,
    dataset_summary: dict[str, Any] | None = None,
) -> ModelVersionRecord:
    t0 = time.perf_counter()
    model, res_mean, res_std, _, train_meta = train_production_hybrid(
        train_frame,
        input_window=config.input_window,
        forecast_horizon=config.forecast_horizon,
        verbose=config.candidate_training_verbose,
    )
    training_duration_seconds = time.perf_counter() - t0

    version_dir = exp_dir / "versions" / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "model.keras"
    model.save(model_path)
    stats = {"res_mean": res_mean, "res_std": res_std}
    with (version_dir / "residual_stats.pkl").open("wb") as fh:
        pickle.dump(stats, fh)

    record = ModelVersionRecord(
        version_id=version_id,
        created_at=utc_now_iso(),
        configuration=config.to_dict(),
        training_dataset_summary={
            "trigger_origin_index": trigger_origin_index,
            "training_duration_seconds": training_duration_seconds,
            "training_metadata": train_meta,
            **(dataset_summary or {}),
        },
        evaluation_summary={},
        deployment_recommendation="pending_evaluation",
        artifact_dir=str(version_dir),
        parent_version=parent_version,
        notes="Offline candidate retrain triggered by AFMLF (original train + accumulated val).",
    )
    (version_dir / "metadata.json").write_text(
        __import__("json").dumps(asdict(record), indent=2) + "\n"
    )
    return record


def load_model_bundle(version_dir: Path) -> tuple[Any, float, float]:
    import tensorflow as tf

    with (version_dir / "residual_stats.pkl").open("rb") as fh:
        stats = pickle.load(fh)
    model = tf.keras.models.load_model(version_dir / "model.keras")
    return model, float(stats["res_mean"]), float(stats["res_std"])
