"""Production preprocessing — container split then per-container 80/20 temporal split."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from production.utils.config import (
    MIN_ROWS_RESAMPLED,
    TRAIN_SPLIT_RATIO,
    UNSEEN_RAW_PATH,
)


def preprocess_train_containers(
    df_resampled: pd.DataFrame,
    train_container_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, MinMaxScaler], dict[str, Any]]:
    """
    Apply frozen 80/20 temporal split and MinMax scaling for training containers.

    Scalers are fit on each container's train portion only.
    """
    train_list: list[pd.DataFrame] = []
    val_list: list[pd.DataFrame] = []
    scalers: dict[str, MinMaxScaler] = {}
    removed: list[str] = []

    train_set = set(train_container_ids)
    subset = df_resampled[df_resampled["container_id"].isin(train_set)]

    for cid, group in subset.groupby("container_id"):
        group = group.sort_values("time_stamp").copy()
        if len(group) < MIN_ROWS_RESAMPLED:
            removed.append(str(cid))
            continue

        group["cpu"] = group["cpu_usage"]
        split_idx = int(len(group) * TRAIN_SPLIT_RATIO)
        train_part = group.iloc[:split_idx].copy()
        val_part = group.iloc[split_idx:].copy()

        if train_part["cpu"].nunique() <= 1 or train_part["cpu"].std() == 0:
            removed.append(str(cid))
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_part["cpu_scaled"] = scaler.fit_transform(train_part[["cpu"]])
        val_part["cpu_scaled"] = scaler.transform(val_part[["cpu"]])
        scalers[str(cid)] = scaler

        cpu_mean = float(train_part["cpu_scaled"].mean())
        cpu_std = float(train_part["cpu_scaled"].std())
        cpu_max = float(train_part["cpu_scaled"].max())
        cpu_min = float(train_part["cpu_scaled"].min())

        for part in (train_part, val_part):
            part["cpu_mean"] = cpu_mean
            part["cpu_std"] = cpu_std
            part["cpu_max"] = cpu_max
            part["cpu_min"] = cpu_min
            part["hour"] = part["time_stamp"].dt.hour

        train_list.append(train_part)
        val_list.append(val_part)

    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)

    meta: dict[str, Any] = {
        "n_train_containers_requested": len(train_container_ids),
        "n_train_containers_retained": train_df["container_id"].nunique(),
        "n_removed_degenerate": len(removed),
        "removed_container_ids": removed,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_split_ratio": TRAIN_SPLIT_RATIO,
    }
    return train_df, val_df, scalers, meta


def extract_unseen_resampled(
    df_resampled: pd.DataFrame,
    unseen_container_ids: list[str],
) -> pd.DataFrame:
    """Store full resampled timelines for unseen containers (no GRU preprocessing)."""
    unseen = df_resampled[
        df_resampled["container_id"].isin(unseen_container_ids)
    ].copy()
    unseen = unseen.sort_values(["container_id", "time_stamp"])
    UNSEEN_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    unseen.to_parquet(UNSEEN_RAW_PATH, index=False)
    return unseen


def temporal_split_single_container(
    group: pd.DataFrame,
    train_split_ratio: float = TRAIN_SPLIT_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one container timeline into historical (train) and holdout (val) portions."""
    group = group.sort_values("time_stamp").copy()
    group["cpu"] = group["cpu_usage"]
    split_idx = int(len(group) * train_split_ratio)
    return group.iloc[:split_idx].copy(), group.iloc[split_idx:].copy()
