"""Fixed container-level train/unseen split for production."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from production.utils.config import (
    CONTAINER_SPLIT_METADATA_PATH,
    CONTAINER_SPLIT_SEED,
    TRAIN_CONTAINER_FRACTION,
    TRAIN_CONTAINER_IDS_PATH,
    UNSEEN_CONTAINER_FRACTION,
    UNSEEN_CONTAINER_IDS_PATH,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def load_or_create_container_split(
    all_container_ids: list[str] | np.ndarray,
    force_recreate: bool = False,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """
    Load persisted container split or create a new one.

    Once saved, the split is immutable unless ``force_recreate=True``.
    """
    if (
        not force_recreate
        and TRAIN_CONTAINER_IDS_PATH.exists()
        and UNSEEN_CONTAINER_IDS_PATH.exists()
        and CONTAINER_SPLIT_METADATA_PATH.exists()
    ):
        with TRAIN_CONTAINER_IDS_PATH.open() as fh:
            train_ids = json.load(fh)["container_ids"]
        with UNSEEN_CONTAINER_IDS_PATH.open() as fh:
            unseen_ids = json.load(fh)["container_ids"]
        with CONTAINER_SPLIT_METADATA_PATH.open() as fh:
            meta = json.load(fh)
        return train_ids, unseen_ids, meta

    rng = np.random.default_rng(CONTAINER_SPLIT_SEED)
    ids = sorted(str(c) for c in all_container_ids)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    n_train = int(len(shuffled) * TRAIN_CONTAINER_FRACTION)
    train_ids = sorted(shuffled[:n_train])
    unseen_ids = sorted(shuffled[n_train:])

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": CONTAINER_SPLIT_SEED,
        "train_fraction": TRAIN_CONTAINER_FRACTION,
        "unseen_fraction": UNSEEN_CONTAINER_FRACTION,
        "n_total_containers": len(ids),
        "n_train_containers": len(train_ids),
        "n_unseen_containers": len(unseen_ids),
        "immutable": True,
        "note": (
            "Container split is fixed after first creation. "
            "Unseen containers never participate in production GRU training."
        ),
    }

    _write_json(TRAIN_CONTAINER_IDS_PATH, {"container_ids": train_ids})
    _write_json(UNSEEN_CONTAINER_IDS_PATH, {"container_ids": unseen_ids})
    _write_json(CONTAINER_SPLIT_METADATA_PATH, meta)

    return train_ids, unseen_ids, meta
