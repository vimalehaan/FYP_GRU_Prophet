"""Deterministic per-container RNG streams for CSRLE."""

from __future__ import annotations

import hashlib

import numpy as np


def container_seed(
    container_id: str,
    global_seed: int,
    stream: str,
) -> int:
    """Map (container_id, global_seed, stream) to a stable 32-bit seed."""
    payload = f"{global_seed}|{container_id}|{stream}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def container_rng(
    container_id: str,
    global_seed: int,
    stream: str,
) -> np.random.Generator:
    return np.random.default_rng(
        container_seed(container_id, global_seed, stream)
    )


def uniform_init_pair(
    container_id: str,
    global_seed: int,
    stream: str,
    low: float,
    high: float,
) -> tuple[float, float]:
    rng = container_rng(container_id, global_seed, stream)
    return float(rng.uniform(low, high)), float(rng.uniform(low, high))
