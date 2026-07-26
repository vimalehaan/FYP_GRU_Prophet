"""Ridge Residual Analysis (RRA) — read-only diagnostic experiment utilities."""

from utils.rra.pipeline import (
    compute_container_ridge_analysis,
    train_ridge_diagnostic_model,
)

__all__ = [
    "compute_container_ridge_analysis",
    "train_ridge_diagnostic_model",
]
