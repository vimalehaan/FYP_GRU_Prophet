"""Page-Hinkley change detection on error streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PageHinkleyResult:
    statistic: float
    alarm: bool
    cumulative_min: float


def page_hinkley_test(
    values: list[float] | np.ndarray,
    delta: float = 0.005,
    lambd: float = 50.0,
) -> PageHinkleyResult:
    """
    Page-Hinkley test for mean increase in a stream.

    Returns alarm=True when cumulative deviation exceeds lambda.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return PageHinkleyResult(statistic=0.0, alarm=False, cumulative_min=0.0)

    mean = float(np.mean(x))
    cumsum = 0.0
    min_cumsum = 0.0
    max_stat = 0.0

    for value in x:
        cumsum += value - mean - delta
        min_cumsum = min(min_cumsum, cumsum)
        stat = cumsum - min_cumsum
        max_stat = max(max_stat, stat)

    return PageHinkleyResult(
        statistic=float(max_stat),
        alarm=bool(max_stat > lambd),
        cumulative_min=float(min_cumsum),
    )
