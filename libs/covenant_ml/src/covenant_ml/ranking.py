"""Shared binary-score validation and tie-aware ranks for classification metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def binary_counts(labels: NDArray[np.int64], scores: NDArray[np.float64]) -> tuple[int, int]:
    """Validate aligned finite binary-score evidence and count its classes.

    Args:
        labels: One-dimensional zero/one labels.
        scores: One finite ranking score per label.

    Returns:
        Positive and negative counts; the consuming metric decides required counts.

    Raises:
        ValueError: CVML-METRIC-001 for shape, length, label or finiteness errors.
    """
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("[CVML-METRIC-001] metrics require aligned one-dimensional vectors")
    positive: NDArray[np.bool_] = labels == 1
    negative: NDArray[np.bool_] = labels == 0
    binary: NDArray[np.bool_] = positive | negative
    finite: NDArray[np.bool_] = np.isfinite(scores)
    if not bool(np.all(binary)) or not bool(np.all(finite)):
        raise ValueError("[CVML-METRIC-001] metrics require binary labels and finite scores")
    return int(np.sum(positive)), int(np.sum(negative))


def midranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rank values from 1, giving tied values the mean of their ranks.

    Args:
        values: Values to rank.

    Returns:
        One rank per value, in the input's order.
    """
    order: NDArray[np.intp] = np.argsort(values, kind="mergesort")
    ordered: NDArray[np.float64] = values[order]
    unique_parts = np.unique(ordered, return_index=True, return_counts=True)
    first: NDArray[np.intp] = unique_parts[1]
    counts: NDArray[np.intp] = unique_parts[2]
    mean_rank: NDArray[np.float64] = first.astype(np.float64) + (counts + 1) / 2.0
    ranked_in_order: NDArray[np.float64] = np.repeat(mean_rank, counts)
    ranks: NDArray[np.float64] = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranked_in_order
    return ranks


__all__ = ["binary_counts", "midranks"]
