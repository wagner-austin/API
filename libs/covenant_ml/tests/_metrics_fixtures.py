"""Shared fixtures and helpers for test_metrics splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _make_int_array(values: list[int]) -> NDArray[np.int64]:
    """Create int64 array from values."""
    arr: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _make_float_array(values: list[float]) -> NDArray[np.float64]:
    """Create float64 array from values."""
    arr: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def _make_binary_arrays(
    n_samples: int = 10,
    positive_ratio: float = 0.5,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    """Create deterministic binary arrays for testing."""
    y_true: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    y_pred: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    y_prob: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)

    n_positive = int(n_samples * positive_ratio)

    for i in range(n_samples):
        if i < n_positive:
            y_true[i] = 1
            if i < n_positive // 2:
                y_pred[i] = 1
                y_prob[i] = 0.8
            else:
                y_pred[i] = 0
                y_prob[i] = 0.3
        else:
            y_true[i] = 0
            if i < n_samples - 1:
                y_pred[i] = 0
                y_prob[i] = 0.2
            else:
                y_pred[i] = 1
                y_prob[i] = 0.6

    return y_true, y_pred, y_prob
