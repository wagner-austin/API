"""Shared fixtures and helpers for test_netcdf_temporal splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.datasets.types_temporal import (
    TemporalFeatureConfig,
)


def _val(arr: NDArray[np.float64], i: int, j: int, k: int) -> float:
    """Extract a typed float from a 3D NDArray (avoids mypy Any from indexing)."""
    row: NDArray[np.float64] = arr[i, j]
    return float(row.flat[k])


def _val2(arr: NDArray[np.float64], i: int, j: int) -> float:
    """Extract a typed float from a 2D NDArray (avoids mypy Any from indexing)."""
    row: NDArray[np.float64] = arr[i]
    return float(row.flat[j])


def _i64(values: list[int]) -> NDArray[np.int64]:
    """Create int64 array from typed list (avoids mypy list[Any] error)."""
    result: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for idx, v in enumerate(values):
        result[idx] = v
    return result


def _f64(values: list[float]) -> NDArray[np.float64]:
    """Create float64 array from typed list (avoids mypy list[Any] error)."""
    result: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for idx, v in enumerate(values):
        result[idx] = v
    return result


def _f64_2d(values: list[list[float]]) -> NDArray[np.float64]:
    """Create 2D float64 array from nested list (avoids mypy list[Any] error)."""
    rows = len(values)
    cols = len(values[0])
    result: NDArray[np.float64] = np.zeros((rows, cols), dtype=np.float64)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            result[i, j] = v
    return result


def _repeat_i64(segments: list[tuple[int, int]]) -> NDArray[np.int64]:
    """Create int64 array by repeating values (avoids np.full/np.concatenate Any).

    Args:
        segments: List of (value, count) pairs.

    Returns:
        1D int64 array with each value repeated count times.
    """
    total = sum(count for _, count in segments)
    result: NDArray[np.int64] = np.zeros(total, dtype=np.int64)
    offset = 0
    for value, count in segments:
        for i in range(count):
            result[offset + i] = value
        offset += count
    return result


def _max_abs(arr: NDArray[np.float64]) -> float:
    """Compute max absolute value (avoids mypy Any from np.abs/np.max)."""
    flat: NDArray[np.float64] = arr.ravel()
    max_val: float = 0.0
    for i in range(int(flat.shape[0])):
        val = abs(float(flat.flat[i]))
        if val > max_val:
            max_val = val
    return max_val


def _variance(arr: NDArray[np.float64]) -> float:
    """Compute variance (avoids mypy Any from .var())."""
    flat: NDArray[np.float64] = arr.ravel()
    n = int(flat.shape[0])
    total: float = 0.0
    for i in range(n):
        total += float(flat.flat[i])
    mean = total / n
    sq_sum: float = 0.0
    for i in range(n):
        diff = float(flat.flat[i]) - mean
        sq_sum += diff * diff
    return sq_sum / n


def _has_nan(arr: NDArray[np.float64]) -> bool:
    """Check if array has any NaN values (avoids mypy Any from np.isnan)."""
    flat: NDArray[np.float64] = arr.ravel()
    for i in range(int(flat.shape[0])):
        val = float(flat.flat[i])
        if val != val:  # NaN != NaN
            return True
    return False


def _make_config(compute_ar1: bool = True) -> TemporalFeatureConfig:
    """Create a temporal feature config for testing."""
    return TemporalFeatureConfig(
        n_fourier_harmonics=3,
        hot_cutoff_percentile=95.0,
        cold_cutoff_percentile=5.0,
        season="warm",
        season_months=(6, 7, 8),
        compute_ar1=compute_ar1,
    )
