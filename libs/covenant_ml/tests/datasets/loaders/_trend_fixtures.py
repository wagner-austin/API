"""Shared fixtures and helpers for test_netcdf_trend_testing splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _f64(values: list[float]) -> NDArray[np.float64]:
    """Create float64 array from typed list."""
    result: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for idx, v in enumerate(values):
        result[idx] = v
    return result


def _f64_2d(values: list[list[float]]) -> NDArray[np.float64]:
    """Create 2D float64 array from nested list."""
    rows = len(values)
    cols = len(values[0])
    result: NDArray[np.float64] = np.zeros((rows, cols), dtype=np.float64)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            result[i, j] = v
    return result


def _f64_3d(values: list[list[list[float]]]) -> NDArray[np.float64]:
    """Create 3D float64 array from nested list."""
    d0 = len(values)
    d1 = len(values[0])
    d2 = len(values[0][0])
    result: NDArray[np.float64] = np.zeros((d0, d1, d2), dtype=np.float64)
    for i in range(d0):
        for j in range(d1):
            for k in range(d2):
                result.flat[i * d1 * d2 + j * d2 + k] = values[i][j][k]
    return result


def _flat(arr: NDArray[np.float64], idx: int) -> float:
    """Extract a typed float from flat index."""
    return float(arr.flat[idx])


def _val2(arr: NDArray[np.float64], i: int, j: int) -> float:
    """Extract a typed float from a 2D NDArray."""
    return float(arr.flat[i * int(arr.shape[1]) + j])
