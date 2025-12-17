"""Helper functions for creating typed numpy arrays in tests.

These helpers ensure strict typing compliance when creating test arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_float64_1d(values: list[float]) -> NDArray[np.float64]:
    """Create a 1D float64 array from a list of floats.

    Args:
        values: List of float values.

    Returns:
        1D numpy array with float64 dtype.
    """
    n = len(values)
    result: NDArray[np.float64] = np.zeros(n, dtype=np.float64)
    for i, v in enumerate(values):
        result[i] = v
    return result


def make_float64_2d(rows: list[list[float]]) -> NDArray[np.float64]:
    """Create a 2D float64 array from a list of rows.

    Args:
        rows: List of rows, where each row is a list of floats.

    Returns:
        2D numpy array with float64 dtype.
    """
    n_rows = len(rows)
    if n_rows == 0:
        return np.zeros((0, 0), dtype=np.float64)
    n_cols = len(rows[0])
    result: NDArray[np.float64] = np.zeros((n_rows, n_cols), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            result[i, j] = v
    return result


def get_float(arr: NDArray[np.float64], i: int, j: int | None = None) -> float:
    """Extract a float value from an array using flat iteration.

    Args:
        arr: Source array.
        i: First index (row for 2D, element for 1D).
        j: Second index (column for 2D, None for 1D).

    Returns:
        Float value at the specified position.
    """
    if j is None:
        # 1D array: iterate to position i
        for idx, val in enumerate(arr.flat):
            if idx == i:
                return float(val.item())
        raise IndexError(f"Index {i} out of bounds for array of size {arr.size}")
    # 2D array: compute flat index
    n_cols = int(arr.shape[1])
    flat_idx = i * n_cols + j
    for idx, val in enumerate(arr.flat):
        if idx == flat_idx:
            return float(val.item())
    raise IndexError(f"Index ({i}, {j}) out of bounds for array of shape {arr.shape}")


def assert_close(actual: float, expected: float, tol: float = 1e-10) -> None:
    """Assert two floats are close within tolerance.

    Args:
        actual: Actual value.
        expected: Expected value.
        tol: Tolerance for comparison.

    Raises:
        AssertionError: If values differ by more than tolerance.
    """
    diff = abs(actual - expected)
    assert diff < tol, f"Expected {expected}, got {actual}, diff={diff}"
