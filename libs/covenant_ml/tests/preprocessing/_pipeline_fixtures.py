"""Shared fixtures and helpers for test_pipeline splits."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _arr(*values: float) -> NDArray[np.float64]:
    """Create 1D float64 array from values.

    Uses *args to ensure proper typing (no list[Any]).
    """
    result: NDArray[np.float64] = np.array(values, dtype=np.float64)
    return result


def _col(*values: float) -> NDArray[np.float64]:
    """Create single-column 2D array from values.

    Returns array of shape (len(values), 1).
    """
    arr_1d = _arr(*values)
    result: NDArray[np.float64] = arr_1d.reshape(-1, 1)
    return result


def _stack_cols(*cols: NDArray[np.float64]) -> NDArray[np.float64]:
    """Stack 1D arrays as columns."""
    result: NDArray[np.float64] = np.column_stack(cols)
    return result


def _all_finite(arr: NDArray[np.float64]) -> bool:
    """Check if all values in array are finite (not NaN or inf)."""
    finite_mask: NDArray[np.bool_] = np.isfinite(arr)
    all_result: np.bool_ = np.all(finite_mask)
    return bool(all_result)


def _get_val(arr: NDArray[np.float64], row: int, col: int) -> float:
    """Get scalar value from 2D array with proper typing."""
    return float(arr.item((row, col)))


def _get_1d(arr: NDArray[np.float64], idx: int) -> float:
    """Get scalar value from 1D array with proper typing."""
    return float(arr.item(idx))


def _is_nan(value: float) -> bool:
    """Check if a scalar value is NaN."""
    import math

    return math.isnan(value)


def _any_equal(arr: NDArray[np.float64], value: float) -> bool:
    """Check if any element in array equals value."""
    eq_mask: NDArray[np.bool_] = arr == value
    any_result: np.bool_ = np.any(eq_mask)
    return bool(any_result)


def _max_abs(arr: NDArray[np.float64]) -> float:
    """Get max absolute value in array using iteration."""
    abs_arr: NDArray[np.float64] = np.abs(arr)
    max_val = 0.0
    for val in abs_arr.flat:
        v = float(val)
        if v > max_val:
            max_val = v
    return max_val


def _std(arr: NDArray[np.float64]) -> float:
    """Compute standard deviation using iteration."""
    n = int(arr.shape[0])
    if n == 0:
        return 0.0
    # Compute mean
    total = 0.0
    for val in arr.flat:
        total += float(val)
    mean = total / n
    # Compute variance
    var_sum = 0.0
    for val in arr.flat:
        diff = float(val) - mean
        var_sum += diff * diff
    return math.sqrt(var_sum / n)


def _make_test_matrix(n_rows: int, n_cols: int, seed: int = 42) -> NDArray[np.float64]:
    """Create test matrix with random but reproducible values."""
    rng = np.random.default_rng(seed)
    result: NDArray[np.float64] = rng.standard_normal((n_rows, n_cols)).astype(np.float64)
    return result


def _make_simple_data() -> NDArray[np.float64]:
    """Create simple test data with known properties (5 rows, 3 cols)."""
    col0 = _arr(1.0, 2.0, 3.0, 4.0, 5.0)
    col1 = _arr(11.0, 12.0, 13.0, 14.0, 15.0)
    col2 = _arr(21.0, 22.0, 23.0, 24.0, 25.0)
    return _stack_cols(col0, col1, col2)


def _make_data_with_outliers() -> NDArray[np.float64]:
    """Create test data with extreme outliers."""
    col0 = _arr(1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -50.0)
    col1 = _arr(10.0, 20.0, 30.0, 40.0, 50.0, 1000.0, -500.0)
    return _stack_cols(col0, col1)


def _make_data_with_special_codes() -> NDArray[np.float64]:
    """Create test data with special codes (96, 98)."""
    col0 = _arr(1.0, 2.0, 3.0, 4.0, 5.0)
    col1 = _arr(96.0, 20.0, 98.0, 40.0, 50.0)
    return _stack_cols(col0, col1)


def _make_data_with_nan() -> NDArray[np.float64]:
    """Create test data with NaN values."""
    col0 = _arr(1.0, 2.0, np.nan, 4.0, 5.0)
    col1 = _arr(np.nan, 20.0, 30.0, 40.0, 50.0)
    return _stack_cols(col0, col1)


def _make_labels(n_samples: int) -> NDArray[np.int64]:
    """Create dummy labels for API consistency."""
    return np.zeros(n_samples, dtype=np.int64)
