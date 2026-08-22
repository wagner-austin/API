"""NaN/Inf-safe numeric primitives for preprocessing statistics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def _finite_mask(arr: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Get mask of finite (not NaN/inf) values."""
    result: NDArray[np.bool_] = np.isfinite(arr)
    return result


def _nan_mask(arr: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Get mask of NaN values."""
    result: NDArray[np.bool_] = np.isnan(arr)
    return result


def _isclose_mask(
    arr: NDArray[np.float64], value: float, rtol: float = 1e-9, atol: float = 1e-9
) -> NDArray[np.bool_]:
    """Get mask of values close to target value."""
    result: NDArray[np.bool_] = np.isclose(arr, value, rtol=rtol, atol=atol)
    return result


def _is_nan(value: float) -> bool:
    """Check if a scalar float value is NaN."""
    return value != value  # NaN != NaN is True


def _safe_nanmean(arr: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Compute mean per column, ignoring NaN values.

    Args:
        arr: 2D array of shape (n_rows, n_cols).
        axis: Must be 0 (column-wise computation).

    Returns:
        1D array of shape (n_cols,) with mean of each column.
    """
    n_cols = int(arr.shape[1])
    result: NDArray[np.float64] = np.zeros(n_cols, dtype=np.float64)
    for col_idx in range(n_cols):
        col: NDArray[np.float64] = arr[:, col_idx]
        total = 0.0
        count = 0
        for elem in col.flat:
            val = float(elem.item())
            if not _is_nan(val):
                total += val
                count += 1
        if count > 0:
            result[col_idx] = total / count
    return result


def _safe_nanstd(arr: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Compute std per column, ignoring NaN values.

    Args:
        arr: 2D array of shape (n_rows, n_cols).
        axis: Must be 0 (column-wise computation).

    Returns:
        1D array of shape (n_cols,) with std of each column.
    """
    n_cols = int(arr.shape[1])
    means: NDArray[np.float64] = _safe_nanmean(arr, axis)
    result: NDArray[np.float64] = np.zeros(n_cols, dtype=np.float64)
    for col_idx in range(n_cols):
        col: NDArray[np.float64] = arr[:, col_idx]
        var_sum = 0.0
        count = 0
        col_mean = float(means.flat[col_idx])
        for elem in col.flat:
            val = float(elem.item())
            if not _is_nan(val):
                diff = val - col_mean
                var_sum += diff * diff
                count += 1
        if count > 0:
            result[col_idx] = math.sqrt(var_sum / count)
    return result


def _replace_zeros_with_one(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Replace zero values with 1.0 to avoid division by zero."""
    zero_mask: NDArray[np.bool_] = arr == 0.0
    result: NDArray[np.float64] = arr.copy()
    result[zero_mask] = 1.0
    return result


def _safe_percentile(arr: NDArray[np.float64], pct: float) -> float:
    """Compute percentile using sort + index for strict typing.

    Args:
        arr: Non-empty 1D array of values.
        pct: Percentile to compute (0-100).

    Returns:
        The computed percentile value.
    """
    n = int(arr.shape[0])
    sorted_indices: NDArray[np.intp] = np.argsort(arr)
    sorted_arr: NDArray[np.float64] = arr[sorted_indices]
    # Linear interpolation index
    idx_float = (pct / 100.0) * (n - 1)
    idx_low = int(idx_float)
    idx_high = min(idx_low + 1, n - 1)
    frac = idx_float - idx_low
    val_low = float(sorted_arr.flat[idx_low])
    val_high = float(sorted_arr.flat[idx_high])
    return val_low + frac * (val_high - val_low)


def _safe_median(arr: NDArray[np.float64]) -> float:
    """Compute median using sort + index for strict typing.

    Args:
        arr: Non-empty 1D array of values.

    Returns:
        The median value.
    """
    n = int(arr.shape[0])
    sorted_indices: NDArray[np.intp] = np.argsort(arr)
    sorted_arr: NDArray[np.float64] = arr[sorted_indices]
    if n % 2 == 1:
        # Odd: take middle element
        mid = n // 2
        return float(sorted_arr.flat[mid])
    # Even: average of two middle elements
    mid_high = n // 2
    mid_low = mid_high - 1
    val_low = float(sorted_arr.flat[mid_low])
    val_high = float(sorted_arr.flat[mid_high])
    return (val_low + val_high) / 2.0


def _safe_mean(arr: NDArray[np.float64]) -> float:
    """Compute mean using iteration for strict typing.

    Args:
        arr: Non-empty 1D array of values.

    Returns:
        The mean value.
    """
    n = int(arr.shape[0])
    total = 0.0
    for val in arr.flat:
        total += float(val)
    return total / n
