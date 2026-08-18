"""Typed numpy helpers for the growth-policy tests.

Under ``disallow_any_expr`` a bare numpy reduction is an ``Any`` expression:
``np.mean`` returns ``floating[Any]`` and array construction infers loosely
unless the destination is annotated. Rather than repeating the same narrowing
in every assertion, the handful of shapes these tests need are declared once
here and every test reads through them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def floats(values: list[float]) -> NDArray[np.float64]:
    """Build a float array.

    Args:
        values: Values to hold.

    Returns:
        The array.
    """
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    return array


def ints(values: list[int]) -> NDArray[np.int64]:
    """Build an integer array.

    Args:
        values: Values to hold.

    Returns:
        The array.
    """
    array: NDArray[np.int64] = np.asarray(values, dtype=np.int64)
    return array


def mean_of(values: NDArray[np.float64]) -> float:
    """Average a float array.

    Uses an explicit sum over length rather than ``np.mean``, whose return type
    carries ``Any``.

    Args:
        values: Values to average. Must be non-empty.

    Returns:
        The arithmetic mean.
    """
    total: float = float(np.sum(values))
    return total / len(values)


def positive_rate(labels: NDArray[np.int64]) -> float:
    """Report the share of labels equal to one.

    Args:
        labels: Binary labels. Must be non-empty.

    Returns:
        The positive rate.
    """
    total: int = int(np.sum(labels))
    return float(total) / len(labels)


def as_float_list(values: NDArray[np.float64]) -> list[float]:
    """Convert a float array to a list.

    Args:
        values: Array to convert.

    Returns:
        The values as Python floats.
    """
    result: list[float] = []
    for index in range(len(values)):
        item: np.float64 = values[index]
        result.append(float(item))
    return result


def as_int_list(values: NDArray[np.int64]) -> list[int]:
    """Convert an integer array to a list.

    Args:
        values: Array to convert.

    Returns:
        The values as Python ints.
    """
    result: list[int] = []
    for index in range(len(values)):
        item: np.int64 = values[index]
        result.append(int(item))
    return result


def columns_of(values: NDArray[np.float64]) -> int:
    """Report a 2-D array's column count.

    Args:
        values: Array to measure.

    Returns:
        The number of columns.
    """
    return int(values.shape[1])


def equal(left: NDArray[np.float64], right: NDArray[np.float64]) -> bool:
    """Compare two float arrays element for element.

    Args:
        left: First array.
        right: Second array.

    Returns:
        True when the arrays match exactly.
    """
    return bool(np.array_equal(left, right))


def select(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Take the rows a mask selects.

    Args:
        values: Array to filter.
        mask: Boolean mask.

    Returns:
        The selected values.
    """
    selected: NDArray[np.float64] = values[mask]
    return selected


def label_mask(labels: NDArray[np.int64], value: int) -> NDArray[np.bool_]:
    """Build a mask selecting one label value.

    Args:
        labels: Binary labels.
        value: Label to select.

    Returns:
        The mask.
    """
    mask: NDArray[np.bool_] = labels == value
    return mask


__all__ = [
    "as_float_list",
    "as_int_list",
    "columns_of",
    "equal",
    "floats",
    "ints",
    "label_mask",
    "mean_of",
    "positive_rate",
    "select",
]
