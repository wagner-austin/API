"""Contracts and typed array helpers shared by the option tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.options.types import AsianPutContract

#: The parent study's base contract: one year, 50 dates, every fifth session.
BASE = AsianPutContract(spot=100.0, strike=100.0, years=1.0, exercise_dates=50, sessions_per_date=5)

#: A small contract whose paths can be written out by hand.
SMALL = AsianPutContract(spot=100.0, strike=100.0, years=1.0, exercise_dates=3, sessions_per_date=2)


def contract_with(**changes: float) -> AsianPutContract:
    """Return :data:`SMALL` with the floating fields replaced.

    Args:
        **changes: New ``spot``, ``strike`` or ``years``.

    Returns:
        The changed contract.
    """
    return AsianPutContract(
        spot=changes.get("spot", SMALL["spot"]),
        strike=changes.get("strike", SMALL["strike"]),
        years=changes.get("years", SMALL["years"]),
        exercise_dates=SMALL["exercise_dates"],
        sessions_per_date=SMALL["sessions_per_date"],
    )


def floats(values: list[float]) -> NDArray[np.float64]:
    """Build a float vector from a typed list.

    Args:
        values: The values.

    Returns:
        A float64 vector.
    """
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    return array


def matrix(rows: list[list[float]]) -> NDArray[np.float64]:
    """Build a float table from typed rows.

    Args:
        rows: Equal-length rows.

    Returns:
        A two-dimensional float64 array.
    """
    array: NDArray[np.float64] = np.asarray(rows, dtype=np.float64)
    return array


def listed(array: NDArray[np.float64]) -> list[float]:
    """Read every value of a float array in row order.

    Args:
        array: Any float64 array.

    Returns:
        Its values as Python floats.
    """
    return [float(array.flat[index]) for index in range(int(array.size))]


def listed_ints(array: NDArray[np.int64]) -> list[int]:
    """Read every value of an integer array in row order.

    Args:
        array: Any int64 array.

    Returns:
        Its values as Python ints.
    """
    return [int(array.flat[index]) for index in range(int(array.size))]


def mean(array: NDArray[np.float64]) -> float:
    """Average a float vector.

    Args:
        array: A nonempty vector.

    Returns:
        Its arithmetic mean.
    """
    return float(np.sum(array)) / len(array)


__all__ = [
    "BASE",
    "SMALL",
    "contract_with",
    "floats",
    "listed",
    "listed_ints",
    "matrix",
    "mean",
]
