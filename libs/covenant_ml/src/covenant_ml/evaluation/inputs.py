"""Numerical preconditions shared by the evaluation instruments.

These pure checks reject malformed evidence before numpy broadcasting or a
nonfinite estimate can turn it into a plausible-looking interval.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .types import ERR_INVALID_ARGUMENT, ERR_TOO_FEW_UNITS


def require_observations(values: NDArray[np.float64], name: str) -> None:
    """Require a finite, one-dimensional vector.

    Args:
        values: Observations, before arithmetic or broadcasting.
        name: Field named in a refusal.

    Raises:
        ValueError: If the array is not a vector of finite values.
    """
    if values.ndim != 1:
        raise ValueError(f"[{ERR_INVALID_ARGUMENT}] {name} must be one-dimensional")
    finite: NDArray[np.bool_] = np.isfinite(values)
    if not bool(np.all(finite)):
        raise ValueError(f"[{ERR_INVALID_ARGUMENT}] {name} must contain only finite values")


def require_interval_inputs(estimate: float, std_error: float, n_units: int) -> None:
    """Require a finite estimate, nonnegative finite error and two units.

    Args:
        estimate: Point estimate.
        std_error: Estimated standard error.
        n_units: Units used to estimate uncertainty.

    Raises:
        ValueError: If a numerical precondition fails.
    """
    if not math.isfinite(estimate):
        raise ValueError(f"[{ERR_INVALID_ARGUMENT}] estimate must be finite")
    if not math.isfinite(std_error) or std_error < 0.0:
        raise ValueError(f"[{ERR_INVALID_ARGUMENT}] standard error must be finite and nonnegative")
    if n_units < 2:
        raise ValueError(f"[{ERR_TOO_FEW_UNITS}] uncertainty needs at least two units")


__all__ = ["require_interval_inputs", "require_observations"]
