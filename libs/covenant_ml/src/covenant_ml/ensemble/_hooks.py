"""The solver seam the ensemble optimizers minimize through.

Strict typing only: no Any, no casts, no stubs.

``minimize`` is bound to scipy, so a caller reaches the real solver without
anything being wired first. Tests rebind this module's attribute and restore
it afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

_ConstraintDict = dict[str, str | Callable[[NDArray[np.float64]], float]]
_OptionsDict = dict[str, int | float]


class _OptimizeResultProtocol(Protocol):
    """Protocol for scipy.optimize.OptimizeResult."""

    x: NDArray[np.float64]
    fun: float
    nit: int
    success: bool


class _MinimizeFnProtocol(Protocol):
    """Protocol for scipy.optimize.minimize function."""

    def __call__(
        self,
        fun: Callable[[NDArray[np.float64]], float],
        x0: NDArray[np.float64],
        method: str,
        bounds: tuple[tuple[float, float], ...],
        constraints: tuple[_ConstraintDict, ...],
        options: _OptionsDict,
    ) -> _OptimizeResultProtocol:
        """Minimize a function."""
        ...


def _real_minimize(
    fun: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    method: str,
    bounds: tuple[tuple[float, float], ...],
    constraints: tuple[_ConstraintDict, ...],
    options: _OptionsDict,
) -> _OptimizeResultProtocol:
    """Minimize via scipy, which is imported on the call rather than on import.

    Args:
        fun: Objective to minimize.
        x0: Initial guess.
        method: Solver name.
        bounds: Per-variable bounds.
        constraints: Constraint specifications.
        options: Solver options.

    Returns:
        The solver's result.
    """
    scipy_opt = __import__("scipy.optimize", fromlist=["minimize"])
    minimize_fn: _MinimizeFnProtocol = scipy_opt.minimize
    # Keywords, not position: scipy's third positional parameter is args, so
    # passing method there sends bounds on to method and the solver fails with
    # "'tuple' object has no attribute 'lower'".
    return minimize_fn(
        fun=fun,
        x0=x0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )


minimize: _MinimizeFnProtocol = _real_minimize


__all__ = [
    "minimize",
]
