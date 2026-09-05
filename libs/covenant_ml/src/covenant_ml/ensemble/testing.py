"""Test utilities for ensemble module.

Provides fake scipy minimize implementation for testing without real scipy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from covenant_ml.ensemble._hooks import (
    _OptionsDict,
)


class FakeOptimizeResult:
    """Fake scipy OptimizeResult for testing."""

    def __init__(
        self,
        x: NDArray[np.float64],
        fun: float,
        nit: int,
        success: bool,
    ) -> None:
        """Initialize fake result.

        Args:
            x: Solution array.
            fun: Objective function value at solution.
            nit: Number of iterations.
            success: Whether optimization converged.
        """
        self.x = x
        self.fun = fun
        self.nit = nit
        self.success = success


def fake_minimize(
    fun: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    method: str,
    bounds: tuple[tuple[float, float], ...],
    constraints: tuple[dict[str, str | Callable[[NDArray[np.float64]], float]], ...],
    options: _OptionsDict,
) -> FakeOptimizeResult:
    """Fake minimize that does simple grid search.

    This is a simplified optimizer for testing that:
    1. Evaluates the objective at initial weights
    2. Tries a few random perturbations
    3. Returns the best result found

    Args:
        fun: Objective function to minimize.
        x0: Initial guess.
        method: Optimization method (ignored).
        bounds: Parameter bounds (used for clipping).
        constraints: Constraints (sum-to-one enforced).
        options: Options (maxiter used).

    Returns:
        FakeOptimizeResult with best solution found.
    """
    n_params = len(x0)
    max_iter = int(options.get("maxiter", 100))

    # Limit iterations for testing
    n_trials = min(max_iter, 20)

    best_x = x0.copy()
    best_fun = fun(x0)
    n_iter = 1

    rng = np.random.default_rng(42)

    for _ in range(n_trials):
        # Generate random weights
        raw_weights: NDArray[np.float64] = rng.random(n_params).astype(np.float64)

        # Normalize to sum to 1
        trial_x: NDArray[np.float64] = raw_weights / float(np.sum(raw_weights))

        # Evaluate
        trial_fun = fun(trial_x)
        n_iter += 1

        if trial_fun < best_fun:
            best_x = trial_x
            best_fun = trial_fun

    return FakeOptimizeResult(
        x=best_x,
        fun=best_fun,
        nit=n_iter,
        success=True,
    )


__all__ = [
    "FakeOptimizeResult",
    "fake_minimize",
]
