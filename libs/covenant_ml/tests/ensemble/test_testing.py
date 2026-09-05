"""Tests for ensemble testing utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.ensemble._hooks import (
    _ConstraintDict,
    _OptionsDict,
)
from covenant_ml.ensemble.testing import FakeOptimizeResult, fake_minimize


def _float_array(*values: float) -> NDArray[np.float64]:
    """Create typed float64 array from values.

    Args:
        *values: Float values for the array.

    Returns:
        NDArray of float64.
    """
    return np.array(values, dtype=np.float64)


def _sum_constraint(w: NDArray[np.float64]) -> float:
    """Constraint function requiring weights to sum to 1.

    Args:
        w: Weight array.

    Returns:
        Difference from 1.0 (should be 0 for valid weights).
    """
    return float(np.sum(w)) - 1.0


class TestFakeOptimizeResult:
    """Tests for FakeOptimizeResult class."""

    def test_create_result(self) -> None:
        """FakeOptimizeResult stores all attributes."""
        x = _float_array(0.5, 0.5)
        result = FakeOptimizeResult(x=x, fun=-0.8, nit=10, success=True)

        assert np.allclose(result.x, x)
        assert result.fun == -0.8
        assert result.nit == 10
        assert result.success is True


class TestFakeMinimize:
    """Tests for fake_minimize function."""

    def test_basic_minimization(self) -> None:
        """fake_minimize returns valid result structure."""

        # Simple quadratic objective centered at (0.3, 0.7)
        # This ensures the random search can find improvements
        def objective(w: NDArray[np.float64]) -> float:
            target = _float_array(0.3, 0.7)
            diff = w - target
            return float(np.sum(diff**2))

        x0 = _float_array(0.5, 0.5)
        bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
        constraint: _ConstraintDict = {"type": "eq", "fun": _sum_constraint}
        constraints: tuple[_ConstraintDict, ...] = (constraint,)
        options: _OptionsDict = {"maxiter": 100, "ftol": 1e-6}

        result = fake_minimize(
            fun=objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        # Should return a result
        assert result.x.shape == (2,)
        assert np.isclose(float(np.sum(result.x)), 1.0, atol=0.01)
        assert result.nit > 0
        assert result.success is True

    def test_finds_improvement(self) -> None:
        """fake_minimize finds improvements over initial solution.

        This test specifically exercises the improvement branch (lines 93-94)
        by using an objective where equal weights give a poor score.
        """

        # Objective that strongly prefers [0.2, 0.8] over [0.5, 0.5]
        # The random search with seed 42 should find an improvement
        def objective(w: NDArray[np.float64]) -> float:
            # Penalize weights far from [0.2, 0.8]
            target = _float_array(0.2, 0.8)
            diff = w - target
            return float(np.sum(diff**2))

        x0 = _float_array(0.5, 0.5)  # Initial is far from optimal
        bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
        constraint: _ConstraintDict = {"type": "eq", "fun": _sum_constraint}
        constraints: tuple[_ConstraintDict, ...] = (constraint,)
        options: _OptionsDict = {"maxiter": 100}

        # Initial objective value
        initial_fun = objective(x0)

        result = fake_minimize(
            fun=objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        # The optimizer should find a better solution
        assert result.fun < initial_fun
        # And the result should be different from initial
        assert not np.allclose(result.x, x0)

    def test_respects_maxiter_option(self) -> None:
        """fake_minimize uses maxiter from options."""
        call_count = 0

        def counting_objective(w: NDArray[np.float64]) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.sum(w**2))

        x0 = _float_array(0.5, 0.5)
        bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
        constraints: tuple[_ConstraintDict, ...] = ()
        options: _OptionsDict = {"maxiter": 5}  # Low maxiter

        result = fake_minimize(
            fun=counting_objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        # Should have limited iterations (1 initial + up to 5 trials)
        assert call_count <= 6  # 1 for initial + 5 for trials
        assert result.nit <= 6

    def test_three_models(self) -> None:
        """fake_minimize works with 3 parameters."""

        def objective(w: NDArray[np.float64]) -> float:
            return float(np.sum((w - 0.33) ** 2))

        x0 = _float_array(0.33, 0.33, 0.34)
        bounds: tuple[tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
        constraints: tuple[_ConstraintDict, ...] = ()
        options: _OptionsDict = {"maxiter": 20}

        result = fake_minimize(
            fun=objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        assert result.x.shape == (3,)
        assert result.success is True
