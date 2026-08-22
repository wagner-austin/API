"""Shared fixtures and helpers for test_lightgbm_objective splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams


def _make_test_data(
    n_samples: int = 100, n_features: int = 5, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create test dataset for optimization."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    # Create imbalanced labels: ~30% positive
    n_positive = n_samples // 3
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_positive] = 1
    rng.shuffle(y)
    return x, y, [f"feat_{i}" for i in range(n_features)]


def _make_positive_data(x: NDArray[np.float64], offset: float) -> NDArray[np.float64]:
    """Make data positive by taking absolute value and adding offset.

    Uses direct type annotation to satisfy mypy, same pattern as features.py.
    """
    abs_x: NDArray[np.float64] = np.abs(x)
    result: NDArray[np.float64] = abs_x + offset
    return result


def _make_default_int_params() -> SampledIntParams:
    """Create default integer parameters for testing.

    Note: max_depth is not included because LightGBM optimization uses
    fixed max_depth=-1 (unlimited) to let num_leaves control tree complexity.
    """
    return SampledIntParams(
        n_estimators=10,
        num_leaves=8,
        min_child_samples=5,
    )


def _make_default_float_params() -> SampledFloatParams:
    """Create default float parameters for testing."""
    return SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
    )


def _make_default_string_params() -> SampledStringParams:
    """Create default string parameters for testing (empty for LightGBM)."""
    return SampledStringParams()
