"""Shared fixtures and helpers for test_regression_optimizer splits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from covenant_ml.ensemble.regression_types import (
    RegressionOptimizationConfig,
)
from covenant_ml.ensemble.types import ModelOOFPredictions

_RegressionMetric = Literal["neg_rmse", "neg_mae", "r_squared"]

_ObjectiveFnType = Callable[[NDArray[np.float64]], float]

_ConstraintDict = dict[str, str | _ObjectiveFnType]


def _float_array(values: tuple[float, ...]) -> NDArray[np.float64]:
    """Create float64 array from tuple."""
    result: NDArray[np.float64] = np.zeros(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _int_array(values: tuple[int, ...]) -> NDArray[np.int64]:
    """Create int64 array from tuple."""
    result: NDArray[np.int64] = np.zeros(len(values), dtype=np.int64)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _make_model_predictions(
    name: str,
    predictions: tuple[float, ...],
) -> ModelOOFPredictions:
    """Create ModelOOFPredictions for testing."""
    n_samples = len(predictions)
    preds: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for i, v in enumerate(predictions):
        preds[i] = v

    fold_indices: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)
    return ModelOOFPredictions(
        model_name=name,
        predictions=preds,
        fold_indices=fold_indices,
    )


def _make_test_config(
    metric: _RegressionMetric = "neg_rmse",
) -> RegressionOptimizationConfig:
    """Create test optimization config."""
    return RegressionOptimizationConfig(
        metric=metric,
        method="SLSQP",
        max_iterations=100,
        tolerance=1e-6,
        random_state=42,
    )
