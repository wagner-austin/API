"""Shared fixtures and helpers for test_regression_runner splits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _make_intp_array(values: tuple[int, ...]) -> NDArray[np.intp]:
    """Create intp array from tuple of ints."""
    result: NDArray[np.intp] = np.zeros(len(values), dtype=np.intp)
    for i, v in enumerate(values):
        result[i] = v
    return result


def _get_val(arr: NDArray[np.float64], idx: int) -> float:
    """Get value at index with proper typing."""
    return float(arr.item(idx))


class SimpleLinearRegressor:
    """Simple linear regressor for testing.

    Predicts y = slope * x[0] + intercept, where slope and intercept
    are computed from training data.
    """

    def __init__(self, slope: float, intercept: float) -> None:
        """Initialize regressor.

        Args:
            slope: Coefficient for first feature.
            intercept: Bias term.
        """
        self._slope = slope
        self._intercept = intercept

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix.

        Returns:
            Predicted values of shape (n_samples,).
        """
        n_samples = int(x.shape[0])
        preds: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            feat0 = float(x.item((i, 0)))
            preds[i] = self._slope * feat0 + self._intercept
        return preds


def simple_regressor_trainer(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_val: NDArray[np.float64],
    y_val: NDArray[np.float64],
    fold_number: int,
) -> SimpleLinearRegressor:
    """Train simple regressor for testing.

    Computes slope and intercept from training data using first feature.

    Args:
        x_train: Training features.
        y_train: Training targets.
        x_val: Validation features (unused).
        y_val: Validation targets (unused).
        fold_number: Fold number (unused).

    Returns:
        Trained SimpleLinearRegressor.
    """
    _ = x_val, y_val, fold_number

    n_train = int(x_train.shape[0])

    # Compute means
    x_sum = 0.0
    y_sum = 0.0
    for i in range(n_train):
        x_sum += float(x_train.item((i, 0)))
        y_sum += float(y_train.item(i))
    x_mean = x_sum / max(1, n_train)
    y_mean = y_sum / max(1, n_train)

    # Compute slope via covariance / variance
    cov_sum = 0.0
    var_sum = 0.0
    for i in range(n_train):
        xd = float(x_train.item((i, 0))) - x_mean
        yd = float(y_train.item(i)) - y_mean
        cov_sum += xd * yd
        var_sum += xd * xd

    slope = cov_sum / max(1e-10, var_sum)
    intercept = y_mean - slope * x_mean

    return SimpleLinearRegressor(slope, intercept)
