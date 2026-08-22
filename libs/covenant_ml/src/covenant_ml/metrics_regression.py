"""Regression evaluation metrics (MSE, RMSE, MAE, R^2, MAPE)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from covenant_ml.types_regression import RegressionMetrics


def compute_mse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Compute mean squared error.

    Args:
        y_true: True continuous values, shape (n_samples,).
        y_pred: Predicted continuous values, shape (n_samples,).

    Returns:
        Mean squared error (lower is better, minimum 0.0).
    """
    diff: NDArray[np.float64] = y_true - y_pred
    squared: NDArray[np.float64] = diff * diff
    return float(np.sum(squared)) / len(squared)


def compute_rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Compute root mean squared error.

    Args:
        y_true: True continuous values, shape (n_samples,).
        y_pred: Predicted continuous values, shape (n_samples,).

    Returns:
        Root mean squared error (lower is better, minimum 0.0).
    """
    return math.sqrt(compute_mse(y_true, y_pred))


def compute_mae(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Compute mean absolute error.

    Args:
        y_true: True continuous values, shape (n_samples,).
        y_pred: Predicted continuous values, shape (n_samples,).

    Returns:
        Mean absolute error (lower is better, minimum 0.0).
    """
    abs_diff: NDArray[np.float64] = np.abs(y_true - y_pred)
    return float(np.sum(abs_diff)) / len(abs_diff)


def compute_r_squared(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """Compute coefficient of determination (R-squared).

    R² = 1 - SS_res / SS_tot where SS_res = sum((y - y_hat)²)
    and SS_tot = sum((y - y_mean)²).

    Args:
        y_true: True continuous values, shape (n_samples,).
        y_pred: Predicted continuous values, shape (n_samples,).

    Returns:
        R-squared (1.0 = perfect, 0.0 = mean baseline, negative = worse than mean).
    """
    residuals: NDArray[np.float64] = y_true - y_pred
    ss_res = float(np.sum(residuals * residuals))
    mean_true = float(np.sum(y_true)) / len(y_true)
    deviations: NDArray[np.float64] = y_true - mean_true
    ss_tot = float(np.sum(deviations * deviations))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_mape(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    eps: float = 1e-8,
) -> float:
    """Compute mean absolute percentage error.

    Uses eps to protect against division by near-zero true values.

    Args:
        y_true: True continuous values, shape (n_samples,).
        y_pred: Predicted continuous values, shape (n_samples,).
        eps: Small value added to denominator to avoid division by zero.

    Returns:
        Mean absolute percentage error (lower is better, minimum 0.0).
    """
    # Compute |y_true| + eps without np.abs (returns untyped in strict mypy)
    abs_true: NDArray[np.float64] = y_true.copy()
    neg_mask: NDArray[np.bool_] = y_true < 0
    abs_true[neg_mask] = -abs_true[neg_mask]
    abs_true = abs_true + eps
    # Compute |y_true - y_pred|
    diff: NDArray[np.float64] = y_true - y_pred
    abs_diff: NDArray[np.float64] = diff.copy()
    neg_diff_mask: NDArray[np.bool_] = diff < 0
    abs_diff[neg_diff_mask] = -abs_diff[neg_diff_mask]
    abs_pct_error: NDArray[np.float64] = abs_diff / abs_true
    return float(np.sum(abs_pct_error)) / len(abs_pct_error)


def compute_all_regression_metrics(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> RegressionMetrics:
    """Compute all regression evaluation metrics.

    Args:
        y_true: True continuous values, shape (n_samples,).
        y_pred: Predicted continuous values, shape (n_samples,).

    Returns:
        RegressionMetrics with mse, rmse, mae, r_squared, mape.
    """
    mse = compute_mse(y_true, y_pred)
    return RegressionMetrics(
        mse=mse,
        rmse=math.sqrt(mse),
        mae=compute_mae(y_true, y_pred),
        r_squared=compute_r_squared(y_true, y_pred),
        mape=compute_mape(y_true, y_pred),
    )


def format_regression_metrics_str(metrics: RegressionMetrics) -> str:
    """Format regression metrics as a human-readable string.

    Args:
        metrics: Computed regression metrics.

    Returns:
        Formatted string like "MSE=0.0123 RMSE=0.1109 MAE=0.0891 R²=0.9456 MAPE=0.0345".
    """
    return (
        f"MSE={metrics['mse']:.4f} "
        f"RMSE={metrics['rmse']:.4f} "
        f"MAE={metrics['mae']:.4f} "
        f"R²={metrics['r_squared']:.4f} "
        f"MAPE={metrics['mape']:.4f}"
    )


__all__ = [
    "compute_all_regression_metrics",
    "compute_mae",
    "compute_mape",
    "compute_mse",
    "compute_r_squared",
    "compute_rmse",
    "format_regression_metrics_str",
]
