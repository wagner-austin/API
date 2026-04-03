"""Loss function stubs for cleargbm_rs.

Mirrors ``pyo3_module/loss_fns.rs``.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm_rs._constants import NOT_BUILT_MSG


def binary_log_loss_rs(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> float:
    """Compute mean binary cross-entropy (log loss).

    Args:
        y_true: True labels as int64 array (0 or 1).
        y_pred: Predicted probabilities as float64 array.

    Returns:
        Mean loss value.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def binary_log_loss_gradients_rs(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute gradients of binary log loss (p - y).

    Args:
        y_true: True labels as int64 array (0 or 1).
        y_pred: Predicted probabilities as float64 array.

    Returns:
        Gradient array (float64).

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def binary_log_loss_hessians_rs(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute hessians of binary log loss (p * (1-p)).

    Args:
        y_true: True labels as int64 array (0 or 1).
        y_pred: Predicted probabilities as float64 array.

    Returns:
        Hessian array (float64).

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def binary_log_loss_initial_prediction_rs(
    y_true: NDArray[np.int64],
) -> float:
    """Compute initial prediction (log-odds of positive class rate).

    Args:
        y_true: True labels as int64 array (0 or 1).

    Returns:
        Initial prediction in log-odds space.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def sigmoid_array_rs(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply sigmoid to each element of an array.

    Args:
        x: Input array (log-odds).

    Returns:
        Array of probabilities.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


__all__ = [
    "binary_log_loss_gradients_rs",
    "binary_log_loss_hessians_rs",
    "binary_log_loss_initial_prediction_rs",
    "binary_log_loss_rs",
    "sigmoid_array_rs",
]
