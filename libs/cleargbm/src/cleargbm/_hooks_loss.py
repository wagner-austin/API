"""Loss function backend hooks for cleargbm.

Binary log loss computation, gradients, hessians, and initial prediction.
Tests inject fakes, production uses real implementations.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class BinaryLogLossBackend(Protocol):
    """Protocol for binary log loss computation backend."""

    def __call__(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> float:
        """Compute mean binary cross-entropy loss.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities in (0, 1).

        Returns:
            Mean loss value.
        """
        ...


class BinaryLogLossGradientsBackend(Protocol):
    """Protocol for binary log loss gradients backend."""

    def __call__(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute gradients of binary log loss (p - y).

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities in (0, 1).

        Returns:
            Gradient for each sample.
        """
        ...


class BinaryLogLossHessiansBackend(Protocol):
    """Protocol for binary log loss hessians backend."""

    def __call__(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute hessians of binary log loss (p * (1-p)).

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities in (0, 1).

        Returns:
            Hessian for each sample.
        """
        ...


class BinaryLogLossInitialPredictionBackend(Protocol):
    """Protocol for binary log loss initial prediction backend."""

    def __call__(
        self,
        y_true: NDArray[np.int64],
    ) -> float:
        """Compute initial prediction (log-odds of positive class rate).

        Args:
            y_true: True labels (0 or 1).

        Returns:
            Initial prediction in log-odds space.
        """
        ...


def _default_binary_log_loss(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> float:
    """Python binary log loss implementation.

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted probabilities in (0, 1).

    Returns:
        Mean loss value.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
        ValueError: If y_true is empty.
    """
    n_true: int = int(y_true.shape[0])
    n_pred: int = int(y_pred.shape[0])
    if n_true != n_pred:
        raise ValueError(f"y_true and y_pred must have same length, got {n_true} and {n_pred}")
    if n_true == 0:
        raise ValueError("y_true must not be empty")

    eps = 1e-15
    p_clipped: NDArray[np.float64] = np.clip(y_pred, eps, 1.0 - eps)
    y_float: NDArray[np.float64] = y_true.astype(np.float64)
    losses: NDArray[np.float64] = -(
        y_float * np.log(p_clipped) + (1.0 - y_float) * np.log(1.0 - p_clipped)
    )
    total_loss: float = float(np.sum(losses))
    mean_loss: float = total_loss / n_true
    return mean_loss


def _default_binary_log_loss_gradients(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Python binary log loss gradients implementation (p - y).

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted probabilities in (0, 1).

    Returns:
        Gradient for each sample.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    n_true: int = int(y_true.shape[0])
    n_pred: int = int(y_pred.shape[0])
    if n_true != n_pred:
        raise ValueError(f"y_true and y_pred must have same length, got {n_true} and {n_pred}")
    y_float: NDArray[np.float64] = y_true.astype(np.float64)
    result: NDArray[np.float64] = y_pred - y_float
    return result


def _default_binary_log_loss_hessians(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Python binary log loss hessians implementation (p * (1-p)).

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted probabilities in (0, 1).

    Returns:
        Hessian for each sample.

    Raises:
        ValueError: If y_true and y_pred have different lengths.
    """
    n_true: int = int(y_true.shape[0])
    n_pred: int = int(y_pred.shape[0])
    if n_true != n_pred:
        raise ValueError(f"y_true and y_pred must have same length, got {n_true} and {n_pred}")
    eps = 1e-15
    p_clipped: NDArray[np.float64] = np.clip(y_pred, eps, 1.0 - eps)
    result: NDArray[np.float64] = p_clipped * (1.0 - p_clipped)
    return result


def _default_binary_log_loss_initial_prediction(
    y_true: NDArray[np.int64],
) -> float:
    """Python binary log loss initial prediction implementation.

    Computes log-odds of positive class rate.

    Args:
        y_true: True labels (0 or 1).

    Returns:
        Initial prediction in log-odds space.

    Raises:
        ValueError: If y_true is empty.
        ValueError: If all labels are the same.
    """
    n_total: int = int(y_true.shape[0])
    if n_total == 0:
        raise ValueError("y_true must not be empty")

    n_positive: int = int(np.sum(y_true))
    p_positive: float = n_positive / n_total

    eps = 1e-15
    if p_positive < eps:
        raise ValueError("Cannot compute initial prediction: all labels are 0")
    if p_positive > 1.0 - eps:
        raise ValueError("Cannot compute initial prediction: all labels are 1")

    return math.log(p_positive / (1.0 - p_positive))


# Module-level hooks for loss function backend.
# Production sets these to Rust implementations at startup.
_binary_log_loss_backend: BinaryLogLossBackend = _default_binary_log_loss
_binary_log_loss_gradients_backend: BinaryLogLossGradientsBackend = (
    _default_binary_log_loss_gradients
)
_binary_log_loss_hessians_backend: BinaryLogLossHessiansBackend = _default_binary_log_loss_hessians
_binary_log_loss_initial_prediction_backend: BinaryLogLossInitialPredictionBackend = (
    _default_binary_log_loss_initial_prediction
)


def binary_log_loss(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> float:
    """Compute mean binary cross-entropy loss.

    Delegates to the active backend hook.

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted probabilities in (0, 1).

    Returns:
        Mean loss value.
    """
    return _binary_log_loss_backend(y_true, y_pred)


def binary_log_loss_gradients(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute gradients of binary log loss.

    Delegates to the active backend hook.

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted probabilities in (0, 1).

    Returns:
        Gradient for each sample.
    """
    return _binary_log_loss_gradients_backend(y_true, y_pred)


def binary_log_loss_hessians(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute hessians of binary log loss.

    Delegates to the active backend hook.

    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted probabilities in (0, 1).

    Returns:
        Hessian for each sample.
    """
    return _binary_log_loss_hessians_backend(y_true, y_pred)


def binary_log_loss_initial_prediction(
    y_true: NDArray[np.int64],
) -> float:
    """Compute initial prediction for binary log loss.

    Delegates to the active backend hook.

    Args:
        y_true: True labels (0 or 1).

    Returns:
        Initial prediction in log-odds space.
    """
    return _binary_log_loss_initial_prediction_backend(y_true)


__all__ = [
    "BinaryLogLossBackend",
    "BinaryLogLossGradientsBackend",
    "BinaryLogLossHessiansBackend",
    "BinaryLogLossInitialPredictionBackend",
    "binary_log_loss",
    "binary_log_loss_gradients",
    "binary_log_loss_hessians",
    "binary_log_loss_initial_prediction",
]
