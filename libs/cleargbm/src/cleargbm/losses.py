"""Loss functions with gradients and hessians for gradient boosting.

Uses numpy arrays for efficient vectorized operations.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm._test_hooks import sigmoid as _sigmoid_hook
from cleargbm._test_hooks import sigmoid_array as _sigmoid_array_hook


class LossFunction(Protocol):
    """Protocol for loss functions used in gradient boosting."""

    def loss(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> float:
        """Compute mean loss.

        Args:
            y_true: True labels (0 or 1).
            y_pred: Predicted probabilities.

        Returns:
            Mean loss value.
        """
        ...

    def gradients(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute gradients (first derivative of loss w.r.t. predictions).

        Args:
            y_true: True labels.
            y_pred: Predicted probabilities.

        Returns:
            Gradient for each sample.
        """
        ...

    def hessians(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute hessians (second derivative for Newton step).

        Args:
            y_true: True labels.
            y_pred: Predicted probabilities.

        Returns:
            Hessian for each sample.
        """
        ...

    def initial_prediction(
        self,
        y_true: NDArray[np.int64],
    ) -> float:
        """Compute initial prediction (before any trees).

        Args:
            y_true: True labels.

        Returns:
            Initial prediction in raw score space (log-odds for classification).
        """
        ...


def sigmoid(x: float) -> float:
    """Compute sigmoid function with numerical stability.

    Uses the active backend (Rust when available, Python fallback).

    Args:
        x: Input value (log-odds).

    Returns:
        Probability in [0, 1].

    Examples:
        >>> sigmoid(0.0)
        0.5
        >>> sigmoid(100.0) < 1.0
        True
        >>> sigmoid(-100.0) > 0.0
        True
    """
    return _sigmoid_hook(x)


def sigmoid_array(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute sigmoid function for array with numerical stability.

    Uses the active backend (Rust when available, Python fallback).

    Args:
        x: Input array (log-odds).

    Returns:
        Probabilities in [0, 1].
    """
    return _sigmoid_array_hook(x)


class BinaryLogLoss:
    """Binary cross-entropy (log loss) for binary classification.

    loss = -[y * log(p) + (1-y) * log(1-p)]
    gradient = p - y  (derivative of loss w.r.t. raw prediction)
    hessian = p * (1-p)  (second derivative)
    initial = log(p_mean / (1 - p_mean))  (log-odds of positive class rate)
    """

    def loss(
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

        # Clip predictions to avoid log(0)
        eps = 1e-15
        p_clipped: NDArray[np.float64] = np.clip(y_pred, eps, 1.0 - eps)

        # Vectorized log loss computation
        y_float: NDArray[np.float64] = y_true.astype(np.float64)
        losses: NDArray[np.float64] = -(
            y_float * np.log(p_clipped) + (1.0 - y_float) * np.log(1.0 - p_clipped)
        )
        total_loss: float = float(np.sum(losses))
        mean_loss: float = total_loss / n_true
        return mean_loss

    def gradients(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute gradients (p - y).

        The gradient of log loss with respect to the raw prediction
        (before sigmoid) is simply p - y, where p is the predicted probability.

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

    def hessians(
        self,
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute hessians (p * (1-p)).

        The second derivative of log loss is p * (1-p), which is always
        positive and represents the curvature of the loss function.

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
        # Clip to avoid numerical issues at boundaries
        eps = 1e-15
        p_clipped: NDArray[np.float64] = np.clip(y_pred, eps, 1.0 - eps)
        result: NDArray[np.float64] = p_clipped * (1.0 - p_clipped)
        return result

    def initial_prediction(
        self,
        y_true: NDArray[np.int64],
    ) -> float:
        """Compute initial prediction (log-odds of positive class rate).

        Args:
            y_true: True labels (0 or 1).

        Returns:
            Initial prediction in log-odds space.

        Raises:
            ValueError: If y_true is empty.
            ValueError: If all labels are the same (cannot compute log-odds).
        """
        n_total: int = int(y_true.shape[0])
        if n_total == 0:
            raise ValueError("y_true must not be empty")

        n_positive: int = int(np.sum(y_true))
        p_positive: float = n_positive / n_total

        # Handle edge cases where all samples are one class
        eps = 1e-15
        if p_positive < eps:
            raise ValueError("Cannot compute initial prediction: all labels are 0")
        if p_positive > 1.0 - eps:
            raise ValueError("Cannot compute initial prediction: all labels are 1")

        # Return log-odds
        return math.log(p_positive / (1.0 - p_positive))


def compute_raw_predictions(
    base_prediction: float,
    tree_predictions: tuple[NDArray[np.float64], ...],
    learning_rate: float,
) -> NDArray[np.float64]:
    """Compute raw predictions from base and tree contributions.

    Args:
        base_prediction: Initial prediction (log-odds).
        tree_predictions: Predictions from each tree.
        learning_rate: Learning rate multiplier for trees.

    Returns:
        Raw predictions (log-odds) for each sample.

    Raises:
        ValueError: If tree_predictions is empty.
        ValueError: If tree predictions have inconsistent lengths.
    """
    if len(tree_predictions) == 0:
        raise ValueError("tree_predictions must not be empty")

    n_samples: int = int(tree_predictions[0].shape[0])
    for i, preds in enumerate(tree_predictions):
        preds_len: int = int(preds.shape[0])
        if preds_len != n_samples:
            raise ValueError(
                f"All tree predictions must have same length. "
                f"tree_predictions[0] has {n_samples}, "
                f"tree_predictions[{i}] has {preds_len}"
            )

    # Start with base prediction for all samples
    raw_preds: NDArray[np.float64] = np.full(n_samples, base_prediction, dtype=np.float64)

    # Add scaled tree contributions
    for tree_preds in tree_predictions:
        raw_preds = raw_preds + learning_rate * tree_preds

    return raw_preds


def raw_to_proba(raw_predictions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert raw predictions (log-odds) to probabilities.

    Args:
        raw_predictions: Raw predictions in log-odds space.

    Returns:
        Probabilities in [0, 1].
    """
    return sigmoid_array(raw_predictions)


__all__ = [
    "BinaryLogLoss",
    "LossFunction",
    "compute_raw_predictions",
    "raw_to_proba",
    "sigmoid",
    "sigmoid_array",
]
