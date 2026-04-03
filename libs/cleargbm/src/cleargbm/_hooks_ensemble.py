"""Ensemble prediction backend hooks for cleargbm.

Raw ensemble prediction and probability conversion. Tests inject fakes,
production uses real implementations.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm._hooks_prediction import predict_tree
from cleargbm._hooks_sigmoid import sigmoid
from cleargbm.types import DecisionTree


class PredictRawBackend(Protocol):
    """Protocol for ensemble raw prediction backend."""

    def __call__(
        self,
        trees: tuple[DecisionTree, ...],
        features: NDArray[np.float64],
        base_prediction: float,
        learning_rate: float,
    ) -> NDArray[np.float64]:
        """Predict raw scores from an ensemble of trees.

        Args:
            trees: Trained decision trees.
            features: Feature matrix (n_samples, n_features).
            base_prediction: Initial prediction before any tree contributions.
            learning_rate: Shrinkage factor for tree contributions.

        Returns:
            Raw predictions (log-odds) for each sample.
        """
        ...


class PredictProbaBackend(Protocol):
    """Protocol for probability prediction backend."""

    def __call__(
        self,
        raw_predictions: NDArray[np.float64],
    ) -> tuple[tuple[float, float], ...]:
        """Convert raw predictions to class probabilities.

        Args:
            raw_predictions: Raw predictions (log-odds).

        Returns:
            Tuple of (prob_class_0, prob_class_1) per sample.
        """
        ...


def _default_predict_raw(
    trees: tuple[DecisionTree, ...],
    features: NDArray[np.float64],
    base_prediction: float,
    learning_rate: float,
) -> NDArray[np.float64]:
    """Python ensemble raw prediction implementation.

    Loops over trees, calling the predict_tree hook for each.

    Args:
        trees: Trained decision trees.
        features: Feature matrix (n_samples, n_features).
        base_prediction: Initial prediction before any tree contributions.
        learning_rate: Shrinkage factor for tree contributions.

    Returns:
        Raw predictions (log-odds) for each sample.
    """
    n_samples: int = int(features.shape[0])
    raw_preds: NDArray[np.float64] = np.full(n_samples, base_prediction, dtype=np.float64)
    for tree in trees:
        tree_preds = predict_tree(tree, features)
        raw_preds = raw_preds + learning_rate * tree_preds
    return raw_preds


def _default_predict_proba(
    raw_predictions: NDArray[np.float64],
) -> tuple[tuple[float, float], ...]:
    """Python probability prediction implementation.

    Applies sigmoid to each raw prediction to get class probabilities.

    Args:
        raw_predictions: Raw predictions (log-odds).

    Returns:
        Tuple of (prob_class_0, prob_class_1) per sample.
    """
    n_samples: int = int(raw_predictions.shape[0])
    result: list[tuple[float, float]] = []
    for i in range(n_samples):
        raw_val: float = raw_predictions.item(i)
        prob_1 = sigmoid(raw_val)
        prob_0 = 1.0 - prob_1
        result.append((prob_0, prob_1))
    return tuple(result)


# Module-level hooks for ensemble prediction backend.
# Production sets these to Rust implementations at startup.
_predict_raw_backend: PredictRawBackend = _default_predict_raw
_predict_proba_backend: PredictProbaBackend = _default_predict_proba


def predict_raw_ensemble(
    trees: tuple[DecisionTree, ...],
    features: NDArray[np.float64],
    base_prediction: float,
    learning_rate: float,
) -> NDArray[np.float64]:
    """Predict raw scores from an ensemble of trees.

    Delegates to the active backend hook.

    Args:
        trees: Trained decision trees.
        features: Feature matrix (n_samples, n_features).
        base_prediction: Initial prediction before any tree contributions.
        learning_rate: Shrinkage factor for tree contributions.

    Returns:
        Raw predictions (log-odds) for each sample.
    """
    return _predict_raw_backend(trees, features, base_prediction, learning_rate)


def predict_proba_from_raw(
    raw_predictions: NDArray[np.float64],
) -> tuple[tuple[float, float], ...]:
    """Convert raw predictions to class probabilities.

    Delegates to the active backend hook.

    Args:
        raw_predictions: Raw predictions (log-odds).

    Returns:
        Tuple of (prob_class_0, prob_class_1) per sample.
    """
    return _predict_proba_backend(raw_predictions)


__all__ = [
    "PredictProbaBackend",
    "PredictRawBackend",
    "predict_proba_from_raw",
    "predict_raw_ensemble",
]
