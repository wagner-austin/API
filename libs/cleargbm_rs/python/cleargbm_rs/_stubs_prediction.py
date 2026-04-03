"""Prediction function stubs for cleargbm_rs.

Mirrors ``pyo3_module/prediction_fns.rs``.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm_rs._constants import NOT_BUILT_MSG
from cleargbm_rs._stubs_tree import PyTree


def sigmoid_rs(x: float) -> float:
    """Compute sigmoid (logistic) function with numerical stability.

    Args:
        x: Input value (log-odds).

    Returns:
        Probability in [0, 1].

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def predict_single_rs(tree: PyTree, features: NDArray[np.float64]) -> float:
    """Predict leaf value for a single sample.

    Args:
        tree: Rust PyTree instance.
        features: 1D feature array for a single sample.

    Returns:
        Scalar prediction value.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def predict_tree_rs(
    tree: PyTree,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict leaf values for a batch of samples using a single tree.

    Args:
        tree: Rust PyTree instance.
        features: 2D feature matrix (n_samples, n_features).

    Returns:
        1D array of predictions.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def predict_ensemble_rs(
    trees: list[PyTree],
    features: NDArray[np.float64],
    base_prediction: float,
    learning_rate: float,
) -> NDArray[np.float64]:
    """Predict raw scores from an ensemble of trees.

    Args:
        trees: List of PyTree instances.
        features: 2D feature matrix (n_samples, n_features).
        base_prediction: Initial prediction before tree contributions.
        learning_rate: Shrinkage factor in (0, 1].

    Returns:
        1D array of raw predictions (log-odds).

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def predict_proba_rs(
    raw_predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert raw predictions to binary class probabilities.

    Args:
        raw_predictions: 1D array of raw predictions (log-odds).

    Returns:
        2D array of shape (n_samples, 2) with (prob_class_0, prob_class_1).

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


__all__ = [
    "predict_ensemble_rs",
    "predict_proba_rs",
    "predict_single_rs",
    "predict_tree_rs",
    "sigmoid_rs",
]
