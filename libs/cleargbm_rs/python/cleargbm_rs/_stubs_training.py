"""Training function stubs for cleargbm_rs.

Mirrors ``pyo3_module/training_fns.rs``.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm_rs._constants import NOT_BUILT_MSG


class PyGbmModel:
    """Opaque wrapper around a trained Rust GradientBoostingModel.

    Created by ``train_gradient_boosting_rs`` and consumed by
    ``predict_proba_model_rs`` and ``predict_raw_model_rs``.

    Avoids serialization overhead by keeping the model in Rust memory.

    Raises:
        ImportError: Always, when native extension is not built.
    """

    def __init__(self) -> None:
        """Create a PyGbmModel.

        Raises:
            ImportError: Always, when native extension is not built.
        """
        raise ImportError(NOT_BUILT_MSG)


def train_gradient_boosting_rs(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    config: dict[str, int | float | bool | list[int] | None],
    feature_names: list[str],
) -> PyGbmModel:
    """Train a gradient boosting model on binary classification data.

    Runs the full training loop in Rust: binning, iterative tree construction,
    gradient/hessian computation, optional early stopping, and model assembly.

    Args:
        x_train: 2D training feature matrix (n_samples, n_features).
        y_train: 1D training labels (0 or 1).
        x_val: Optional 2D validation feature matrix.
        y_val: Optional 1D validation labels.
        config: Training hyperparameters dict.
        feature_names: List of feature name strings.

    Returns:
        Trained PyGbmModel wrapping the Rust model.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def predict_proba_model_rs(
    model: PyGbmModel,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict class probabilities using a trained model.

    Args:
        model: Trained PyGbmModel.
        features: 2D feature matrix (n_samples, n_features).

    Returns:
        2D array of shape (n_samples, 2) with (prob_class_0, prob_class_1).

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def predict_raw_model_rs(
    model: PyGbmModel,
    features: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict raw log-odds using a trained model.

    Args:
        model: Trained PyGbmModel.
        features: 2D feature matrix (n_samples, n_features).

    Returns:
        1D array of raw predictions (log-odds).

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


__all__ = [
    "PyGbmModel",
    "predict_proba_model_rs",
    "predict_raw_model_rs",
    "train_gradient_boosting_rs",
]
