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


def py_gbm_model_to_json_rs(model: PyGbmModel) -> str:
    """Serialize a PyGbmModel to a JSON string.

    Args:
        model: Trained PyGbmModel.

    Returns:
        JSON string representation of the model. Round-trips through
        ``py_gbm_model_from_json_rs`` without loss beyond one ULP on float text
        representation; per-sample predictions match at 1e-15 tolerance (see
        Rust test ``test_model_roundtrip_predictions_identical``).

    Raises:
        ImportError: When native extension is not built.
        RuntimeError: If serialization fails at native layer.
    """
    raise ImportError(NOT_BUILT_MSG)


def py_gbm_model_from_json_rs(json_str: str) -> PyGbmModel:
    """Deserialize a PyGbmModel from a JSON string.

    Args:
        json_str: JSON string previously produced by ``py_gbm_model_to_json_rs``.

    Returns:
        A new PyGbmModel wrapping the decoded ensemble.

    Raises:
        ImportError: When native extension is not built.
        RuntimeError: On parse failures or on validation errors from the model's
            config validator (e.g. an invalid ``learning_rate`` value in the
            payload).
    """
    raise ImportError(NOT_BUILT_MSG)


def py_gbm_model_feature_importances_rs(model: PyGbmModel) -> list[tuple[str, float]]:
    """Return per-feature split-count importance, normalized to sum to 1.0.

    A feature that never appears at an internal (split) node has importance 0.0.
    If the ensemble has zero internal nodes (every tree is a single leaf), every
    feature has importance 0.0.

    Args:
        model: Trained PyGbmModel.

    Returns:
        List of ``(feature_name, importance)`` pairs in feature-index order.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def py_gbm_model_n_trees_rs(model: PyGbmModel) -> int:
    """Return the number of trees in a PyGbmModel ensemble.

    Args:
        model: Trained PyGbmModel.

    Returns:
        Tree count (equal to ``n_estimators`` unless early stopping trimmed the
        ensemble).

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def py_gbm_model_n_classes_rs(model: PyGbmModel) -> int:
    """Return the number of classes in a PyGbmModel.

    Always ``2`` for binary classification; the current library only trains
    binary classifiers.

    Args:
        model: Trained PyGbmModel.

    Returns:
        Class count (2).

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


__all__ = [
    "PyGbmModel",
    "predict_proba_model_rs",
    "predict_raw_model_rs",
    "py_gbm_model_feature_importances_rs",
    "py_gbm_model_from_json_rs",
    "py_gbm_model_n_classes_rs",
    "py_gbm_model_n_trees_rs",
    "py_gbm_model_to_json_rs",
    "train_gradient_boosting_rs",
]
