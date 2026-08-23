"""ClearGBM multiclass training + prediction API.

The ``multiclass_softmax`` surface, split from :mod:`cleargbm.ensemble`
because it is a different contract, not a variant of the binary one: labels
are class indices in ``[0, n_classes)``, each boosting round trains
``n_classes`` trees, and prediction is a trio — a raw score matrix, a
softmax probability matrix, and an argmax class vector — rather than the
single-score pair.

Public API:

- :func:`train_gradient_boosting_multiclass` — train a multiclass softmax
  ensemble; returns an opaque ``PyGbmModel`` handle.
- :func:`predict_raw_multiclass` — per-sample raw per-class scores.
- :func:`predict_proba_multiclass` — per-sample class probabilities
  (softmax over the raw scores; rows sum to 1).
- :func:`predict_class` — per-sample argmax class index (ties resolve to
  the lowest index).

Strict typing only: no ``Any``, no ``cast``, no ``type: ignore``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm._rust import (
    PyGbmModelProto,
    predict_class_model_rs,
    predict_proba_multiclass_model_rs,
    predict_raw_multiclass_model_rs,
    train_gradient_boosting_multiclass_rs,
)
from cleargbm.ensemble import _config_to_rust_dict, _validate_training_inputs
from cleargbm.types import GradientBoostingConfig


def train_gradient_boosting_multiclass(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    *,
    sample_weight: NDArray[np.float64] | None = None,
    val_sample_weight: NDArray[np.float64] | None = None,
) -> PyGbmModelProto:
    """Train a multiclass softmax gradient boosting ensemble.

    Validates input shapes then runs the entire training loop as a single
    Rust call. Each boosting round trains ``config["n_classes"]`` trees —
    one per class against its softmax gradient — and early stopping
    truncates whole rounds so the stored tree count is always a multiple
    of the class count.

    Weights are data, not configuration: ``sample_weight=None`` weighs
    every row 1 and is bit-identical to weightless history. The Rust
    boundary validates weight length and positivity.

    Args:
        x_train: Training feature matrix ``(n_samples, n_features)``.
        y_train: Class labels in ``[0, n_classes)``, shape ``(n_samples,)``.
        x_val: Optional validation feature matrix.
        y_val: Optional validation labels.
        config: Training configuration; ``config["objective"]`` must be
            ``"multiclass_softmax"`` with ``config["n_classes"]`` an int
            >= 2 (the Rust boundary enforces the pairing).
        feature_names: Feature name tuple; length must match
            ``x_train.shape[1]``.
        sample_weight: Optional per-row training weights (finite, > 0),
            shape ``(n_samples,)``.
        val_sample_weight: Optional per-row evaluation weights for the
            validation split; requires ``x_val``/``y_val``.

    Returns:
        Trained ``PyGbmModel`` handle.

    Raises:
        ValueError: On any input shape or feature-name mismatch, or an
            invalid weight.
        RuntimeError: Propagated from the native trainer on Rust-side
            error, including an out-of-range label or an
            objective/``n_classes`` pairing violation.
    """
    _validate_training_inputs(x_train, y_train, feature_names)
    rust_config = _config_to_rust_dict(config)
    names_list: list[str] = list(feature_names)
    return train_gradient_boosting_multiclass_rs(
        x_train,
        y_train,
        sample_weight,
        x_val,
        y_val,
        val_sample_weight,
        rust_config,
        names_list,
    )


def predict_raw_multiclass(
    model: PyGbmModelProto,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict raw per-class scores for a batch of samples.

    The score matrix before the softmax: each row holds one un-normalized
    score per class (the class base score plus its trees' contributions).

    Args:
        model: Trained multiclass ``PyGbmModel`` handle.
        x: Feature matrix ``(n_samples, n_features)``.

    Returns:
        2D array of shape ``(n_samples, n_classes)``.

    Raises:
        ValueError: If ``x`` is empty.
        RuntimeError: Propagated from the native predictor on Rust-side
            error, including a model not trained under
            ``multiclass_softmax``.
    """
    if int(x.shape[0]) == 0:
        raise ValueError("x must not be empty")
    return predict_raw_multiclass_model_rs(model, x)


def predict_proba_multiclass(
    model: PyGbmModelProto,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict per-class probabilities for a batch of samples.

    The softmax of each row of raw scores (max-subtracted for numerical
    stability); every row sums to 1.

    Args:
        model: Trained multiclass ``PyGbmModel`` handle.
        x: Feature matrix ``(n_samples, n_features)``.

    Returns:
        2D array of shape ``(n_samples, n_classes)``; rows sum to 1.

    Raises:
        ValueError: If ``x`` is empty.
        RuntimeError: Propagated from the native predictor on Rust-side
            error, including a model not trained under
            ``multiclass_softmax``.
    """
    if int(x.shape[0]) == 0:
        raise ValueError("x must not be empty")
    return predict_proba_multiclass_model_rs(model, x)


def predict_class(
    model: PyGbmModelProto,
    x: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Predict class labels for a batch of samples.

    The argmax over each row's raw scores; ties resolve to the lowest
    class index, deterministically.

    Args:
        model: Trained multiclass ``PyGbmModel`` handle.
        x: Feature matrix ``(n_samples, n_features)``.

    Returns:
        1D array of class indices, one per sample.

    Raises:
        ValueError: If ``x`` is empty.
        RuntimeError: Propagated from the native predictor on Rust-side
            error, including a model not trained under
            ``multiclass_softmax``.
    """
    if int(x.shape[0]) == 0:
        raise ValueError("x must not be empty")
    return predict_class_model_rs(model, x)


__all__ = [
    "predict_class",
    "predict_proba_multiclass",
    "predict_raw_multiclass",
    "train_gradient_boosting_multiclass",
]
