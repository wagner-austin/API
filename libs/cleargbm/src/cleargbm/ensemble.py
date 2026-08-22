"""ClearGBM public training + prediction API.

Rust is the only compute path. The Python surface validates inputs at the
boundary and dispatches directly into the ``cleargbm_rs`` extension module.
There is no Python-fallback implementation and no per-primitive hook layer —
the entire training loop and prediction pipeline runs inside a single native
Rust call.

Public API:

- :func:`train_gradient_boosting` — train an ensemble; returns an opaque
  ``PyGbmModel`` handle.
- :func:`predict_proba` — per-sample class probabilities from a trained model.
- :func:`predict_raw` — per-sample raw log-odds from a trained model.

Strict typing only: no ``Any``, no ``cast``, no ``type: ignore``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm._rust import (
    PyGbmModelProto,
    predict_proba_model_rs,
    predict_raw_model_rs,
    py_gbm_model_to_json_rs,
    train_gradient_boosting_rs,
)
from cleargbm.types import GradientBoostingConfig


def _validate_training_inputs(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    feature_names: tuple[str, ...],
) -> None:
    """Validate training input shapes at the Python boundary.

    Args:
        x_train: Training feature matrix.
        y_train: Training labels.
        feature_names: Feature name tuple.

    Raises:
        ValueError: If ``x_train`` is empty, if row counts don't match, or if
            the feature-name count doesn't match the column count.
    """
    n_train: int = int(x_train.shape[0])
    if n_train == 0:
        raise ValueError("x_train must not be empty")
    n_y: int = int(y_train.shape[0])
    if n_train != n_y:
        raise ValueError(f"x_train and y_train must have same length, got {n_train} and {n_y}")
    n_features: int = int(x_train.shape[1])
    if n_features != len(feature_names):
        raise ValueError(
            f"x_train has {n_features} features but {len(feature_names)} feature names provided"
        )


def _config_to_rust_dict(
    config: GradientBoostingConfig,
) -> dict[str, int | float | bool | str | list[int] | None]:
    """Translate a Python ``GradientBoostingConfig`` into the Rust-side dict.

    The Rust training function extracts 15 hyperparameter fields plus
    ``n_jobs`` from the dict it receives. ``n_jobs`` selects the worker-thread
    policy for the run and is deliberately not part of the Rust
    ``GradientBoostingConfig``: it does not change the fitted model, and that
    config is serialized into the saved model.

    ``growth_strategy`` crosses as the same string on both sides
    (``"depth_wise"`` / ``"leaf_wise"``) rather than being re-encoded, so the
    policy has exactly one spelling everywhere it appears.

    Every config field crosses to Rust; nothing is dropped. ``max_features``
    became a real per-split feature budget on 2026-08-22, and
    ``track_contributions`` was removed from the config outright the same
    day — contribution extraction is a post-hoc explainer capability over
    the saved model, not a training knob, and a config field the trainer
    ignores is the defect class this codebase spent the day eradicating.

    ``monotonic_constraints`` is stored as a tuple of ints on the Python side
    and as a list of enum-variant strings on the Rust side; this function
    performs that translation.

    Args:
        config: Python-side config.

    Returns:
        A dict shaped for ``cleargbm_rs.train_gradient_boosting_rs``.
    """
    mc = config["monotonic_constraints"]
    mc_list: list[int] | None = list(mc) if mc is not None else None
    return {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_bins": config["max_bins"],
        "subsample": config["subsample"],
        "random_state": config["random_state"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "monotonic_constraints": mc_list,
        "early_stopping_rounds": config["early_stopping_rounds"],
        "n_jobs": config["n_jobs"],
        "growth_strategy": config["growth_strategy"],
        "num_leaves": config["num_leaves"],
        "scale_pos_weight": config["scale_pos_weight"],
        "max_features": config["max_features"],
    }


def train_gradient_boosting(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
) -> PyGbmModelProto:
    """Train a binary-classification gradient boosting ensemble.

    Validates inputs then runs the entire training loop as a single Rust call.
    The returned handle is opaque to Python; pass it to :func:`predict_proba`,
    :func:`predict_raw`, or the module-level ``cleargbm_rs.py_gbm_model_*_rs``
    functions for JSON persistence and feature-importance extraction.

    Args:
        x_train: Training feature matrix ``(n_samples, n_features)``.
        y_train: Training labels (``0`` or ``1``), shape ``(n_samples,)``.
        x_val: Optional validation feature matrix.
        y_val: Optional validation labels.
        config: Training configuration.
        feature_names: Feature name tuple; length must match
            ``x_train.shape[1]``.

    Returns:
        Trained ``PyGbmModel`` handle.

    Raises:
        ValueError: On any input shape or feature-name mismatch.
        RuntimeError: Propagated from the native trainer on Rust-side error.
    """
    _validate_training_inputs(x_train, y_train, feature_names)
    rust_config = _config_to_rust_dict(config)
    names_list: list[str] = list(feature_names)
    return train_gradient_boosting_rs(
        x_train,
        y_train,
        x_val,
        y_val,
        rust_config,
        names_list,
    )


def predict_proba(
    model: PyGbmModelProto,
    x: NDArray[np.float64],
) -> tuple[tuple[float, float], ...]:
    """Predict class probabilities for a batch of samples.

    Args:
        model: Trained ``PyGbmModel`` handle.
        x: Feature matrix ``(n_samples, n_features)``.

    Returns:
        Tuple of ``(prob_class_0, prob_class_1)`` per sample.

    Raises:
        ValueError: If ``x`` is empty.
        RuntimeError: Propagated from the native predictor on Rust-side error.
    """
    if int(x.shape[0]) == 0:
        raise ValueError("x must not be empty")
    return predict_proba_model_rs(model, x)


def predict_raw(
    model: PyGbmModelProto,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict raw log-odds scores for a batch of samples.

    Args:
        model: Trained ``PyGbmModel`` handle.
        x: Feature matrix ``(n_samples, n_features)``.

    Returns:
        1D array of raw log-odds predictions, one per sample.

    Raises:
        ValueError: If ``x`` is empty.
        RuntimeError: Propagated from the native predictor on Rust-side error.
    """
    if int(x.shape[0]) == 0:
        raise ValueError("x must not be empty")
    return predict_raw_model_rs(model, x)


def export_model_json(model: PyGbmModelProto) -> str:
    """Serialize a trained model to its JSON representation.

    Exposes the full ensemble structure -- every tree, node, split threshold,
    leaf value and per-node sample count -- so callers can inspect what the
    model learned rather than treating it as a black box. This is the
    public entry point for model introspection and for round-tripping a model
    through :func:`load_model_json`.

    Args:
        model: Trained ``PyGbmModel`` handle.

    Returns:
        JSON document describing the ensemble.

    Raises:
        RuntimeError: Propagated from the native serializer on Rust-side error.
    """
    return py_gbm_model_to_json_rs(model)


__all__ = [
    "PyGbmModelProto",
    "export_model_json",
    "predict_proba",
    "predict_raw",
    "train_gradient_boosting",
]
