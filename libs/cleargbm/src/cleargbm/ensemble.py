"""ClearGBM public training + prediction API.

Rust is the only compute path. The Python surface validates inputs at the
boundary and dispatches directly into the ``cleargbm_rs`` extension module.
There is no Python-fallback implementation and no per-primitive hook layer —
the entire training loop and prediction pipeline runs inside a single native
Rust call.

Public API:

- :func:`train_gradient_boosting` — train a binary-classification ensemble;
  returns an opaque ``PyGbmModel`` handle.
- :func:`train_gradient_boosting_regression` — train a squared-error
  regression ensemble on continuous ``f64`` targets.
- :func:`predict_proba` — per-sample class probabilities from a trained
  binary model (rejected for regression models).
- :func:`predict_raw` — per-sample raw scores from a trained model. Under
  ``binary_log_loss`` these are log-odds; under ``squared_error`` they ARE
  the predictions — this is the regression inference function.

The ``multiclass_softmax`` surface (its training entry and prediction trio)
lives in :mod:`cleargbm.ensemble_multiclass`, and the ``lambdarank``
training entry lives in :mod:`cleargbm.ensemble_ranking` — different
contracts, not variants of this one. A ranking model IS scored here: its
:func:`predict_raw` output is the ranking key.

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
    train_gradient_boosting_regression_rs,
    train_gradient_boosting_rs,
)
from cleargbm.types import GradientBoostingConfig


def _validate_training_inputs(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64] | NDArray[np.float64],
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

    The Rust training entries extract the full hyperparameter field set plus
    ``n_jobs`` from the dict they receive. ``n_jobs`` selects the
    worker-thread policy for the run and is deliberately not part of the Rust
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
        "objective": config["objective"],
        "scale_pos_weight": config["scale_pos_weight"],
        "max_features": config["max_features"],
        "colsample_bytree": config["colsample_bytree"],
        "categorical_features": (
            list(config["categorical_features"])
            if config["categorical_features"] is not None
            else None
        ),
        "n_classes": config["n_classes"],
        "lambdarank_truncation_level": config["lambdarank_truncation_level"],
        "goss_top_rate": config["goss_top_rate"],
        "goss_other_rate": config["goss_other_rate"],
    }


def train_gradient_boosting(
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
    """Train a binary-classification gradient boosting ensemble.

    Validates inputs then runs the entire training loop as a single Rust call.
    The returned handle is opaque to Python; pass it to :func:`predict_proba`,
    :func:`predict_raw`, or the module-level ``cleargbm_rs.py_gbm_model_*_rs``
    functions for JSON persistence and feature-importance extraction.

    Weights are data, not configuration: ``sample_weight=None`` weighs
    every row 1 and is bit-identical to weightless history, so the keyword
    default cannot silently change semantics the way a config default
    could. The Rust boundary validates weight length and positivity.

    Args:
        x_train: Training feature matrix ``(n_samples, n_features)``.
        y_train: Training labels (``0`` or ``1``), shape ``(n_samples,)``.
        x_val: Optional validation feature matrix.
        y_val: Optional validation labels.
        config: Training configuration; ``config["objective"]`` must be
            ``"binary_log_loss"`` (the Rust boundary rejects a mismatch
            between entry and objective).
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
        RuntimeError: Propagated from the native trainer on Rust-side error.
    """
    _validate_training_inputs(x_train, y_train, feature_names)
    rust_config = _config_to_rust_dict(config)
    names_list: list[str] = list(feature_names)
    return train_gradient_boosting_rs(
        x_train,
        y_train,
        sample_weight,
        x_val,
        y_val,
        val_sample_weight,
        rust_config,
        names_list,
    )


def train_gradient_boosting_regression(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.float64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    *,
    sample_weight: NDArray[np.float64] | None = None,
    val_sample_weight: NDArray[np.float64] | None = None,
) -> PyGbmModelProto:
    """Train a squared-error regression gradient boosting ensemble.

    Targets are continuous ``f64`` values; each must be finite. Predictions
    come from :func:`predict_raw` — under ``squared_error`` the raw score IS
    the prediction, and :func:`predict_proba` is rejected for the returned
    model.

    Weights are data, not configuration: ``sample_weight=None`` weighs
    every row 1 and is bit-identical to weightless history — the honest
    encoding of instrument or measurement confidence in scientific corpora.

    Args:
        x_train: Training feature matrix ``(n_samples, n_features)``.
        y_train: Continuous training targets, shape ``(n_samples,)``.
        x_val: Optional validation feature matrix.
        y_val: Optional continuous validation targets.
        config: Training configuration; ``config["objective"]`` must be
            ``"squared_error"`` (the Rust boundary rejects a mismatch
            between entry and objective).
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
        RuntimeError: Propagated from the native trainer on Rust-side error,
            including non-finite targets and objective/entry mismatches.
    """
    _validate_training_inputs(x_train, y_train, feature_names)
    rust_config = _config_to_rust_dict(config)
    names_list: list[str] = list(feature_names)
    return train_gradient_boosting_regression_rs(
        x_train,
        y_train,
        sample_weight,
        x_val,
        y_val,
        val_sample_weight,
        rust_config,
        names_list,
    )


def predict_proba(
    model: PyGbmModelProto,
    x: NDArray[np.float64],
) -> tuple[tuple[float, float], ...]:
    """Predict class probabilities for a batch of samples.

    Only meaningful for a model trained under ``binary_log_loss``; a
    ``squared_error`` model's raw scores are predictions, not log-odds, so
    the native layer rejects the call rather than squashing them through a
    sigmoid.

    Args:
        model: Trained ``PyGbmModel`` handle.
        x: Feature matrix ``(n_samples, n_features)``.

    Returns:
        Tuple of ``(prob_class_0, prob_class_1)`` per sample.

    Raises:
        ValueError: If ``x`` is empty.
        RuntimeError: Propagated from the native predictor on Rust-side
            error, including a model whose objective is ``squared_error``.
    """
    if int(x.shape[0]) == 0:
        raise ValueError("x must not be empty")
    return predict_proba_model_rs(model, x)


def predict_raw(
    model: PyGbmModelProto,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict raw scores for a batch of samples.

    Under ``binary_log_loss`` the raw score is a log-odds; under
    ``squared_error`` it is the prediction itself — this is the regression
    inference function.

    Args:
        model: Trained ``PyGbmModel`` handle.
        x: Feature matrix ``(n_samples, n_features)``.

    Returns:
        1D array of raw predictions, one per sample.

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
    "train_gradient_boosting_regression",
]
