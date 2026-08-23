"""ClearGBM continued-training API.

More boosting rounds on top of an existing trained model. The design
inverts LightGBM's ``init_model`` shape on purpose: LightGBM bakes the old
model's raw predictions into the dataset as an init score and returns a
booster holding only the NEW trees — a delta model that excludes its own
baseline. Here the continuation's starting scores are initialized the same
way (the existing model's predictions over the continuation data), but the
new trees are APPENDED, so the returned handle is one self-contained model
whose embedded config states the combined round budget.

The continuation trains under the model's OWN embedded config — objective,
learning rate, tree shape, sampling knobs, early stopping. The caller
states only the data and the additional round budget. Supported for the
single-score objectives (``"binary_log_loss"``, ``"squared_error"``); a
multiclass or ranking model is refused with the scope named.

Strict typing only: no ``Any``, no ``cast``, no ``type: ignore``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm._rust import (
    PyGbmModelProto,
    continue_gradient_boosting_regression_rs,
    continue_gradient_boosting_rs,
)


def continue_gradient_boosting(
    model: PyGbmModelProto,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    additional_rounds: int,
    *,
    sample_weight: NDArray[np.float64] | None = None,
    val_sample_weight: NDArray[np.float64] | None = None,
    n_jobs: int = 1,
) -> PyGbmModelProto:
    """Continue a binary-classification model with more boosting rounds.

    Args:
        model: The existing trained ``PyGbmModel`` (objective
            ``"binary_log_loss"``); it is not modified.
        x_train: Continuation feature matrix ``(n_samples, n_features)``;
            the column count must match the model's features.
        y_train: Continuation labels (``0`` or ``1``).
        x_val: Optional validation feature matrix for the embedded
            config's early stopping.
        y_val: Optional validation labels.
        additional_rounds: New boosting rounds to run (>= 1).
        sample_weight: Optional per-row training weights (finite, > 0).
        val_sample_weight: Optional per-row evaluation weights; requires
            ``x_val``/``y_val``.
        n_jobs: Worker-thread policy for this run (1 = sequential, -1 =
            all cores); runtime state, never persisted.

    Returns:
        A NEW self-contained ``PyGbmModel``: the existing trees plus the
        continuation trees.

    Raises:
        ValueError: On shape mismatches, an invalid label/weight, a
            partial validation pair, a non-positive round budget, or a
            model whose objective is not ``"binary_log_loss"``.
        RuntimeError: Propagated from the native trainer on Rust-side
            error.
    """
    return continue_gradient_boosting_rs(
        model,
        x_train,
        y_train,
        sample_weight,
        x_val,
        y_val,
        val_sample_weight,
        additional_rounds,
        n_jobs,
    )


def continue_gradient_boosting_regression(
    model: PyGbmModelProto,
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.float64] | None,
    additional_rounds: int,
    *,
    sample_weight: NDArray[np.float64] | None = None,
    val_sample_weight: NDArray[np.float64] | None = None,
    n_jobs: int = 1,
) -> PyGbmModelProto:
    """Continue a squared-error regression model with more boosting rounds.

    Args:
        model: The existing trained ``PyGbmModel`` (objective
            ``"squared_error"``); it is not modified.
        x_train: Continuation feature matrix ``(n_samples, n_features)``;
            the column count must match the model's features.
        y_train: Continuous continuation targets, each finite.
        x_val: Optional validation feature matrix for the embedded
            config's early stopping.
        y_val: Optional continuous validation targets.
        additional_rounds: New boosting rounds to run (>= 1).
        sample_weight: Optional per-row training weights (finite, > 0).
        val_sample_weight: Optional per-row evaluation weights; requires
            ``x_val``/``y_val``.
        n_jobs: Worker-thread policy for this run (1 = sequential, -1 =
            all cores); runtime state, never persisted.

    Returns:
        A NEW self-contained ``PyGbmModel``: the existing trees plus the
        continuation trees.

    Raises:
        ValueError: On shape mismatches, an invalid target/weight, a
            partial validation pair, a non-positive round budget, or a
            model whose objective is not ``"squared_error"``.
        RuntimeError: Propagated from the native trainer on Rust-side
            error.
    """
    return continue_gradient_boosting_regression_rs(
        model,
        x_train,
        y_train,
        sample_weight,
        x_val,
        y_val,
        val_sample_weight,
        additional_rounds,
        n_jobs,
    )


__all__ = [
    "continue_gradient_boosting",
    "continue_gradient_boosting_regression",
]
