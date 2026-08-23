"""ClearGBM LambdaMART ranking training API.

The ``lambdarank`` training surface, split from :mod:`cleargbm.ensemble`
because its data contract differs: relevance labels are integer grades in
``[0, 31]`` (gain = ``2^label - 1``), rows travel with query group sizes
that partition them exactly, and there is no separate prediction surface —
the existing :func:`cleargbm.ensemble.predict_raw` scores a ranking model,
its raw score being the ranking key (documents sort by it, descending).

There is no validation-weight argument: NDCG is a per-query metric, and a
per-document evaluation weight has no defined meaning for it. Training
weights remain per-row data and multiply each row's lambda and hessian.

Strict typing only: no ``Any``, no ``cast``, no ``type: ignore``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm._rust import PyGbmModelProto, train_gradient_boosting_ranking_rs
from cleargbm.ensemble import _config_to_rust_dict, _validate_training_inputs
from cleargbm.types import GradientBoostingConfig


def train_gradient_boosting_ranking(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    group: NDArray[np.int64],
    x_val: NDArray[np.float64] | None,
    y_val: NDArray[np.int64] | None,
    val_group: NDArray[np.int64] | None,
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    *,
    sample_weight: NDArray[np.float64] | None = None,
) -> PyGbmModelProto:
    """Train a LambdaMART ranking gradient boosting ensemble.

    Validates input shapes then runs the entire training loop as a single
    Rust call: per query, documents sort by their running score and a
    truncation-bounded pair scan produces the lambda gradients
    (Burges 2010 / LightGBM's formulation, sigma fixed at 1, lambda
    normalization always on). Early stopping, when configured, minimizes
    ``1 - mean NDCG`` at the truncation level over the validation queries.

    Args:
        x_train: Training feature matrix ``(n_samples, n_features)``.
        y_train: Relevance grades in ``[0, 31]``, shape ``(n_samples,)``.
        group: Documents per query, in row order; sizes must partition the
            rows exactly, each query holding 1..=10000 documents.
        x_val: Optional validation feature matrix.
        y_val: Optional validation labels.
        val_group: Optional validation query group sizes; the three
            validation arguments travel together — all or none.
        config: Training configuration; ``config["objective"]`` must be
            ``"lambdarank"`` with ``config["lambdarank_truncation_level"]``
            an int >= 1 (the Rust boundary enforces the pairing).
        feature_names: Feature name tuple; length must match
            ``x_train.shape[1]``.
        sample_weight: Optional per-row training weights (finite, > 0),
            shape ``(n_samples,)``.

    Returns:
        Trained ``PyGbmModel`` handle; score it with
        :func:`cleargbm.ensemble.predict_raw`.

    Raises:
        ValueError: On any input shape or feature-name mismatch, an invalid
            label, group, or weight, or a partial validation triple.
        RuntimeError: Propagated from the native trainer on Rust-side error.
    """
    _validate_training_inputs(x_train, y_train, feature_names)
    rust_config = _config_to_rust_dict(config)
    names_list: list[str] = list(feature_names)
    return train_gradient_boosting_ranking_rs(
        x_train,
        y_train,
        group,
        sample_weight,
        x_val,
        y_val,
        val_group,
        rust_config,
        names_list,
    )


__all__ = [
    "train_gradient_boosting_ranking",
]
