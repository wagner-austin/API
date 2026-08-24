"""Tests for the ``cleargbm.ensemble_multiclass`` public API.

These exercise the multiclass Python boundary in front of the Rust core:
end-to-end training on a small three-cluster dataset, the prediction trio
(raw scores, probabilities, argmax classes), and the boundary rejections.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cleargbm.ensemble_multiclass import (
    predict_class,
    predict_proba_multiclass,
    predict_raw_multiclass,
    train_gradient_boosting_multiclass,
)
from cleargbm.types import GradientBoostingConfig


def _make_multiclass_config(
    n_estimators: int = 5,
    n_classes: int = 3,
    early_stopping_rounds: int | None = None,
) -> GradientBoostingConfig:
    """Return a minimal valid multiclass training config."""
    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=2,
        learning_rate=0.3,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        colsample_bytree=None,
        categorical_features=None,
        n_classes=n_classes,
        lambdarank_truncation_level=None,
        goss_top_rate=None,
        goss_other_rate=None,
        quantized_gradient_bins=None,
        max_bins=16,
        subsample=1.0,
        random_state=42,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=1,
        early_stopping_rounds=early_stopping_rounds,
        growth_strategy="depth_wise",
        num_leaves=None,
        objective="multiclass_softmax",
        scale_pos_weight=None,
    )


def _cluster_data() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Three well-separated clusters at x = 0, 10, 20, three rows each."""
    x: NDArray[np.float64] = np.zeros((9, 2), dtype=np.float64)
    y: NDArray[np.int64] = np.zeros(9, dtype=np.int64)
    for i in range(9):
        x[i, 0] = float(10 * (i // 3) + i % 3)
        y[i] = i // 3
    return x, y


class TestTrainMulticlass:
    """Training on separable clusters and the boundary rejections."""

    def test_learns_three_separable_clusters(self) -> None:
        """Every training row must classify back to its own cluster."""
        x, y = _cluster_data()
        model = train_gradient_boosting_multiclass(
            x, y, None, None, _make_multiclass_config(), ("f0", "f1")
        )
        predicted = predict_class(model, x)
        assert np.array_equal(predicted, y)

    def test_train_with_validation_and_weights_runs(self) -> None:
        """The fully-populated call: weights, a validation split, val weights."""
        x, y = _cluster_data()
        weights = np.ones(9, dtype=np.float64)
        model = train_gradient_boosting_multiclass(
            x,
            y,
            x,
            y,
            _make_multiclass_config(early_stopping_rounds=2),
            ("f0", "f1"),
            sample_weight=weights,
            val_sample_weight=weights,
        )
        assert predict_class(model, x).shape == (9,)

    def test_train_empty_x_raises_at_boundary(self) -> None:
        """The Python boundary rejects an empty matrix before Rust sees it."""
        x = np.empty((0, 2), dtype=np.float64)
        y = np.empty((0,), dtype=np.int64)
        with pytest.raises(ValueError, match="x_train must not be empty"):
            train_gradient_boosting_multiclass(
                x, y, None, None, _make_multiclass_config(), ("f0", "f1")
            )

    def test_out_of_range_label_is_rejected_by_rust(self) -> None:
        """A label >= n_classes propagates the Rust-side rejection."""
        x, y = _cluster_data()
        bad = y.copy()
        bad[0] = 7
        with pytest.raises(ValueError, match=r"labels must be < n_classes \(3\), got 7 at index 0"):
            train_gradient_boosting_multiclass(
                x, bad, None, None, _make_multiclass_config(), ("f0", "f1")
            )


class TestPredictTrio:
    """The three prediction surfaces agree with each other."""

    def test_raw_scores_have_one_column_per_class(self) -> None:
        """Raw prediction is a (n_samples, n_classes) matrix."""
        x, y = _cluster_data()
        model = train_gradient_boosting_multiclass(
            x, y, None, None, _make_multiclass_config(), ("f0", "f1")
        )
        raw = predict_raw_multiclass(model, x)
        assert raw.shape == (9, 3)

    def test_probabilities_are_normalized_rows(self) -> None:
        """Each probability row sums to 1 with entries in [0, 1]."""
        x, y = _cluster_data()
        model = train_gradient_boosting_multiclass(
            x, y, None, None, _make_multiclass_config(), ("f0", "f1")
        )
        probas = predict_proba_multiclass(model, x)
        assert probas.shape == (9, 3)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)
        ones: NDArray[np.float64] = np.ones(3, dtype=np.float64)
        row_sums: NDArray[np.float64] = probas @ ones
        deviation: NDArray[np.float64] = np.abs(row_sums - 1.0)
        assert float(np.max(deviation)) < 1e-12

    def test_class_is_the_argmax_of_raw_scores(self) -> None:
        """The class vector must equal the row-wise argmax of the raw matrix."""
        x, y = _cluster_data()
        model = train_gradient_boosting_multiclass(
            x, y, None, None, _make_multiclass_config(), ("f0", "f1")
        )
        raw = predict_raw_multiclass(model, x)
        classes = predict_class(model, x)
        expected: NDArray[np.intp] = np.argmax(raw, axis=1)
        assert np.array_equal(classes, expected)

    def test_each_predictor_rejects_an_empty_matrix(self) -> None:
        """All three prediction entries refuse an empty input."""
        x, y = _cluster_data()
        model = train_gradient_boosting_multiclass(
            x, y, None, None, _make_multiclass_config(), ("f0", "f1")
        )
        empty = np.empty((0, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="x must not be empty"):
            predict_raw_multiclass(model, empty)
        with pytest.raises(ValueError, match="x must not be empty"):
            predict_proba_multiclass(model, empty)
        with pytest.raises(ValueError, match="x must not be empty"):
            predict_class(model, empty)
