"""Tests for the ``cleargbm.ensemble_continued`` public API.

These exercise the continuation Python boundary in front of the Rust core:
the split-equals-fresh identity, artifact growth, and the boundary
rejections.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cleargbm.ensemble import (
    predict_raw,
    train_gradient_boosting,
    train_gradient_boosting_regression,
)
from cleargbm.ensemble_continued import (
    continue_gradient_boosting,
    continue_gradient_boosting_regression,
)
from cleargbm.ensemble_multiclass import train_gradient_boosting_multiclass
from tests.conftest import make_config


def _binary_data() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Eight separable rows on two features."""
    x: NDArray[np.float64] = np.zeros((8, 2), dtype=np.float64)
    y: NDArray[np.int64] = np.zeros(8, dtype=np.int64)
    for i in range(8):
        x[i, 0] = float(i) / 8.0
        x[i, 1] = float(i) / 8.0
        y[i] = 1 if i >= 4 else 0
    return x, y


class TestContinueBinary:
    """Continuation extends the model and matches a fresh longer run."""

    def test_three_plus_three_equals_a_fresh_six_round_run(self) -> None:
        """Split training reproduces one long run bit for bit."""
        x, y = _binary_data()
        base = train_gradient_boosting(x, y, None, None, make_config(n_estimators=3), ("f0", "f1"))
        continued = continue_gradient_boosting(base, x, y, None, None, 3)
        fresh = train_gradient_boosting(x, y, None, None, make_config(n_estimators=6), ("f0", "f1"))
        continued_preds = predict_raw(continued, x)
        fresh_preds = predict_raw(fresh, x)
        assert np.array_equal(continued_preds, fresh_preds)

    def test_continuation_accepts_weights_and_validation(self) -> None:
        """The fully-populated call runs and returns a scoring model."""
        x, y = _binary_data()
        base = train_gradient_boosting(x, y, None, None, make_config(n_estimators=2), ("f0", "f1"))
        weights = np.ones(8, dtype=np.float64)
        continued = continue_gradient_boosting(
            base,
            x,
            y,
            x,
            y,
            2,
            sample_weight=weights,
            val_sample_weight=weights,
        )
        assert predict_raw(continued, x).shape == (8,)

    def test_zero_rounds_is_rejected(self) -> None:
        """A continuation must add at least one round."""
        x, y = _binary_data()
        base = train_gradient_boosting(x, y, None, None, make_config(n_estimators=2), ("f0", "f1"))
        with pytest.raises(ValueError, match="additional_rounds"):
            continue_gradient_boosting(base, x, y, None, None, 0)

    def test_a_multiclass_model_is_refused_with_the_scope_named(self) -> None:
        """Continuation supports single-score objectives only, stated."""
        x: NDArray[np.float64] = np.zeros((9, 2), dtype=np.float64)
        y: NDArray[np.int64] = np.zeros(9, dtype=np.int64)
        for i in range(9):
            x[i, 0] = float(10 * (i // 3) + i % 3)
            y[i] = i // 3
        mc_model = train_gradient_boosting_multiclass(
            x,
            y,
            None,
            None,
            make_config(
                n_estimators=2,
                n_classes=3,
                max_bins=16,
                objective="multiclass_softmax",
                scale_pos_weight=None,
            ),
            ("f0", "f1"),
        )
        with pytest.raises(ValueError, match=r"multiclass_softmax.* not implemented"):
            continue_gradient_boosting(mc_model, x, y, None, None, 1)


class TestContinueRegression:
    """The regression continuation mirrors the binary one."""

    def test_two_plus_two_equals_a_fresh_four_round_run(self) -> None:
        """Split regression training reproduces one long run bit for bit."""
        x: NDArray[np.float64] = np.zeros((12, 2), dtype=np.float64)
        y: NDArray[np.float64] = np.zeros(12, dtype=np.float64)
        for i in range(12):
            x[i, 0] = float(i)
            y[i] = 2.0 * float(i)
        config2 = make_config(n_estimators=2, objective="squared_error", scale_pos_weight=None)
        config4 = make_config(n_estimators=4, objective="squared_error", scale_pos_weight=None)
        base = train_gradient_boosting_regression(x, y, None, None, config2, ("f0", "f1"))
        continued = continue_gradient_boosting_regression(base, x, y, None, None, 2)
        fresh = train_gradient_boosting_regression(x, y, None, None, config4, ("f0", "f1"))
        assert np.array_equal(predict_raw(continued, x), predict_raw(fresh, x))

    def test_wrong_label_kind_is_rejected(self) -> None:
        """Continuing a binary model through the regression entry fails."""
        x, y = _binary_data()
        base = train_gradient_boosting(x, y, None, None, make_config(n_estimators=2), ("f0", "f1"))
        targets = y.astype(np.float64)
        with pytest.raises(ValueError, match="binary \\(u8\\) labels"):
            continue_gradient_boosting_regression(base, x, targets, None, None, 1)
