"""Tests for cleargbm.losses module.

Built from scratch - uses only Python stdlib (no numpy).
"""

from __future__ import annotations

import math

import pytest

from cleargbm.losses import (
    BinaryLogLoss,
    compute_raw_predictions,
    raw_to_proba,
    sigmoid,
    sigmoid_array,
)


class TestSigmoid:
    """Tests for sigmoid function."""

    def test_sigmoid_zero_returns_half(self) -> None:
        """sigmoid(0) should return 0.5."""
        result = sigmoid(0.0)
        assert abs(result - 0.5) < 1e-10

    def test_sigmoid_large_positive_approaches_one(self) -> None:
        """sigmoid(large positive) should approach 1.0."""
        result = sigmoid(100.0)
        # At 100.0, sigmoid is effectively 1.0 due to float precision
        assert result >= 0.99999

    def test_sigmoid_large_negative_approaches_zero(self) -> None:
        """sigmoid(large negative) should approach 0.0."""
        result = sigmoid(-100.0)
        assert result < 0.00001
        assert result > 0.0

    def test_sigmoid_extreme_positive_no_overflow(self) -> None:
        """sigmoid(very large) should not overflow (returns 1.0 due to clipping)."""
        result = sigmoid(1000.0)
        # At extreme values, returns 1.0 due to clipping at -500/+500
        assert result == 1.0

    def test_sigmoid_extreme_negative_no_underflow(self) -> None:
        """sigmoid(very small) should not underflow to exactly 0."""
        result = sigmoid(-1000.0)
        assert result > 0.0
        assert result < 0.00001

    def test_sigmoid_one_known_value(self) -> None:
        """sigmoid(1) should equal 1/(1+e^-1)."""
        result = sigmoid(1.0)
        expected = 1.0 / (1.0 + math.exp(-1.0))
        assert abs(result - expected) < 1e-10

    def test_sigmoid_negative_one_known_value(self) -> None:
        """sigmoid(-1) should equal 1/(1+e^1)."""
        result = sigmoid(-1.0)
        expected = 1.0 / (1.0 + math.exp(1.0))
        assert abs(result - expected) < 1e-10


class TestSigmoidArray:
    """Tests for sigmoid_array function."""

    def test_sigmoid_array_empty(self) -> None:
        """sigmoid_array([]) should return empty tuple."""
        result = sigmoid_array(())
        assert result == ()

    def test_sigmoid_array_single_element(self) -> None:
        """sigmoid_array with single element should work."""
        result = sigmoid_array((0.0,))
        assert len(result) == 1
        assert abs(result[0] - 0.5) < 1e-10

    def test_sigmoid_array_multiple_elements(self) -> None:
        """sigmoid_array with multiple elements should apply sigmoid to each."""
        result = sigmoid_array((-100.0, 0.0, 100.0))
        assert len(result) == 3
        assert result[0] < 0.00001  # sigmoid(-100)
        assert abs(result[1] - 0.5) < 1e-10  # sigmoid(0)
        assert result[2] > 0.99999  # sigmoid(100)


class TestBinaryLogLoss:
    """Tests for BinaryLogLoss class."""

    def test_loss_perfect_prediction_class_one(self) -> None:
        """Loss should be near zero for perfect predictions (label=1, pred=1)."""
        loss_fn = BinaryLogLoss()
        y_true = (1,)
        y_pred = (0.9999,)  # Near-perfect prediction

        loss = loss_fn.loss(y_true, y_pred)

        assert loss < 0.001

    def test_loss_perfect_prediction_class_zero(self) -> None:
        """Loss should be near zero for perfect predictions (label=0, pred=0)."""
        loss_fn = BinaryLogLoss()
        y_true = (0,)
        y_pred = (0.0001,)  # Near-perfect prediction

        loss = loss_fn.loss(y_true, y_pred)

        assert loss < 0.001

    def test_loss_terrible_prediction(self) -> None:
        """Loss should be high for bad predictions."""
        loss_fn = BinaryLogLoss()
        y_true = (1,)
        y_pred = (0.01,)  # Terrible prediction

        loss = loss_fn.loss(y_true, y_pred)

        assert loss > 4.0  # -log(0.01) ≈ 4.6

    def test_loss_mismatched_lengths_raises(self) -> None:
        """Loss should raise ValueError for mismatched lengths."""
        loss_fn = BinaryLogLoss()

        with pytest.raises(ValueError, match="same length"):
            loss_fn.loss((1, 0), (0.5,))

    def test_loss_empty_raises(self) -> None:
        """Loss should raise ValueError for empty input."""
        loss_fn = BinaryLogLoss()

        with pytest.raises(ValueError, match="not be empty"):
            loss_fn.loss((), ())

    def test_gradients_perfect_prediction(self) -> None:
        """Gradient should be near zero when prediction equals label."""
        loss_fn = BinaryLogLoss()
        y_true = (1, 0)
        y_pred = (1.0, 0.0)  # Perfect predictions (will be clipped internally)

        gradients = loss_fn.gradients(y_true, y_pred)

        # p - y should be ~0 for perfect predictions
        assert len(gradients) == 2
        assert abs(gradients[0]) < 0.0001  # 1.0 - 1 ≈ 0
        assert abs(gradients[1]) < 0.0001  # 0.0 - 0 ≈ 0

    def test_gradients_over_prediction(self) -> None:
        """Gradient should be positive when over-predicting (pred > label)."""
        loss_fn = BinaryLogLoss()
        y_true = (0,)  # Label is 0
        y_pred = (0.8,)  # Predicted 0.8

        gradients = loss_fn.gradients(y_true, y_pred)

        # p - y = 0.8 - 0 = 0.8 > 0
        assert gradients[0] > 0

    def test_gradients_under_prediction(self) -> None:
        """Gradient should be negative when under-predicting (pred < label)."""
        loss_fn = BinaryLogLoss()
        y_true = (1,)  # Label is 1
        y_pred = (0.2,)  # Predicted 0.2

        gradients = loss_fn.gradients(y_true, y_pred)

        # p - y = 0.2 - 1 = -0.8 < 0
        assert gradients[0] < 0

    def test_gradients_mismatched_lengths_raises(self) -> None:
        """Gradients should raise ValueError for mismatched lengths."""
        loss_fn = BinaryLogLoss()

        with pytest.raises(ValueError, match="same length"):
            loss_fn.gradients((1, 0, 1), (0.5, 0.5))

    def test_hessians_always_positive(self) -> None:
        """Hessian p*(1-p) should always be positive."""
        loss_fn = BinaryLogLoss()
        y_true = (0, 1, 0, 1)
        y_pred = (0.1, 0.3, 0.7, 0.9)

        hessians = loss_fn.hessians(y_true, y_pred)

        assert len(hessians) == 4
        for h in hessians:
            assert h > 0

    def test_hessians_max_at_half(self) -> None:
        """Hessian should be maximized at p=0.5."""
        loss_fn = BinaryLogLoss()
        y_true = (0, 0, 0)
        y_pred = (0.5, 0.1, 0.9)

        hessians = loss_fn.hessians(y_true, y_pred)

        # h = p * (1-p), max at p=0.5 where h=0.25
        assert abs(hessians[0] - 0.25) < 0.0001
        assert hessians[0] > hessians[1]  # 0.25 > 0.09
        assert hessians[0] > hessians[2]  # 0.25 > 0.09

    def test_hessians_known_values(self) -> None:
        """Hessian should equal p*(1-p) for known values."""
        loss_fn = BinaryLogLoss()
        y_true = (0,)
        y_pred = (0.3,)

        hessians = loss_fn.hessians(y_true, y_pred)

        expected = 0.3 * 0.7
        assert abs(hessians[0] - expected) < 1e-10

    def test_hessians_mismatched_lengths_raises(self) -> None:
        """Hessians should raise ValueError for mismatched lengths."""
        loss_fn = BinaryLogLoss()

        with pytest.raises(ValueError, match="same length"):
            loss_fn.hessians((1,), (0.5, 0.5))

    def test_initial_prediction_balanced_classes(self) -> None:
        """Initial prediction should be 0 for 50/50 class balance."""
        loss_fn = BinaryLogLoss()
        y_true = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1)  # 50% positive

        initial = loss_fn.initial_prediction(y_true)

        # log(0.5 / 0.5) = log(1) = 0
        assert abs(initial) < 1e-10

    def test_initial_prediction_mostly_positive(self) -> None:
        """Initial prediction should be positive for mostly positive labels."""
        loss_fn = BinaryLogLoss()
        y_true = (0, 1, 1, 1)  # 75% positive

        initial = loss_fn.initial_prediction(y_true)

        # log(0.75 / 0.25) = log(3) > 0
        assert initial > 0
        expected = math.log(0.75 / 0.25)
        assert abs(initial - expected) < 1e-10

    def test_initial_prediction_mostly_negative(self) -> None:
        """Initial prediction should be negative for mostly negative labels."""
        loss_fn = BinaryLogLoss()
        y_true = (0, 0, 0, 1)  # 25% positive

        initial = loss_fn.initial_prediction(y_true)

        # log(0.25 / 0.75) = log(1/3) < 0
        assert initial < 0
        expected = math.log(0.25 / 0.75)
        assert abs(initial - expected) < 1e-10

    def test_initial_prediction_empty_raises(self) -> None:
        """Initial prediction should raise ValueError for empty input."""
        loss_fn = BinaryLogLoss()

        with pytest.raises(ValueError, match="not be empty"):
            loss_fn.initial_prediction(())

    def test_initial_prediction_all_zeros_raises(self) -> None:
        """Initial prediction should raise ValueError when all labels are 0."""
        loss_fn = BinaryLogLoss()

        with pytest.raises(ValueError, match="all labels are 0"):
            loss_fn.initial_prediction((0, 0, 0))

    def test_initial_prediction_all_ones_raises(self) -> None:
        """Initial prediction should raise ValueError when all labels are 1."""
        loss_fn = BinaryLogLoss()

        with pytest.raises(ValueError, match="all labels are 1"):
            loss_fn.initial_prediction((1, 1, 1))


class TestComputeRawPredictions:
    """Tests for compute_raw_predictions function."""

    def test_single_tree_prediction(self) -> None:
        """Raw prediction with single tree should add tree contribution."""
        base = 0.5
        tree_preds = ((0.1, 0.2, 0.3),)
        learning_rate = 1.0

        result = compute_raw_predictions(base, tree_preds, learning_rate)

        assert len(result) == 3
        assert abs(result[0] - 0.6) < 1e-10  # 0.5 + 1.0 * 0.1
        assert abs(result[1] - 0.7) < 1e-10  # 0.5 + 1.0 * 0.2
        assert abs(result[2] - 0.8) < 1e-10  # 0.5 + 1.0 * 0.3

    def test_multiple_trees(self) -> None:
        """Raw prediction should sum contributions from all trees."""
        base = 0.0
        tree_preds = (
            (0.1, 0.2),
            (0.3, 0.4),
        )
        learning_rate = 1.0

        result = compute_raw_predictions(base, tree_preds, learning_rate)

        assert len(result) == 2
        assert abs(result[0] - 0.4) < 1e-10  # 0 + 0.1 + 0.3
        assert abs(result[1] - 0.6) < 1e-10  # 0 + 0.2 + 0.4

    def test_learning_rate_scaling(self) -> None:
        """Learning rate should scale tree contributions."""
        base = 0.0
        tree_preds = ((1.0, 2.0),)
        learning_rate = 0.1

        result = compute_raw_predictions(base, tree_preds, learning_rate)

        assert abs(result[0] - 0.1) < 1e-10  # 0 + 0.1 * 1.0
        assert abs(result[1] - 0.2) < 1e-10  # 0 + 0.1 * 2.0

    def test_empty_tree_predictions_raises(self) -> None:
        """Should raise ValueError for empty tree_predictions."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_raw_predictions(0.0, (), 0.1)

    def test_inconsistent_lengths_raises(self) -> None:
        """Should raise ValueError for inconsistent tree prediction lengths."""
        tree_preds = (
            (0.1, 0.2, 0.3),
            (0.1, 0.2),  # Different length
        )

        with pytest.raises(ValueError, match="same length"):
            compute_raw_predictions(0.0, tree_preds, 0.1)


class TestRawToProba:
    """Tests for raw_to_proba function."""

    def test_zero_raw_gives_half(self) -> None:
        """Raw prediction of 0 should give probability 0.5."""
        result = raw_to_proba((0.0,))

        assert len(result) == 1
        assert abs(result[0] - 0.5) < 1e-10

    def test_positive_raw_gives_above_half(self) -> None:
        """Positive raw prediction should give probability > 0.5."""
        result = raw_to_proba((1.0,))

        assert result[0] > 0.5

    def test_negative_raw_gives_below_half(self) -> None:
        """Negative raw prediction should give probability < 0.5."""
        result = raw_to_proba((-1.0,))

        assert result[0] < 0.5

    def test_multiple_values(self) -> None:
        """Should convert multiple raw predictions to probabilities."""
        result = raw_to_proba((-2.0, 0.0, 2.0))

        assert len(result) == 3
        assert result[0] < 0.5
        assert abs(result[1] - 0.5) < 1e-10
        assert result[2] > 0.5

    def test_empty_input(self) -> None:
        """Empty input should return empty output."""
        result = raw_to_proba(())

        assert result == ()
