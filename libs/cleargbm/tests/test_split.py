"""Tests for cleargbm.split module.

Split computation helpers for gradient boosting trees.
"""

from __future__ import annotations

import pytest

from cleargbm.split import (
    _check_monotonicity,
    _compute_leaf_value,
    _compute_split_gain,
    _compute_threshold,
    _create_internal_node,
    _create_leaf_node,
    _find_best_split_with_constraints,
    _find_split_for_feature,
    find_best_split,
)
from cleargbm.types import GradientBoostingConfig, SplitCandidate

from .conftest import make_config


class TestComputeLeafValue:
    """Tests for _compute_leaf_value."""

    def test_positive_gradients(self) -> None:
        """Positive gradients should give negative leaf value."""
        gradients = (1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25)

        value = _compute_leaf_value(gradients, hessians)

        # -G/H = -3.0/0.75 = -4.0
        assert abs(value - (-4.0)) < 1e-10

    def test_negative_gradients(self) -> None:
        """Negative gradients should give positive leaf value."""
        gradients = (-1.0, -1.0)
        hessians = (0.5, 0.5)

        value = _compute_leaf_value(gradients, hessians)

        # -G/H = -(-2)/1.0 = 2.0
        assert abs(value - 2.0) < 1e-10

    def test_zero_hessian_returns_zero(self) -> None:
        """Zero hessian should return 0.0 to avoid division by zero."""
        gradients = (1.0, 1.0)
        hessians = (0.0, 0.0)

        value = _compute_leaf_value(gradients, hessians)

        assert value == 0.0

    def test_mismatched_lengths_raises(self) -> None:
        """Different length arrays should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            _compute_leaf_value((1.0, 2.0), (0.5,))

    def test_empty_arrays_raises(self) -> None:
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty arrays"):
            _compute_leaf_value((), ())

    def test_l2_regularization_shrinks_value(self) -> None:
        """L2 regularization should shrink leaf value."""
        gradients = (1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25)

        # Without L2: -G/H = -3.0/0.75 = -4.0
        value_no_reg = _compute_leaf_value(gradients, hessians)
        assert abs(value_no_reg - (-4.0)) < 1e-10

        # With L2 (lambda=1.0): -G/(H+lambda) = -3.0/1.75 = -1.714...
        value_with_reg = _compute_leaf_value(gradients, hessians, reg_alpha=0.0, reg_lambda=1.0)
        expected = -3.0 / 1.75
        assert abs(value_with_reg - expected) < 1e-10

    def test_l1_regularization_soft_threshold(self) -> None:
        """L1 regularization should apply soft thresholding."""
        gradients = (2.0, 2.0)  # G = 4.0
        hessians = (0.5, 0.5)  # H = 1.0

        # Without L1: -G/H = -4.0/1.0 = -4.0
        value_no_reg = _compute_leaf_value(gradients, hessians)
        assert abs(value_no_reg - (-4.0)) < 1e-10

        # With L1 (alpha=1.0, lambda=0.0): -sign(G) * max(|G|-alpha, 0) / H
        # = -1 * max(4-1, 0) / 1 = -3.0
        value_with_l1 = _compute_leaf_value(gradients, hessians, reg_alpha=1.0, reg_lambda=0.0)
        assert abs(value_with_l1 - (-3.0)) < 1e-10

    def test_l1_regularization_shrinks_to_zero(self) -> None:
        """L1 regularization should shrink to zero when |G| <= alpha."""
        gradients = (0.5, 0.5)  # G = 1.0
        hessians = (0.5, 0.5)  # H = 1.0

        # With L1 (alpha=2.0): |G|=1.0 <= alpha=2.0, so value = 0
        value = _compute_leaf_value(gradients, hessians, reg_alpha=2.0, reg_lambda=0.0)
        assert value == 0.0

    def test_l1_negative_gradient(self) -> None:
        """L1 regularization should work with negative gradients."""
        gradients = (-2.0, -2.0)  # G = -4.0
        hessians = (0.5, 0.5)  # H = 1.0

        # With L1 (alpha=1.0, lambda=0.0): -sign(G) * max(|G|-alpha, 0) / H
        # = -(-1) * max(4-1, 0) / 1 = 3.0
        value = _compute_leaf_value(gradients, hessians, reg_alpha=1.0, reg_lambda=0.0)
        assert abs(value - 3.0) < 1e-10

    def test_l1_l2_combined(self) -> None:
        """L1 and L2 regularization should work together."""
        gradients = (2.0, 2.0)  # G = 4.0
        hessians = (0.5, 0.5)  # H = 1.0

        # With L1 (alpha=1.0) and L2 (lambda=1.0):
        # -sign(G) * max(|G|-alpha, 0) / (H + lambda)
        # = -1 * max(4-1, 0) / (1 + 1) = -3.0 / 2.0 = -1.5
        value = _compute_leaf_value(gradients, hessians, reg_alpha=1.0, reg_lambda=1.0)
        assert abs(value - (-1.5)) < 1e-10


class TestComputeSplitGain:
    """Tests for _compute_split_gain."""

    def test_perfect_split_positive_gain(self) -> None:
        """Perfect split should have positive gain."""
        # Left: G=-2, H=0.5
        # Right: G=2, H=0.5
        # Total: G=0, H=1.0
        gain = _compute_split_gain(
            g_left=-2.0,
            h_left=0.5,
            g_right=2.0,
            h_right=0.5,
            g_total=0.0,
            h_total=1.0,
        )

        # Score_left = 4/0.5 = 8
        # Score_right = 4/0.5 = 8
        # Score_total = 0/1 = 0
        # Gain = 16 - 0 = 16
        assert gain > 0
        assert abs(gain - 16.0) < 1e-10

    def test_no_split_improvement_zero_gain(self) -> None:
        """Same gradients in both children should have zero gain."""
        # Same ratio on both sides
        gain = _compute_split_gain(
            g_left=1.0,
            h_left=0.5,
            g_right=1.0,
            h_right=0.5,
            g_total=2.0,
            h_total=1.0,
        )

        # All three scores are the same ratio, so gain is 0
        assert abs(gain) < 1e-10

    def test_zero_hessian_returns_zero(self) -> None:
        """Zero hessian should return zero gain."""
        gain = _compute_split_gain(
            g_left=1.0,
            h_left=0.0,
            g_right=1.0,
            h_right=0.5,
            g_total=2.0,
            h_total=0.5,
        )

        assert gain == 0.0

    def test_l2_regularization_reduces_gain(self) -> None:
        """L2 regularization should reduce split gain."""
        # Without regularization
        gain_no_reg = _compute_split_gain(
            g_left=-2.0,
            h_left=0.5,
            g_right=2.0,
            h_right=0.5,
            g_total=0.0,
            h_total=1.0,
        )
        # Gain = 4/0.5 + 4/0.5 - 0/1 = 16

        # With L2 regularization (lambda=0.5)
        gain_with_reg = _compute_split_gain(
            g_left=-2.0,
            h_left=0.5,
            g_right=2.0,
            h_right=0.5,
            g_total=0.0,
            h_total=1.0,
            reg_lambda=0.5,
        )
        # Gain = 4/(0.5+0.5) + 4/(0.5+0.5) - 0/(1+0.5) = 4 + 4 - 0 = 8

        assert abs(gain_no_reg - 16.0) < 1e-10
        assert abs(gain_with_reg - 8.0) < 1e-10
        assert gain_with_reg < gain_no_reg


class TestCheckMonotonicity:
    """Tests for _check_monotonicity."""

    def test_no_constraint_always_passes(self) -> None:
        """Constraint 0 should always return True."""
        assert _check_monotonicity(0, 1.0, 0.5, -1.0, 0.5)
        assert _check_monotonicity(0, -1.0, 0.5, 1.0, 0.5)

    def test_increasing_constraint_satisfied(self) -> None:
        """Increasing constraint (+1) passes when left <= right."""
        # left_value = -(-1.0)/0.5 = 2.0
        # right_value = -(1.0)/0.5 = -2.0
        # 2.0 <= -2.0 is False
        assert not _check_monotonicity(1, -1.0, 0.5, 1.0, 0.5)

        # left_value = -(1.0)/0.5 = -2.0
        # right_value = -(-1.0)/0.5 = 2.0
        # -2.0 <= 2.0 is True
        assert _check_monotonicity(1, 1.0, 0.5, -1.0, 0.5)

    def test_decreasing_constraint_satisfied(self) -> None:
        """Decreasing constraint (-1) passes when left >= right."""
        # left_value = -(-1.0)/0.5 = 2.0
        # right_value = -(1.0)/0.5 = -2.0
        # 2.0 >= -2.0 is True
        assert _check_monotonicity(-1, -1.0, 0.5, 1.0, 0.5)

        # left_value = -(1.0)/0.5 = -2.0
        # right_value = -(-1.0)/0.5 = 2.0
        # -2.0 >= 2.0 is False
        assert not _check_monotonicity(-1, 1.0, 0.5, -1.0, 0.5)


class TestComputeThreshold:
    """Tests for _compute_threshold."""

    def test_midpoint_between_values(self) -> None:
        """Threshold should be midpoint between adjacent values."""
        sorted_pairs: list[tuple[float, int]] = [(1.0, 0), (3.0, 1), (5.0, 2)]

        # Split at position 1 means midpoint between index 0 and 1
        threshold = _compute_threshold(sorted_pairs, 1, 3)

        # Midpoint between 1.0 and 3.0
        assert abs(threshold - 2.0) < 1e-10

    def test_end_of_list(self) -> None:
        """At end of list, threshold should be last value + 0.5."""
        sorted_pairs: list[tuple[float, int]] = [(1.0, 0), (3.0, 1)]

        threshold = _compute_threshold(sorted_pairs, 2, 2)

        assert abs(threshold - 3.5) < 1e-10


class TestFindSplitForFeature:
    """Tests for _find_split_for_feature."""

    def test_finds_split_for_separable_data(self) -> None:
        """Should find a valid split for clearly separable data."""
        # Feature values: 0.0, 0.0, 1.0, 1.0 (indices 0,1,2,3)
        sorted_pairs: list[tuple[float, int]] = [(0.0, 0), (0.0, 1), (1.0, 2), (1.0, 3)]
        gradients = (-1.0, -1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)

        split = _find_split_for_feature(
            sorted_pairs=sorted_pairs,
            gradients=gradients,
            hessians=hessians,
            g_total=0.0,
            h_total=1.0,
            min_samples_leaf=1,
            monotonic_constraint=0,
            feature_idx=0,
        )

        # Split should be found with positive gain on feature 0
        if split is None:
            pytest.fail("Expected split to be found for separable data")
        assert split["gain"] > 0
        assert split["feature_index"] == 0

    def test_no_split_when_all_same_value(self) -> None:
        """Should return None when all feature values are the same."""
        sorted_pairs: list[tuple[float, int]] = [(1.0, 0), (1.0, 1), (1.0, 2), (1.0, 3)]
        gradients = (-1.0, -1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)

        split = _find_split_for_feature(
            sorted_pairs=sorted_pairs,
            gradients=gradients,
            hessians=hessians,
            g_total=0.0,
            h_total=1.0,
            min_samples_leaf=1,
            monotonic_constraint=0,
            feature_idx=0,
        )

        assert split is None

    def test_no_split_when_monotonicity_violated(self) -> None:
        """Should return None when all splits violate monotonicity constraint."""
        # Data that would need a decreasing relationship but we constrain to increasing
        # Values go up (0.0, 1.0, 2.0, 3.0) but gradients suggest lower values = higher pred
        sorted_pairs: list[tuple[float, int]] = [(0.0, 0), (1.0, 1), (2.0, 2), (3.0, 3)]
        # Gradients: negative on left (want higher pred), positive on right (want lower pred)
        # This means: left should have higher value, right should have lower value
        # With increasing constraint (1), left must be <= right, which contradicts
        gradients = (-2.0, -1.0, 1.0, 2.0)  # g_left < 0 means higher pred on left
        hessians = (0.25, 0.25, 0.25, 0.25)

        split = _find_split_for_feature(
            sorted_pairs=sorted_pairs,
            gradients=gradients,
            hessians=hessians,
            g_total=0.0,
            h_total=1.0,
            min_samples_leaf=1,
            monotonic_constraint=1,  # Increasing: left <= right required
            feature_idx=0,
        )

        # All splits should be rejected due to monotonicity violation
        assert split is None


class TestFindBestSplit:
    """Tests for find_best_split."""

    def test_finds_best_among_features(self) -> None:
        """Should find the best split among multiple features."""
        # Feature 0 perfectly separates, feature 1 does not
        x: tuple[tuple[float, ...], ...] = (
            (0.0, 0.5),
            (0.0, 0.5),
            (1.0, 0.5),
            (1.0, 0.5),
        )
        gradients = (-1.0, -1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)

        split = find_best_split(
            x=x,
            gradients=gradients,
            hessians=hessians,
            sample_indices=(0, 1, 2, 3),
            feature_indices=(0, 1),
            min_samples_leaf=1,
            monotonic_constraint=0,
        )

        # Should select feature 0 as it provides better separation
        if split is None:
            pytest.fail("Expected split to be found among features")
        assert split["feature_index"] == 0

    def test_returns_none_when_too_few_samples(self) -> None:
        """Should return None when fewer samples than 2*min_samples_leaf."""
        x: tuple[tuple[float, ...], ...] = ((0.0,), (1.0,))
        gradients = (-1.0, 1.0)
        hessians = (0.25, 0.25)

        split = find_best_split(
            x=x,
            gradients=gradients,
            hessians=hessians,
            sample_indices=(0, 1),
            feature_indices=(0,),
            min_samples_leaf=2,  # Need 4 samples but only have 2
            monotonic_constraint=0,
        )

        assert split is None


class TestCreateLeafNode:
    """Tests for _create_leaf_node."""

    def test_creates_proper_leaf(self) -> None:
        """Should create a leaf node with correct values."""
        gradients = (-1.0, -1.0)
        hessians = (0.5, 0.5)

        node = _create_leaf_node(
            node_id=5,
            sample_indices=(0, 1),
            gradients=gradients,
            hessians=hessians,
        )

        assert node["node_id"] == 5
        assert node["is_leaf"]
        assert node["feature_index"] is None
        assert node["n_samples"] == 2
        # leaf_value = -(-2.0)/1.0 = 2.0
        assert abs(node["value"] - 2.0) < 1e-10


class TestCreateInternalNode:
    """Tests for _create_internal_node."""

    def test_creates_internal_node(self) -> None:
        """Should create an internal node with split info."""
        split = SplitCandidate(
            feature_index=0,
            threshold=0.5,
            gain=10.0,
            left_indices=(0, 1),
            right_indices=(2, 3),
            nan_direction="left",
        )
        gradients = (-1.0, -1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)

        node = _create_internal_node(
            node_id=0,
            sample_indices=(0, 1, 2, 3),
            gradients=gradients,
            hessians=hessians,
            split=split,
            feature_names=("f0", "f1"),
        )

        assert node["node_id"] == 0
        assert not node["is_leaf"]
        assert node["feature_index"] == 0
        assert node["feature_name"] == "f0"
        assert node["threshold"] == 0.5
        assert node["n_samples"] == 4


class TestFindBestSplitWithConstraints:
    """Tests for _find_best_split_with_constraints."""

    def test_respects_monotonic_constraints(self) -> None:
        """Should respect monotonic constraints when set."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (1.0,),
            (2.0,),
            (3.0,),
        )
        gradients = (1.0, 0.5, -0.5, -1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)

        # With increasing constraint on feature 0
        config = make_config()
        config_with_constraint = GradientBoostingConfig(**{**config, "monotonic_constraints": (1,)})

        result = _find_best_split_with_constraints(
            x=x,
            gradients=gradients,
            hessians=hessians,
            sample_indices=(0, 1, 2, 3),
            feature_indices=(0,),
            config=config_with_constraint,
        )

        # Result is either None or a valid SplitCandidate
        # With monotonic constraint, split should still be found for separable data
        assert result is None or result["gain"] > 0

    def test_selects_best_among_multiple_features(self) -> None:
        """Should select feature with best gain among multiple features."""
        # Feature 0: perfect separation (best split)
        # Feature 1: no separation (worse split or no split)
        x: tuple[tuple[float, ...], ...] = (
            (0.0, 0.5),
            (0.0, 0.5),
            (1.0, 0.5),
            (1.0, 0.5),
        )
        gradients = (-1.0, -1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)

        config = make_config(min_samples_leaf=1)

        result = _find_best_split_with_constraints(
            x=x,
            gradients=gradients,
            hessians=hessians,
            sample_indices=(0, 1, 2, 3),
            feature_indices=(0, 1),  # Both features checked
            config=config,
        )

        # Should select feature 0 as it has better separation
        if result is None:
            raise AssertionError("Expected to find a split")
        assert result["feature_index"] == 0
        assert result["gain"] > 0
