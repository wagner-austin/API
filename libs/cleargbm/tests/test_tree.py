"""Tests for cleargbm.tree module.

Core tree building, prediction, and explanation functions.
Uses numpy arrays for all array operations.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from cleargbm.buffers import HistogramBuffer
from cleargbm.histogram import precompute_feature_bins
from cleargbm.tree import (
    _compute_child_histograms,
    _compute_max_depth,
    _finalize_nodes,
    _get_sample_indices,
    _predict_single,
    _select_features,
    _should_be_leaf,
    _update_parent_child,
    build_tree,
    explain_tree_prediction,
    predict_tree,
)
from cleargbm.types import DecisionTree, TreeNode

from .conftest import make_config


def _float_matrix(data: list[list[float]]) -> NDArray[np.float64]:
    """Create a 2D float array from nested list (helper for strict typing)."""
    return np.array(data, dtype=np.float64)


def _float_array(data: list[float]) -> NDArray[np.float64]:
    """Create a 1D float array from list (helper for strict typing)."""
    return np.array(data, dtype=np.float64)


def _int_array(data: list[int]) -> NDArray[np.int64]:
    """Create a 1D int array from list (helper for strict typing)."""
    return np.array(data, dtype=np.int64)


class TestShouldBeLeaf:
    """Tests for _should_be_leaf."""

    def test_max_depth_reached(self) -> None:
        """Should be leaf when max depth reached."""
        config = make_config(max_depth=2)

        assert _should_be_leaf(depth=2, n_samples=100, config=config)
        assert not _should_be_leaf(depth=1, n_samples=100, config=config)

    def test_too_few_samples_for_split(self) -> None:
        """Should be leaf when too few samples for split."""
        config = make_config(min_samples_split=10)

        assert _should_be_leaf(depth=0, n_samples=5, config=config)
        assert not _should_be_leaf(depth=0, n_samples=15, config=config)

    def test_too_few_samples_for_leaf(self) -> None:
        """Should be leaf when fewer than 2*min_samples_leaf."""
        config = make_config(min_samples_leaf=5)

        assert _should_be_leaf(depth=0, n_samples=8, config=config)
        assert not _should_be_leaf(depth=0, n_samples=15, config=config)


class TestBuildTree:
    """Tests for build_tree."""

    def test_builds_tree_for_simple_data(self) -> None:
        """Should build a tree for simple separable data."""
        x = _float_matrix(
            [
                [0.0, 0.5],
                [0.0, 0.5],
                [1.0, 0.5],
                [1.0, 0.5],
            ]
        )
        gradients = _float_array([-1.0, -1.0, 1.0, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        config = make_config(max_depth=2, min_samples_leaf=1)

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0", "f1"),
        )

        # Should have at least root node plus 2 children (3 nodes minimum)
        assert len(tree["nodes"]) >= 3
        assert tree["n_leaves"] >= 2
        assert tree["feature_names"] == ("f0", "f1")

    def test_empty_x_raises(self) -> None:
        """Should raise ValueError for empty input."""
        x_empty: NDArray[np.float64] = np.zeros((0, 1), dtype=np.float64)
        with pytest.raises(ValueError, match="not be empty"):
            build_tree(
                x=x_empty,
                gradients=_float_array([]),
                hessians=_float_array([]),
                config=make_config(),
                feature_names=(),
            )

    def test_mismatched_gradients_raises(self) -> None:
        """Should raise ValueError for mismatched gradient length."""
        x = _float_matrix([[0.0], [1.0]])
        with pytest.raises(ValueError, match="gradients length"):
            build_tree(
                x=x,
                gradients=_float_array([1.0]),  # Wrong length
                hessians=_float_array([0.5, 0.5]),
                config=make_config(),
                feature_names=("f0",),
            )

    def test_mismatched_hessians_raises(self) -> None:
        """Should raise ValueError for mismatched hessian length."""
        x = _float_matrix([[0.0], [1.0]])
        with pytest.raises(ValueError, match="hessians length"):
            build_tree(
                x=x,
                gradients=_float_array([1.0, -1.0]),
                hessians=_float_array([0.5]),  # Wrong length
                config=make_config(),
                feature_names=("f0",),
            )

    def test_mismatched_feature_names_raises(self) -> None:
        """Should raise ValueError for mismatched feature names."""
        x = _float_matrix([[0.0, 1.0]])
        with pytest.raises(ValueError, match="feature_names length"):
            build_tree(
                x=x,
                gradients=_float_array([1.0]),
                hessians=_float_array([0.5]),
                config=make_config(),
                feature_names=("f0",),  # Only 1 name but 2 features
            )

    def test_with_subsampling(self) -> None:
        """Should work with row subsampling."""
        x = _float_matrix([[float(i)] for i in range(20)])
        gradients = _float_array([1.0 if i < 10 else -1.0 for i in range(20)])
        hessians = _float_array([0.25 for _ in range(20)])
        config = make_config(subsample=0.5)

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0",),
        )

        assert tree["n_leaves"] > 0

    def test_with_max_features(self) -> None:
        """Should work with feature subsampling."""
        x = _float_matrix(
            [
                [0.0, 0.5, 0.3],
                [0.0, 0.5, 0.3],
                [1.0, 0.5, 0.3],
                [1.0, 0.5, 0.3],
            ]
        )
        gradients = _float_array([-1.0, -1.0, 1.0, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        config = make_config(max_features=1)

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0", "f1", "f2"),
        )

        assert tree["n_leaves"] > 0

    def test_with_max_features_equals_n_features(self) -> None:
        """max_features == n_features should use all features."""
        x = _float_matrix(
            [
                [0.0, 0.5, 0.3],
                [0.0, 0.5, 0.3],
                [1.0, 0.5, 0.3],
                [1.0, 0.5, 0.3],
            ]
        )
        gradients = _float_array([-1.0, -1.0, 1.0, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        # max_features = 3 = n_features
        config = make_config(max_features=3)

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0", "f1", "f2"),
        )

        assert tree["n_leaves"] > 0

    def test_with_max_features_exceeds_n_features(self) -> None:
        """max_features > n_features should use all features."""
        x = _float_matrix(
            [
                [0.0, 0.5],
                [0.0, 0.5],
                [1.0, 0.5],
                [1.0, 0.5],
            ]
        )
        gradients = _float_array([-1.0, -1.0, 1.0, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        # max_features = 10 > n_features = 2
        config = make_config(max_features=10)

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0", "f1"),
        )

        assert tree["n_leaves"] > 0

    def test_with_max_features_deterministic(self) -> None:
        """Same seed with max_features should produce same tree."""
        x = _float_matrix(
            [
                [0.0, 0.5, 0.8],
                [0.1, 0.4, 0.7],
                [0.9, 0.6, 0.2],
                [1.0, 0.3, 0.1],
            ]
        )
        gradients = _float_array([-1.0, -0.5, 0.5, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        config = make_config(max_features=1, random_state=42)

        tree1 = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0", "f1", "f2"),
        )

        # Same seed should produce identical tree
        tree2 = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0", "f1", "f2"),
        )

        assert len(tree1["nodes"]) == len(tree2["nodes"])
        for n1, n2 in zip(tree1["nodes"], tree2["nodes"], strict=True):
            assert n1["feature_index"] == n2["feature_index"]
            assert n1["value"] == n2["value"]

    def test_with_precomputed_feature_bins(self) -> None:
        """Should use precomputed feature bins when passed."""
        x = _float_matrix(
            [
                [0.0, 0.5],
                [0.0, 0.5],
                [1.0, 0.5],
                [1.0, 0.5],
            ]
        )
        gradients = _float_array([-1.0, -1.0, 1.0, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        config = make_config(max_depth=2, min_samples_leaf=1)

        # Precompute bins
        feature_bins = precompute_feature_bins(x, config["max_bins"])

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0", "f1"),
            feature_bins=feature_bins,
        )

        # Should have at least root node plus 2 children (3 nodes minimum)
        assert len(tree["nodes"]) >= 3
        assert tree["n_leaves"] >= 2


class TestPredictTree:
    """Tests for predict_tree."""

    def test_predicts_correct_values(self) -> None:
        """Should predict correct values based on tree structure."""
        x = _float_matrix(
            [
                [0.0],
                [0.0],
                [1.0],
                [1.0],
            ]
        )
        gradients = _float_array([-1.0, -1.0, 1.0, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        config = make_config(max_depth=1, min_samples_leaf=1)

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0",),
        )

        # Predict on the training data
        predictions = predict_tree(tree, x)

        n_preds: int = predictions.shape[0]
        assert n_preds == 4
        # First two samples should have same prediction
        pred_0: float = predictions.item(0)
        pred_1: float = predictions.item(1)
        pred_2: float = predictions.item(2)
        pred_3: float = predictions.item(3)
        assert pred_0 == pred_1
        # Last two should have same prediction
        assert pred_2 == pred_3
        # Left and right should be different
        assert pred_0 != pred_2

    def test_predicts_nan_values_using_nan_direction(self) -> None:
        """NaN values should follow nan_direction when predicting."""
        import math

        # Create a tree with explicit nan_direction
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="left",  # NaN goes left
                    value=0.0,
                    n_samples=20,
                    left_child=1,
                    right_child=2,
                ),
                TreeNode(
                    node_id=1,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=-1.0,  # Left leaf value
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
                TreeNode(
                    node_id=2,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=1.0,  # Right leaf value
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        # Predict with NaN value - should go left and get -1.0
        x_nan = _float_matrix([[math.nan]])
        predictions = predict_tree(tree, x_nan)
        n_preds: int = predictions.shape[0]
        assert n_preds == 1
        pred_val: float = predictions.item(0)
        assert abs(pred_val - (-1.0)) < 1e-10


class TestExplainTreePrediction:
    """Tests for explain_tree_prediction."""

    def test_explanation_contains_path(self) -> None:
        """Explanation should contain the path taken through tree."""
        x = _float_matrix(
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ]
        )
        gradients = _float_array([-1.0, -0.5, 0.5, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        config = make_config(max_depth=2, min_samples_leaf=1)

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0",),
        )

        # Explain prediction for first sample
        x_test = _float_array([0.0])
        explanation = explain_tree_prediction(tree, x_test, tree_index=5)

        assert explanation["tree_index"] == 5
        # Path should have entries for internal nodes traversed
        # Prediction should be a float
        assert explanation["prediction"] == explanation["prediction"]  # Not NaN
        assert explanation["n_samples_in_leaf"] > 0

    def test_explanation_right_path(self) -> None:
        """Explanation should track right path correctly."""
        # Build a simple tree where high values go right
        x = _float_matrix(
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
            ]
        )
        gradients = _float_array([-1.0, -0.5, 0.5, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        config = make_config(max_depth=2, min_samples_leaf=1)

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0",),
        )

        # Use a high value to go right
        x_test = _float_array([3.0])
        explanation = explain_tree_prediction(tree, x_test, tree_index=0)

        # Should have at least one path entry where direction is right
        has_right = any(step["direction"] == "right" for step in explanation["path"])
        assert has_right

    def test_explanation_node_missing_split_info(self) -> None:
        """Explanation should handle node with missing split info."""
        # Create a tree with a leaf node that has None for split fields
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=None,  # Missing split info
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=0.5,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=1,
            n_leaves=1,
            feature_names=("f0",),
        )

        x_test = _float_array([0.0])
        explanation = explain_tree_prediction(tree, x_test, tree_index=0)

        # Should return the value from the node
        assert abs(explanation["prediction"] - 0.5) < 1e-10
        assert len(explanation["path"]) == 0  # No splits traversed

    def test_explanation_missing_child(self) -> None:
        """Explanation should handle node with missing child."""
        # Create a tree where the child pointer is None
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="left",
                    value=0.7,  # Return value when child is missing
                    n_samples=10,
                    left_child=None,  # Missing child
                    right_child=None,
                ),
            ),
            max_depth=1,
            n_leaves=1,
            feature_names=("f0",),
        )

        x_test = _float_array([0.0])
        explanation = explain_tree_prediction(tree, x_test, tree_index=0)

        # Should return node's value when child is missing
        assert abs(explanation["prediction"] - 0.7) < 1e-10
        assert len(explanation["path"]) == 1

    def test_nan_value_follows_nan_direction_left(self) -> None:
        """NaN feature value should follow nan_direction in explanation."""
        import math

        # Create a tree that routes NaN left
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="left",
                    value=0.0,
                    n_samples=20,
                    left_child=1,
                    right_child=2,
                ),
                TreeNode(
                    node_id=1,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=-1.0,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
                TreeNode(
                    node_id=2,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=1.0,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        # Pass NaN value - should go left
        x_nan = _float_array([math.nan])
        explanation = explain_tree_prediction(tree, x_nan, tree_index=0)

        # Should predict -1.0 (left child value)
        assert abs(explanation["prediction"] - (-1.0)) < 1e-10
        # Path should show we went left
        assert len(explanation["path"]) == 1
        assert explanation["path"][0]["direction"] == "left"

    def test_nan_value_follows_nan_direction_right(self) -> None:
        """NaN feature value should follow nan_direction='right' in explanation."""
        import math

        # Create a tree that routes NaN right
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="right",
                    value=0.0,
                    n_samples=20,
                    left_child=1,
                    right_child=2,
                ),
                TreeNode(
                    node_id=1,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=-1.0,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
                TreeNode(
                    node_id=2,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=1.0,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        # Pass NaN value - should go right
        x_nan = _float_array([math.nan])
        explanation = explain_tree_prediction(tree, x_nan, tree_index=0)

        # Should predict 1.0 (right child value)
        assert abs(explanation["prediction"] - 1.0) < 1e-10
        # Path should show we went right
        assert len(explanation["path"]) == 1
        assert explanation["path"][0]["direction"] == "right"


class TestHelperFunctions:
    """Tests for various helper functions."""

    def test_get_sample_indices_full(self) -> None:
        """Full sampling should return all indices."""
        from cleargbm._hooks_infra import get_random_state

        rng = get_random_state(42)
        indices = _get_sample_indices(10, 1.0, rng)

        expected = _int_array(list(range(10)))
        assert np.array_equal(indices, expected)

    def test_get_sample_indices_subsample(self) -> None:
        """Subsampling should return fewer indices."""
        from cleargbm._hooks_infra import get_random_state

        rng = get_random_state(42)
        indices = _get_sample_indices(10, 0.5, rng)

        n_indices: int = indices.shape[0]
        assert n_indices == 5
        for i in range(n_indices):
            idx: int = indices.item(i)
            assert 0 <= idx < 10

    def test_select_features_all(self) -> None:
        """All features should be returned when max_features >= n_features."""
        from cleargbm._hooks_infra import get_random_state

        rng = get_random_state(42)
        indices = _select_features(5, 10, rng)

        assert indices == tuple(range(5))

    def test_select_features_subset(self) -> None:
        """Subset of features should be returned."""
        from cleargbm._hooks_infra import get_random_state

        rng = get_random_state(42)
        indices = _select_features(10, 3, rng)

        assert len(indices) == 3
        for idx in indices:
            assert 0 <= idx < 10

    def test_select_features_single(self) -> None:
        """max_features=1 should return exactly one feature."""
        from cleargbm._hooks_infra import get_random_state

        rng = get_random_state(42)
        indices = _select_features(5, 1, rng)

        assert len(indices) == 1
        assert 0 <= indices[0] < 5

    def test_select_features_exceeds_n_features(self) -> None:
        """max_features > n_features should return all features."""
        from cleargbm._hooks_infra import get_random_state

        rng = get_random_state(42)
        indices = _select_features(3, 100, rng)

        # Should return all 3 features in order
        assert indices == (0, 1, 2)

    def test_select_features_equals_n_features(self) -> None:
        """max_features == n_features should return all features."""
        from cleargbm._hooks_infra import get_random_state

        rng = get_random_state(42)
        indices = _select_features(4, 4, rng)

        # Should return all 4 features in order
        assert indices == (0, 1, 2, 3)

    def test_select_features_deterministic(self) -> None:
        """Same seed should produce same feature selection."""
        from cleargbm._hooks_infra import get_random_state

        rng1 = get_random_state(123)
        indices1 = _select_features(10, 3, rng1)

        rng2 = get_random_state(123)
        indices2 = _select_features(10, 3, rng2)

        assert indices1 == indices2

    def test_select_features_different_seeds(self) -> None:
        """Different seeds should (likely) produce different selections."""
        from cleargbm._hooks_infra import get_random_state

        rng1 = get_random_state(42)
        indices1 = _select_features(10, 3, rng1)

        rng2 = get_random_state(99)
        indices2 = _select_features(10, 3, rng2)

        # With 10 choose 3, different seeds should give different results
        assert indices1 != indices2

    def test_select_features_no_replacement(self) -> None:
        """Selected features should be unique (no duplicates)."""
        from cleargbm._hooks_infra import get_random_state

        rng = get_random_state(42)
        indices = _select_features(10, 5, rng)

        # All indices should be unique
        assert len(indices) == len(set(indices))

    def test_update_parent_child_left(self) -> None:
        """Should update left child correctly."""
        node_children: dict[int, tuple[int | None, int | None]] = {0: (None, None)}
        _update_parent_child(node_children, 0, 1, is_left=True)

        assert node_children[0] == (1, None)

    def test_update_parent_child_right(self) -> None:
        """Should update right child correctly."""
        node_children: dict[int, tuple[int | None, int | None]] = {0: (1, None)}
        _update_parent_child(node_children, 0, 2, is_left=False)

        assert node_children[0] == (1, 2)

    def test_update_parent_child_none_parent(self) -> None:
        """Should handle None parent (root node)."""
        node_children: dict[int, tuple[int | None, int | None]] = {}
        _update_parent_child(node_children, None, 0, is_left=None)

        # Should not crash, nothing to update
        assert len(node_children) == 0

    def test_finalize_nodes_updates_children(self) -> None:
        """Should update child pointers in internal nodes."""
        nodes = [
            TreeNode(
                node_id=0,
                is_leaf=False,
                feature_index=0,
                feature_name="f0",
                threshold=0.5,
                nan_direction="left",
                value=0.0,
                n_samples=4,
                left_child=None,
                right_child=None,
            ),
            TreeNode(
                node_id=1,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=1.0,
                n_samples=2,
                left_child=None,
                right_child=None,
            ),
        ]
        node_children: dict[int, tuple[int | None, int | None]] = {0: (1, 2)}

        final = _finalize_nodes(nodes, node_children)

        assert final[0]["left_child"] == 1
        assert final[0]["right_child"] == 2
        assert final[1]["left_child"] is None  # Leaf unchanged

    def test_compute_max_depth(self) -> None:
        """Should compute correct max depth."""
        nodes = [
            TreeNode(
                node_id=0,
                is_leaf=False,
                feature_index=0,
                feature_name="f0",
                threshold=0.5,
                nan_direction="left",
                value=0.0,
                n_samples=4,
                left_child=1,
                right_child=2,
            ),
            TreeNode(
                node_id=1,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=1.0,
                n_samples=2,
                left_child=None,
                right_child=None,
            ),
            TreeNode(
                node_id=2,
                is_leaf=True,
                feature_index=None,
                feature_name=None,
                threshold=None,
                nan_direction=None,
                value=-1.0,
                n_samples=2,
                left_child=None,
                right_child=None,
            ),
        ]

        depth = _compute_max_depth(nodes)

        assert depth == 1  # Root at 0, leaves at 1


class TestPredictSingle:
    """Tests for _predict_single."""

    def test_predicts_from_leaf(self) -> None:
        """Should return leaf value."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=0.5,
                    n_samples=10,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=0,
            n_leaves=1,
            feature_names=("f0",),
        )

        x_test = _float_array([0.0])
        pred = _predict_single(tree, x_test)

        assert pred == 0.5

    def test_navigates_tree(self) -> None:
        """Should navigate left/right based on feature values."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="left",
                    value=0.0,
                    n_samples=10,
                    left_child=1,
                    right_child=2,
                ),
                TreeNode(
                    node_id=1,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=-1.0,
                    n_samples=5,
                    left_child=None,
                    right_child=None,
                ),
                TreeNode(
                    node_id=2,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=1.0,
                    n_samples=5,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        # x[0] = 0.0 <= 0.5, go left
        x_left = _float_array([0.0])
        assert _predict_single(tree, x_left) == -1.0

        # x[0] = 1.0 > 0.5, go right
        x_right = _float_array([1.0])
        assert _predict_single(tree, x_right) == 1.0

    def test_handles_missing_child(self) -> None:
        """Should return current node value when child is None."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="left",
                    value=0.5,
                    n_samples=10,
                    left_child=None,  # Missing child
                    right_child=None,
                ),
            ),
            max_depth=0,
            n_leaves=0,
            feature_names=("f0",),
        )

        x_test = _float_array([0.0])
        pred = _predict_single(tree, x_test)

        assert pred == 0.5

    def test_handles_missing_feature_info(self) -> None:
        """Should return value when feature info is missing in non-leaf."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=None,  # Missing
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=0.25,
                    n_samples=10,
                    left_child=1,
                    right_child=2,
                ),
            ),
            max_depth=0,
            n_leaves=0,
            feature_names=("f0",),
        )

        x_test = _float_array([0.0])
        pred = _predict_single(tree, x_test)

        assert pred == 0.25

    def test_routes_nan_left(self) -> None:
        """Should route NaN to left child when nan_direction is left."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="left",
                    value=0.0,
                    n_samples=10,
                    left_child=1,
                    right_child=2,
                ),
                TreeNode(
                    node_id=1,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=-1.0,
                    n_samples=5,
                    left_child=None,
                    right_child=None,
                ),
                TreeNode(
                    node_id=2,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=1.0,
                    n_samples=5,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        x_nan = _float_array([float("nan")])
        pred = _predict_single(tree, x_nan)

        assert pred == -1.0

    def test_routes_nan_right(self) -> None:
        """Should route NaN to right child when nan_direction is right."""
        tree = DecisionTree(
            nodes=(
                TreeNode(
                    node_id=0,
                    is_leaf=False,
                    feature_index=0,
                    feature_name="f0",
                    threshold=0.5,
                    nan_direction="right",
                    value=0.0,
                    n_samples=10,
                    left_child=1,
                    right_child=2,
                ),
                TreeNode(
                    node_id=1,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=-1.0,
                    n_samples=5,
                    left_child=None,
                    right_child=None,
                ),
                TreeNode(
                    node_id=2,
                    is_leaf=True,
                    feature_index=None,
                    feature_name=None,
                    threshold=None,
                    nan_direction=None,
                    value=1.0,
                    n_samples=5,
                    left_child=None,
                    right_child=None,
                ),
            ),
            max_depth=1,
            n_leaves=2,
            feature_names=("f0",),
        )

        x_nan = _float_array([float("nan")])
        pred = _predict_single(tree, x_nan)

        assert pred == 1.0


class TestComputeChildHistograms:
    """Tests for _compute_child_histograms."""

    def test_computes_using_subtraction(self) -> None:
        """Should compute child histograms using sibling subtraction."""
        x = _float_matrix(
            [
                [0.0],
                [0.0],
                [1.0],
                [1.0],
            ]
        )
        gradients = _float_array([-1.0, -1.0, 1.0, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        config = make_config(max_bins=4)

        feature_bins = precompute_feature_bins(x, config["max_bins"])

        # Create parent histogram
        parent_hist = HistogramBuffer.from_tuples(
            gradient_sums=(0.0, -2.0, 2.0, 0.0),
            hessian_sums=(0.0, 0.5, 0.5, 0.0),
            counts=(0, 2, 2, 0),
        )
        parent_histograms: dict[int, HistogramBuffer] = {0: parent_hist}

        left_indices = _int_array([0, 1])
        right_indices = _int_array([2, 3])

        left_hists, right_hists = _compute_child_histograms(
            left_indices=left_indices,
            right_indices=right_indices,
            gradients=gradients,
            hessians=hessians,
            feature_bins=feature_bins,
            parent_histograms=parent_histograms,
        )

        # Both children should have histograms
        assert 0 in left_hists
        assert 0 in right_hists

        # Sum of child histograms should equal parent
        for i in range(4):
            left_g = left_hists[0].get_gradient_sum(i)
            right_g = right_hists[0].get_gradient_sum(i)
            parent_g = parent_hist.get_gradient_sum(i)
            assert abs(left_g + right_g - parent_g) < 1e-10

    def test_smaller_child_built_directly(self) -> None:
        """Should build smaller child directly and derive larger via subtraction."""
        x = _float_matrix(
            [
                [0.0],
                [0.0],
                [0.0],
                [1.0],
            ]
        )
        gradients = _float_array([-1.0, -1.0, -1.0, 1.0])
        hessians = _float_array([0.25, 0.25, 0.25, 0.25])
        config = make_config(max_bins=4)

        feature_bins = precompute_feature_bins(x, config["max_bins"])

        # Parent has all 4 samples
        parent_hist = HistogramBuffer.from_tuples(
            gradient_sums=(0.0, -3.0, 1.0, 0.0),
            hessian_sums=(0.0, 0.75, 0.25, 0.0),
            counts=(0, 3, 1, 0),
        )
        parent_histograms: dict[int, HistogramBuffer] = {0: parent_hist}

        # Right is smaller (1 sample)
        left_indices = _int_array([0, 1, 2])
        right_indices = _int_array([3])

        left_hists, right_hists = _compute_child_histograms(
            left_indices=left_indices,
            right_indices=right_indices,
            gradients=gradients,
            hessians=hessians,
            feature_bins=feature_bins,
            parent_histograms=parent_histograms,
        )

        # Both children should have valid histograms
        assert 0 in left_hists
        assert 0 in right_hists

        # Verify counts add up
        left_count = sum(left_hists[0].counts_tuple())
        right_count = sum(right_hists[0].counts_tuple())
        assert left_count + right_count == sum(parent_hist.counts_tuple())
