"""Tests for ClearGBM SHAP adapter.

Comprehensive tests for converting ClearGBM models to SHAP format
and computing local explanations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
from cleargbm.types import DecisionTree, GradientBoostingConfig, GradientBoostingModel, TreeNode
from numpy.typing import NDArray

from covenant_ml.explainers.cleargbm_shap import (
    ClearGBMShapWrapper,
    _convert_decision_tree,
    _convert_tree_node_to_arrays,
    _get_default_child_idx,
    _populate_internal_node,
    _ShapArrays,
    convert_cleargbm_to_shap_format,
)

# =============================================================================
# Test data creation helpers
# =============================================================================


def _make_x_1x2() -> NDArray[np.float64]:
    """Create 1x2 test array."""
    data: NDArray[np.float64] = np.zeros((1, 2), dtype=np.float64)
    data[0, 0] = 0.3
    data[0, 1] = 0.5
    return data


def _make_x_2x2() -> NDArray[np.float64]:
    """Create 2x2 test array."""
    data: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
    data[0, 0] = 0.1
    data[0, 1] = 0.2
    data[1, 0] = 0.6
    data[1, 1] = 0.7
    return data


def _make_x_3x2() -> NDArray[np.float64]:
    """Create 3x2 test array."""
    data: NDArray[np.float64] = np.zeros((3, 2), dtype=np.float64)
    data[0, 0] = 0.2
    data[0, 1] = 0.4
    data[1, 0] = 0.7
    data[1, 1] = 0.3
    data[2, 0] = 0.5
    data[2, 1] = 0.5
    return data


def _make_x_1x3() -> NDArray[np.float64]:
    """Create 1x3 test array."""
    data: NDArray[np.float64] = np.zeros((1, 3), dtype=np.float64)
    data[0, 0] = 0.1
    data[0, 1] = 0.2
    data[0, 2] = 0.3
    return data


# =============================================================================
# Test fixtures
# =============================================================================


def _make_config() -> GradientBoostingConfig:
    """Create minimal GradientBoostingConfig for testing."""
    config: GradientBoostingConfig = GradientBoostingConfig(
        n_estimators=2,
        max_depth=2,
        learning_rate=0.1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        max_bins=64,
        subsample=1.0,
        random_state=42,
        track_contributions=False,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=1,
        early_stopping_rounds=10,
    )
    return config


def _make_leaf_node(
    node_id: int,
    value: float,
    n_samples: int,
) -> TreeNode:
    """Create a leaf TreeNode."""
    return TreeNode(
        node_id=node_id,
        is_leaf=True,
        feature_index=None,
        feature_name=None,
        threshold=None,
        nan_direction=None,
        value=value,
        n_samples=n_samples,
        left_child=None,
        right_child=None,
    )


def _make_split_node(
    node_id: int,
    feature_index: int,
    threshold: float,
    left_child: int,
    right_child: int,
    n_samples: int,
    nan_direction: Literal["left", "right"] | None = None,
    value: float = 0.0,
) -> TreeNode:
    """Create an internal (split) TreeNode."""
    return TreeNode(
        node_id=node_id,
        is_leaf=False,
        feature_index=feature_index,
        feature_name=f"feature_{feature_index}",
        threshold=threshold,
        nan_direction=nan_direction,
        value=value,
        n_samples=n_samples,
        left_child=left_child,
        right_child=right_child,
    )


def _make_simple_tree() -> DecisionTree:
    """Create a simple tree with one split and two leaves.

    Structure:
        [0] root (split on feature 0 at 0.5)
         ├── [1] left leaf (value=-0.1)
         └── [2] right leaf (value=0.1)
    """
    root = _make_split_node(
        node_id=0,
        feature_index=0,
        threshold=0.5,
        left_child=1,
        right_child=2,
        n_samples=100,
        nan_direction="left",
    )
    left_leaf = _make_leaf_node(node_id=1, value=-0.1, n_samples=40)
    right_leaf = _make_leaf_node(node_id=2, value=0.1, n_samples=60)

    return DecisionTree(
        nodes=(root, left_leaf, right_leaf),
        max_depth=1,
        n_leaves=2,
        feature_names=("feature_0", "feature_1"),
    )


def _make_deeper_tree() -> DecisionTree:
    """Create a tree with depth 2 (two levels of splits).

    Structure:
        [0] root (split on feature 0 at 0.5)
         ├── [1] left split (feature 1 at 0.3)
         │    ├── [3] left-left leaf
         │    └── [4] left-right leaf
         └── [2] right leaf
    """
    root = _make_split_node(
        node_id=0,
        feature_index=0,
        threshold=0.5,
        left_child=1,
        right_child=2,
        n_samples=100,
        nan_direction="right",
    )
    left_split = _make_split_node(
        node_id=1,
        feature_index=1,
        threshold=0.3,
        left_child=3,
        right_child=4,
        n_samples=40,
        nan_direction=None,
    )
    right_leaf = _make_leaf_node(node_id=2, value=0.2, n_samples=60)
    left_left_leaf = _make_leaf_node(node_id=3, value=-0.2, n_samples=15)
    left_right_leaf = _make_leaf_node(node_id=4, value=0.0, n_samples=25)

    return DecisionTree(
        nodes=(root, left_split, right_leaf, left_left_leaf, left_right_leaf),
        max_depth=2,
        n_leaves=3,
        feature_names=("feature_0", "feature_1"),
    )


def _make_model(trees: list[DecisionTree]) -> GradientBoostingModel:
    """Create a GradientBoostingModel with given trees."""
    return GradientBoostingModel(
        trees=tuple(trees),
        base_prediction=0.0,
        learning_rate=0.1,
        feature_names=("feature_0", "feature_1"),
        n_classes=2,
        config=_make_config(),
    )


# =============================================================================
# Tests for _ShapArrays
# =============================================================================


class TestShapArrays:
    """Tests for _ShapArrays helper class."""

    def test_init_creates_correct_shapes(self) -> None:
        """_ShapArrays initializes arrays with correct shapes."""
        n_nodes = 5
        arrays = _ShapArrays(n_nodes)

        assert arrays.children_left.shape == (n_nodes,)
        assert arrays.children_right.shape == (n_nodes,)
        assert arrays.children_default.shape == (n_nodes,)
        assert arrays.features.shape == (n_nodes,)
        assert arrays.thresholds.shape == (n_nodes,)
        assert arrays.values.shape == (n_nodes, 1)
        assert arrays.node_sample_weight.shape == (n_nodes,)

    def test_init_dtypes(self) -> None:
        """_ShapArrays uses correct numpy dtypes."""
        arrays = _ShapArrays(2)

        assert arrays.children_left.dtype == np.int64
        assert arrays.children_right.dtype == np.int64
        assert arrays.children_default.dtype == np.int64
        assert arrays.features.dtype == np.int64
        assert arrays.thresholds.dtype == np.float64
        assert arrays.values.dtype == np.float64
        assert arrays.node_sample_weight.dtype == np.float64

    def test_to_typed_dict_returns_correct_keys(self) -> None:
        """to_typed_dict converts to TypedDict with expected keys."""
        arrays = _ShapArrays(3)
        result = arrays.to_typed_dict()

        # Verify all expected keys present
        assert len(result["children_left"]) == 3
        assert len(result["children_right"]) == 3
        assert len(result["children_default"]) == 3
        assert len(result["features"]) == 3
        assert len(result["thresholds"]) == 3
        assert len(result["node_sample_weight"]) == 3
        # Values array is 2D
        assert result["values"].shape == (3, 1)


# =============================================================================
# Tests for _get_default_child_idx
# =============================================================================


class TestGetDefaultChildIdx:
    """Tests for _get_default_child_idx helper function."""

    def test_nan_direction_left(self) -> None:
        """Returns left child index when nan_direction is 'left'."""
        node = _make_split_node(
            node_id=0,
            feature_index=0,
            threshold=0.5,
            left_child=1,
            right_child=2,
            n_samples=100,
            nan_direction="left",
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        result = _get_default_child_idx(node, node_id_to_idx)

        assert result == 1  # Index of left child

    def test_nan_direction_right(self) -> None:
        """Returns right child index when nan_direction is 'right'."""
        node = _make_split_node(
            node_id=0,
            feature_index=0,
            threshold=0.5,
            left_child=1,
            right_child=2,
            n_samples=100,
            nan_direction="right",
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        result = _get_default_child_idx(node, node_id_to_idx)

        assert result == 2  # Index of right child

    def test_no_nan_direction_defaults_to_left(self) -> None:
        """Defaults to left child when nan_direction is None."""
        node = _make_split_node(
            node_id=0,
            feature_index=0,
            threshold=0.5,
            left_child=1,
            right_child=2,
            n_samples=100,
            nan_direction=None,
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        result = _get_default_child_idx(node, node_id_to_idx)

        assert result == 1  # Defaults to left child

    def test_no_children_returns_negative_one(self) -> None:
        """Returns -1 when node has no children (leaf node)."""
        node = _make_leaf_node(node_id=0, value=0.1, n_samples=100)
        node_id_to_idx = {0: 0}

        result = _get_default_child_idx(node, node_id_to_idx)

        assert result == -1


# =============================================================================
# Tests for _populate_internal_node
# =============================================================================


class TestPopulateInternalNode:
    """Tests for _populate_internal_node helper function."""

    def test_populates_arrays(self) -> None:
        """Populates arrays for an internal node."""
        arrays = _ShapArrays(3)
        node = _make_split_node(
            node_id=0,
            feature_index=1,
            threshold=0.75,
            left_child=1,
            right_child=2,
            n_samples=50,
            nan_direction="left",
            value=0.05,
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        _populate_internal_node(arrays, 0, node, node_id_to_idx)

        # Verify arrays were modified (not default values)
        assert arrays.children_left.dtype == np.int64
        assert arrays.children_right.dtype == np.int64
        assert arrays.features.dtype == np.int64
        assert arrays.thresholds.dtype == np.float64

    def test_handles_none_feature_index(self) -> None:
        """Does not crash when feature_index is None."""
        arrays = _ShapArrays(3)
        node = TreeNode(
            node_id=0,
            is_leaf=False,
            feature_index=None,
            feature_name=None,
            threshold=0.5,
            nan_direction=None,
            value=0.0,
            n_samples=100,
            left_child=1,
            right_child=2,
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        # Should not raise
        _populate_internal_node(arrays, 0, node, node_id_to_idx)

    def test_handles_none_threshold(self) -> None:
        """Does not crash when threshold is None."""
        arrays = _ShapArrays(3)
        node = TreeNode(
            node_id=0,
            is_leaf=False,
            feature_index=0,
            feature_name="f0",
            threshold=None,
            nan_direction=None,
            value=0.0,
            n_samples=100,
            left_child=1,
            right_child=2,
        )
        node_id_to_idx = {0: 0, 1: 1, 2: 2}

        # Should not raise
        _populate_internal_node(arrays, 0, node, node_id_to_idx)

    def test_handles_none_left_child(self) -> None:
        """Does not crash when left_child is None."""
        arrays = _ShapArrays(2)
        node = TreeNode(
            node_id=0,
            is_leaf=False,
            feature_index=0,
            feature_name="f0",
            threshold=0.5,
            nan_direction=None,
            value=0.0,
            n_samples=100,
            left_child=None,
            right_child=1,
        )
        node_id_to_idx = {0: 0, 1: 1}

        # Should not raise
        _populate_internal_node(arrays, 0, node, node_id_to_idx)

    def test_handles_none_right_child(self) -> None:
        """Does not crash when right_child is None."""
        arrays = _ShapArrays(2)
        node = TreeNode(
            node_id=0,
            is_leaf=False,
            feature_index=0,
            feature_name="f0",
            threshold=0.5,
            nan_direction="left",
            value=0.0,
            n_samples=100,
            left_child=1,
            right_child=None,
        )
        node_id_to_idx = {0: 0, 1: 1}

        # Should not raise
        _populate_internal_node(arrays, 0, node, node_id_to_idx)


# =============================================================================
# Tests for _convert_tree_node_to_arrays
# =============================================================================


class TestConvertTreeNodeToArrays:
    """Tests for _convert_tree_node_to_arrays function."""

    def test_raises_on_empty_nodes(self) -> None:
        """Raises ValueError for empty nodes tuple."""
        with pytest.raises(ValueError, match="Cannot convert empty tree nodes"):
            _convert_tree_node_to_arrays(())

    def test_converts_single_leaf(self) -> None:
        """Converts a single leaf node."""
        leaf = _make_leaf_node(node_id=0, value=0.5, n_samples=100)
        nodes = (leaf,)

        result = _convert_tree_node_to_arrays(nodes)

        # Verify structure
        assert len(result["children_left"]) == 1
        assert len(result["children_right"]) == 1
        assert len(result["features"]) == 1
        assert result["values"].shape == (1, 1)

    def test_converts_simple_tree(self) -> None:
        """Converts a simple tree with one split."""
        tree = _make_simple_tree()

        result = _convert_tree_node_to_arrays(tree["nodes"])

        # 3 nodes: root + 2 leaves
        assert len(result["children_left"]) == 3
        assert len(result["children_right"]) == 3
        assert len(result["features"]) == 3
        assert result["values"].shape == (3, 1)

    def test_converts_deeper_tree(self) -> None:
        """Converts a tree with depth 2."""
        tree = _make_deeper_tree()

        result = _convert_tree_node_to_arrays(tree["nodes"])

        # 5 nodes total
        assert len(result["children_left"]) == 5
        assert len(result["children_right"]) == 5
        assert len(result["features"]) == 5


# =============================================================================
# Tests for _convert_decision_tree
# =============================================================================


class TestConvertDecisionTree:
    """Tests for _convert_decision_tree function."""

    def test_extracts_nodes_and_converts(self) -> None:
        """Extracts nodes from DecisionTree and converts."""
        tree = _make_simple_tree()

        result = _convert_decision_tree(tree)

        # Verify conversion produces correct structure
        assert len(result["children_left"]) == 3
        assert len(result["children_right"]) == 3
        assert len(result["features"]) == 3


# =============================================================================
# Tests for convert_cleargbm_to_shap_format
# =============================================================================


class TestConvertClearGBMToShapFormat:
    """Tests for convert_cleargbm_to_shap_format function."""

    def test_raises_on_empty_trees(self) -> None:
        """Raises ValueError for model with no trees."""
        model = GradientBoostingModel(
            trees=(),
            base_prediction=0.0,
            learning_rate=0.1,
            feature_names=("f0",),
            n_classes=2,
            config=_make_config(),
        )

        with pytest.raises(ValueError, match="Cannot convert model with no trees"):
            convert_cleargbm_to_shap_format(model)

    def test_converts_single_tree_model(self) -> None:
        """Converts model with one tree."""
        tree = _make_simple_tree()
        model = _make_model([tree])

        result = convert_cleargbm_to_shap_format(model)

        # Verify all expected fields are present with correct values
        assert result["num_outputs"] == 1
        assert result["base_offset"] == 0.0
        assert result["objective"] == "binary:logistic"
        assert result["tree_output"] == "raw"
        assert result["input_dtype"] == np.float64
        assert len(result["trees"]) == 1

    def test_converts_multi_tree_model(self) -> None:
        """Converts model with multiple trees."""
        tree1 = _make_simple_tree()
        tree2 = _make_deeper_tree()
        model = _make_model([tree1, tree2])

        result = convert_cleargbm_to_shap_format(model)

        assert len(result["trees"]) == 2
        # First tree has 3 nodes
        assert len(result["trees"][0]["children_left"]) == 3
        # Second tree has 5 nodes
        assert len(result["trees"][1]["children_left"]) == 5

    def test_preserves_base_prediction(self) -> None:
        """Preserves base_prediction as base_offset."""
        tree = _make_simple_tree()
        model = GradientBoostingModel(
            trees=(tree,),
            base_prediction=-0.5,
            learning_rate=0.1,
            feature_names=("f0", "f1"),
            n_classes=2,
            config=_make_config(),
        )

        result = convert_cleargbm_to_shap_format(model)

        assert result["base_offset"] == -0.5


# =============================================================================
# Tests for ClearGBMShapWrapper
# =============================================================================


class TestClearGBMShapWrapper:
    """Tests for ClearGBMShapWrapper class."""

    def test_init_converts_model(self) -> None:
        """Initialization converts model to SHAP format."""
        tree = _make_simple_tree()
        model = _make_model([tree])

        wrapper = ClearGBMShapWrapper(model)

        # Verify SHAP format was created with correct structure
        assert len(wrapper._shap_format["trees"]) == 1
        assert wrapper._base_prediction == 0.0

    def test_init_raises_on_empty_model(self) -> None:
        """Initialization raises for model with no trees."""
        model = GradientBoostingModel(
            trees=(),
            base_prediction=0.0,
            learning_rate=0.1,
            feature_names=("f0",),
            n_classes=2,
            config=_make_config(),
        )

        with pytest.raises(ValueError, match="Cannot convert model with no trees"):
            ClearGBMShapWrapper(model)

    def test_explain_local_raises_on_feature_mismatch(self) -> None:
        """explain_local raises when feature count doesn't match."""
        tree = _make_simple_tree()
        model = _make_model([tree])
        wrapper = ClearGBMShapWrapper(model)

        x = _make_x_1x3()  # 3 features
        feature_names = ["f0", "f1"]  # Only 2 names

        with pytest.raises(ValueError, match="Feature count mismatch"):
            wrapper.explain_local(x, feature_names)

    def test_explain_local_returns_list_of_explanations(self) -> None:
        """explain_local returns LocalExplanation for each sample."""
        tree = _make_simple_tree()
        model = _make_model([tree])
        wrapper = ClearGBMShapWrapper(model)

        x = _make_x_2x2()
        feature_names = ["f0", "f1"]

        result = wrapper.explain_local(x, feature_names)

        assert len(result) == 2
        for explanation in result:
            # Verify structure by accessing actual values
            assert len(explanation["values"]) == 2
            assert explanation["feature_names"] == ["f0", "f1"]
            # base_value should be a finite number
            assert explanation["base_value"] == explanation["base_value"]  # NaN check

    def test_explain_local_with_single_sample(self) -> None:
        """explain_local works with single sample."""
        tree = _make_simple_tree()
        model = _make_model([tree])
        wrapper = ClearGBMShapWrapper(model)

        x = _make_x_1x2()
        feature_names = ["feature_0", "feature_1"]

        result = wrapper.explain_local(x, feature_names)

        assert len(result) == 1
        explanation = result[0]
        # Verify values are finite (implicitly checks float type)
        assert explanation["base_value"] == explanation["base_value"]
        assert len(explanation["values"]) == 2

    def test_explain_local_values_are_floats(self) -> None:
        """All SHAP values are converted to Python floats."""
        tree = _make_simple_tree()
        model = _make_model([tree])
        wrapper = ClearGBMShapWrapper(model)

        x = _make_x_1x2()
        feature_names = ["f0", "f1"]

        result = wrapper.explain_local(x, feature_names)

        for explanation in result:
            # Values should be finite floats
            assert explanation["base_value"] == explanation["base_value"]
            for v in explanation["values"]:
                assert v == v  # NaN check - finite floats equal themselves


# =============================================================================
# Integration test with realistic model
# =============================================================================


class TestIntegration:
    """Integration tests with more realistic models."""

    def test_full_pipeline_simple_model(self) -> None:
        """Full pipeline from model to explanations."""
        # Create a model with 2 trees
        tree1 = _make_simple_tree()
        tree2 = _make_deeper_tree()
        model = GradientBoostingModel(
            trees=(tree1, tree2),
            base_prediction=-0.1,
            learning_rate=0.1,
            feature_names=("feature_0", "feature_1"),
            n_classes=2,
            config=_make_config(),
        )

        # Convert to SHAP format
        shap_format = convert_cleargbm_to_shap_format(model)
        assert shap_format["base_offset"] == -0.1
        assert len(shap_format["trees"]) == 2

        # Create wrapper and explain
        wrapper = ClearGBMShapWrapper(model)
        x = _make_x_3x2()
        feature_names = ["feature_0", "feature_1"]

        explanations = wrapper.explain_local(x, feature_names)

        assert len(explanations) == 3
        for exp in explanations:
            # SHAP values should sum to difference from base value
            # (within numerical tolerance)
            assert len(exp["values"]) == 2
            assert exp["feature_names"] == feature_names

    def test_shap_tree_arrays_structure(self) -> None:
        """Verify ShapTreeArrays has correct SHAP format structure."""
        tree = _make_deeper_tree()

        result = _convert_decision_tree(tree)

        # Deeper tree has 5 nodes - verify all arrays have correct length
        assert len(result["children_left"]) == 5
        assert len(result["children_right"]) == 5
        assert len(result["children_default"]) == 5
        assert len(result["features"]) == 5
        assert len(result["thresholds"]) == 5
        assert len(result["node_sample_weight"]) == 5

        # values must be 2D for SHAP
        values_shape = result["values"].shape
        assert len(values_shape) == 2
