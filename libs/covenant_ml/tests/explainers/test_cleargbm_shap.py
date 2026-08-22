"""Tests for ClearGBM SHAP adapter.

Comprehensive tests for converting ClearGBM models to SHAP format
and computing local explanations.
"""

from __future__ import annotations

import numpy as np
import pytest
from cleargbm.types import GradientBoostingModel, TreeNode

from covenant_ml.explainers.cleargbm_shap import (
    _convert_decision_tree,
    _convert_tree_node_to_arrays,
    _get_default_child_idx,
    _populate_internal_node,
    _ShapArrays,
    convert_cleargbm_to_shap_format,
)
from tests.explainers._cleargbm_shap_fixtures import (
    _make_config,
    _make_deeper_tree,
    _make_leaf_node,
    _make_model,
    _make_simple_tree,
    _make_split_node,
)


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
