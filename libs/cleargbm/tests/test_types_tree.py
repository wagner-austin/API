"""Tests for cleargbm.types: tree structures."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    DecisionTree,
    JSONDict,
    JSONTypeError,
    SplitCondition,
    TreeNode,
    TreePredictionExplanation,
    decode_decision_tree,
    decode_split_condition,
    decode_tree_node,
    decode_tree_prediction_explanation,
    encode_decision_tree,
    encode_split_condition,
    encode_tree_node,
    encode_tree_prediction_explanation,
)

# =============================================================================
# SplitCondition Tests
# =============================================================================


class TestSplitCondition:
    """Tests for SplitCondition encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        original: SplitCondition = {
            "feature_index": 2,
            "feature_name": "debt_ratio",
            "threshold": 2.5,
            "direction": "left",
        }
        encoded = encode_split_condition(original)
        decoded = decode_split_condition(encoded)

        assert decoded["feature_index"] == 2
        assert decoded["feature_name"] == "debt_ratio"
        assert decoded["threshold"] == 2.5
        assert decoded["direction"] == "left"

    def test_decode_direction_right(self) -> None:
        """Direction 'right' should decode correctly."""
        raw: JSONDict = {
            "feature_index": 0,
            "feature_name": "x",
            "threshold": 1.0,
            "direction": "right",
        }
        decoded = decode_split_condition(raw)
        assert decoded["direction"] == "right"

    def test_decode_invalid_direction(self) -> None:
        """Invalid direction should raise ValueError."""
        raw: JSONDict = {
            "feature_index": 0,
            "feature_name": "x",
            "threshold": 1.0,
            "direction": "up",
        }
        with pytest.raises(ValueError, match="direction must be 'left' or 'right'"):
            decode_split_condition(raw)

    def test_decode_missing_key(self) -> None:
        """Missing key should raise KeyError."""
        raw: JSONDict = {
            "feature_index": 0,
            "feature_name": "x",
            "threshold": 1.0,
            # missing direction
        }
        with pytest.raises(KeyError):
            decode_split_condition(raw)

    def test_decode_wrong_type(self) -> None:
        """Wrong type should raise TypeError."""
        raw: JSONDict = {
            "feature_index": "not an int",
            "feature_name": "x",
            "threshold": 1.0,
            "direction": "left",
        }
        with pytest.raises(JSONTypeError, match="feature_index must be int"):
            decode_split_condition(raw)

    def test_decode_negative_feature_index(self) -> None:
        """Negative feature_index should raise ValueError."""
        raw: JSONDict = {
            "feature_index": -1,
            "feature_name": "x",
            "threshold": 1.0,
            "direction": "left",
        }
        with pytest.raises(ValueError, match="feature_index must be non-negative"):
            decode_split_condition(raw)


# =============================================================================
# TreeNode Tests
# =============================================================================


class TestTreeNode:
    """Tests for TreeNode encode/decode."""

    def test_encode_decode_leaf_node(self) -> None:
        """Encode then decode leaf node should preserve data."""
        original: TreeNode = {
            "node_id": 1,
            "is_leaf": True,
            "feature_index": None,
            "feature_name": None,
            "threshold": None,
            "nan_direction": None,
            "value": 0.75,
            "n_samples": 50,
            "left_child": None,
            "right_child": None,
        }
        encoded = encode_tree_node(original)
        decoded = decode_tree_node(encoded)

        assert decoded["node_id"] == 1
        assert decoded["is_leaf"] is True
        assert decoded["feature_index"] is None
        assert decoded["nan_direction"] is None
        assert decoded["value"] == 0.75
        assert decoded["n_samples"] == 50

    def test_encode_decode_split_node(self) -> None:
        """Encode then decode split node should preserve data."""
        original: TreeNode = {
            "node_id": 0,
            "is_leaf": False,
            "feature_index": 2,
            "feature_name": "coverage",
            "threshold": 1.5,
            "nan_direction": "left",
            "value": 0.0,
            "n_samples": 100,
            "left_child": 1,
            "right_child": 2,
        }
        encoded = encode_tree_node(original)
        decoded = decode_tree_node(encoded)

        assert decoded["node_id"] == 0
        assert decoded["is_leaf"] is False
        assert decoded["feature_index"] == 2
        assert decoded["feature_name"] == "coverage"
        assert decoded["threshold"] == 1.5
        assert decoded["nan_direction"] == "left"
        assert decoded["left_child"] == 1
        assert decoded["right_child"] == 2

    def test_decode_bool_as_int_fails(self) -> None:
        """Boolean value where int expected should raise TypeError."""
        raw: JSONDict = {
            "node_id": True,  # bool, not int
            "is_leaf": True,
            "feature_index": None,
            "feature_name": None,
            "threshold": None,
            "value": 0.0,
            "n_samples": 10,
            "left_child": None,
            "right_child": None,
        }
        with pytest.raises(JSONTypeError, match="node_id must be int"):
            decode_tree_node(raw)

    def test_decode_int_as_float_coerced(self) -> None:
        """Integer value where float expected should be coerced."""
        raw: JSONDict = {
            "node_id": 0,
            "is_leaf": True,
            "feature_index": None,
            "feature_name": None,
            "threshold": None,
            "value": 1,  # int, will be coerced to float
            "n_samples": 10,
            "left_child": None,
            "right_child": None,
        }
        decoded = decode_tree_node(raw)
        assert decoded["value"] == 1.0

    def test_decode_nan_direction_wrong_type_fails(self) -> None:
        """Non-string nan_direction should raise JSONTypeError."""
        raw: JSONDict = {
            "node_id": 0,
            "is_leaf": False,
            "feature_index": 0,
            "feature_name": "f0",
            "threshold": 0.5,
            "nan_direction": 123,  # Wrong type - should be str
            "value": 0.0,
            "n_samples": 10,
            "left_child": 1,
            "right_child": 2,
        }
        with pytest.raises(JSONTypeError, match="nan_direction must be str or None"):
            decode_tree_node(raw)

    def test_decode_nan_direction_invalid_value_fails(self) -> None:
        """Invalid nan_direction value should raise ValueError."""
        raw: JSONDict = {
            "node_id": 0,
            "is_leaf": False,
            "feature_index": 0,
            "feature_name": "f0",
            "threshold": 0.5,
            "nan_direction": "center",  # Invalid - must be "left" or "right"
            "value": 0.0,
            "n_samples": 10,
            "left_child": 1,
            "right_child": 2,
        }
        with pytest.raises(ValueError, match="nan_direction must be 'left' or 'right'"):
            decode_tree_node(raw)


# =============================================================================
# DecisionTree Tests
# =============================================================================


class TestDecisionTree:
    """Tests for DecisionTree encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        node: TreeNode = {
            "node_id": 0,
            "is_leaf": True,
            "feature_index": None,
            "feature_name": None,
            "threshold": None,
            "value": 0.5,
            "n_samples": 100,
            "left_child": None,
            "right_child": None,
            "nan_direction": None,
        }
        original: DecisionTree = {
            "nodes": (node,),
            "max_depth": 0,
            "n_leaves": 1,
            "feature_names": ("x", "y"),
        }
        encoded = encode_decision_tree(original)
        decoded = decode_decision_tree(encoded)

        assert len(decoded["nodes"]) == 1
        assert decoded["max_depth"] == 0
        assert decoded["n_leaves"] == 1
        assert decoded["feature_names"] == ("x", "y")

    def test_decode_nodes_not_list(self) -> None:
        """nodes not a list should raise TypeError."""
        raw: JSONDict = {
            "nodes": "not a list",
            "max_depth": 0,
            "n_leaves": 1,
            "feature_names": ["x"],
        }
        with pytest.raises(JSONTypeError, match="nodes must be list"):
            decode_decision_tree(raw)

    def test_decode_node_not_dict(self) -> None:
        """Node not a dict should raise TypeError."""
        raw: JSONDict = {
            "nodes": ["not a dict"],
            "max_depth": 0,
            "n_leaves": 1,
            "feature_names": ["x"],
        }
        with pytest.raises(JSONTypeError, match=r"nodes\[0\] must be dict"):
            decode_decision_tree(raw)

    def test_decode_feature_names_not_list(self) -> None:
        """feature_names not a list should raise TypeError."""
        node_raw: JSONDict = {
            "node_id": 0,
            "is_leaf": True,
            "feature_index": None,
            "feature_name": None,
            "threshold": None,
            "value": 0.0,
            "n_samples": 10,
            "left_child": None,
            "right_child": None,
        }
        raw: JSONDict = {
            "nodes": [node_raw],
            "max_depth": 0,
            "n_leaves": 1,
            "feature_names": "not a list",
        }
        with pytest.raises(JSONTypeError, match="feature_names must be list"):
            decode_decision_tree(raw)

    def test_decode_feature_name_not_str(self) -> None:
        """Feature name not a string should raise TypeError."""
        node_raw: JSONDict = {
            "node_id": 0,
            "is_leaf": True,
            "feature_index": None,
            "feature_name": None,
            "threshold": None,
            "value": 0.0,
            "n_samples": 10,
            "left_child": None,
            "right_child": None,
        }
        raw: JSONDict = {
            "nodes": [node_raw],
            "max_depth": 0,
            "n_leaves": 1,
            "feature_names": [123],  # not a string
        }
        with pytest.raises(JSONTypeError, match=r"feature_names\[0\] must be str"):
            decode_decision_tree(raw)


# =============================================================================
# TreePredictionExplanation Tests
# =============================================================================


class TestTreePredictionExplanation:
    """Tests for TreePredictionExplanation encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        split: SplitCondition = {
            "feature_index": 0,
            "feature_name": "x",
            "threshold": 1.0,
            "direction": "left",
        }
        original: TreePredictionExplanation = {
            "tree_index": 5,
            "prediction": 0.25,
            "path": (split,),
            "leaf_node_id": 3,
            "n_samples_in_leaf": 42,
        }
        encoded = encode_tree_prediction_explanation(original)
        decoded = decode_tree_prediction_explanation(encoded)

        assert decoded["tree_index"] == 5
        assert decoded["prediction"] == 0.25
        assert len(decoded["path"]) == 1
        assert decoded["path"][0]["feature_name"] == "x"
        assert decoded["leaf_node_id"] == 3
        assert decoded["n_samples_in_leaf"] == 42

    def test_decode_path_not_list(self) -> None:
        """path not a list should raise TypeError."""
        raw: JSONDict = {
            "tree_index": 0,
            "prediction": 0.0,
            "path": "not a list",
            "leaf_node_id": 0,
            "n_samples_in_leaf": 10,
        }
        with pytest.raises(JSONTypeError, match="path must be list"):
            decode_tree_prediction_explanation(raw)

    def test_decode_path_item_not_dict(self) -> None:
        """path item not a dict should raise TypeError."""
        raw: JSONDict = {
            "tree_index": 0,
            "prediction": 0.0,
            "path": ["not a dict"],
            "leaf_node_id": 0,
            "n_samples_in_leaf": 10,
        }
        with pytest.raises(JSONTypeError, match=r"path\[0\] must be dict"):
            decode_tree_prediction_explanation(raw)
