"""Tests for cleargbm.types: JSON helper error paths."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    JSONTypeError,
    decode_gradient_boosting_model,
    decode_split_condition,
    decode_tree_node,
)

# =============================================================================
# Helper Function Error Path Tests
# =============================================================================


class TestHelperErrorPaths:
    """Tests for helper function error paths to ensure 100% coverage."""

    def test_require_str_wrong_type(self) -> None:
        """_require_str with non-string should raise JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_name must be str, got int"):
            decode_split_condition(
                {
                    "feature_index": 0,
                    "feature_name": 123,  # wrong type
                    "threshold": 1.0,
                    "direction": "left",
                }
            )

    def test_require_float_bool_rejected(self) -> None:
        """_require_float with bool should raise JSONTypeError."""
        with pytest.raises(JSONTypeError, match="threshold must be float, got bool"):
            decode_split_condition(
                {
                    "feature_index": 0,
                    "feature_name": "x",
                    "threshold": True,  # bool, not float
                    "direction": "left",
                }
            )

    def test_require_float_wrong_type(self) -> None:
        """_require_float with string should raise JSONTypeError."""
        with pytest.raises(JSONTypeError, match="threshold must be float"):
            decode_split_condition(
                {
                    "feature_index": 0,
                    "feature_name": "x",
                    "threshold": "not a float",
                    "direction": "left",
                }
            )

    def test_require_bool_wrong_type(self) -> None:
        """_require_bool with non-bool should raise JSONTypeError."""
        with pytest.raises(JSONTypeError, match="is_leaf must be bool"):
            decode_tree_node(
                {
                    "node_id": 0,
                    "is_leaf": "yes",  # string, not bool
                    "feature_index": None,
                    "feature_name": None,
                    "threshold": None,
                    "value": 0.0,
                    "n_samples": 10,
                    "left_child": None,
                    "right_child": None,
                }
            )

    def test_get_optional_int_bool_rejected(self) -> None:
        """_get_optional_int with bool should raise JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_index must be int or None"):
            decode_tree_node(
                {
                    "node_id": 0,
                    "is_leaf": True,
                    "feature_index": True,  # bool, not int
                    "feature_name": None,
                    "threshold": None,
                    "value": 0.0,
                    "n_samples": 10,
                    "left_child": None,
                    "right_child": None,
                }
            )

    def test_get_optional_int_wrong_type(self) -> None:
        """_get_optional_int with string should raise JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_index must be int or None"):
            decode_tree_node(
                {
                    "node_id": 0,
                    "is_leaf": True,
                    "feature_index": "not an int",
                    "feature_name": None,
                    "threshold": None,
                    "value": 0.0,
                    "n_samples": 10,
                    "left_child": None,
                    "right_child": None,
                }
            )

    def test_get_optional_float_bool_rejected(self) -> None:
        """_get_optional_float with bool should raise JSONTypeError."""
        with pytest.raises(JSONTypeError, match="threshold must be float or None, got bool"):
            decode_tree_node(
                {
                    "node_id": 0,
                    "is_leaf": True,
                    "feature_index": None,
                    "feature_name": None,
                    "threshold": False,  # bool, not float
                    "value": 0.0,
                    "n_samples": 10,
                    "left_child": None,
                    "right_child": None,
                }
            )

    def test_get_optional_float_wrong_type(self) -> None:
        """_get_optional_float with string should raise JSONTypeError."""
        with pytest.raises(JSONTypeError, match="threshold must be float or None"):
            decode_tree_node(
                {
                    "node_id": 0,
                    "is_leaf": True,
                    "feature_index": None,
                    "feature_name": None,
                    "threshold": "not a float",
                    "value": 0.0,
                    "n_samples": 10,
                    "left_child": None,
                    "right_child": None,
                }
            )

    def test_get_optional_str_wrong_type(self) -> None:
        """_get_optional_str with non-string should raise JSONTypeError."""
        with pytest.raises(JSONTypeError, match="feature_name must be str or None"):
            decode_tree_node(
                {
                    "node_id": 0,
                    "is_leaf": True,
                    "feature_index": None,
                    "feature_name": 123,  # int, not str
                    "threshold": None,
                    "value": 0.0,
                    "n_samples": 10,
                    "left_child": None,
                    "right_child": None,
                }
            )

    def test_decode_model_tree_not_dict(self) -> None:
        """decode_gradient_boosting_model with tree not dict should raise."""
        with pytest.raises(JSONTypeError, match=r"trees\[0\] must be dict"):
            decode_gradient_boosting_model(
                {
                    "trees": ["not a dict"],
                    "base_prediction": 0.0,
                    "learning_rate": 0.1,
                    "feature_names": ["x"],
                    "n_classes": 2,
                    "config": {
                        "n_estimators": 1,
                        "max_depth": 1,
                        "learning_rate": 0.1,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "max_features": None,
                        "max_bins": 64,
                        "subsample": 1.0,
                        "random_state": 0,
                        "track_contributions": False,
                        "monotonic_constraints": None,
                    },
                }
            )

    def test_decode_model_feature_name_not_str(self) -> None:
        """decode_gradient_boosting_model with feature_name not str should raise."""
        with pytest.raises(JSONTypeError, match=r"feature_names\[0\] must be str"):
            decode_gradient_boosting_model(
                {
                    "trees": [],
                    "base_prediction": 0.0,
                    "learning_rate": 0.1,
                    "feature_names": [123],  # int, not str
                    "n_classes": 2,
                    "config": {
                        "n_estimators": 1,
                        "max_depth": 1,
                        "learning_rate": 0.1,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "max_features": None,
                        "max_bins": 64,
                        "subsample": 1.0,
                        "random_state": 0,
                        "track_contributions": False,
                        "monotonic_constraints": None,
                    },
                }
            )

    def test_decode_model_feature_names_not_list(self) -> None:
        """decode_gradient_boosting_model with feature_names not list should raise."""
        with pytest.raises(JSONTypeError, match="feature_names must be list"):
            decode_gradient_boosting_model(
                {
                    "trees": [],
                    "base_prediction": 0.0,
                    "learning_rate": 0.1,
                    "feature_names": "not a list",  # string, not list
                    "n_classes": 2,
                    "config": {
                        "n_estimators": 1,
                        "max_depth": 1,
                        "learning_rate": 0.1,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1,
                        "max_features": None,
                        "max_bins": 64,
                        "subsample": 1.0,
                        "random_state": 0,
                        "track_contributions": False,
                        "monotonic_constraints": None,
                    },
                }
            )

    def test_missing_optional_int_key(self) -> None:
        """_get_optional_int returns None for missing key."""
        from cleargbm.types import decode_tree_node

        # Tree node with missing optional keys (feature_index, left_child, right_child)
        result = decode_tree_node(
            {
                "node_id": 0,
                "is_leaf": True,
                # feature_index key is missing
                "feature_name": None,
                "threshold": None,
                "value": 0.5,
                "n_samples": 10,
                # left_child key is missing
                # right_child key is missing
            }
        )
        assert result["feature_index"] is None
        assert result["left_child"] is None
        assert result["right_child"] is None

    def test_missing_optional_float_key(self) -> None:
        """_get_optional_float returns None for missing key."""
        from cleargbm.types import decode_tree_node

        # Tree node with missing threshold key
        result = decode_tree_node(
            {
                "node_id": 0,
                "is_leaf": True,
                "feature_index": None,
                "feature_name": None,
                # threshold key is missing
                "value": 0.5,
                "n_samples": 10,
                "left_child": None,
                "right_child": None,
            }
        )
        assert result["threshold"] is None

    def test_missing_optional_str_key(self) -> None:
        """_get_optional_str returns None for missing key."""
        from cleargbm.types import decode_tree_node

        # Tree node with missing feature_name key
        result = decode_tree_node(
            {
                "node_id": 0,
                "is_leaf": True,
                "feature_index": None,
                # feature_name key is missing
                "threshold": None,
                "value": 0.5,
                "n_samples": 10,
                "left_child": None,
                "right_child": None,
            }
        )
        assert result["feature_name"] is None

    def test_optional_float_int_coercion(self) -> None:
        """_get_optional_float coerces int to float."""
        from cleargbm.types import decode_tree_node

        # Tree node with threshold as int (should be coerced to float)
        result = decode_tree_node(
            {
                "node_id": 0,
                "is_leaf": False,
                "feature_index": 0,
                "feature_name": "x",
                "threshold": 5,  # int, should be coerced to float
                "value": 0.5,
                "n_samples": 10,
                "left_child": 1,
                "right_child": 2,
            }
        )
        # The int 5 should be coerced to float 5.0
        # This test covers line 256: return float(value)
        assert result["threshold"] == 5.0
