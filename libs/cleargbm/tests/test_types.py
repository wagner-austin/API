"""Tests for cleargbm.types module."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    DecisionTree,
    FeatureContribution,
    FloatBufferData,
    GradientBoostingConfig,
    GradientBoostingModel,
    HistogramBufferData,
    IntBufferData,
    JSONDict,
    JSONTypeError,
    PredictionExplanation,
    Rule,
    SplitCondition,
    TimingResult,
    TrainingProgress,
    TreeNode,
    TreePredictionExplanation,
    TuningReport,
    decode_decision_tree,
    decode_feature_contribution,
    decode_float_buffer_data,
    decode_gradient_boosting_config,
    decode_gradient_boosting_model,
    decode_histogram_buffer_data,
    decode_int_buffer_data,
    decode_prediction_explanation,
    decode_rule,
    decode_split_condition,
    decode_timing_result,
    decode_training_progress,
    decode_tree_node,
    decode_tree_prediction_explanation,
    decode_tuning_report,
    encode_decision_tree,
    encode_feature_contribution,
    encode_float_buffer_data,
    encode_gradient_boosting_config,
    encode_gradient_boosting_model,
    encode_histogram_buffer_data,
    encode_int_buffer_data,
    encode_prediction_explanation,
    encode_rule,
    encode_split_condition,
    encode_timing_result,
    encode_training_progress,
    encode_tree_node,
    encode_tree_prediction_explanation,
    encode_tuning_report,
    require_n_jobs,
    require_non_negative_float,
    require_non_negative_int,
    require_positive_float,
    require_positive_int,
    require_unit_float,
)

# =============================================================================
# Validation Helpers Tests
# =============================================================================


class TestRequirePositiveInt:
    """Tests for require_positive_int."""

    def test_accepts_positive(self) -> None:
        """Positive integers should pass."""
        assert require_positive_int(1, "x") == 1
        assert require_positive_int(100, "x") == 100

    def test_rejects_zero(self) -> None:
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match="x must be positive, got 0"):
            require_positive_int(0, "x")

    def test_rejects_negative(self) -> None:
        """Negative integers should raise ValueError."""
        with pytest.raises(ValueError, match="x must be positive, got -5"):
            require_positive_int(-5, "x")


class TestRequireNonNegativeInt:
    """Tests for require_non_negative_int."""

    def test_accepts_positive(self) -> None:
        """Positive integers should pass."""
        assert require_non_negative_int(1, "x") == 1

    def test_accepts_zero(self) -> None:
        """Zero should pass."""
        assert require_non_negative_int(0, "x") == 0

    def test_rejects_negative(self) -> None:
        """Negative integers should raise ValueError."""
        with pytest.raises(ValueError, match="x must be non-negative, got -1"):
            require_non_negative_int(-1, "x")


class TestRequirePositiveFloat:
    """Tests for require_positive_float."""

    def test_accepts_positive(self) -> None:
        """Positive floats should pass."""
        assert require_positive_float(0.1, "x") == 0.1
        assert require_positive_float(100.5, "x") == 100.5

    def test_rejects_zero(self) -> None:
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be positive, got 0\.0"):
            require_positive_float(0.0, "x")

    def test_rejects_negative(self) -> None:
        """Negative floats should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be positive, got -0\.5"):
            require_positive_float(-0.5, "x")


class TestRequireUnitFloat:
    """Tests for require_unit_float."""

    def test_accepts_in_range(self) -> None:
        """Values in (0, 1] should pass."""
        assert require_unit_float(0.5, "x") == 0.5
        assert require_unit_float(1.0, "x") == 1.0
        assert require_unit_float(0.001, "x") == 0.001

    def test_rejects_zero(self) -> None:
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be in \(0, 1\], got 0.0"):
            require_unit_float(0.0, "x")

    def test_rejects_greater_than_one(self) -> None:
        """Values > 1 should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be in \(0, 1\], got 1.5"):
            require_unit_float(1.5, "x")

    def test_rejects_negative(self) -> None:
        """Negative values should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be in \(0, 1\], got -0.1"):
            require_unit_float(-0.1, "x")


class TestRequireNonNegativeFloat:
    """Tests for require_non_negative_float."""

    def test_accepts_positive(self) -> None:
        """Positive floats should pass."""
        assert require_non_negative_float(0.5, "x") == 0.5

    def test_accepts_zero(self) -> None:
        """Zero should pass."""
        assert require_non_negative_float(0.0, "x") == 0.0

    def test_rejects_negative(self) -> None:
        """Negative floats should raise ValueError."""
        with pytest.raises(ValueError, match=r"x must be non-negative, got -0\.1"):
            require_non_negative_float(-0.1, "x")


class TestRequireNJobs:
    """Tests for require_n_jobs."""

    def test_accepts_positive(self) -> None:
        """Positive integers should pass."""
        assert require_n_jobs(1, "n_jobs") == 1
        assert require_n_jobs(4, "n_jobs") == 4
        assert require_n_jobs(100, "n_jobs") == 100

    def test_accepts_minus_one(self) -> None:
        """-1 should pass (use all cores)."""
        assert require_n_jobs(-1, "n_jobs") == -1

    def test_rejects_zero(self) -> None:
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive, got 0"):
            require_n_jobs(0, "n_jobs")

    def test_rejects_negative_other_than_minus_one(self) -> None:
        """Negative values other than -1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive, got -2"):
            require_n_jobs(-2, "n_jobs")
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive, got -5"):
            require_n_jobs(-5, "n_jobs")


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


# =============================================================================
# GradientBoostingConfig Tests
# =============================================================================


class TestGradientBoostingConfig:
    """Tests for GradientBoostingConfig encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        original: GradientBoostingConfig = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": 3,
            "max_bins": 64,
            "subsample": 0.8,
            "random_state": 42,
            "track_contributions": True,
            "monotonic_constraints": (1, -1, 0),
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": 4,
            "early_stopping_rounds": 10,
        }
        encoded = encode_gradient_boosting_config(original)
        decoded = decode_gradient_boosting_config(encoded)

        assert decoded["n_estimators"] == 100
        assert decoded["max_depth"] == 4
        assert decoded["learning_rate"] == 0.1
        assert decoded["min_samples_split"] == 10
        assert decoded["min_samples_leaf"] == 5
        assert decoded["max_features"] == 3
        assert decoded["max_bins"] == 64
        assert decoded["subsample"] == 0.8
        assert decoded["random_state"] == 42
        assert decoded["track_contributions"] is True
        assert decoded["monotonic_constraints"] == (1, -1, 0)
        assert decoded["reg_alpha"] == 0.1
        assert decoded["reg_lambda"] == 1.0
        assert decoded["n_jobs"] == 4
        assert decoded["early_stopping_rounds"] == 10

    def test_encode_decode_with_none_optionals(self) -> None:
        """None values for optional fields should roundtrip."""
        original: GradientBoostingConfig = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "track_contributions": False,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
        }
        encoded = encode_gradient_boosting_config(original)
        decoded = decode_gradient_boosting_config(encoded)

        assert decoded["max_features"] is None
        assert decoded["monotonic_constraints"] is None
        assert decoded["reg_alpha"] == 0.0
        assert decoded["reg_lambda"] == 0.0
        assert decoded["n_jobs"] == 1
        assert decoded["early_stopping_rounds"] is None

    def test_decode_invalid_monotonic_constraint_value(self) -> None:
        """Monotonic constraint value not in {-1, 0, 1} should raise ValueError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "track_contributions": False,
            "monotonic_constraints": [2],  # invalid value
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
        }
        with pytest.raises(ValueError, match=r"monotonic_constraints\[0\] must be -1, 0, or 1"):
            decode_gradient_boosting_config(raw)

    def test_decode_monotonic_constraints_not_list(self) -> None:
        """monotonic_constraints not a list should raise TypeError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "track_contributions": False,
            "monotonic_constraints": "not a list",
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
        }
        with pytest.raises(JSONTypeError, match="monotonic_constraints must be list or None"):
            decode_gradient_boosting_config(raw)

    def test_decode_monotonic_constraint_not_int(self) -> None:
        """Monotonic constraint item not an int should raise TypeError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "track_contributions": False,
            "monotonic_constraints": ["not an int"],
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
        }
        with pytest.raises(JSONTypeError, match=r"monotonic_constraints\[0\] must be int"):
            decode_gradient_boosting_config(raw)

    def test_decode_bool_in_monotonic_constraints_fails(self) -> None:
        """Boolean in monotonic_constraints should raise TypeError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "track_contributions": False,
            "monotonic_constraints": [True],  # bool, not int
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
        }
        with pytest.raises(JSONTypeError, match=r"monotonic_constraints\[0\] must be int"):
            decode_gradient_boosting_config(raw)

    def test_encode_decode_n_jobs_minus_one(self) -> None:
        """n_jobs=-1 (all cores) should roundtrip."""
        original: GradientBoostingConfig = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "track_contributions": False,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": -1,
            "early_stopping_rounds": None,
        }
        encoded = encode_gradient_boosting_config(original)
        decoded = decode_gradient_boosting_config(encoded)

        assert decoded["n_jobs"] == -1

    def test_decode_n_jobs_invalid(self) -> None:
        """Invalid n_jobs should raise ValueError."""
        raw: JSONDict = {
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "track_contributions": False,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 0,  # invalid
            "early_stopping_rounds": None,
        }
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive"):
            decode_gradient_boosting_config(raw)


# =============================================================================
# GradientBoostingModel Tests
# =============================================================================


class TestGradientBoostingModel:
    """Tests for GradientBoostingModel encode/decode."""

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
        tree: DecisionTree = {
            "nodes": (node,),
            "max_depth": 0,
            "n_leaves": 1,
            "feature_names": ("x",),
        }
        config: GradientBoostingConfig = {
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
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
        }
        original: GradientBoostingModel = {
            "trees": (tree,),
            "base_prediction": -0.5,
            "learning_rate": 0.1,
            "feature_names": ("x",),
            "n_classes": 2,
            "config": config,
        }
        encoded = encode_gradient_boosting_model(original)
        decoded = decode_gradient_boosting_model(encoded)

        assert len(decoded["trees"]) == 1
        assert decoded["base_prediction"] == -0.5
        assert decoded["learning_rate"] == 0.1
        assert decoded["feature_names"] == ("x",)
        assert decoded["n_classes"] == 2

    def test_decode_trees_not_list(self) -> None:
        """trees not a list should raise TypeError."""
        raw: JSONDict = {
            "trees": "not a list",
            "base_prediction": 0.0,
            "learning_rate": 0.1,
            "feature_names": ["x"],
            "n_classes": 2,
            "config": {},
        }
        with pytest.raises(JSONTypeError, match="trees must be list"):
            decode_gradient_boosting_model(raw)

    def test_decode_config_not_dict(self) -> None:
        """config not a dict should raise TypeError."""
        raw: JSONDict = {
            "trees": [],
            "base_prediction": 0.0,
            "learning_rate": 0.1,
            "feature_names": ["x"],
            "n_classes": 2,
            "config": "not a dict",
        }
        with pytest.raises(JSONTypeError, match="config must be dict"):
            decode_gradient_boosting_model(raw)


# =============================================================================
# FeatureContribution Tests
# =============================================================================


class TestFeatureContribution:
    """Tests for FeatureContribution encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        original: FeatureContribution = {
            "feature_name": "debt_ratio",
            "feature_index": 0,
            "total_contribution": 0.35,
            "n_splits": 7,
        }
        encoded = encode_feature_contribution(original)
        decoded = decode_feature_contribution(encoded)

        assert decoded["feature_name"] == "debt_ratio"
        assert decoded["feature_index"] == 0
        assert decoded["total_contribution"] == 0.35
        assert decoded["n_splits"] == 7


# =============================================================================
# PredictionExplanation Tests
# =============================================================================


class TestPredictionExplanation:
    """Tests for PredictionExplanation encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        split: SplitCondition = {
            "feature_index": 0,
            "feature_name": "x",
            "threshold": 1.0,
            "direction": "left",
        }
        tree_contrib: TreePredictionExplanation = {
            "tree_index": 0,
            "prediction": 0.1,
            "path": (split,),
            "leaf_node_id": 1,
            "n_samples_in_leaf": 10,
        }
        feature_contrib: FeatureContribution = {
            "feature_name": "x",
            "feature_index": 0,
            "total_contribution": 0.1,
            "n_splits": 1,
        }
        original: PredictionExplanation = {
            "final_probability": 0.75,
            "base_prediction": 0.0,
            "tree_contributions": (tree_contrib,),
            "top_features": (feature_contrib,),
        }
        encoded = encode_prediction_explanation(original)
        decoded = decode_prediction_explanation(encoded)

        assert decoded["final_probability"] == 0.75
        assert decoded["base_prediction"] == 0.0
        assert len(decoded["tree_contributions"]) == 1
        assert len(decoded["top_features"]) == 1

    def test_decode_tree_contributions_not_list(self) -> None:
        """tree_contributions not a list should raise TypeError."""
        raw: JSONDict = {
            "final_probability": 0.5,
            "base_prediction": 0.0,
            "tree_contributions": "not a list",
            "top_features": [],
        }
        with pytest.raises(JSONTypeError, match="tree_contributions must be list"):
            decode_prediction_explanation(raw)

    def test_decode_top_features_not_list(self) -> None:
        """top_features not a list should raise TypeError."""
        raw: JSONDict = {
            "final_probability": 0.5,
            "base_prediction": 0.0,
            "tree_contributions": [],
            "top_features": "not a list",
        }
        with pytest.raises(JSONTypeError, match="top_features must be list"):
            decode_prediction_explanation(raw)


# =============================================================================
# Rule Tests
# =============================================================================


class TestRule:
    """Tests for Rule encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        original: Rule = {
            "conditions": ("debt_ratio > 2.5", "coverage < 1.2"),
            "prediction_contribution": 0.25,
            "n_samples": 150,
            "importance": 0.85,
        }
        encoded = encode_rule(original)
        decoded = decode_rule(encoded)

        assert decoded["conditions"] == ("debt_ratio > 2.5", "coverage < 1.2")
        assert decoded["prediction_contribution"] == 0.25
        assert decoded["n_samples"] == 150
        assert decoded["importance"] == 0.85

    def test_decode_conditions_not_list(self) -> None:
        """conditions not a list should raise TypeError."""
        raw: JSONDict = {
            "conditions": "not a list",
            "prediction_contribution": 0.0,
            "n_samples": 0,
            "importance": 0.0,
        }
        with pytest.raises(JSONTypeError, match="conditions must be list"):
            decode_rule(raw)

    def test_decode_condition_not_str(self) -> None:
        """condition item not a string should raise TypeError."""
        raw: JSONDict = {
            "conditions": [123],  # not a string
            "prediction_contribution": 0.0,
            "n_samples": 0,
            "importance": 0.0,
        }
        with pytest.raises(JSONTypeError, match=r"conditions\[0\] must be str"):
            decode_rule(raw)


# =============================================================================
# TrainingProgress Tests
# =============================================================================


class TestTrainingProgress:
    """Tests for TrainingProgress encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        original: TrainingProgress = {
            "tree_index": 50,
            "total_trees": 100,
            "train_loss": 0.35,
            "val_loss": 0.42,
        }
        encoded = encode_training_progress(original)
        decoded = decode_training_progress(encoded)

        assert decoded["tree_index"] == 50
        assert decoded["total_trees"] == 100
        assert decoded["train_loss"] == 0.35
        assert decoded["val_loss"] == 0.42

    def test_encode_decode_with_none_val_loss(self) -> None:
        """None val_loss should roundtrip."""
        original: TrainingProgress = {
            "tree_index": 10,
            "total_trees": 50,
            "train_loss": 0.5,
            "val_loss": None,
        }
        encoded = encode_training_progress(original)
        decoded = decode_training_progress(encoded)

        assert decoded["val_loss"] is None


# =============================================================================
# TimingResult Tests
# =============================================================================


class TestTimingResult:
    """Tests for TimingResult encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        original: TimingResult = {
            "n_jobs": 4,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.5,
            "trees_per_second": 3.33,
        }
        encoded = encode_timing_result(original)
        decoded = decode_timing_result(encoded)

        assert decoded["n_jobs"] == 4
        assert decoded["max_bins"] == 64
        assert decoded["max_depth"] == 4
        assert decoded["learning_rate"] == 0.1
        assert decoded["elapsed_seconds"] == 1.5
        assert decoded["trees_per_second"] == 3.33

    def test_decode_n_jobs_minus_one(self) -> None:
        """n_jobs=-1 (all cores) should decode correctly."""
        raw: JSONDict = {
            "n_jobs": -1,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.0,
            "trees_per_second": 5.0,
        }
        decoded = decode_timing_result(raw)
        assert decoded["n_jobs"] == -1

    def test_decode_invalid_n_jobs(self) -> None:
        """Invalid n_jobs should raise ValueError."""
        raw: JSONDict = {
            "n_jobs": 0,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.0,
            "trees_per_second": 5.0,
        }
        with pytest.raises(ValueError, match="n_jobs must be -1 or positive"):
            decode_timing_result(raw)

    def test_decode_invalid_max_bins(self) -> None:
        """Invalid max_bins should raise ValueError."""
        raw: JSONDict = {
            "n_jobs": 1,
            "max_bins": 0,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.0,
            "trees_per_second": 5.0,
        }
        with pytest.raises(ValueError, match="max_bins must be positive"):
            decode_timing_result(raw)

    def test_decode_negative_elapsed_seconds(self) -> None:
        """Negative elapsed_seconds should raise ValueError."""
        raw: JSONDict = {
            "n_jobs": 1,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": -1.0,
            "trees_per_second": 5.0,
        }
        with pytest.raises(ValueError, match="elapsed_seconds must be non-negative"):
            decode_timing_result(raw)

    def test_decode_missing_key(self) -> None:
        """Missing key should raise KeyError."""
        raw: JSONDict = {
            "n_jobs": 1,
            "max_bins": 64,
            # missing max_depth
            "learning_rate": 0.1,
            "elapsed_seconds": 1.0,
            "trees_per_second": 5.0,
        }
        with pytest.raises(KeyError):
            decode_timing_result(raw)


# =============================================================================
# TuningReport Tests
# =============================================================================


class TestTuningReport:
    """Tests for TuningReport encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should preserve data."""
        config: GradientBoostingConfig = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "track_contributions": False,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 2,
            "early_stopping_rounds": None,
        }
        timing_result: TimingResult = {
            "n_jobs": 2,
            "max_bins": 64,
            "max_depth": 4,
            "learning_rate": 0.1,
            "elapsed_seconds": 1.5,
            "trees_per_second": 3.33,
        }
        original: TuningReport = {
            "best_config": config,
            "timing_results": (timing_result,),
            "sample_size": 1000,
            "n_features": 10,
            "recommended_n_jobs": 2,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.8,
            "total_tune_time_seconds": 30.5,
        }
        encoded = encode_tuning_report(original)
        decoded = decode_tuning_report(encoded)

        assert decoded["best_config"]["n_jobs"] == 2
        assert len(decoded["timing_results"]) == 1
        assert decoded["timing_results"][0]["n_jobs"] == 2
        assert decoded["sample_size"] == 1000
        assert decoded["n_features"] == 10
        assert decoded["recommended_n_jobs"] == 2
        assert decoded["recommended_max_bins"] == 64
        assert decoded["parallel_speedup"] == 1.8
        assert decoded["total_tune_time_seconds"] == 30.5

    def test_decode_timing_results_not_list(self) -> None:
        """timing_results not a list should raise JSONTypeError."""
        config_raw: JSONDict = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "track_contributions": False,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
        }
        raw: JSONDict = {
            "best_config": config_raw,
            "timing_results": "not a list",
            "sample_size": 100,
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.0,
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(JSONTypeError, match="timing_results must be list"):
            decode_tuning_report(raw)

    def test_decode_timing_result_not_dict(self) -> None:
        """timing_results item not a dict should raise JSONTypeError."""
        config_raw: JSONDict = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "track_contributions": False,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
        }
        raw: JSONDict = {
            "best_config": config_raw,
            "timing_results": ["not a dict"],
            "sample_size": 100,
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.0,
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(JSONTypeError, match=r"timing_results\[0\] must be dict"):
            decode_tuning_report(raw)

    def test_decode_invalid_sample_size(self) -> None:
        """Invalid sample_size should raise ValueError."""
        config_raw: JSONDict = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "track_contributions": False,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
        }
        raw: JSONDict = {
            "best_config": config_raw,
            "timing_results": [],
            "sample_size": 0,  # invalid
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.0,
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(ValueError, match="sample_size must be positive"):
            decode_tuning_report(raw)

    def test_decode_negative_parallel_speedup(self) -> None:
        """Negative parallel_speedup should raise ValueError."""
        config_raw: JSONDict = {
            "n_estimators": 5,
            "max_depth": 4,
            "learning_rate": 0.1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 42,
            "track_contributions": False,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
        }
        raw: JSONDict = {
            "best_config": config_raw,
            "timing_results": [],
            "sample_size": 100,
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": -0.5,  # invalid
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(ValueError, match="parallel_speedup must be non-negative"):
            decode_tuning_report(raw)

    def test_decode_best_config_not_dict(self) -> None:
        """best_config not a dict should raise JSONTypeError."""
        raw: JSONDict = {
            "best_config": "not a dict",
            "timing_results": [],
            "sample_size": 100,
            "n_features": 5,
            "recommended_n_jobs": 1,
            "recommended_max_bins": 64,
            "parallel_speedup": 1.0,
            "total_tune_time_seconds": 10.0,
        }
        with pytest.raises(JSONTypeError, match="best_config must be dict"):
            decode_tuning_report(raw)


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


# =============================================================================
# Buffer Type Tests
# =============================================================================


class TestFloatBufferData:
    """Tests for FloatBufferData encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip."""
        data: FloatBufferData = FloatBufferData(
            values=(1.0, 2.0, 3.0),
            size=3,
        )
        encoded: JSONDict = encode_float_buffer_data(data["values"], data["size"])
        decoded: FloatBufferData = decode_float_buffer_data(encoded)
        assert decoded["values"] == (1.0, 2.0, 3.0)
        assert decoded["size"] == 3

    def test_decode_coerces_int_to_float(self) -> None:
        """Test decode coerces int values to float."""
        raw: JSONDict = {"values": [1, 2, 3], "size": 3}
        decoded: FloatBufferData = decode_float_buffer_data(raw)
        assert decoded["values"] == (1.0, 2.0, 3.0)

    def test_decode_raises_on_missing_size(self) -> None:
        """Test decode raises KeyError for missing size."""
        raw: JSONDict = {"values": [1.0]}
        with pytest.raises(KeyError):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_missing_values(self) -> None:
        """Test decode raises KeyError for missing values."""
        raw: JSONDict = {"size": 3}
        with pytest.raises(KeyError):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_non_list_values(self) -> None:
        """Test decode raises JSONTypeError for non-list values."""
        raw: JSONDict = {"values": "not a list", "size": 3}
        with pytest.raises(JSONTypeError, match="values must be list"):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_bool_value(self) -> None:
        """Test decode raises JSONTypeError for bool in values."""
        raw: JSONDict = {"values": [True, 2.0], "size": 2}
        with pytest.raises(JSONTypeError, match=r"values\[0\] must be float, got bool"):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_string_value(self) -> None:
        """Test decode raises JSONTypeError for string in values."""
        raw: JSONDict = {"values": [1.0, "not a float"], "size": 2}
        with pytest.raises(JSONTypeError, match=r"values\[1\] must be float"):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_size_mismatch(self) -> None:
        """Test decode raises ValueError for values/size mismatch."""
        raw: JSONDict = {"values": [1.0, 2.0], "size": 3}
        with pytest.raises(ValueError, match="values length 2 != size 3"):
            decode_float_buffer_data(raw)

    def test_decode_raises_on_non_positive_size(self) -> None:
        """Test decode raises ValueError for non-positive size."""
        raw: JSONDict = {"values": [], "size": 0}
        with pytest.raises(ValueError, match="size must be positive"):
            decode_float_buffer_data(raw)


class TestIntBufferData:
    """Tests for IntBufferData encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip."""
        data: IntBufferData = IntBufferData(
            values=(1, 2, 3),
            size=3,
        )
        encoded: JSONDict = encode_int_buffer_data(data["values"], data["size"])
        decoded: IntBufferData = decode_int_buffer_data(encoded)
        assert decoded["values"] == (1, 2, 3)
        assert decoded["size"] == 3

    def test_decode_raises_on_missing_size(self) -> None:
        """Test decode raises KeyError for missing size."""
        raw: JSONDict = {"values": [1]}
        with pytest.raises(KeyError):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_missing_values(self) -> None:
        """Test decode raises KeyError for missing values."""
        raw: JSONDict = {"size": 3}
        with pytest.raises(KeyError):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_non_list_values(self) -> None:
        """Test decode raises JSONTypeError for non-list values."""
        raw: JSONDict = {"values": "not a list", "size": 3}
        with pytest.raises(JSONTypeError, match="values must be list"):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_bool_value(self) -> None:
        """Test decode raises JSONTypeError for bool in values."""
        raw: JSONDict = {"values": [True, 2], "size": 2}
        with pytest.raises(JSONTypeError, match=r"values\[0\] must be int"):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_float_value(self) -> None:
        """Test decode raises JSONTypeError for float in values."""
        raw: JSONDict = {"values": [1, 2.5], "size": 2}
        with pytest.raises(JSONTypeError, match=r"values\[1\] must be int"):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_size_mismatch(self) -> None:
        """Test decode raises ValueError for values/size mismatch."""
        raw: JSONDict = {"values": [1, 2], "size": 3}
        with pytest.raises(ValueError, match="values length 2 != size 3"):
            decode_int_buffer_data(raw)

    def test_decode_raises_on_non_positive_size(self) -> None:
        """Test decode raises ValueError for non-positive size."""
        raw: JSONDict = {"values": [], "size": 0}
        with pytest.raises(ValueError, match="size must be positive"):
            decode_int_buffer_data(raw)


class TestHistogramBufferData:
    """Tests for HistogramBufferData encode/decode."""

    def test_encode_decode_roundtrip(self) -> None:
        """Test encode/decode roundtrip."""
        data: HistogramBufferData = HistogramBufferData(
            gradient_sums=(1.0, 2.0, 3.0),
            hessian_sums=(0.5, 1.0, 1.5),
            counts=(1, 2, 3),
            n_bins=3,
        )
        encoded: JSONDict = encode_histogram_buffer_data(
            data["gradient_sums"],
            data["hessian_sums"],
            data["counts"],
            data["n_bins"],
        )
        decoded: HistogramBufferData = decode_histogram_buffer_data(encoded)
        assert decoded["gradient_sums"] == (1.0, 2.0, 3.0)
        assert decoded["hessian_sums"] == (0.5, 1.0, 1.5)
        assert decoded["counts"] == (1, 2, 3)
        assert decoded["n_bins"] == 3

    def test_decode_coerces_int_to_float(self) -> None:
        """Test decode coerces int values to float for gradient/hessian sums."""
        raw: JSONDict = {
            "gradient_sums": [1, 2, 3],
            "hessian_sums": [1, 2, 3],
            "counts": [1, 2, 3],
            "n_bins": 3,
        }
        decoded: HistogramBufferData = decode_histogram_buffer_data(raw)
        assert decoded["gradient_sums"] == (1.0, 2.0, 3.0)
        assert decoded["hessian_sums"] == (1.0, 2.0, 3.0)

    def test_decode_raises_on_missing_gradient_sums(self) -> None:
        """Test decode raises KeyError for missing gradient_sums."""
        raw: JSONDict = {"hessian_sums": [1.0], "counts": [1], "n_bins": 1}
        with pytest.raises(KeyError):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_missing_hessian_sums(self) -> None:
        """Test decode raises KeyError for missing hessian_sums."""
        raw: JSONDict = {"gradient_sums": [1.0], "counts": [1], "n_bins": 1}
        with pytest.raises(KeyError):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_missing_counts(self) -> None:
        """Test decode raises KeyError for missing counts."""
        raw: JSONDict = {"gradient_sums": [1.0], "hessian_sums": [1.0], "n_bins": 1}
        with pytest.raises(KeyError):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_missing_n_bins(self) -> None:
        """Test decode raises KeyError for missing n_bins."""
        raw: JSONDict = {"gradient_sums": [1.0], "hessian_sums": [1.0], "counts": [1]}
        with pytest.raises(KeyError):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_non_list_gradient_sums(self) -> None:
        """Test decode raises JSONTypeError for non-list gradient_sums."""
        raw: JSONDict = {
            "gradient_sums": "not a list",
            "hessian_sums": [1.0],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match="gradient_sums must be list"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_non_list_hessian_sums(self) -> None:
        """Test decode raises JSONTypeError for non-list hessian_sums."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": "not a list",
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match="hessian_sums must be list"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_non_list_counts(self) -> None:
        """Test decode raises JSONTypeError for non-list counts."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": [1.0],
            "counts": "not a list",
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match="counts must be list"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_bool_in_gradient_sums(self) -> None:
        """Test decode raises JSONTypeError for bool in gradient_sums."""
        raw: JSONDict = {
            "gradient_sums": [True],
            "hessian_sums": [1.0],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"gradient_sums\[0\] must be float, got bool"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_string_in_gradient_sums(self) -> None:
        """Test decode raises JSONTypeError for string in gradient_sums."""
        raw: JSONDict = {
            "gradient_sums": ["not a float"],
            "hessian_sums": [1.0],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"gradient_sums\[0\] must be float"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_bool_in_hessian_sums(self) -> None:
        """Test decode raises JSONTypeError for bool in hessian_sums."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": [True],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"hessian_sums\[0\] must be float, got bool"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_string_in_hessian_sums(self) -> None:
        """Test decode raises JSONTypeError for string in hessian_sums."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": ["not a float"],
            "counts": [1],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"hessian_sums\[0\] must be float"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_bool_in_counts(self) -> None:
        """Test decode raises JSONTypeError for bool in counts."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": [1.0],
            "counts": [True],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"counts\[0\] must be int"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_float_in_counts(self) -> None:
        """Test decode raises JSONTypeError for float in counts."""
        raw: JSONDict = {
            "gradient_sums": [1.0],
            "hessian_sums": [1.0],
            "counts": [1.5],
            "n_bins": 1,
        }
        with pytest.raises(JSONTypeError, match=r"counts\[0\] must be int"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_gradient_sums_length_mismatch(self) -> None:
        """Test decode raises ValueError for gradient_sums/n_bins mismatch."""
        raw: JSONDict = {
            "gradient_sums": [1.0, 2.0],
            "hessian_sums": [1.0, 2.0, 3.0],
            "counts": [1, 2, 3],
            "n_bins": 3,
        }
        with pytest.raises(ValueError, match="gradient_sums length 2 != n_bins 3"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_hessian_sums_length_mismatch(self) -> None:
        """Test decode raises ValueError for hessian_sums/n_bins mismatch."""
        raw: JSONDict = {
            "gradient_sums": [1.0, 2.0, 3.0],
            "hessian_sums": [1.0, 2.0],
            "counts": [1, 2, 3],
            "n_bins": 3,
        }
        with pytest.raises(ValueError, match="hessian_sums length 2 != n_bins 3"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_counts_length_mismatch(self) -> None:
        """Test decode raises ValueError for counts/n_bins mismatch."""
        raw: JSONDict = {
            "gradient_sums": [1.0, 2.0, 3.0],
            "hessian_sums": [1.0, 2.0, 3.0],
            "counts": [1, 2],
            "n_bins": 3,
        }
        with pytest.raises(ValueError, match="counts length 2 != n_bins 3"):
            decode_histogram_buffer_data(raw)

    def test_decode_raises_on_non_positive_n_bins(self) -> None:
        """Test decode raises ValueError for non-positive n_bins."""
        raw: JSONDict = {
            "gradient_sums": [],
            "hessian_sums": [],
            "counts": [],
            "n_bins": 0,
        }
        with pytest.raises(ValueError, match="n_bins must be positive"):
            decode_histogram_buffer_data(raw)
