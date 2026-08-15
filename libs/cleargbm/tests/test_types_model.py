"""Tests for cleargbm.types: model configuration and payloads."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    DecisionTree,
    GradientBoostingConfig,
    GradientBoostingModel,
    JSONDict,
    JSONTypeError,
    TreeNode,
    decode_gradient_boosting_config,
    decode_gradient_boosting_model,
    encode_gradient_boosting_config,
    encode_gradient_boosting_model,
)

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
