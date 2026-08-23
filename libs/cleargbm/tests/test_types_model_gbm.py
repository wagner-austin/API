"""Tests for GradientBoostingModel encode/decode."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    DecisionTree,
    GradientBoostingConfig,
    GradientBoostingModel,
    JSONDict,
    JSONTypeError,
    TreeNode,
    decode_gradient_boosting_model,
    encode_gradient_boosting_model,
)

# =============================================================================
# GradientBoostingConfig Tests
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
            "colsample_bytree": None,
            "categorical_features": None,
            "max_bins": 64,
            "subsample": 1.0,
            "random_state": 0,
            "monotonic_constraints": None,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "n_jobs": 1,
            "early_stopping_rounds": None,
            "growth_strategy": "depth_wise",
            "num_leaves": None,
            "objective": "binary_log_loss",
            "scale_pos_weight": 1.0,
        }
        original: GradientBoostingModel = {
            "trees": (tree,),
            "base_prediction": -0.5,
            "learning_rate": 0.1,
            "feature_names": ("x",),
            "config": config,
        }
        encoded = encode_gradient_boosting_model(original)
        decoded = decode_gradient_boosting_model(encoded)

        assert len(decoded["trees"]) == 1
        assert decoded["base_prediction"] == -0.5
        assert decoded["learning_rate"] == 0.1
        assert decoded["feature_names"] == ("x",)

    def test_decode_trees_not_list(self) -> None:
        """trees not a list should raise TypeError."""
        raw: JSONDict = {
            "trees": "not a list",
            "base_prediction": 0.0,
            "learning_rate": 0.1,
            "feature_names": ["x"],
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
            "config": "not a dict",
        }
        with pytest.raises(JSONTypeError, match="config must be dict"):
            decode_gradient_boosting_model(raw)
