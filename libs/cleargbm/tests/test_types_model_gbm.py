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
    decode_gradient_boosting_config,
    decode_gradient_boosting_model,
    encode_gradient_boosting_model,
)


def _make_raw_config(n_classes: int | None, objective: str) -> JSONDict:
    """Return a raw config dict with the given objective pairing."""
    return {
        "n_estimators": 1,
        "max_depth": 1,
        "learning_rate": 0.1,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "colsample_bytree": None,
        "categorical_features": None,
        "n_classes": n_classes,
        "lambdarank_truncation_level": None,
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
        "objective": objective,
        "scale_pos_weight": None,
    }


def _make_raw_multiclass_model() -> JSONDict:
    """Return a raw multiclass model dict with per-class base scores."""
    return {
        "trees": [],
        "base_prediction": None,
        "class_base_predictions": [0.25, -0.5, 0.75],
        "learning_rate": 0.1,
        "feature_names": ["x"],
        "config": _make_raw_config(3, "multiclass_softmax"),
    }


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
            "n_classes": None,
            "lambdarank_truncation_level": None,
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
            "class_base_predictions": None,
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


class TestConfigNClasses:
    """Decode of the ``n_classes`` config field."""

    def test_decode_carries_a_valid_class_count(self) -> None:
        """An int >= 2 passes through the decode unchanged."""
        decoded = decode_gradient_boosting_config(_make_raw_config(3, "multiclass_softmax"))
        assert decoded["n_classes"] == 3

    def test_decode_rejects_a_class_count_below_two(self) -> None:
        """One class cannot describe a classification problem."""
        with pytest.raises(ValueError, match="n_classes must be >= 2 when set, got 1"):
            decode_gradient_boosting_config(_make_raw_config(1, "multiclass_softmax"))

    def test_decode_carries_a_valid_truncation_level(self) -> None:
        """An int >= 1 passes through the decode unchanged."""
        raw = _make_raw_config(None, "lambdarank")
        raw["lambdarank_truncation_level"] = 10
        decoded = decode_gradient_boosting_config(raw)
        assert decoded["lambdarank_truncation_level"] == 10

    def test_decode_rejects_a_zero_truncation_level(self) -> None:
        """A truncation position must be positive when set."""
        raw = _make_raw_config(None, "lambdarank")
        raw["lambdarank_truncation_level"] = 0
        with pytest.raises(ValueError, match="lambdarank_truncation_level must be positive"):
            decode_gradient_boosting_config(raw)


class TestClassBasePredictions:
    """Decode of the mutually exclusive base-score spellings."""

    def test_decode_multiclass_model_roundtrip(self) -> None:
        """A per-class base vector decodes to a float tuple."""
        decoded = decode_gradient_boosting_model(_make_raw_multiclass_model())
        assert decoded["base_prediction"] is None
        assert decoded["class_base_predictions"] == (0.25, -0.5, 0.75)

    def test_encode_carries_the_class_base_list(self) -> None:
        """Encoding a multiclass model spells the base vector as a list."""
        decoded = decode_gradient_boosting_model(_make_raw_multiclass_model())
        encoded = encode_gradient_boosting_model(decoded)
        assert encoded["base_prediction"] is None
        assert encoded["class_base_predictions"] == [0.25, -0.5, 0.75]

    def test_decode_rejects_a_non_list_base_vector(self) -> None:
        """class_base_predictions must be a list or None."""
        raw = _make_raw_multiclass_model()
        raw["class_base_predictions"] = "not a list"
        with pytest.raises(JSONTypeError, match="class_base_predictions must be list or None"):
            decode_gradient_boosting_model(raw)

    def test_decode_rejects_a_non_float_member(self) -> None:
        """Each per-class base score must be numeric; bool is not a score."""
        raw = _make_raw_multiclass_model()
        raw["class_base_predictions"] = [0.25, True, 0.75]
        with pytest.raises(JSONTypeError, match=r"class_base_predictions\[1\] must be float"):
            decode_gradient_boosting_model(raw)

    def test_decode_rejects_both_base_spellings_set(self) -> None:
        """A model cannot carry a scalar base and a per-class base at once."""
        raw = _make_raw_multiclass_model()
        raw["base_prediction"] = 0.5
        with pytest.raises(ValueError, match="exactly one of base_prediction"):
            decode_gradient_boosting_model(raw)

    def test_decode_rejects_neither_base_spelling_set(self) -> None:
        """A model with no base score at all is malformed."""
        raw = _make_raw_multiclass_model()
        raw["class_base_predictions"] = None
        with pytest.raises(ValueError, match="exactly one of base_prediction"):
            decode_gradient_boosting_model(raw)
