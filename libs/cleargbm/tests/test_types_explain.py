"""Tests for cleargbm.types: explanation payloads."""

from __future__ import annotations

import pytest

from cleargbm.types import (
    FeatureContribution,
    JSONDict,
    JSONTypeError,
    PredictionExplanation,
    Rule,
    SplitCondition,
    TreePredictionExplanation,
    decode_feature_contribution,
    decode_prediction_explanation,
    decode_rule,
    encode_feature_contribution,
    encode_prediction_explanation,
    encode_rule,
)

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
