"""Explanation type definitions for ClearGBM.

Provides FeatureContribution, PredictionExplanation, and Rule TypedDicts
with their encode/decode functions.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

from typing import TypedDict

from cleargbm._types_json import (
    JSONDict,
    JSONTypeError,
    _as_json_dict,
    _require_float,
    _require_int,
    _require_str,
    require_non_negative_float,
    require_non_negative_int,
)
from cleargbm._types_tree import (
    TreePredictionExplanation,
    decode_tree_prediction_explanation,
    encode_tree_prediction_explanation,
)

# =============================================================================
# Explanation Types
# =============================================================================


class FeatureContribution(TypedDict):
    """Contribution of a single feature to the prediction."""

    feature_name: str
    feature_index: int
    total_contribution: float
    n_splits: int


def encode_feature_contribution(contrib: FeatureContribution) -> JSONDict:
    """Encode FeatureContribution to JSON-serializable dict.

    Args:
        contrib: Contribution to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "feature_name": contrib["feature_name"],
        "feature_index": contrib["feature_index"],
        "total_contribution": contrib["total_contribution"],
        "n_splits": contrib["n_splits"],
    }


def decode_feature_contribution(raw: JSONDict) -> FeatureContribution:
    """Decode raw dict to FeatureContribution.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated FeatureContribution.

    Raises:
        KeyError: If required key is missing.
        TypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    feature_name = _require_str(raw, "feature_name")
    feature_index = require_non_negative_int(_require_int(raw, "feature_index"), "feature_index")
    total_contribution = _require_float(raw, "total_contribution")
    n_splits = require_non_negative_int(_require_int(raw, "n_splits"), "n_splits")

    return FeatureContribution(
        feature_name=feature_name,
        feature_index=feature_index,
        total_contribution=total_contribution,
        n_splits=n_splits,
    )


class PredictionExplanation(TypedDict):
    """Full explanation for a gradient boosting prediction."""

    final_probability: float
    base_prediction: float
    tree_contributions: tuple[TreePredictionExplanation, ...]
    top_features: tuple[FeatureContribution, ...]


def encode_prediction_explanation(
    explanation: PredictionExplanation,
) -> JSONDict:
    """Encode PredictionExplanation to JSON-serializable dict.

    Args:
        explanation: Explanation to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "final_probability": explanation["final_probability"],
        "base_prediction": explanation["base_prediction"],
        "tree_contributions": [
            encode_tree_prediction_explanation(t) for t in explanation["tree_contributions"]
        ],
        "top_features": [encode_feature_contribution(f) for f in explanation["top_features"]],
    }


def decode_prediction_explanation(
    raw: JSONDict,
) -> PredictionExplanation:
    """Decode raw dict to PredictionExplanation.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated PredictionExplanation.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    final_probability = _require_float(raw, "final_probability")
    base_prediction = _require_float(raw, "base_prediction")

    tree_contributions_raw = raw["tree_contributions"]
    if not isinstance(tree_contributions_raw, list):
        raise JSONTypeError(
            f"tree_contributions must be list, got {type(tree_contributions_raw).__name__}"
        )
    tree_contributions: list[TreePredictionExplanation] = []
    for i, tc_raw in enumerate(tree_contributions_raw):
        tc_dict = _as_json_dict(tc_raw, f"tree_contributions[{i}]")
        tree_contributions.append(decode_tree_prediction_explanation(tc_dict))

    top_features_raw = raw["top_features"]
    if not isinstance(top_features_raw, list):
        raise JSONTypeError(f"top_features must be list, got {type(top_features_raw).__name__}")
    top_features: list[FeatureContribution] = []
    for i, fc_raw in enumerate(top_features_raw):
        fc_dict = _as_json_dict(fc_raw, f"top_features[{i}]")
        top_features.append(decode_feature_contribution(fc_dict))

    return PredictionExplanation(
        final_probability=final_probability,
        base_prediction=base_prediction,
        tree_contributions=tuple(tree_contributions),
        top_features=tuple(top_features),
    )


class Rule(TypedDict):
    """Human-readable decision rule."""

    conditions: tuple[str, ...]
    prediction_contribution: float
    n_samples: int
    importance: float


def encode_rule(rule: Rule) -> JSONDict:
    """Encode Rule to JSON-serializable dict.

    Args:
        rule: Rule to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "conditions": list(rule["conditions"]),
        "prediction_contribution": rule["prediction_contribution"],
        "n_samples": rule["n_samples"],
        "importance": rule["importance"],
    }


def decode_rule(raw: JSONDict) -> Rule:
    """Decode raw dict to Rule.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated Rule.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    conditions_raw = raw["conditions"]
    if not isinstance(conditions_raw, list):
        raise JSONTypeError(f"conditions must be list, got {type(conditions_raw).__name__}")
    conditions: list[str] = []
    for i, cond in enumerate(conditions_raw):
        if not isinstance(cond, str):
            raise JSONTypeError(f"conditions[{i}] must be str, got {type(cond).__name__}")
        conditions.append(cond)

    prediction_contribution = _require_float(raw, "prediction_contribution")
    n_samples = require_non_negative_int(_require_int(raw, "n_samples"), "n_samples")
    importance = require_non_negative_float(_require_float(raw, "importance"), "importance")

    return Rule(
        conditions=tuple(conditions),
        prediction_contribution=prediction_contribution,
        n_samples=n_samples,
        importance=importance,
    )


__all__ = [
    "FeatureContribution",
    "PredictionExplanation",
    "Rule",
    "decode_feature_contribution",
    "decode_prediction_explanation",
    "decode_rule",
    "encode_feature_contribution",
    "encode_prediction_explanation",
    "encode_rule",
]
