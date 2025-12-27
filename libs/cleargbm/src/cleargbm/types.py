"""Type definitions for ClearGBM.

All data structures are immutable TypedDicts. Every TypedDict has encode/decode
functions with require_* validation.

Uses numpy arrays for efficient data representation.
"""

from __future__ import annotations

from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

# JSON type aliases (recursive type for strict typing)
JSONValue = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None
JSONDict = dict[str, JSONValue]


class JSONTypeError(TypeError):
    """Raised when JSON value has unexpected type during decoding."""


# =============================================================================
# Validation Helpers
# =============================================================================


def require_positive_int(value: int, name: str) -> int:
    """Validate that value is a positive integer.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_non_negative_int(value: int, name: str) -> int:
    """Validate that value is a non-negative integer.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def require_positive_float(value: float, name: str) -> float:
    """Validate that value is a positive float.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_unit_float(value: float, name: str) -> float:
    """Validate that value is in (0, 1].

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is not in (0, 1].
    """
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def require_non_negative_float(value: float, name: str) -> float:
    """Validate that value is a non-negative float.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is negative.
    """
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def require_n_jobs(value: int, name: str) -> int:
    """Validate n_jobs: must be -1 (all cores) or positive.

    Args:
        value: The value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated value.

    Raises:
        ValueError: If value is not -1 or positive.
    """
    if value != -1 and value <= 0:
        raise ValueError(f"{name} must be -1 or positive, got {value}")
    return value


# =============================================================================
# Raw Dict Extraction Helpers
# =============================================================================


def _require_str(raw: JSONDict, key: str) -> str:
    """Extract and validate string from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        String value.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value is not a string.
    """
    value = raw[key]
    if not isinstance(value, str):
        raise JSONTypeError(f"{key} must be str, got {type(value).__name__}")
    return value


def _require_int(raw: JSONDict, key: str) -> int:
    """Extract and validate int from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Integer value.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value is not an int.
    """
    value = raw[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise JSONTypeError(f"{key} must be int, got {type(value).__name__}")
    return value


def _require_float(raw: JSONDict, key: str) -> float:
    """Extract and validate float from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Float value.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value is not a float or int.
    """
    value = raw[key]
    if isinstance(value, bool):
        raise JSONTypeError(f"{key} must be float, got bool")
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise JSONTypeError(f"{key} must be float, got {type(value).__name__}")
    return value


def _require_bool(raw: JSONDict, key: str) -> bool:
    """Extract and validate bool from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Boolean value.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value is not a bool.
    """
    value = raw[key]
    if not isinstance(value, bool):
        raise JSONTypeError(f"{key} must be bool, got {type(value).__name__}")
    return value


def _get_optional_int(raw: JSONDict, key: str) -> int | None:
    """Extract optional int from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Integer value or None if key not present or value is None.

    Raises:
        JSONTypeError: If value is present but not int or None.
    """
    if key not in raw:
        return None
    value = raw[key]
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise JSONTypeError(f"{key} must be int or None, got {type(value).__name__}")
    return value


def _get_optional_float(raw: JSONDict, key: str) -> float | None:
    """Extract optional float from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        Float value or None if key not present or value is None.

    Raises:
        JSONTypeError: If value is present but not float/int or None.
    """
    if key not in raw:
        return None
    value = raw[key]
    if value is None:
        return None
    if isinstance(value, bool):
        raise JSONTypeError(f"{key} must be float or None, got bool")
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, float):
        raise JSONTypeError(f"{key} must be float or None, got {type(value).__name__}")
    return value


def _get_optional_str(raw: JSONDict, key: str) -> str | None:
    """Extract optional string from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        String value or None if key not present or value is None.

    Raises:
        JSONTypeError: If value is present but not str or None.
    """
    if key not in raw:
        return None
    value = raw[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise JSONTypeError(f"{key} must be str or None, got {type(value).__name__}")
    return value


# =============================================================================
# Tree Structures
# =============================================================================


class SplitCondition(TypedDict):
    """A single split condition in a decision tree path."""

    feature_index: int
    feature_name: str
    threshold: float
    direction: Literal["left", "right"]


def encode_split_condition(split: SplitCondition) -> JSONDict:
    """Encode SplitCondition to JSON-serializable dict.

    Args:
        split: Split condition to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "feature_index": split["feature_index"],
        "feature_name": split["feature_name"],
        "threshold": split["threshold"],
        "direction": split["direction"],
    }


def decode_split_condition(raw: JSONDict) -> SplitCondition:
    """Decode raw dict to SplitCondition.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated SplitCondition.

    Raises:
        KeyError: If required key is missing.
        TypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    feature_index = require_non_negative_int(_require_int(raw, "feature_index"), "feature_index")
    feature_name = _require_str(raw, "feature_name")
    threshold = _require_float(raw, "threshold")
    direction_raw = _require_str(raw, "direction")
    if direction_raw not in ("left", "right"):
        raise ValueError(f"direction must be 'left' or 'right', got {direction_raw!r}")
    direction: Literal["left", "right"] = "left" if direction_raw == "left" else "right"

    return SplitCondition(
        feature_index=feature_index,
        feature_name=feature_name,
        threshold=threshold,
        direction=direction,
    )


class TreeNode(TypedDict):
    """A node in the decision tree."""

    node_id: int
    is_leaf: bool
    # Split info (None for leaf nodes)
    feature_index: int | None
    feature_name: str | None
    threshold: float | None
    nan_direction: Literal["left", "right"] | None  # Direction for NaN values
    # Leaf info
    value: float  # prediction value (always present, 0.0 for non-leaf)
    n_samples: int
    # Tree structure (None for leaf nodes)
    left_child: int | None
    right_child: int | None


def encode_tree_node(node: TreeNode) -> JSONDict:
    """Encode TreeNode to JSON-serializable dict.

    Args:
        node: Tree node to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "node_id": node["node_id"],
        "is_leaf": node["is_leaf"],
        "feature_index": node["feature_index"],
        "feature_name": node["feature_name"],
        "threshold": node["threshold"],
        "nan_direction": node["nan_direction"],
        "value": node["value"],
        "n_samples": node["n_samples"],
        "left_child": node["left_child"],
        "right_child": node["right_child"],
    }


def decode_tree_node(raw: JSONDict) -> TreeNode:
    """Decode raw dict to TreeNode.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated TreeNode.

    Raises:
        KeyError: If required key is missing.
        TypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    node_id = require_non_negative_int(_require_int(raw, "node_id"), "node_id")
    is_leaf = _require_bool(raw, "is_leaf")
    feature_index = _get_optional_int(raw, "feature_index")
    feature_name = _get_optional_str(raw, "feature_name")
    threshold = _get_optional_float(raw, "threshold")
    value = _require_float(raw, "value")
    n_samples = require_non_negative_int(_require_int(raw, "n_samples"), "n_samples")
    left_child = _get_optional_int(raw, "left_child")
    right_child = _get_optional_int(raw, "right_child")

    # Parse nan_direction
    nan_direction: Literal["left", "right"] | None = None
    if "nan_direction" in raw and raw["nan_direction"] is not None:
        nan_dir_raw = raw["nan_direction"]
        if not isinstance(nan_dir_raw, str):
            type_name = type(nan_dir_raw).__name__
            raise JSONTypeError(f"nan_direction must be str or None, got {type_name}")
        if nan_dir_raw not in ("left", "right"):
            raise ValueError(f"nan_direction must be 'left' or 'right', got {nan_dir_raw!r}")
        nan_direction = "left" if nan_dir_raw == "left" else "right"

    return TreeNode(
        node_id=node_id,
        is_leaf=is_leaf,
        feature_index=feature_index,
        feature_name=feature_name,
        threshold=threshold,
        nan_direction=nan_direction,
        value=value,
        n_samples=n_samples,
        left_child=left_child,
        right_child=right_child,
    )


class DecisionTree(TypedDict):
    """Complete decision tree structure."""

    nodes: tuple[TreeNode, ...]
    max_depth: int
    n_leaves: int
    feature_names: tuple[str, ...]


def encode_decision_tree(tree: DecisionTree) -> JSONDict:
    """Encode DecisionTree to JSON-serializable dict.

    Args:
        tree: Decision tree to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "nodes": [encode_tree_node(n) for n in tree["nodes"]],
        "max_depth": tree["max_depth"],
        "n_leaves": tree["n_leaves"],
        "feature_names": list(tree["feature_names"]),
    }


def _as_json_dict(value: JSONValue, context: str) -> JSONDict:
    """Convert JSONValue to JSONDict with validation.

    Args:
        value: Value to convert.
        context: Context for error messages.

    Returns:
        The value as a JSONDict.

    Raises:
        JSONTypeError: If value is not a dict.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{context} must be dict, got {type(value).__name__}")
    return value


def decode_decision_tree(raw: JSONDict) -> DecisionTree:
    """Decode raw dict to DecisionTree.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated DecisionTree.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    nodes_raw = raw["nodes"]
    if not isinstance(nodes_raw, list):
        raise JSONTypeError(f"nodes must be list, got {type(nodes_raw).__name__}")
    nodes: list[TreeNode] = []
    for i, node_raw in enumerate(nodes_raw):
        node_dict = _as_json_dict(node_raw, f"nodes[{i}]")
        nodes.append(decode_tree_node(node_dict))

    max_depth = require_non_negative_int(_require_int(raw, "max_depth"), "max_depth")
    n_leaves = require_positive_int(_require_int(raw, "n_leaves"), "n_leaves")

    feature_names_raw = raw["feature_names"]
    if not isinstance(feature_names_raw, list):
        raise JSONTypeError(f"feature_names must be list, got {type(feature_names_raw).__name__}")
    feature_names: list[str] = []
    for i, name in enumerate(feature_names_raw):
        if not isinstance(name, str):
            raise JSONTypeError(f"feature_names[{i}] must be str, got {type(name).__name__}")
        feature_names.append(name)

    return DecisionTree(
        nodes=tuple(nodes),
        max_depth=max_depth,
        n_leaves=n_leaves,
        feature_names=tuple(feature_names),
    )


class TreePredictionExplanation(TypedDict):
    """Explanation for a single tree's prediction."""

    tree_index: int
    prediction: float
    path: tuple[SplitCondition, ...]
    leaf_node_id: int
    n_samples_in_leaf: int


def encode_tree_prediction_explanation(
    explanation: TreePredictionExplanation,
) -> JSONDict:
    """Encode TreePredictionExplanation to JSON-serializable dict.

    Args:
        explanation: Explanation to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "tree_index": explanation["tree_index"],
        "prediction": explanation["prediction"],
        "path": [encode_split_condition(s) for s in explanation["path"]],
        "leaf_node_id": explanation["leaf_node_id"],
        "n_samples_in_leaf": explanation["n_samples_in_leaf"],
    }


def decode_tree_prediction_explanation(
    raw: JSONDict,
) -> TreePredictionExplanation:
    """Decode raw dict to TreePredictionExplanation.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated TreePredictionExplanation.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    tree_index = require_non_negative_int(_require_int(raw, "tree_index"), "tree_index")
    prediction = _require_float(raw, "prediction")
    leaf_node_id = require_non_negative_int(_require_int(raw, "leaf_node_id"), "leaf_node_id")
    n_samples_in_leaf = require_non_negative_int(
        _require_int(raw, "n_samples_in_leaf"), "n_samples_in_leaf"
    )

    path_raw = raw["path"]
    if not isinstance(path_raw, list):
        raise JSONTypeError(f"path must be list, got {type(path_raw).__name__}")
    path: list[SplitCondition] = []
    for i, split_raw in enumerate(path_raw):
        split_dict = _as_json_dict(split_raw, f"path[{i}]")
        path.append(decode_split_condition(split_dict))

    return TreePredictionExplanation(
        tree_index=tree_index,
        prediction=prediction,
        path=tuple(path),
        leaf_node_id=leaf_node_id,
        n_samples_in_leaf=n_samples_in_leaf,
    )


# =============================================================================
# Configuration
# =============================================================================


class GradientBoostingConfig(TypedDict):
    """Configuration for gradient boosting training.

    Args:
        n_estimators: Number of boosting rounds (maximum if early stopping enabled).
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage factor for updates.
        min_samples_split: Minimum samples to split a node.
        min_samples_leaf: Minimum samples in a leaf.
        max_features: Max features per split (None = all).
        max_bins: Number of histogram bins for split finding (64-256 typical).
        subsample: Row subsampling ratio.
        random_state: Random seed.
        track_contributions: Track feature contributions.
        monotonic_constraints: Per-feature constraints (-1, 0, +1).
        reg_alpha: L1 regularization term on leaf weights (default: 0.0).
        reg_lambda: L2 regularization term on leaf weights (default: 1.0).
        n_jobs: Number of parallel workers (1 = sequential, -1 = all cores).
        early_stopping_rounds: Stop training after this many rounds without
            improvement on validation loss. None disables early stopping.
            Requires validation data to be provided during training.
    """

    n_estimators: int
    max_depth: int
    learning_rate: float
    min_samples_split: int
    min_samples_leaf: int
    max_features: int | None
    max_bins: int
    subsample: float
    random_state: int
    track_contributions: bool
    monotonic_constraints: tuple[int, ...] | None
    reg_alpha: float
    reg_lambda: float
    n_jobs: int
    early_stopping_rounds: int | None


def encode_gradient_boosting_config(config: GradientBoostingConfig) -> JSONDict:
    """Encode GradientBoostingConfig to JSON-serializable dict.

    Args:
        config: Config to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "learning_rate": config["learning_rate"],
        "min_samples_split": config["min_samples_split"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_features": config["max_features"],
        "max_bins": config["max_bins"],
        "subsample": config["subsample"],
        "random_state": config["random_state"],
        "track_contributions": config["track_contributions"],
        "monotonic_constraints": (
            list(config["monotonic_constraints"])
            if config["monotonic_constraints"] is not None
            else None
        ),
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "n_jobs": config["n_jobs"],
        "early_stopping_rounds": config["early_stopping_rounds"],
    }


def decode_gradient_boosting_config(raw: JSONDict) -> GradientBoostingConfig:
    """Decode raw dict to GradientBoostingConfig.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated GradientBoostingConfig.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    n_estimators = require_positive_int(_require_int(raw, "n_estimators"), "n_estimators")
    max_depth = require_positive_int(_require_int(raw, "max_depth"), "max_depth")
    learning_rate = require_positive_float(_require_float(raw, "learning_rate"), "learning_rate")
    min_samples_split = require_positive_int(
        _require_int(raw, "min_samples_split"), "min_samples_split"
    )
    min_samples_leaf = require_positive_int(
        _require_int(raw, "min_samples_leaf"), "min_samples_leaf"
    )
    max_features = _get_optional_int(raw, "max_features")
    if max_features is not None:
        max_features = require_positive_int(max_features, "max_features")
    max_bins = require_positive_int(_require_int(raw, "max_bins"), "max_bins")
    subsample = require_unit_float(_require_float(raw, "subsample"), "subsample")
    random_state = _require_int(raw, "random_state")
    track_contributions = _require_bool(raw, "track_contributions")

    monotonic_constraints: tuple[int, ...] | None = None
    if "monotonic_constraints" in raw and raw["monotonic_constraints"] is not None:
        mc_raw = raw["monotonic_constraints"]
        if not isinstance(mc_raw, list):
            raise JSONTypeError(
                f"monotonic_constraints must be list or None, got {type(mc_raw).__name__}"
            )
        mc_list: list[int] = []
        for i, val in enumerate(mc_raw):
            if not isinstance(val, int) or isinstance(val, bool):
                raise JSONTypeError(
                    f"monotonic_constraints[{i}] must be int, got {type(val).__name__}"
                )
            if val not in (-1, 0, 1):
                raise ValueError(f"monotonic_constraints[{i}] must be -1, 0, or 1, got {val}")
            mc_list.append(val)
        monotonic_constraints = tuple(mc_list)

    reg_alpha = require_non_negative_float(_require_float(raw, "reg_alpha"), "reg_alpha")
    reg_lambda = require_non_negative_float(_require_float(raw, "reg_lambda"), "reg_lambda")
    n_jobs = require_n_jobs(_require_int(raw, "n_jobs"), "n_jobs")

    # early_stopping_rounds: None or positive int
    early_stopping_rounds = _get_optional_int(raw, "early_stopping_rounds")
    if early_stopping_rounds is not None:
        early_stopping_rounds = require_positive_int(early_stopping_rounds, "early_stopping_rounds")

    return GradientBoostingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_bins=max_bins,
        subsample=subsample,
        random_state=random_state,
        track_contributions=track_contributions,
        monotonic_constraints=monotonic_constraints,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        early_stopping_rounds=early_stopping_rounds,
    )


# =============================================================================
# Model
# =============================================================================


class GradientBoostingModel(TypedDict):
    """Trained gradient boosting model."""

    trees: tuple[DecisionTree, ...]
    base_prediction: float
    learning_rate: float
    feature_names: tuple[str, ...]
    n_classes: int
    config: GradientBoostingConfig


def encode_gradient_boosting_model(model: GradientBoostingModel) -> JSONDict:
    """Encode GradientBoostingModel to JSON-serializable dict.

    Args:
        model: Model to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "trees": [encode_decision_tree(t) for t in model["trees"]],
        "base_prediction": model["base_prediction"],
        "learning_rate": model["learning_rate"],
        "feature_names": list(model["feature_names"]),
        "n_classes": model["n_classes"],
        "config": encode_gradient_boosting_config(model["config"]),
    }


def decode_gradient_boosting_model(raw: JSONDict) -> GradientBoostingModel:
    """Decode raw dict to GradientBoostingModel.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated GradientBoostingModel.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    trees_raw = raw["trees"]
    if not isinstance(trees_raw, list):
        raise JSONTypeError(f"trees must be list, got {type(trees_raw).__name__}")
    trees: list[DecisionTree] = []
    for i, tree_raw in enumerate(trees_raw):
        tree_dict = _as_json_dict(tree_raw, f"trees[{i}]")
        trees.append(decode_decision_tree(tree_dict))

    base_prediction = _require_float(raw, "base_prediction")
    learning_rate = require_positive_float(_require_float(raw, "learning_rate"), "learning_rate")
    n_classes = require_positive_int(_require_int(raw, "n_classes"), "n_classes")

    feature_names_raw = raw["feature_names"]
    if not isinstance(feature_names_raw, list):
        raise JSONTypeError(f"feature_names must be list, got {type(feature_names_raw).__name__}")
    feature_names: list[str] = []
    for i, name in enumerate(feature_names_raw):
        if not isinstance(name, str):
            raise JSONTypeError(f"feature_names[{i}] must be str, got {type(name).__name__}")
        feature_names.append(name)

    config_dict = _as_json_dict(raw["config"], "config")
    config = decode_gradient_boosting_config(config_dict)

    return GradientBoostingModel(
        trees=tuple(trees),
        base_prediction=base_prediction,
        learning_rate=learning_rate,
        feature_names=tuple(feature_names),
        n_classes=n_classes,
        config=config,
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


def decode_prediction_explanation(raw: JSONDict) -> PredictionExplanation:
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


# =============================================================================
# Training Progress
# =============================================================================


class TrainingProgress(TypedDict):
    """Progress update during training."""

    tree_index: int
    total_trees: int
    train_loss: float
    val_loss: float | None


def encode_training_progress(progress: TrainingProgress) -> JSONDict:
    """Encode TrainingProgress to JSON-serializable dict.

    Args:
        progress: Progress to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "tree_index": progress["tree_index"],
        "total_trees": progress["total_trees"],
        "train_loss": progress["train_loss"],
        "val_loss": progress["val_loss"],
    }


def decode_training_progress(raw: JSONDict) -> TrainingProgress:
    """Decode raw dict to TrainingProgress.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated TrainingProgress.

    Raises:
        KeyError: If required key is missing.
        TypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    tree_index = require_non_negative_int(_require_int(raw, "tree_index"), "tree_index")
    total_trees = require_positive_int(_require_int(raw, "total_trees"), "total_trees")
    train_loss = require_non_negative_float(_require_float(raw, "train_loss"), "train_loss")
    val_loss = _get_optional_float(raw, "val_loss")
    if val_loss is not None:
        val_loss = require_non_negative_float(val_loss, "val_loss")

    return TrainingProgress(
        tree_index=tree_index,
        total_trees=total_trees,
        train_loss=train_loss,
        val_loss=val_loss,
    )


# =============================================================================
# Internal Types (for tree building)
# =============================================================================


class SplitCandidate(TypedDict):
    """A potential split to evaluate during tree building."""

    feature_index: int
    threshold: float
    gain: float
    left_indices: NDArray[np.int64]
    right_indices: NDArray[np.int64]
    nan_direction: Literal["left", "right"]


# =============================================================================
# Tuning Types
# =============================================================================


class TimingResult(TypedDict):
    """Timing result for a single configuration.

    Args:
        n_jobs: Number of parallel workers used.
        max_bins: Number of histogram bins used.
        max_depth: Maximum tree depth used.
        learning_rate: Learning rate used.
        elapsed_seconds: Time taken in seconds.
        trees_per_second: Training throughput.
    """

    n_jobs: int
    max_bins: int
    max_depth: int
    learning_rate: float
    elapsed_seconds: float
    trees_per_second: float


def encode_timing_result(result: TimingResult) -> JSONDict:
    """Encode TimingResult to JSON-serializable dict.

    Args:
        result: Result to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "n_jobs": result["n_jobs"],
        "max_bins": result["max_bins"],
        "max_depth": result["max_depth"],
        "learning_rate": result["learning_rate"],
        "elapsed_seconds": result["elapsed_seconds"],
        "trees_per_second": result["trees_per_second"],
    }


def decode_timing_result(raw: JSONDict) -> TimingResult:
    """Decode raw dict to TimingResult.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated TimingResult.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    n_jobs = require_n_jobs(_require_int(raw, "n_jobs"), "n_jobs")
    max_bins = require_positive_int(_require_int(raw, "max_bins"), "max_bins")
    max_depth = require_positive_int(_require_int(raw, "max_depth"), "max_depth")
    learning_rate = require_positive_float(_require_float(raw, "learning_rate"), "learning_rate")
    elapsed_seconds = require_non_negative_float(
        _require_float(raw, "elapsed_seconds"), "elapsed_seconds"
    )
    trees_per_second = require_non_negative_float(
        _require_float(raw, "trees_per_second"), "trees_per_second"
    )

    return TimingResult(
        n_jobs=n_jobs,
        max_bins=max_bins,
        max_depth=max_depth,
        learning_rate=learning_rate,
        elapsed_seconds=elapsed_seconds,
        trees_per_second=trees_per_second,
    )


class TuningReport(TypedDict):
    """Complete autotuning report with recommendations.

    Args:
        best_config: Recommended configuration based on tuning.
        timing_results: All timing results from the grid search.
        sample_size: Number of samples used for tuning.
        n_features: Number of features in the dataset.
        recommended_n_jobs: Best n_jobs value found.
        recommended_max_bins: Best max_bins value found.
        parallel_speedup: Speedup ratio vs sequential (1.0 = no speedup).
        total_tune_time_seconds: Total time spent tuning.
    """

    best_config: GradientBoostingConfig
    timing_results: tuple[TimingResult, ...]
    sample_size: int
    n_features: int
    recommended_n_jobs: int
    recommended_max_bins: int
    parallel_speedup: float
    total_tune_time_seconds: float


def encode_tuning_report(report: TuningReport) -> JSONDict:
    """Encode TuningReport to JSON-serializable dict.

    Args:
        report: Report to encode.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "best_config": encode_gradient_boosting_config(report["best_config"]),
        "timing_results": [encode_timing_result(r) for r in report["timing_results"]],
        "sample_size": report["sample_size"],
        "n_features": report["n_features"],
        "recommended_n_jobs": report["recommended_n_jobs"],
        "recommended_max_bins": report["recommended_max_bins"],
        "parallel_speedup": report["parallel_speedup"],
        "total_tune_time_seconds": report["total_tune_time_seconds"],
    }


def decode_tuning_report(raw: JSONDict) -> TuningReport:
    """Decode raw dict to TuningReport.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated TuningReport.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If value fails validation.
    """
    config_dict = _as_json_dict(raw["best_config"], "best_config")
    best_config = decode_gradient_boosting_config(config_dict)

    timing_results_raw = raw["timing_results"]
    if not isinstance(timing_results_raw, list):
        raise JSONTypeError(f"timing_results must be list, got {type(timing_results_raw).__name__}")
    timing_results: list[TimingResult] = []
    for i, tr_raw in enumerate(timing_results_raw):
        tr_dict = _as_json_dict(tr_raw, f"timing_results[{i}]")
        timing_results.append(decode_timing_result(tr_dict))

    sample_size = require_positive_int(_require_int(raw, "sample_size"), "sample_size")
    n_features = require_positive_int(_require_int(raw, "n_features"), "n_features")
    recommended_n_jobs = require_n_jobs(
        _require_int(raw, "recommended_n_jobs"), "recommended_n_jobs"
    )
    recommended_max_bins = require_positive_int(
        _require_int(raw, "recommended_max_bins"), "recommended_max_bins"
    )
    parallel_speedup = require_non_negative_float(
        _require_float(raw, "parallel_speedup"), "parallel_speedup"
    )
    total_tune_time_seconds = require_non_negative_float(
        _require_float(raw, "total_tune_time_seconds"), "total_tune_time_seconds"
    )

    return TuningReport(
        best_config=best_config,
        timing_results=tuple(timing_results),
        sample_size=sample_size,
        n_features=n_features,
        recommended_n_jobs=recommended_n_jobs,
        recommended_max_bins=recommended_max_bins,
        parallel_speedup=parallel_speedup,
        total_tune_time_seconds=total_tune_time_seconds,
    )


# =============================================================================
# Buffer Serialization Types
# =============================================================================


class FloatBufferData(TypedDict):
    """Serialized FloatBuffer for JSON persistence.

    Args:
        values: Tuple of float values.
        size: Number of elements.
    """

    values: tuple[float, ...]
    size: int


def encode_float_buffer_data(values: tuple[float, ...], size: int) -> JSONDict:
    """Encode FloatBuffer data to JSON-serializable dict.

    Args:
        values: Buffer values as tuple.
        size: Buffer size.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "values": list(values),
        "size": size,
    }


def decode_float_buffer_data(raw: JSONDict) -> FloatBufferData:
    """Decode raw dict to FloatBufferData.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated FloatBufferData.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If validation fails.
    """
    size = require_positive_int(_require_int(raw, "size"), "size")

    values_raw = raw["values"]
    if not isinstance(values_raw, list):
        raise JSONTypeError(f"values must be list, got {type(values_raw).__name__}")

    values: list[float] = []
    for i, val in enumerate(values_raw):
        if isinstance(val, bool):
            raise JSONTypeError(f"values[{i}] must be float, got bool")
        if isinstance(val, int):
            values.append(float(val))
        elif isinstance(val, float):
            values.append(val)
        else:
            raise JSONTypeError(f"values[{i}] must be float, got {type(val).__name__}")

    if len(values) != size:
        raise ValueError(f"values length {len(values)} != size {size}")

    return FloatBufferData(values=tuple(values), size=size)


class IntBufferData(TypedDict):
    """Serialized IntBuffer for JSON persistence.

    Args:
        values: Tuple of int values.
        size: Number of elements.
    """

    values: tuple[int, ...]
    size: int


def encode_int_buffer_data(values: tuple[int, ...], size: int) -> JSONDict:
    """Encode IntBuffer data to JSON-serializable dict.

    Args:
        values: Buffer values as tuple.
        size: Buffer size.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "values": list(values),
        "size": size,
    }


def decode_int_buffer_data(raw: JSONDict) -> IntBufferData:
    """Decode raw dict to IntBufferData.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated IntBufferData.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If validation fails.
    """
    size = require_positive_int(_require_int(raw, "size"), "size")

    values_raw = raw["values"]
    if not isinstance(values_raw, list):
        raise JSONTypeError(f"values must be list, got {type(values_raw).__name__}")

    values: list[int] = []
    for i, val in enumerate(values_raw):
        if not isinstance(val, int) or isinstance(val, bool):
            raise JSONTypeError(f"values[{i}] must be int, got {type(val).__name__}")
        values.append(val)

    if len(values) != size:
        raise ValueError(f"values length {len(values)} != size {size}")

    return IntBufferData(values=tuple(values), size=size)


class HistogramBufferData(TypedDict):
    """Serialized HistogramBuffer for JSON persistence.

    Args:
        gradient_sums: Gradient sum per bin.
        hessian_sums: Hessian sum per bin.
        counts: Sample count per bin.
        n_bins: Number of bins.
    """

    gradient_sums: tuple[float, ...]
    hessian_sums: tuple[float, ...]
    counts: tuple[int, ...]
    n_bins: int


def encode_histogram_buffer_data(
    gradient_sums: tuple[float, ...],
    hessian_sums: tuple[float, ...],
    counts: tuple[int, ...],
    n_bins: int,
) -> JSONDict:
    """Encode HistogramBuffer data to JSON-serializable dict.

    Args:
        gradient_sums: Gradient sums per bin.
        hessian_sums: Hessian sums per bin.
        counts: Sample counts per bin.
        n_bins: Number of bins.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "gradient_sums": list(gradient_sums),
        "hessian_sums": list(hessian_sums),
        "counts": list(counts),
        "n_bins": n_bins,
    }


def _require_float_list(raw: JSONDict, key: str) -> list[float]:
    """Extract and validate a list of floats from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        List of float values.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value has wrong type.
    """
    values_raw = raw[key]
    if not isinstance(values_raw, list):
        raise JSONTypeError(f"{key} must be list, got {type(values_raw).__name__}")

    result: list[float] = []
    for i, val in enumerate(values_raw):
        if isinstance(val, bool):
            raise JSONTypeError(f"{key}[{i}] must be float, got bool")
        if isinstance(val, int):
            result.append(float(val))
        elif isinstance(val, float):
            result.append(val)
        else:
            raise JSONTypeError(f"{key}[{i}] must be float, got {type(val).__name__}")
    return result


def _require_int_list(raw: JSONDict, key: str) -> list[int]:
    """Extract and validate a list of ints from raw dict.

    Args:
        raw: Raw dictionary.
        key: Key to extract.

    Returns:
        List of int values.

    Raises:
        KeyError: If key not present.
        JSONTypeError: If value has wrong type.
    """
    values_raw = raw[key]
    if not isinstance(values_raw, list):
        raise JSONTypeError(f"{key} must be list, got {type(values_raw).__name__}")

    result: list[int] = []
    for i, val in enumerate(values_raw):
        if not isinstance(val, int) or isinstance(val, bool):
            raise JSONTypeError(f"{key}[{i}] must be int, got {type(val).__name__}")
        result.append(val)
    return result


def decode_histogram_buffer_data(raw: JSONDict) -> HistogramBufferData:
    """Decode raw dict to HistogramBufferData.

    Args:
        raw: Raw dictionary from JSON.

    Returns:
        Validated HistogramBufferData.

    Raises:
        KeyError: If required key is missing.
        JSONTypeError: If value has wrong type.
        ValueError: If validation fails.
    """
    n_bins = require_positive_int(_require_int(raw, "n_bins"), "n_bins")
    gradient_sums = _require_float_list(raw, "gradient_sums")
    hessian_sums = _require_float_list(raw, "hessian_sums")
    counts = _require_int_list(raw, "counts")

    # Validate lengths
    if len(gradient_sums) != n_bins:
        raise ValueError(f"gradient_sums length {len(gradient_sums)} != n_bins {n_bins}")
    if len(hessian_sums) != n_bins:
        raise ValueError(f"hessian_sums length {len(hessian_sums)} != n_bins {n_bins}")
    if len(counts) != n_bins:
        raise ValueError(f"counts length {len(counts)} != n_bins {n_bins}")

    return HistogramBufferData(
        gradient_sums=tuple(gradient_sums),
        hessian_sums=tuple(hessian_sums),
        counts=tuple(counts),
        n_bins=n_bins,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "DecisionTree",
    "FeatureContribution",
    "FloatBufferData",
    "GradientBoostingConfig",
    "GradientBoostingModel",
    "HistogramBufferData",
    "IntBufferData",
    "JSONDict",
    "JSONTypeError",
    "JSONValue",
    "PredictionExplanation",
    "Rule",
    "SplitCandidate",
    "SplitCondition",
    "TimingResult",
    "TrainingProgress",
    "TreeNode",
    "TreePredictionExplanation",
    "TuningReport",
    "decode_decision_tree",
    "decode_feature_contribution",
    "decode_float_buffer_data",
    "decode_gradient_boosting_config",
    "decode_gradient_boosting_model",
    "decode_histogram_buffer_data",
    "decode_int_buffer_data",
    "decode_prediction_explanation",
    "decode_rule",
    "decode_split_condition",
    "decode_timing_result",
    "decode_training_progress",
    "decode_tree_node",
    "decode_tree_prediction_explanation",
    "decode_tuning_report",
    "encode_decision_tree",
    "encode_feature_contribution",
    "encode_float_buffer_data",
    "encode_gradient_boosting_config",
    "encode_gradient_boosting_model",
    "encode_histogram_buffer_data",
    "encode_int_buffer_data",
    "encode_prediction_explanation",
    "encode_rule",
    "encode_split_condition",
    "encode_timing_result",
    "encode_training_progress",
    "encode_tree_node",
    "encode_tree_prediction_explanation",
    "encode_tuning_report",
    "require_n_jobs",
    "require_non_negative_float",
    "require_non_negative_int",
    "require_positive_float",
    "require_positive_int",
    "require_unit_float",
]
