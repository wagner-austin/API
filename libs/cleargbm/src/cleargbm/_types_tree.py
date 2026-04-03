"""Tree-related type definitions for ClearGBM.

Provides binning structures (BinEdges, FeatureBins), decision tree structures
(SplitCondition, TreeNode, DecisionTree, TreePredictionExplanation), and
the internal SplitCandidate used during tree building.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, TypedDict

import numpy as np
from numpy.typing import NDArray

from cleargbm._types_json import (
    JSONDict,
    JSONTypeError,
    _as_json_dict,
    _get_optional_float,
    _get_optional_int,
    _get_optional_str,
    _require_bool,
    _require_float,
    _require_int,
    _require_str,
    require_non_negative_int,
    require_positive_int,
)

# =============================================================================
# Histogram Binning Structures
# =============================================================================


class BinEdges(NamedTuple):
    """Bin edges for a single feature.

    Args:
        edges: Tuple of K-1 threshold values defining K bins.
               Values <= edges[0] go to bin 0, values > edges[-1] go to bin K-1.
    """

    edges: tuple[float, ...]


class FeatureBins(NamedTuple):
    """Precomputed bin assignments for all samples across all features.

    Args:
        bin_edges: Bin edges for each feature.
        sample_bins: Per-sample bin ID for each feature (2D array).
                     sample_bins[sample_idx, feature_idx] = bin_id
    """

    bin_edges: tuple[BinEdges, ...]
    sample_bins: NDArray[np.int64]  # Shape: (n_samples, n_features)


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


__all__ = [
    "BinEdges",
    "DecisionTree",
    "FeatureBins",
    "SplitCandidate",
    "SplitCondition",
    "TreeNode",
    "TreePredictionExplanation",
    "decode_decision_tree",
    "decode_split_condition",
    "decode_tree_node",
    "decode_tree_prediction_explanation",
    "encode_decision_tree",
    "encode_split_condition",
    "encode_tree_node",
    "encode_tree_prediction_explanation",
]
