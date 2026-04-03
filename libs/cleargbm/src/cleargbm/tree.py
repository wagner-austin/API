"""Decision tree implementation for gradient boosting.

Uses numpy arrays for efficient data representation.
Uses histogram binning for O(K) split finding instead of O(n log n).
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from cleargbm._hooks_infra import RandomStateProtocol, WorkerPoolProtocol, get_random_state
from cleargbm._hooks_prediction import predict_tree as _predict_tree_hook
from cleargbm.buffers import HistogramBuffer
from cleargbm.histogram import (
    build_histogram,
    precompute_feature_bins,
    subtract_histogram,
)
from cleargbm.parallel import _find_best_histogram_split_with_cache
from cleargbm.split import _compute_leaf_value, _create_leaf_node
from cleargbm.types import (
    DecisionTree,
    FeatureBins,
    GradientBoostingConfig,
    SplitCondition,
    TreeNode,
    TreePredictionExplanation,
)


def _should_be_leaf(
    depth: int,
    n_samples: int,
    config: GradientBoostingConfig,
) -> bool:
    """Check if node should be a leaf.

    Args:
        depth: Current depth.
        n_samples: Number of samples in node.
        config: Training configuration.

    Returns:
        True if should be leaf, False otherwise.
    """
    return (
        depth >= config["max_depth"]
        or n_samples < config["min_samples_split"]
        or n_samples < 2 * config["min_samples_leaf"]
    )


def build_tree(
    x: NDArray[np.float64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    feature_bins: FeatureBins | None = None,
    pool: WorkerPoolProtocol | None = None,
) -> DecisionTree:
    """Build a single decision tree using histogram binning.

    Args:
        x: Feature matrix (n_samples, n_features).
        gradients: Gradient for each sample.
        hessians: Hessian for each sample.
        config: Training configuration.
        feature_names: Names for each feature.
        feature_bins: Precomputed feature bins (optional, computed if None).
        pool: Worker pool for parallel histogram building (optional).

    Returns:
        Trained decision tree.

    Raises:
        ValueError: If input shapes are inconsistent.
    """
    n_samples: int = int(x.shape[0])
    if n_samples == 0:
        raise ValueError("x must not be empty")

    n_features: int = int(x.shape[1])
    n_gradients: int = int(gradients.shape[0])
    n_hessians: int = int(hessians.shape[0])
    if n_gradients != n_samples:
        raise ValueError(f"gradients length {n_gradients} != x rows {n_samples}")
    if n_hessians != n_samples:
        raise ValueError(f"hessians length {n_hessians} != x rows {n_samples}")
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names length {len(feature_names)} != x cols {n_features}")

    rng = get_random_state(config["random_state"])
    max_features = config["max_features"] if config["max_features"] is not None else n_features

    # Precompute feature bins for histogram-based split finding (O(K) instead of O(n log n))
    if feature_bins is None:
        feature_bins = precompute_feature_bins(x, config["max_bins"])

    # Get subsampling indices
    sample_indices = _get_sample_indices(n_samples, config["subsample"], rng)

    # Build tree using histogram-based split finding
    nodes, n_leaves, node_children = _build_tree_with_histograms(
        x,
        gradients,
        hessians,
        sample_indices,
        config,
        feature_names,
        n_features,
        max_features,
        rng,
        feature_bins,
        pool,
    )

    # Finalize nodes with child pointers
    final_nodes = _finalize_nodes(nodes, node_children)

    max_depth_found = _compute_max_depth(final_nodes) if final_nodes else 0

    return DecisionTree(
        nodes=tuple(final_nodes),
        max_depth=max_depth_found,
        n_leaves=n_leaves,
        feature_names=feature_names,
    )


def _get_sample_indices(
    n_samples: int,
    subsample: float,
    rng: RandomStateProtocol,
) -> NDArray[np.int64]:
    """Get sample indices for tree building.

    Args:
        n_samples: Total number of samples.
        subsample: Subsampling ratio.
        rng: Random state.

    Returns:
        Numpy array of sample indices.
    """
    if subsample < 1.0:
        n_subsample = max(1, int(n_samples * subsample))
        choice_result = rng.choice(n_samples, size=n_subsample, replace=False)
        indices: NDArray[np.int64] = np.asarray(choice_result, dtype=np.int64)
        return indices
    return np.arange(n_samples, dtype=np.int64)


def _build_tree_with_histograms(
    x: NDArray[np.float64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    sample_indices: NDArray[np.int64],
    config: GradientBoostingConfig,
    feature_names: tuple[str, ...],
    n_features: int,
    max_features: int,
    rng: RandomStateProtocol,
    feature_bins: FeatureBins,
    pool: WorkerPoolProtocol | None,
) -> tuple[list[TreeNode], int, dict[int, tuple[int | None, int | None]]]:
    """Build tree nodes using histogram-based split finding.

    Uses O(K) histogram scans instead of O(n log n) sorting for each split.
    Applies sibling histogram subtraction for 2x histogram building speedup.

    Args:
        x: Feature matrix.
        gradients: Gradients.
        hessians: Hessians.
        sample_indices: Initial sample indices.
        config: Configuration.
        feature_names: Feature names.
        n_features: Number of features.
        max_features: Max features to consider.
        rng: Random state.
        feature_bins: Precomputed feature bins.
        pool: Worker pool for parallel histogram building (optional).

    Returns:
        Tuple of (nodes list, number of leaves, child pointers dict).
    """
    nodes: list[TreeNode] = []
    next_node_id = 0
    n_leaves = 0
    node_children: dict[int, tuple[int | None, int | None]] = {}

    # Stack entries: (sample_indices, depth, parent_id, is_left_child, precomputed_histograms)
    # precomputed_histograms is a dict mapping feature_idx -> HistogramBuffer, or None
    stack: list[
        tuple[NDArray[np.int64], int, int | None, bool | None, dict[int, HistogramBuffer] | None]
    ] = [(sample_indices, 0, None, None, None)]

    while stack:
        current_indices, depth, parent_id, is_left, cached_histograms = stack.pop()
        node_id = next_node_id
        next_node_id += 1

        _update_parent_child(node_children, parent_id, node_id, is_left)

        reg_alpha = config["reg_alpha"]
        reg_lambda = config["reg_lambda"]

        n_samples_current = current_indices.shape[0]
        if _should_be_leaf(depth, n_samples_current, config):
            nodes.append(
                _create_leaf_node(
                    node_id, current_indices, gradients, hessians, reg_alpha, reg_lambda
                )
            )
            n_leaves += 1
            continue

        # Find best split using histograms (with optional cached histograms)
        feature_indices = _select_features(n_features, max_features, rng)
        best_split, parent_histograms = _find_best_histogram_split_with_cache(
            current_indices,
            gradients,
            hessians,
            feature_indices,
            config,
            feature_bins,
            cached_histograms,
            pool,
        )

        if best_split is None:
            nodes.append(
                _create_leaf_node(
                    node_id, current_indices, gradients, hessians, reg_alpha, reg_lambda
                )
            )
            n_leaves += 1
            continue

        # Create internal node - compute leaf value from gradients/hessians at current indices
        grads_node: NDArray[np.float64] = gradients[current_indices]
        hess_node: NDArray[np.float64] = hessians[current_indices]
        node_value = _compute_leaf_value(grads_node, hess_node, reg_alpha, reg_lambda)
        nodes.append(
            TreeNode(
                node_id=node_id,
                is_leaf=False,
                feature_index=best_split["feature_index"],
                feature_name=feature_names[best_split["feature_index"]],
                threshold=best_split["threshold"],
                nan_direction=best_split["nan_direction"],
                value=node_value,
                n_samples=n_samples_current,
                left_child=None,
                right_child=None,
            )
        )
        node_children[node_id] = (None, None)

        # Compute child histograms using sibling subtraction trick
        left_indices = best_split["left_indices"]
        right_indices = best_split["right_indices"]
        left_histograms, right_histograms = _compute_child_histograms(
            left_indices,
            right_indices,
            gradients,
            hessians,
            feature_bins,
            parent_histograms,
        )

        # Add children to stack with precomputed histograms
        stack.append((right_indices, depth + 1, node_id, False, right_histograms))
        stack.append((left_indices, depth + 1, node_id, True, left_histograms))

    return nodes, n_leaves, node_children


def _compute_child_histograms(
    left_indices: NDArray[np.int64],
    right_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    feature_bins: FeatureBins,
    parent_histograms: dict[int, HistogramBuffer],
) -> tuple[dict[int, HistogramBuffer], dict[int, HistogramBuffer]]:
    """Compute histograms for both children using sibling subtraction.

    Builds histogram for smaller child and derives larger child via subtraction.
    This gives 2x speedup on histogram building per split.

    Args:
        left_indices: Sample indices for left child.
        right_indices: Sample indices for right child.
        gradients: All gradients.
        hessians: All hessians.
        feature_bins: Precomputed feature bins.
        parent_histograms: Parent's histograms per feature.

    Returns:
        Tuple of (left_histograms, right_histograms).
    """
    left_histograms: dict[int, HistogramBuffer] = {}
    right_histograms: dict[int, HistogramBuffer] = {}

    # Determine which child is smaller
    n_left: int = int(left_indices.shape[0])
    n_right: int = int(right_indices.shape[0])
    left_is_smaller: bool = n_left <= n_right
    smaller_indices: NDArray[np.int64] = left_indices if left_is_smaller else right_indices

    for feat_idx, parent_hist in parent_histograms.items():
        n_bins = parent_hist.n_bins

        # Build histogram for smaller child
        # sample_bins is now 2D: (n_samples, n_features), access column for this feature
        feat_bins: NDArray[np.int64] = feature_bins.sample_bins[:, feat_idx]
        smaller_hist = build_histogram(
            smaller_indices,
            gradients,
            hessians,
            feat_bins,
            n_bins,
        )

        # Derive larger child via subtraction
        larger_hist = subtract_histogram(parent_hist, smaller_hist)

        if left_is_smaller:
            left_histograms[feat_idx] = smaller_hist
            right_histograms[feat_idx] = larger_hist
        else:
            left_histograms[feat_idx] = larger_hist
            right_histograms[feat_idx] = smaller_hist

    return left_histograms, right_histograms


def _update_parent_child(
    node_children: dict[int, tuple[int | None, int | None]],
    parent_id: int | None,
    node_id: int,
    is_left: bool | None,
) -> None:
    """Update parent's child pointer.

    Args:
        node_children: Dictionary of node children.
        parent_id: Parent node ID.
        node_id: Current node ID.
        is_left: Whether this is the left child.
    """
    if parent_id is not None:
        left_child, right_child = node_children.get(parent_id, (None, None))
        if is_left:
            node_children[parent_id] = (node_id, right_child)
        else:
            node_children[parent_id] = (left_child, node_id)


def _select_features(
    n_features: int,
    max_features: int,
    rng: RandomStateProtocol,
) -> tuple[int, ...]:
    """Select features for splitting.

    Args:
        n_features: Total number of features.
        max_features: Maximum features to select.
        rng: Random state.

    Returns:
        Tuple of feature indices.
    """
    if max_features < n_features:
        return rng.choice(n_features, size=max_features, replace=False)
    return tuple(range(n_features))


def _finalize_nodes(
    nodes: list[TreeNode],
    node_children: dict[int, tuple[int | None, int | None]],
) -> list[TreeNode]:
    """Finalize nodes with proper child pointers.

    Args:
        nodes: List of nodes.
        node_children: Dictionary of child pointers.

    Returns:
        List of nodes with updated child pointers.
    """
    final_nodes: list[TreeNode] = []
    for node in nodes:
        if node["is_leaf"]:
            final_nodes.append(node)
        else:
            left_child, right_child = node_children.get(node["node_id"], (None, None))
            updated_node = TreeNode(
                node_id=node["node_id"],
                is_leaf=node["is_leaf"],
                feature_index=node["feature_index"],
                feature_name=node["feature_name"],
                threshold=node["threshold"],
                nan_direction=node["nan_direction"],
                value=node["value"],
                n_samples=node["n_samples"],
                left_child=left_child,
                right_child=right_child,
            )
            final_nodes.append(updated_node)
    return final_nodes


def _compute_max_depth(nodes: list[TreeNode]) -> int:
    """Compute maximum depth of tree.

    Args:
        nodes: List of tree nodes.

    Returns:
        Maximum depth.
    """

    def depth_of(node_id: int, current_depth: int) -> int:
        node = nodes[node_id]
        if node["is_leaf"]:
            return current_depth
        left_depth = depth_of(node["left_child"] or 0, current_depth + 1)
        right_depth = depth_of(node["right_child"] or 0, current_depth + 1)
        return max(left_depth, right_depth)

    return depth_of(0, 0)


def _predict_single(tree: DecisionTree, x_single: NDArray[np.float64]) -> float:
    """Get prediction for a single sample.

    Args:
        tree: Trained decision tree.
        x_single: Single sample feature vector (1D array).

    Returns:
        Prediction value.
    """
    nodes = tree["nodes"]
    node_id = 0

    while True:
        node = nodes[node_id]
        if node["is_leaf"]:
            return node["value"]

        feature_idx = node["feature_index"]
        threshold = node["threshold"]

        if feature_idx is None or threshold is None:
            return node["value"]

        feature_value: float = x_single.item(feature_idx)

        # Handle NaN values using stored nan_direction
        if math.isnan(feature_value):
            nan_dir = node["nan_direction"]
            next_id = node["left_child"] if nan_dir == "left" else node["right_child"]
        elif feature_value <= threshold:
            next_id = node["left_child"]
        else:
            next_id = node["right_child"]

        if next_id is None:
            return node["value"]

        node_id = next_id


def predict_tree(
    tree: DecisionTree,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Get predictions from tree for all samples.

    Uses the active backend (Rust when available, Python fallback).

    Args:
        tree: Trained decision tree.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Prediction array for each sample.
    """
    return _predict_tree_hook(tree, x)


def explain_tree_prediction(
    tree: DecisionTree,
    x_single: NDArray[np.float64],
    tree_index: int = 0,
) -> TreePredictionExplanation:
    """Explain prediction for a single sample.

    Args:
        tree: Trained decision tree.
        x_single: Single sample feature vector (n_features,).
        tree_index: Index of this tree in ensemble.

    Returns:
        Explanation with path through tree.
    """
    nodes = tree["nodes"]
    node_id = 0
    path: list[SplitCondition] = []

    while True:
        node = nodes[node_id]
        if node["is_leaf"]:
            value = node["value"]
            return TreePredictionExplanation(
                tree_index=tree_index,
                prediction=value if value is not None else 0.0,
                path=tuple(path),
                leaf_node_id=node_id,
                n_samples_in_leaf=node["n_samples"],
            )

        feature_idx = node["feature_index"]
        threshold = node["threshold"]
        feature_name = node["feature_name"]

        if feature_idx is None or threshold is None or feature_name is None:
            return TreePredictionExplanation(
                tree_index=tree_index,
                prediction=node["value"],
                path=tuple(path),
                leaf_node_id=node_id,
                n_samples_in_leaf=node["n_samples"],
            )

        feature_value: float = x_single.item(feature_idx)

        # Handle NaN values using stored nan_direction
        if math.isnan(feature_value):
            nan_direction = node["nan_direction"]
            if nan_direction == "left":
                direction: Literal["left", "right"] = "left"
                next_id = node["left_child"]
            else:
                direction = "right"
                next_id = node["right_child"]
        elif feature_value <= threshold:
            direction = "left"
            next_id = node["left_child"]
        else:
            direction = "right"
            next_id = node["right_child"]

        path.append(
            SplitCondition(
                feature_index=feature_idx,
                feature_name=feature_name,
                threshold=threshold,
                direction=direction,
            )
        )

        if next_id is None:
            return TreePredictionExplanation(
                tree_index=tree_index,
                prediction=node["value"],
                path=tuple(path),
                leaf_node_id=node_id,
                n_samples_in_leaf=node["n_samples"],
            )

        node_id = next_id


__all__ = [
    "_build_tree_with_histograms",
    "_compute_child_histograms",
    "_compute_max_depth",
    "_finalize_nodes",
    "_get_sample_indices",
    "_predict_single",
    "_select_features",
    "_should_be_leaf",
    "_update_parent_child",
    "build_tree",
    "explain_tree_prediction",
    "predict_tree",
]
