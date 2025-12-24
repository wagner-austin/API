"""Split computation helpers for gradient boosting trees.

Contains leaf value computation, split gain calculation, and split finding logic.
"""

from __future__ import annotations

from operator import itemgetter

from cleargbm.types import (
    FloatArray,
    FloatMatrix,
    GradientBoostingConfig,
    SplitCandidate,
    TreeNode,
)


def _compute_leaf_value(
    gradients: FloatArray,
    hessians: FloatArray,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> float:
    """Compute optimal leaf value from gradients and hessians with regularization.

    The optimal value minimizes the regularized loss:
    - Without regularization: leaf = -G / H
    - With L1 (alpha): leaf = -sign(G) * max(|G| - alpha, 0) / (H + lambda)
    - With L2 (lambda): adds lambda to the denominator

    Args:
        gradients: Gradients for samples in this leaf.
        hessians: Hessians for samples in this leaf.
        reg_alpha: L1 regularization term (default: 0.0).
        reg_lambda: L2 regularization term (default: 0.0).

    Returns:
        Optimal leaf prediction value.

    Raises:
        ValueError: If gradients and hessians have different lengths.
        ValueError: If inputs are empty.
    """
    if len(gradients) != len(hessians):
        raise ValueError(
            f"gradients and hessians must have same length, "
            f"got {len(gradients)} and {len(hessians)}"
        )
    if len(gradients) == 0:
        raise ValueError("Cannot compute leaf value from empty arrays")

    g_sum = sum(gradients)
    h_sum = sum(hessians)

    # L2 regularization: add lambda to hessian sum
    h_reg = h_sum + reg_lambda

    # Avoid division by zero
    eps = 1e-10
    if abs(h_reg) < eps:
        return 0.0

    # L1 regularization: soft threshold on gradient
    if reg_alpha > 0.0:
        # Soft thresholding: -sign(G) * max(|G| - alpha, 0) / (H + lambda)
        abs_g = abs(g_sum)
        if abs_g <= reg_alpha:
            return 0.0
        sign_g = 1.0 if g_sum > 0.0 else -1.0
        return -sign_g * (abs_g - reg_alpha) / h_reg

    # Standard case (no L1): -G / (H + lambda)
    return -g_sum / h_reg


def _compute_split_gain(
    g_left: float,
    h_left: float,
    g_right: float,
    h_right: float,
    g_total: float,
    h_total: float,
    reg_lambda: float = 0.0,
) -> float:
    """Compute gain from a split with L2 regularization.

    Without regularization: Gain = G_L^2/H_L + G_R^2/H_R - G^2/H
    With L2 regularization: Gain = G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - G^2/(H + lambda)

    Args:
        g_left: Sum of gradients in left child.
        h_left: Sum of hessians in left child.
        g_right: Sum of gradients in right child.
        h_right: Sum of hessians in right child.
        g_total: Total sum of gradients.
        h_total: Total sum of hessians.
        reg_lambda: L2 regularization term (default: 0.0).

    Returns:
        Split gain (higher is better).
    """
    eps = 1e-10

    # Add L2 regularization to hessian sums
    h_left_reg = h_left + reg_lambda
    h_right_reg = h_right + reg_lambda
    h_total_reg = h_total + reg_lambda

    # Avoid division by zero
    if abs(h_left_reg) < eps or abs(h_right_reg) < eps or abs(h_total_reg) < eps:
        return 0.0

    score_left = (g_left * g_left) / h_left_reg
    score_right = (g_right * g_right) / h_right_reg
    score_total = (g_total * g_total) / h_total_reg

    return score_left + score_right - score_total


def _check_monotonicity(
    monotonic_constraint: int,
    g_left: float,
    h_left: float,
    g_right: float,
    h_right: float,
) -> bool:
    """Check if split satisfies monotonicity constraint.

    Args:
        monotonic_constraint: -1, 0, or +1.
        g_left: Sum of gradients in left child.
        h_left: Sum of hessians in left child.
        g_right: Sum of gradients in right child.
        h_right: Sum of hessians in right child.

    Returns:
        True if constraint is satisfied, False otherwise.
    """
    if monotonic_constraint == 0:
        return True

    left_value = -g_left / max(h_left, 1e-10)
    right_value = -g_right / max(h_right, 1e-10)

    if monotonic_constraint > 0:
        return left_value <= right_value
    return left_value >= right_value


def _find_split_for_feature(
    sorted_pairs: list[tuple[float, int]],
    gradients: FloatArray,
    hessians: FloatArray,
    g_total: float,
    h_total: float,
    min_samples_leaf: int,
    monotonic_constraint: int,
    feature_idx: int,
    reg_lambda: float = 0.0,
) -> SplitCandidate | None:
    """Find best split for a single feature.

    Args:
        sorted_pairs: List of (feature_value, sample_index) sorted by value.
        gradients: Gradient for each sample.
        hessians: Hessian for each sample.
        g_total: Total gradient sum.
        h_total: Total hessian sum.
        min_samples_leaf: Minimum samples required in each leaf.
        monotonic_constraint: Constraint on split direction.
        feature_idx: Index of feature being split.
        reg_lambda: L2 regularization term (default: 0.0).

    Returns:
        Best split candidate for this feature, or None.
    """
    n_samples = len(sorted_pairs)
    best_gain = 0.0
    best_split_pos: int = -1
    best_threshold: float = 0.0

    g_left = 0.0
    h_left = 0.0

    for split_pos in range(min_samples_leaf, n_samples - min_samples_leaf + 1):
        _, sample_idx = sorted_pairs[split_pos - 1]
        g_left += gradients[sample_idx]
        h_left += hessians[sample_idx]

        g_right = g_total - g_left
        h_right = h_total - h_left

        # Skip if same feature value as next sample
        # Note: split_pos is always < n_samples due to loop bounds
        curr_val = sorted_pairs[split_pos - 1][0]
        next_val = sorted_pairs[split_pos][0]
        if abs(curr_val - next_val) < 1e-10:
            continue

        # Check monotonicity
        if not _check_monotonicity(monotonic_constraint, g_left, h_left, g_right, h_right):
            continue

        gain = _compute_split_gain(g_left, h_left, g_right, h_right, g_total, h_total, reg_lambda)

        if gain > best_gain:
            best_gain = gain
            best_split_pos = split_pos
            # Compute threshold only when we find a better split
            best_threshold = _compute_threshold(sorted_pairs, split_pos, n_samples)
    # Only create tuples once for the best split found
    if best_split_pos < 0:
        return None

    left_indices = tuple(idx for _, idx in sorted_pairs[:best_split_pos])
    right_indices = tuple(idx for _, idx in sorted_pairs[best_split_pos:])
    # Exact split finding defaults NaN to left (histogram path learns optimal direction)
    return SplitCandidate(
        feature_index=feature_idx,
        threshold=best_threshold,
        gain=best_gain,
        left_indices=left_indices,
        right_indices=right_indices,
        nan_direction="left",
    )


def _compute_threshold(
    sorted_pairs: list[tuple[float, int]],
    split_pos: int,
    n_samples: int,
) -> float:
    """Compute split threshold.

    Args:
        sorted_pairs: Sorted (value, index) pairs.
        split_pos: Position of split.
        n_samples: Total number of samples.

    Returns:
        Threshold value for split.
    """
    if split_pos < n_samples:
        return (sorted_pairs[split_pos - 1][0] + sorted_pairs[split_pos][0]) / 2.0
    return sorted_pairs[split_pos - 1][0] + 0.5


def find_best_split(
    x: FloatMatrix,
    gradients: FloatArray,
    hessians: FloatArray,
    sample_indices: tuple[int, ...],
    feature_indices: tuple[int, ...],
    min_samples_leaf: int,
    monotonic_constraint: int,
    reg_lambda: float = 0.0,
) -> SplitCandidate | None:
    """Find the best split for current node.

    Args:
        x: Feature matrix (all samples, all features).
        gradients: Gradient for each sample.
        hessians: Hessian for each sample.
        sample_indices: Indices of samples in this node.
        feature_indices: Which features to consider for splitting.
        min_samples_leaf: Minimum samples required in each leaf.
        monotonic_constraint: Constraint on split direction (-1, 0, +1).
        reg_lambda: L2 regularization term (default: 0.0).

    Returns:
        Best split candidate, or None if no valid split exists.
    """
    n_samples = len(sample_indices)
    if n_samples < 2 * min_samples_leaf:
        return None

    g_total = sum(gradients[i] for i in sample_indices)
    h_total = sum(hessians[i] for i in sample_indices)

    best_split: SplitCandidate | None = None

    for feature_idx in feature_indices:
        feature_values: list[tuple[float, int]] = [(x[i][feature_idx], i) for i in sample_indices]
        sorted_pairs: list[tuple[float, int]] = sorted(feature_values, key=itemgetter(0))

        split = _find_split_for_feature(
            sorted_pairs,
            gradients,
            hessians,
            g_total,
            h_total,
            min_samples_leaf,
            monotonic_constraint,
            feature_idx,
            reg_lambda,
        )
        if split is not None and (best_split is None or split["gain"] > best_split["gain"]):
            best_split = split

    return best_split


def _create_leaf_node(
    node_id: int,
    sample_indices: tuple[int, ...],
    gradients: FloatArray,
    hessians: FloatArray,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> TreeNode:
    """Create a leaf node with regularization.

    Args:
        node_id: ID for this node.
        sample_indices: Indices of samples in this node.
        gradients: All gradients.
        hessians: All hessians.
        reg_alpha: L1 regularization term (default: 0.0).
        reg_lambda: L2 regularization term (default: 0.0).

    Returns:
        Leaf TreeNode.
    """
    # Use map + __getitem__ for faster tuple creation
    node_gradients = tuple(gradients[i] for i in sample_indices)
    node_hessians = tuple(hessians[i] for i in sample_indices)
    leaf_value = _compute_leaf_value(node_gradients, node_hessians, reg_alpha, reg_lambda)

    return TreeNode(
        node_id=node_id,
        is_leaf=True,
        feature_index=None,
        feature_name=None,
        threshold=None,
        nan_direction=None,
        value=leaf_value,
        n_samples=len(sample_indices),
        left_child=None,
        right_child=None,
    )


def _create_internal_node(
    node_id: int,
    sample_indices: tuple[int, ...],
    gradients: FloatArray,
    hessians: FloatArray,
    split: SplitCandidate,
    feature_names: tuple[str, ...],
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
) -> TreeNode:
    """Create an internal (non-leaf) node with regularization.

    Args:
        node_id: Node ID.
        sample_indices: Sample indices.
        gradients: Gradients.
        hessians: Hessians.
        split: Split candidate.
        feature_names: Feature names.
        reg_alpha: L1 regularization term (default: 0.0).
        reg_lambda: L2 regularization term (default: 0.0).

    Returns:
        Internal TreeNode.
    """
    # Use map + __getitem__ for faster tuple creation
    node_gradients = tuple(gradients[i] for i in sample_indices)
    node_hessians = tuple(hessians[i] for i in sample_indices)
    leaf_value = _compute_leaf_value(node_gradients, node_hessians, reg_alpha, reg_lambda)

    return TreeNode(
        node_id=node_id,
        is_leaf=False,
        feature_index=split["feature_index"],
        feature_name=feature_names[split["feature_index"]],
        threshold=split["threshold"],
        nan_direction=split["nan_direction"],
        value=leaf_value,
        n_samples=len(sample_indices),
        left_child=None,
        right_child=None,
    )


def _find_best_split_with_constraints(
    x: FloatMatrix,
    gradients: FloatArray,
    hessians: FloatArray,
    sample_indices: tuple[int, ...],
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
) -> SplitCandidate | None:
    """Find best split considering monotonic constraints.

    Args:
        x: Feature matrix.
        gradients: Gradients.
        hessians: Hessians.
        sample_indices: Sample indices for this node.
        feature_indices: Features to consider.
        config: Configuration.

    Returns:
        Best split candidate or None.
    """
    constraints = config["monotonic_constraints"]
    best_split: SplitCandidate | None = None

    for feat_idx in feature_indices:
        constraint = 0 if constraints is None else constraints[feat_idx]
        split = find_best_split(
            x,
            gradients,
            hessians,
            sample_indices,
            (feat_idx,),
            config["min_samples_leaf"],
            constraint,
        )
        if split is not None and (best_split is None or split["gain"] > best_split["gain"]):
            best_split = split

    return best_split


__all__ = [
    "_check_monotonicity",
    "_compute_leaf_value",
    "_compute_split_gain",
    "_compute_threshold",
    "_create_internal_node",
    "_create_leaf_node",
    "_find_best_split_with_constraints",
    "_find_split_for_feature",
    "find_best_split",
]
