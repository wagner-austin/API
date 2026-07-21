"""Histogram-based split finding for gradient boosting.

LightGBM-style histogram binning for O(K) split finding instead of O(n log n).
Bins each feature into K cuts once, then builds gradient/hessian histograms
per feature and scans bins with prefix sums to evaluate splits.

Supports NaN values via dedicated NaN bin (last bin in histogram).
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from cleargbm._hooks_binning import precompute_feature_bins as _precompute_feature_bins_hook
from cleargbm._hooks_histogram import build_histogram as _build_histogram_hook
from cleargbm._hooks_histogram import subtract_histogram as _subtract_histogram_hook
from cleargbm.buffers import HistogramBuffer
from cleargbm.types import BinEdges, FeatureBins

# NaN bin is always the last bin (index = n_regular_bins)
NAN_BIN_OFFSET: Literal[1] = 1


class HistogramSplit(NamedTuple):
    """Best split found from histogram scan.

    Args:
        feature_index: Which feature to split on.
        bin_index: Split at this bin (samples in bins <= bin_index go left).
        threshold: Actual threshold value from bin edges.
        gain: Split gain.
        nan_direction: Direction for NaN values ("left" or "right").
    """

    feature_index: int
    bin_index: int
    threshold: float
    gain: float
    nan_direction: Literal["left", "right"]


def compute_bin_edges(
    x: NDArray[np.float64],
    n_features: int,
    max_bins: int,
) -> tuple[BinEdges, ...]:
    """Compute bin edges for each feature using quantiles.

    Computes K-1 edges to create K bins per feature. Uses approximate
    quantiles to handle large datasets efficiently. NaN values are
    excluded from quantile computation.

    Args:
        x: Feature matrix (n_samples, n_features).
        n_features: Number of features.
        max_bins: Maximum number of bins per feature.

    Returns:
        Tuple of BinEdges, one per feature.
    """
    result: list[BinEdges] = []

    for feat_idx in range(n_features):
        # Extract feature column
        col: NDArray[np.float64] = x[:, feat_idx]

        # Filter out NaN values
        valid_mask: NDArray[np.bool_] = ~np.isnan(col)
        valid_values: NDArray[np.float64] = col[valid_mask]

        # Handle case where all values are NaN
        if valid_values.size == 0:
            result.append(BinEdges(edges=()))
            continue

        sorted_values: NDArray[np.float64] = np.sort(valid_values)
        n_valid = sorted_values.size

        # Compute quantile edges
        n_edges = max_bins - 1
        edges: list[float] = []

        for edge_idx in range(1, n_edges + 1):
            # Quantile position: q ranges from 1/max_bins to (max_bins-1)/max_bins < 1
            # So pos is always in [0, n_valid-1]
            q = edge_idx / max_bins
            pos = int(q * (n_valid - 1))
            # Use .item(idx) on array for proper typing (returns Python float)
            edge_value: float = sorted_values.item(pos)
            # Avoid duplicate edges
            if not edges or edge_value > edges[-1]:
                edges.append(edge_value)

        result.append(BinEdges(edges=tuple(edges)))

    return tuple(result)


def _assign_bin(value: float, edges: tuple[float, ...], nan_bin: int) -> int:
    """Assign a value to a bin using binary search.

    NaN values are assigned to the dedicated NaN bin (last bin).

    Args:
        value: Feature value (may be NaN).
        edges: Sorted bin edges.
        nan_bin: Bin index for NaN values.

    Returns:
        Bin index (0 to len(edges) for regular values, nan_bin for NaN).
    """
    # NaN values go to dedicated NaN bin
    if math.isnan(value):
        return nan_bin

    # Binary search for the correct bin
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) // 2
        if value <= edges[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def bin_samples(
    x: NDArray[np.float64],
    bin_edges: tuple[BinEdges, ...],
) -> NDArray[np.int64]:
    """Assign each sample to bins for each feature.

    NaN values are assigned to a dedicated NaN bin (last bin).

    Args:
        x: Feature matrix (n_samples, n_features).
        bin_edges: Bin edges for each feature.

    Returns:
        2D array of shape (n_samples, n_features) with bin IDs.
        sample_bins[i, f] = bin ID for sample i on feature f.
        NaN values get bin ID = len(edges) + 1.
    """
    n_samples = x.shape[0]
    n_features = len(bin_edges)
    result: NDArray[np.int64] = np.zeros((n_samples, n_features), dtype=np.int64)

    for feat_idx in range(n_features):
        edges = bin_edges[feat_idx].edges
        # NaN bin is after all regular bins: len(edges) + 1
        # Regular bins: 0 to len(edges), NaN bin: len(edges) + 1
        nan_bin = len(edges) + NAN_BIN_OFFSET
        # Extract feature column for proper typing
        feat_col: NDArray[np.float64] = x[:, feat_idx]
        for i in range(n_samples):
            # Use .item(idx) on array for proper typing
            val: float = feat_col.item(i)
            result[i, feat_idx] = _assign_bin(val, edges, nan_bin)

    return result


def precompute_feature_bins(
    x: NDArray[np.float64],
    max_bins: int,
) -> FeatureBins:
    """Precompute all bin assignments for the dataset.

    Call once at the start of training. Subsequent split finding uses
    the precomputed bins for O(K) instead of O(n log n) per split.

    Delegates to the active backend hook.

    Args:
        x: Feature matrix (n_samples, n_features).
        max_bins: Maximum number of bins per feature.

    Returns:
        FeatureBins containing edges and per-sample bin assignments.
    """
    return _precompute_feature_bins_hook(x, max_bins)


def build_histogram(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    sample_bins: NDArray[np.int64],
    n_bins: int,
) -> HistogramBuffer:
    """Build gradient/hessian histogram for one feature in a node.

    Uses the active backend (Rust when available, Python fallback).

    Args:
        sample_indices: Indices of samples in this node.
        gradients: Gradient for each sample (full dataset).
        hessians: Hessian for each sample (full dataset).
        sample_bins: Bin ID for each sample on this feature (1D array).
        n_bins: Number of bins.

    Returns:
        HistogramBuffer with gradient/hessian sums per bin.
    """
    return _build_histogram_hook(sample_indices, gradients, hessians, sample_bins, n_bins)


def subtract_histogram(parent: HistogramBuffer, child: HistogramBuffer) -> HistogramBuffer:
    """Compute sibling histogram by subtraction.

    sibling = parent - child (the histogram trick for 2x speedup).

    Uses the active backend (Rust when available, Python fallback).

    Args:
        parent: Parent node histogram buffer.
        child: One child's histogram buffer.

    Returns:
        Other child's histogram buffer.
    """
    return _subtract_histogram_hook(parent, child)


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

    Without regularization: Gain = G_L^2/H_L + G_R^2/H_R - G^2/H.
    With L2:                Gain = G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda).

    Args:
        g_left: Sum of gradients in left child.
        h_left: Sum of hessians in left child.
        g_right: Sum of gradients in right child.
        h_right: Sum of hessians in right child.
        g_total: Total sum of gradients.
        h_total: Total sum of hessians.
        reg_lambda: L2 regularization term added to each hessian sum (default: 0.0,
            preserving the pre-regularization gain formula).

    Returns:
        Split gain (higher is better).
    """
    eps = 1e-10

    h_left_reg = h_left + reg_lambda
    h_right_reg = h_right + reg_lambda
    h_total_reg = h_total + reg_lambda

    if abs(h_left_reg) < eps or abs(h_right_reg) < eps or abs(h_total_reg) < eps:
        return 0.0

    score_left = (g_left * g_left) / h_left_reg
    score_right = (g_right * g_right) / h_right_reg
    score_total = (g_total * g_total) / h_total_reg

    return score_left + score_right - score_total


def _check_monotonicity_constraint(
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

    eps = 1e-10
    left_value = -g_left / max(h_left, eps)
    right_value = -g_right / max(h_right, eps)

    if monotonic_constraint > 0:
        return left_value <= right_value
    return left_value >= right_value


def _evaluate_nan_direction(
    nan_dir: str,
    g_left_base: float,
    h_left_base: float,
    n_left_base: int,
    g_nan: float,
    h_nan: float,
    n_nan: int,
) -> tuple[float, float, int]:
    """Compute left child stats for a given NaN direction.

    Args:
        nan_dir: "left" or "right".
        g_left_base: Base gradient sum for left (regular bins only).
        h_left_base: Base hessian sum for left (regular bins only).
        n_left_base: Base sample count for left (regular bins only).
        g_nan: Gradient sum for NaN bin.
        h_nan: Hessian sum for NaN bin.
        n_nan: Sample count for NaN bin.

    Returns:
        Tuple of (g_left, h_left, n_left) including NaN if direction is left.
    """
    if nan_dir == "left":
        return g_left_base + g_nan, h_left_base + h_nan, n_left_base + n_nan
    return g_left_base, h_left_base, n_left_base


def find_best_split_from_histogram(
    histogram: HistogramBuffer,
    bin_edges: BinEdges,
    feature_index: int,
    min_samples_leaf: int,
    monotonic_constraint: int,
    reg_lambda: float = 0.0,
) -> HistogramSplit | None:
    """Find best split by scanning histogram bins.

    Uses prefix sums for O(K) split finding. Handles NaN values by evaluating
    both NaN-goes-left and NaN-goes-right scenarios for each split.

    Args:
        histogram: Gradient/hessian histogram buffer (includes NaN bin as last bin).
        bin_edges: Bin edges for threshold lookup.
        feature_index: Feature index.
        min_samples_leaf: Minimum samples in each leaf.
        monotonic_constraint: -1, 0, or +1.
        reg_lambda: L2 regularization term forwarded to the gain formula. Matches
            the treatment applied to leaf values in ``split.py::_compute_leaf_value``
            and the exact-path gain in ``split.py::_compute_split_gain`` (default: 0.0).

    Returns:
        Best split or None if no valid split.
    """
    n_bins = histogram.n_bins
    edges = bin_edges.edges
    n_regular_bins = len(edges) + 1  # Bins 0..len(edges)
    nan_bin_idx = n_regular_bins  # NaN bin is at index len(edges) + 1

    # Extract NaN bin stats (if histogram has NaN bin)
    has_nan_bin = n_bins > n_regular_bins
    g_nan = histogram.get_gradient_sum(nan_bin_idx) if has_nan_bin else 0.0
    h_nan = histogram.get_hessian_sum(nan_bin_idx) if has_nan_bin else 0.0
    n_nan = histogram.get_count(nan_bin_idx) if has_nan_bin else 0

    # Compute totals for regular bins only
    g_regular = 0.0
    h_regular = 0.0
    n_regular = 0
    for i in range(n_regular_bins):
        g_regular += histogram.get_gradient_sum(i)
        h_regular += histogram.get_hessian_sum(i)
        n_regular += histogram.get_count(i)

    # Total including NaN
    g_total = g_regular + g_nan
    h_total = h_regular + h_nan
    n_total = n_regular + n_nan

    if n_total < 2 * min_samples_leaf:
        return None

    best_gain = 0.0
    best_bin = -1
    best_threshold = 0.0
    best_nan_direction: Literal["left", "right"] = "left"

    # Prefix sums for left side (regular bins only)
    g_left_base = 0.0
    h_left_base = 0.0
    n_left_base = 0

    # Scan regular bins (split after each bin)
    for bin_idx in range(n_regular_bins - 1):
        g_left_base += histogram.get_gradient_sum(bin_idx)
        h_left_base += histogram.get_hessian_sum(bin_idx)
        n_left_base += histogram.get_count(bin_idx)

        # Try both NaN directions and pick the best
        for nan_dir in ("left", "right"):
            g_left, h_left, n_left = _evaluate_nan_direction(
                nan_dir, g_left_base, h_left_base, n_left_base, g_nan, h_nan, n_nan
            )
            n_right = n_total - n_left

            # Check min_samples_leaf constraint
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            g_right = g_total - g_left
            h_right = h_total - h_left

            # Check monotonicity constraint
            if not _check_monotonicity_constraint(
                monotonic_constraint, g_left, h_left, g_right, h_right
            ):
                continue

            gain = _compute_split_gain(
                g_left, h_left, g_right, h_right, g_total, h_total, reg_lambda
            )

            if gain > best_gain:
                best_gain = gain
                best_bin = bin_idx
                best_threshold = edges[bin_idx] if bin_idx < len(edges) else 0.0
                best_nan_direction = "left" if nan_dir == "left" else "right"

    if best_bin < 0:
        return None

    return HistogramSplit(
        feature_index=feature_index,
        bin_index=best_bin,
        threshold=best_threshold,
        gain=best_gain,
        nan_direction=best_nan_direction,
    )


def partition_by_bin(
    sample_indices: NDArray[np.int64],
    sample_bins: NDArray[np.int64],
    split_bin: int,
    nan_bin: int,
    nan_direction: Literal["left", "right"],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Partition samples into left/right based on bin split.

    Samples in bins <= split_bin go left, others go right.
    NaN samples (in nan_bin) go to the specified nan_direction.

    Args:
        sample_indices: Indices of samples in this node.
        sample_bins: Bin ID for each sample on the split feature (1D array).
        split_bin: Split at this bin (inclusive left).
        nan_bin: Bin ID for NaN samples.
        nan_direction: Direction for NaN samples ("left" or "right").

    Returns:
        Tuple of (left_indices, right_indices) as numpy arrays.
    """
    # Get bin IDs for the samples in this node
    bins_for_node: NDArray[np.int64] = sample_bins[sample_indices]

    # Create masks for partitioning
    is_nan: NDArray[np.bool_] = bins_for_node == nan_bin
    is_left_regular: NDArray[np.bool_] = (bins_for_node <= split_bin) & ~is_nan
    is_right_regular: NDArray[np.bool_] = (bins_for_node > split_bin) & ~is_nan

    if nan_direction == "left":
        left_mask: NDArray[np.bool_] = is_left_regular | is_nan
        right_mask: NDArray[np.bool_] = is_right_regular
    else:
        left_mask = is_left_regular
        right_mask = is_right_regular | is_nan

    left_indices: NDArray[np.int64] = sample_indices[left_mask]
    right_indices: NDArray[np.int64] = sample_indices[right_mask]

    return left_indices, right_indices


__all__ = [
    "NAN_BIN_OFFSET",
    "HistogramBuffer",
    "HistogramSplit",
    "_check_monotonicity_constraint",
    "_evaluate_nan_direction",
    "build_histogram",
    "compute_bin_edges",
    "find_best_split_from_histogram",
    "partition_by_bin",
    "precompute_feature_bins",
    "subtract_histogram",
]
