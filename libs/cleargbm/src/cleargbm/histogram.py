"""Histogram-based split finding for gradient boosting.

LightGBM-style histogram binning for O(K) split finding instead of O(n log n).
Bins each feature into K cuts once, then builds gradient/hessian histograms
per feature and scans bins with prefix sums to evaluate splits.

Supports NaN values via dedicated NaN bin (last bin in histogram).

Built from scratch - uses only Python stdlib (no numpy).
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

from cleargbm.types import FloatArray, FloatMatrix, IntArray

# NaN bin is always the last bin (index = n_regular_bins)
NAN_BIN_OFFSET: Literal[1] = 1


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
        sample_bins: Per-sample bin ID for each feature.
                     sample_bins[feature_idx][sample_idx] = bin_id
    """

    bin_edges: tuple[BinEdges, ...]
    sample_bins: tuple[IntArray, ...]


class Histogram(NamedTuple):
    """Gradient/hessian histogram for a single feature.

    Args:
        gradient_sums: Sum of gradients in each bin.
        hessian_sums: Sum of hessians in each bin.
        counts: Number of samples in each bin.
    """

    gradient_sums: tuple[float, ...]
    hessian_sums: tuple[float, ...]
    counts: tuple[int, ...]


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
    x: FloatMatrix,
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
    n_samples = len(x)
    result: list[BinEdges] = []

    for feat_idx in range(n_features):
        # Extract feature values, excluding NaN
        values: list[float] = [
            x[i][feat_idx] for i in range(n_samples) if not math.isnan(x[i][feat_idx])
        ]

        # Handle case where all values are NaN
        if not values:
            result.append(BinEdges(edges=()))
            continue

        sorted_values = sorted(values)
        n_valid = len(sorted_values)

        # Compute quantile edges
        n_edges = max_bins - 1
        edges: list[float] = []

        for edge_idx in range(1, n_edges + 1):
            # Quantile position: q ranges from 1/max_bins to (max_bins-1)/max_bins < 1
            # So pos is always in [0, n_valid-1]
            q = edge_idx / max_bins
            pos = int(q * (n_valid - 1))
            edge_value = sorted_values[pos]
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
    x: FloatMatrix,
    bin_edges: tuple[BinEdges, ...],
) -> tuple[IntArray, ...]:
    """Assign each sample to bins for each feature.

    NaN values are assigned to a dedicated NaN bin (last bin).

    Args:
        x: Feature matrix (n_samples, n_features).
        bin_edges: Bin edges for each feature.

    Returns:
        Tuple of IntArray, one per feature. sample_bins[f][i] = bin ID for
        sample i on feature f. NaN values get bin ID = len(edges) + 1.
    """
    n_samples = len(x)
    n_features = len(bin_edges)
    result: list[IntArray] = []

    for feat_idx in range(n_features):
        edges = bin_edges[feat_idx].edges
        # NaN bin is after all regular bins: len(edges) + 1
        # Regular bins: 0 to len(edges), NaN bin: len(edges) + 1
        nan_bin = len(edges) + NAN_BIN_OFFSET
        bins: list[int] = [_assign_bin(x[i][feat_idx], edges, nan_bin) for i in range(n_samples)]
        result.append(tuple(bins))

    return tuple(result)


def precompute_feature_bins(
    x: FloatMatrix,
    max_bins: int,
) -> FeatureBins:
    """Precompute all bin assignments for the dataset.

    Call once at the start of training. Subsequent split finding uses
    the precomputed bins for O(K) instead of O(n log n) per split.

    Args:
        x: Feature matrix (n_samples, n_features).
        max_bins: Maximum number of bins per feature.

    Returns:
        FeatureBins containing edges and per-sample bin assignments.
    """
    n_features = len(x[0]) if x else 0
    edges = compute_bin_edges(x, n_features, max_bins)
    sample_bins = bin_samples(x, edges)
    return FeatureBins(bin_edges=edges, sample_bins=sample_bins)


def build_histogram(
    sample_indices: tuple[int, ...],
    gradients: FloatArray,
    hessians: FloatArray,
    sample_bins: IntArray,
    n_bins: int,
) -> Histogram:
    """Build gradient/hessian histogram for one feature in a node.

    Args:
        sample_indices: Indices of samples in this node.
        gradients: Gradient for each sample (full dataset).
        hessians: Hessian for each sample (full dataset).
        sample_bins: Bin ID for each sample on this feature.
        n_bins: Number of bins.

    Returns:
        Histogram with gradient/hessian sums per bin.
    """
    g_sums: list[float] = [0.0] * n_bins
    h_sums: list[float] = [0.0] * n_bins
    counts: list[int] = [0] * n_bins

    for idx in sample_indices:
        bin_id = sample_bins[idx]
        g_sums[bin_id] += gradients[idx]
        h_sums[bin_id] += hessians[idx]
        counts[bin_id] += 1

    return Histogram(
        gradient_sums=tuple(g_sums),
        hessian_sums=tuple(h_sums),
        counts=tuple(counts),
    )


def subtract_histogram(parent: Histogram, child: Histogram) -> Histogram:
    """Compute sibling histogram by subtraction.

    sibling = parent - child (the histogram trick for 2x speedup).

    Args:
        parent: Parent node histogram.
        child: One child's histogram.

    Returns:
        Other child's histogram.
    """
    # Use map + sub for faster tuple creation
    g_sums = tuple(p - c for p, c in zip(parent.gradient_sums, child.gradient_sums, strict=True))
    h_sums = tuple(p - c for p, c in zip(parent.hessian_sums, child.hessian_sums, strict=True))
    counts = tuple(p - c for p, c in zip(parent.counts, child.counts, strict=True))
    return Histogram(gradient_sums=g_sums, hessian_sums=h_sums, counts=counts)


def _compute_split_gain(
    g_left: float,
    h_left: float,
    g_right: float,
    h_right: float,
    g_total: float,
    h_total: float,
) -> float:
    """Compute gain from a split.

    Gain = (G_L^2/H_L + G_R^2/H_R) - (G^2/H)

    Args:
        g_left: Sum of gradients in left child.
        h_left: Sum of hessians in left child.
        g_right: Sum of gradients in right child.
        h_right: Sum of hessians in right child.
        g_total: Total sum of gradients.
        h_total: Total sum of hessians.

    Returns:
        Split gain (higher is better).
    """
    eps = 1e-10

    if abs(h_left) < eps or abs(h_right) < eps or abs(h_total) < eps:
        return 0.0

    score_left = (g_left * g_left) / h_left
    score_right = (g_right * g_right) / h_right
    score_total = (g_total * g_total) / h_total

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
    histogram: Histogram,
    bin_edges: BinEdges,
    feature_index: int,
    min_samples_leaf: int,
    monotonic_constraint: int,
) -> HistogramSplit | None:
    """Find best split by scanning histogram bins.

    Uses prefix sums for O(K) split finding. Handles NaN values by evaluating
    both NaN-goes-left and NaN-goes-right scenarios for each split.

    Args:
        histogram: Gradient/hessian histogram (includes NaN bin as last bin).
        bin_edges: Bin edges for threshold lookup.
        feature_index: Feature index.
        min_samples_leaf: Minimum samples in each leaf.
        monotonic_constraint: -1, 0, or +1.

    Returns:
        Best split or None if no valid split.
    """
    n_bins = len(histogram.gradient_sums)
    edges = bin_edges.edges
    n_regular_bins = len(edges) + 1  # Bins 0..len(edges)
    nan_bin_idx = n_regular_bins  # NaN bin is at index len(edges) + 1

    # Extract NaN bin stats (if histogram has NaN bin)
    has_nan_bin = n_bins > n_regular_bins
    g_nan = histogram.gradient_sums[nan_bin_idx] if has_nan_bin else 0.0
    h_nan = histogram.hessian_sums[nan_bin_idx] if has_nan_bin else 0.0
    n_nan = histogram.counts[nan_bin_idx] if has_nan_bin else 0

    # Compute totals for regular bins only
    g_regular = sum(histogram.gradient_sums[:n_regular_bins])
    h_regular = sum(histogram.hessian_sums[:n_regular_bins])
    n_regular = sum(histogram.counts[:n_regular_bins])

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
        g_left_base += histogram.gradient_sums[bin_idx]
        h_left_base += histogram.hessian_sums[bin_idx]
        n_left_base += histogram.counts[bin_idx]

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

            gain = _compute_split_gain(g_left, h_left, g_right, h_right, g_total, h_total)

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
    sample_indices: tuple[int, ...],
    sample_bins: IntArray,
    split_bin: int,
    nan_bin: int,
    nan_direction: Literal["left", "right"],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partition samples into left/right based on bin split.

    Samples in bins <= split_bin go left, others go right.
    NaN samples (in nan_bin) go to the specified nan_direction.

    Args:
        sample_indices: Indices of samples in this node.
        sample_bins: Bin ID for each sample on the split feature.
        split_bin: Split at this bin (inclusive left).
        nan_bin: Bin ID for NaN samples.
        nan_direction: Direction for NaN samples ("left" or "right").

    Returns:
        Tuple of (left_indices, right_indices).
    """
    left: list[int] = []
    right: list[int] = []

    for idx in sample_indices:
        bin_id = sample_bins[idx]
        if bin_id == nan_bin:
            # NaN sample: route based on nan_direction
            if nan_direction == "left":
                left.append(idx)
            else:
                right.append(idx)
        elif bin_id <= split_bin:
            left.append(idx)
        else:
            right.append(idx)

    return tuple(left), tuple(right)


__all__ = [
    "NAN_BIN_OFFSET",
    "BinEdges",
    "FeatureBins",
    "Histogram",
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
