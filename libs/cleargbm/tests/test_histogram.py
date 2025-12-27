"""Tests for cleargbm.histogram module.

LightGBM-style histogram binning for O(K) split finding.
Uses numpy arrays for all array operations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm.buffers import HistogramBuffer
from cleargbm.histogram import (
    BinEdges,
    FeatureBins,
    HistogramSplit,
    _assign_bin,
    _compute_split_gain,
    bin_samples,
    build_histogram,
    compute_bin_edges,
    find_best_split_from_histogram,
    partition_by_bin,
    precompute_feature_bins,
    subtract_histogram,
)


def _float_matrix(data: list[list[float]]) -> NDArray[np.float64]:
    """Create a 2D float array from nested list (helper for strict typing)."""
    return np.array(data, dtype=np.float64)


def _float_array(data: list[float]) -> NDArray[np.float64]:
    """Create a 1D float array from list (helper for strict typing)."""
    return np.array(data, dtype=np.float64)


def _int_array(data: list[int]) -> NDArray[np.int64]:
    """Create a 1D int array from list (helper for strict typing)."""
    return np.array(data, dtype=np.int64)


def _int_matrix(data: list[list[int]]) -> NDArray[np.int64]:
    """Create a 2D int array from nested list (helper for strict typing)."""
    return np.array(data, dtype=np.int64)


def _approx_equal(a: float, b: float, tol: float = 1e-10) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) < tol


def _approx_tuple_equal(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-10) -> bool:
    """Check if two float tuples are approximately equal."""
    return len(a) == len(b) and all(_approx_equal(x, y, tol) for x, y in zip(a, b, strict=True))


class TestBinEdges:
    """Tests for BinEdges namedtuple."""

    def test_create_bin_edges(self) -> None:
        """Should create BinEdges with edge values."""
        edges = BinEdges(edges=(0.5, 1.5, 2.5))
        assert edges.edges == (0.5, 1.5, 2.5)

    def test_empty_edges(self) -> None:
        """Should handle empty edges (single bin)."""
        edges = BinEdges(edges=())
        assert edges.edges == ()


class TestComputeBinEdges:
    """Tests for compute_bin_edges function."""

    def test_computes_quantile_edges(self) -> None:
        """Should compute quantile-based bin edges."""
        # 4 samples, 4 bins means 3 edges
        x = _float_matrix([[0.0], [1.0], [2.0], [3.0]])
        edges = compute_bin_edges(x, n_features=1, max_bins=4)

        assert len(edges) == 1
        # Edges should be between values
        for edge in edges[0].edges:
            assert 0.0 <= edge <= 3.0

    def test_handles_single_feature(self) -> None:
        """Should handle single feature correctly."""
        x = _float_matrix([[0.0], [0.5], [1.0], [1.5]])
        edges = compute_bin_edges(x, n_features=1, max_bins=2)

        assert len(edges) == 1
        # With 2 bins, we expect 1 edge (the midpoint)
        assert len(edges[0].edges) == 1

    def test_handles_multiple_features(self) -> None:
        """Should compute edges for each feature independently."""
        x = _float_matrix(
            [
                [0.0, 10.0],
                [1.0, 20.0],
                [2.0, 30.0],
                [3.0, 40.0],
            ]
        )
        edges = compute_bin_edges(x, n_features=2, max_bins=4)

        assert len(edges) == 2

    def test_avoids_duplicate_edges(self) -> None:
        """Should avoid duplicate edges when values repeat."""
        # All same values for first feature
        x = _float_matrix([[1.0], [1.0], [1.0], [1.0]])
        edges = compute_bin_edges(x, n_features=1, max_bins=4)

        assert len(edges) == 1
        # No duplicate edges
        if len(edges[0].edges) > 0:
            edge_list = list(edges[0].edges)
            assert edge_list == sorted(set(edge_list))

    def test_all_nan_feature_returns_empty_edges(self) -> None:
        """Feature with all NaN values should return empty edges."""
        import math

        nan = math.nan
        x = _float_matrix([[nan], [nan], [nan], [nan]])
        edges = compute_bin_edges(x, n_features=1, max_bins=4)

        assert len(edges) == 1
        # All NaN means no valid values to bin, so empty edges
        assert edges[0].edges == ()


class TestAssignBin:
    """Tests for _assign_bin helper."""

    def test_assigns_to_first_bin(self) -> None:
        """Values <= first edge should go to bin 0."""
        edges = (0.5, 1.5, 2.5)
        nan_bin = len(edges) + 1  # NaN bin is after regular bins
        assert _assign_bin(0.0, edges, nan_bin) == 0
        assert _assign_bin(0.5, edges, nan_bin) == 0

    def test_assigns_to_middle_bin(self) -> None:
        """Values between edges should go to correct bin."""
        edges = (0.5, 1.5, 2.5)
        nan_bin = len(edges) + 1
        assert _assign_bin(0.6, edges, nan_bin) == 1
        assert _assign_bin(1.0, edges, nan_bin) == 1
        assert _assign_bin(1.5, edges, nan_bin) == 1

    def test_assigns_to_last_bin(self) -> None:
        """Values > last edge should go to last bin."""
        edges = (0.5, 1.5, 2.5)
        nan_bin = len(edges) + 1
        assert _assign_bin(2.6, edges, nan_bin) == 3
        assert _assign_bin(100.0, edges, nan_bin) == 3

    def test_empty_edges_single_bin(self) -> None:
        """Empty edges means all values go to bin 0."""
        nan_bin = 1  # For empty edges, NaN bin is at index 1
        assert _assign_bin(0.0, (), nan_bin) == 0
        assert _assign_bin(100.0, (), nan_bin) == 0

    def test_nan_value_goes_to_nan_bin(self) -> None:
        """NaN values should be assigned to the dedicated NaN bin."""
        import math

        edges = (0.5, 1.5, 2.5)
        nan_bin = len(edges) + 1  # NaN bin is after regular bins (4)
        assert _assign_bin(math.nan, edges, nan_bin) == nan_bin


class TestBinSamples:
    """Tests for bin_samples function."""

    def test_assigns_samples_to_bins(self) -> None:
        """Should assign each sample to correct bin."""
        x = _float_matrix([[0.0], [0.6], [1.6], [2.6]])
        edges = (BinEdges(edges=(0.5, 1.5, 2.5)),)
        sample_bins = bin_samples(x, edges)

        n_features: int = int(sample_bins.shape[1])
        assert n_features == 1
        # Check bin assignments using column slicing
        col0 = sample_bins[:, 0]
        assert col0.item(0) == 0
        assert col0.item(1) == 1
        assert col0.item(2) == 2
        assert col0.item(3) == 3

    def test_handles_multiple_features(self) -> None:
        """Should bin each feature independently."""
        x = _float_matrix([[0.0, 2.6], [2.6, 0.0]])
        edges = (
            BinEdges(edges=(0.5, 1.5, 2.5)),
            BinEdges(edges=(0.5, 1.5, 2.5)),
        )
        sample_bins = bin_samples(x, edges)

        n_features: int = int(sample_bins.shape[1])
        assert n_features == 2
        col0 = sample_bins[:, 0]
        col1 = sample_bins[:, 1]
        # Feature 0: 0.0 -> bin 0, 2.6 -> bin 3
        assert col0.item(0) == 0
        assert col0.item(1) == 3
        # Feature 1: 2.6 -> bin 3, 0.0 -> bin 0
        assert col1.item(0) == 3
        assert col1.item(1) == 0


class TestPrecomputeFeatureBins:
    """Tests for precompute_feature_bins function."""

    def test_precomputes_bins(self) -> None:
        """Should precompute bin edges and sample assignments."""
        x = _float_matrix(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
        bins = precompute_feature_bins(x, max_bins=4)

        # Verify structure
        assert len(bins.bin_edges) == 2
        n_samples: int = int(bins.sample_bins.shape[0])
        n_features: int = int(bins.sample_bins.shape[1])
        assert n_samples == 4
        assert n_features == 2
        # Verify bin assignments are valid bin IDs
        for i in range(n_samples):
            row = bins.sample_bins[i, :]
            for j in range(n_features):
                bin_id: int = row.item(j)
                assert 0 <= bin_id <= 4  # 0-3 for regular bins, 4 for NaN bin

    def test_handles_empty_x(self) -> None:
        """Should handle empty feature matrix."""
        x_empty: NDArray[np.float64] = np.zeros((0, 0), dtype=np.float64)
        bins = precompute_feature_bins(x_empty, max_bins=4)

        assert bins.bin_edges == ()
        n_samples: int = int(bins.sample_bins.shape[0])
        n_features: int = int(bins.sample_bins.shape[1])
        assert n_samples == 0
        assert n_features == 0


class TestBuildHistogram:
    """Tests for build_histogram function."""

    def test_builds_histogram(self) -> None:
        """Should build gradient/hessian histogram."""
        sample_indices = _int_array([0, 1, 2, 3])
        gradients = _float_array([1.0, 2.0, 3.0, 4.0])
        hessians = _float_array([0.1, 0.2, 0.3, 0.4])
        sample_bins = _int_array([0, 0, 1, 1])  # Samples 0,1 in bin 0; samples 2,3 in bin 1

        histogram = build_histogram(
            sample_indices=sample_indices,
            gradients=gradients,
            hessians=hessians,
            sample_bins=sample_bins,
            n_bins=2,
        )

        assert histogram.gradient_sums_tuple() == (3.0, 7.0)  # 1+2=3, 3+4=7
        assert _approx_tuple_equal(histogram.hessian_sums_tuple(), (0.3, 0.7))  # 0.1+0.2, 0.3+0.4
        assert histogram.counts_tuple() == (2, 2)

    def test_handles_subset_of_samples(self) -> None:
        """Should only include specified sample indices."""
        sample_indices = _int_array([0, 2])  # Only samples 0 and 2
        gradients = _float_array([1.0, 2.0, 3.0, 4.0])
        hessians = _float_array([0.1, 0.2, 0.3, 0.4])
        sample_bins = _int_array([0, 0, 1, 1])

        histogram = build_histogram(
            sample_indices=sample_indices,
            gradients=gradients,
            hessians=hessians,
            sample_bins=sample_bins,
            n_bins=2,
        )

        assert histogram.gradient_sums_tuple() == (1.0, 3.0)  # Only samples 0 and 2
        assert _approx_tuple_equal(histogram.hessian_sums_tuple(), (0.1, 0.3))
        assert histogram.counts_tuple() == (1, 1)

    def test_handles_empty_bins(self) -> None:
        """Should handle bins with no samples."""
        sample_indices = _int_array([0, 1])  # Only in bin 0
        gradients = _float_array([1.0, 2.0, 3.0, 4.0])
        hessians = _float_array([0.1, 0.2, 0.3, 0.4])
        sample_bins = _int_array([0, 0, 1, 1])

        histogram = build_histogram(
            sample_indices=sample_indices,
            gradients=gradients,
            hessians=hessians,
            sample_bins=sample_bins,
            n_bins=2,
        )

        assert histogram.gradient_sums_tuple() == (3.0, 0.0)
        assert _approx_tuple_equal(histogram.hessian_sums_tuple(), (0.3, 0.0))
        assert histogram.counts_tuple() == (2, 0)


class TestSubtractHistogram:
    """Tests for subtract_histogram function."""

    def test_subtracts_histograms(self) -> None:
        """Should compute sibling = parent - child."""
        parent = HistogramBuffer.from_tuples(
            gradient_sums=(5.0, 10.0, 15.0),
            hessian_sums=(0.5, 1.0, 1.5),
            counts=(5, 10, 15),
        )
        child = HistogramBuffer.from_tuples(
            gradient_sums=(2.0, 4.0, 6.0),
            hessian_sums=(0.2, 0.4, 0.6),
            counts=(2, 4, 6),
        )

        sibling = subtract_histogram(parent, child)

        assert sibling.gradient_sums_tuple() == (3.0, 6.0, 9.0)
        assert _approx_tuple_equal(sibling.hessian_sums_tuple(), (0.3, 0.6, 0.9))
        assert sibling.counts_tuple() == (3, 6, 9)

    def test_handles_zero_child(self) -> None:
        """Should handle child with zeros (sibling = parent)."""
        parent = HistogramBuffer.from_tuples(
            gradient_sums=(5.0, 10.0),
            hessian_sums=(0.5, 1.0),
            counts=(5, 10),
        )
        child = HistogramBuffer.from_tuples(
            gradient_sums=(0.0, 0.0),
            hessian_sums=(0.0, 0.0),
            counts=(0, 0),
        )

        sibling = subtract_histogram(parent, child)

        assert sibling.gradient_sums_tuple() == parent.gradient_sums_tuple()
        assert sibling.hessian_sums_tuple() == parent.hessian_sums_tuple()
        assert sibling.counts_tuple() == parent.counts_tuple()


class TestComputeSplitGain:
    """Tests for _compute_split_gain function."""

    def test_perfect_split_positive_gain(self) -> None:
        """Perfect split should have positive gain."""
        gain = _compute_split_gain(
            g_left=-2.0,
            h_left=0.5,
            g_right=2.0,
            h_right=0.5,
            g_total=0.0,
            h_total=1.0,
        )

        assert gain > 0

    def test_no_improvement_zero_gain(self) -> None:
        """Same ratio on both sides should have near-zero gain."""
        gain = _compute_split_gain(
            g_left=1.0,
            h_left=0.5,
            g_right=1.0,
            h_right=0.5,
            g_total=2.0,
            h_total=1.0,
        )

        assert abs(gain) < 1e-10

    def test_zero_left_hessian_returns_zero(self) -> None:
        """Zero left hessian should return zero gain."""
        gain = _compute_split_gain(
            g_left=1.0,
            h_left=0.0,  # Zero hessian
            g_right=1.0,
            h_right=0.5,
            g_total=2.0,
            h_total=0.5,
        )

        assert gain == 0.0

    def test_zero_right_hessian_returns_zero(self) -> None:
        """Zero right hessian should return zero gain."""
        gain = _compute_split_gain(
            g_left=1.0,
            h_left=0.5,
            g_right=1.0,
            h_right=0.0,  # Zero hessian
            g_total=2.0,
            h_total=0.5,
        )

        assert gain == 0.0

    def test_zero_total_hessian_returns_zero(self) -> None:
        """Zero total hessian should return zero gain."""
        gain = _compute_split_gain(
            g_left=1.0,
            h_left=0.5,
            g_right=1.0,
            h_right=0.5,
            g_total=2.0,
            h_total=0.0,  # Zero total hessian
        )

        assert gain == 0.0


class TestFindBestSplitFromHistogram:
    """Tests for find_best_split_from_histogram function."""

    def test_finds_split(self) -> None:
        """Should find best split from histogram."""
        # Separable data: bin 0 has negative gradient, bin 1 has positive
        histogram = HistogramBuffer.from_tuples(
            gradient_sums=(-2.0, 2.0),
            hessian_sums=(0.5, 0.5),
            counts=(5, 5),
        )
        bin_edges = BinEdges(edges=(0.5,))

        split = find_best_split_from_histogram(
            histogram=histogram,
            bin_edges=bin_edges,
            feature_index=0,
            min_samples_leaf=1,
            monotonic_constraint=0,
        )

        # Split must be found for this separable data
        if split is None:
            raise AssertionError("Expected split to be found for separable data")
        assert split.feature_index == 0
        assert split.bin_index == 0
        assert split.threshold == 0.5
        assert split.gain > 0

    def test_returns_none_when_too_few_samples(self) -> None:
        """Should return None when fewer than 2*min_samples_leaf."""
        histogram = HistogramBuffer.from_tuples(
            gradient_sums=(-2.0, 2.0),
            hessian_sums=(0.5, 0.5),
            counts=(3, 3),  # Total 6 samples
        )
        bin_edges = BinEdges(edges=(0.5,))

        split = find_best_split_from_histogram(
            histogram=histogram,
            bin_edges=bin_edges,
            feature_index=0,
            min_samples_leaf=5,  # Need 10 total but only have 6
            monotonic_constraint=0,
        )

        assert split is None

    def test_returns_none_when_no_valid_split(self) -> None:
        """Should return None when all samples in one bin."""
        histogram = HistogramBuffer.from_tuples(
            gradient_sums=(0.0, 0.0),
            hessian_sums=(0.0, 0.0),
            counts=(0, 0),  # No samples
        )
        bin_edges = BinEdges(edges=(0.5,))

        split = find_best_split_from_histogram(
            histogram=histogram,
            bin_edges=bin_edges,
            feature_index=0,
            min_samples_leaf=1,
            monotonic_constraint=0,
        )

        assert split is None

    def test_respects_increasing_monotonicity(self) -> None:
        """Should respect increasing monotonic constraint."""
        # Gradient pattern that wants decreasing (higher left, lower right)
        # But constraint is increasing, so split should be rejected
        histogram = HistogramBuffer.from_tuples(
            gradient_sums=(-3.0, 3.0),  # Left wants higher value, right wants lower
            hessian_sums=(0.5, 0.5),
            counts=(5, 5),
        )
        bin_edges = BinEdges(edges=(0.5,))

        split = find_best_split_from_histogram(
            histogram=histogram,
            bin_edges=bin_edges,
            feature_index=0,
            min_samples_leaf=1,
            monotonic_constraint=1,  # Increasing: left <= right required
        )

        # Split should be rejected due to constraint violation
        # left_value = -(-3.0)/0.5 = 6.0
        # right_value = -3.0/0.5 = -6.0
        # 6.0 > -6.0 violates increasing constraint
        assert split is None

    def test_respects_decreasing_monotonicity(self) -> None:
        """Should respect decreasing monotonic constraint."""
        # Gradient pattern that wants increasing (lower left, higher right)
        # But constraint is decreasing, so split should be rejected
        histogram = HistogramBuffer.from_tuples(
            gradient_sums=(3.0, -3.0),  # Left wants lower value, right wants higher
            hessian_sums=(0.5, 0.5),
            counts=(5, 5),
        )
        bin_edges = BinEdges(edges=(0.5,))

        split = find_best_split_from_histogram(
            histogram=histogram,
            bin_edges=bin_edges,
            feature_index=0,
            min_samples_leaf=1,
            monotonic_constraint=-1,  # Decreasing: left >= right required
        )

        # Split should be rejected due to constraint violation
        # left_value = -3.0/0.5 = -6.0
        # right_value = -(-3.0)/0.5 = 6.0
        # -6.0 < 6.0 violates decreasing constraint
        assert split is None

    def test_allows_valid_increasing_split(self) -> None:
        """Should allow splits that satisfy increasing constraint."""
        # Left has lower value, right has higher value - satisfies increasing
        histogram = HistogramBuffer.from_tuples(
            gradient_sums=(3.0, -3.0),  # Left lower, right higher
            hessian_sums=(0.5, 0.5),
            counts=(5, 5),
        )
        bin_edges = BinEdges(edges=(0.5,))

        split = find_best_split_from_histogram(
            histogram=histogram,
            bin_edges=bin_edges,
            feature_index=0,
            min_samples_leaf=1,
            monotonic_constraint=1,  # Increasing
        )

        # left_value = -3.0/0.5 = -6.0
        # right_value = -(-3.0)/0.5 = 6.0
        # -6.0 <= 6.0 satisfies increasing constraint
        if split is None:
            raise AssertionError("Expected split satisfying increasing constraint")
        assert split.gain > 0
        assert split.feature_index == 0

    def test_allows_valid_decreasing_split(self) -> None:
        """Should allow splits that satisfy decreasing constraint."""
        # Left has higher value, right has lower value - satisfies decreasing
        histogram = HistogramBuffer.from_tuples(
            gradient_sums=(-3.0, 3.0),  # Left higher, right lower
            hessian_sums=(0.5, 0.5),
            counts=(5, 5),
        )
        bin_edges = BinEdges(edges=(0.5,))

        split = find_best_split_from_histogram(
            histogram=histogram,
            bin_edges=bin_edges,
            feature_index=0,
            min_samples_leaf=1,
            monotonic_constraint=-1,  # Decreasing
        )

        # left_value = -(-3.0)/0.5 = 6.0
        # right_value = -3.0/0.5 = -6.0
        # 6.0 >= -6.0 satisfies decreasing constraint
        if split is None:
            raise AssertionError("Expected split satisfying decreasing constraint")
        assert split.gain > 0
        assert split.feature_index == 0


class TestPartitionByBin:
    """Tests for partition_by_bin function."""

    def test_partitions_samples(self) -> None:
        """Should partition samples by bin threshold."""
        sample_indices = _int_array([0, 1, 2, 3])
        sample_bins = _int_array([0, 0, 1, 1])  # Samples 0,1 in bin 0; samples 2,3 in bin 1
        nan_bin = 2  # NaN bin after regular bins

        left, right = partition_by_bin(
            sample_indices=sample_indices,
            sample_bins=sample_bins,
            split_bin=0,  # Split after bin 0
            nan_bin=nan_bin,
            nan_direction="left",
        )

        assert np.array_equal(left, _int_array([0, 1]))  # Samples in bins <= 0
        assert np.array_equal(right, _int_array([2, 3]))  # Samples in bins > 0

    def test_handles_all_left(self) -> None:
        """Should handle all samples going left."""
        sample_indices = _int_array([0, 1, 2, 3])
        sample_bins = _int_array([0, 0, 0, 0])
        nan_bin = 2

        left, right = partition_by_bin(
            sample_indices=sample_indices,
            sample_bins=sample_bins,
            split_bin=0,
            nan_bin=nan_bin,
            nan_direction="left",
        )

        assert np.array_equal(left, _int_array([0, 1, 2, 3]))
        assert int(right.shape[0]) == 0

    def test_handles_all_right(self) -> None:
        """Should handle all samples going right."""
        sample_indices = _int_array([0, 1, 2, 3])
        sample_bins = _int_array([2, 2, 2, 2])
        nan_bin = 3

        left, right = partition_by_bin(
            sample_indices=sample_indices,
            sample_bins=sample_bins,
            split_bin=0,
            nan_bin=nan_bin,
            nan_direction="left",
        )

        assert int(left.shape[0]) == 0
        assert np.array_equal(right, _int_array([0, 1, 2, 3]))

    def test_handles_subset_of_indices(self) -> None:
        """Should only partition specified indices."""
        sample_indices = _int_array([0, 2])  # Only samples 0 and 2
        sample_bins = _int_array([0, 0, 1, 1])
        nan_bin = 2

        left, right = partition_by_bin(
            sample_indices=sample_indices,
            sample_bins=sample_bins,
            split_bin=0,
            nan_bin=nan_bin,
            nan_direction="left",
        )

        assert np.array_equal(left, _int_array([0]))  # Sample 0 in bin 0
        assert np.array_equal(right, _int_array([2]))  # Sample 2 in bin 1

    def test_nan_samples_go_left_when_direction_is_left(self) -> None:
        """NaN samples should go left when nan_direction is 'left'."""
        sample_indices = _int_array([0, 1, 2, 3])
        nan_bin = 2
        # Sample 0: bin 0, Sample 1: NaN bin, Sample 2: bin 1, Sample 3: NaN bin
        sample_bins = _int_array([0, nan_bin, 1, nan_bin])

        left, right = partition_by_bin(
            sample_indices=sample_indices,
            sample_bins=sample_bins,
            split_bin=0,
            nan_bin=nan_bin,
            nan_direction="left",
        )

        # Sample 0 (bin 0) and samples 1, 3 (NaN) go left
        assert np.array_equal(left, _int_array([0, 1, 3]))
        # Sample 2 (bin 1 > 0) goes right
        assert np.array_equal(right, _int_array([2]))

    def test_nan_samples_go_right_when_direction_is_right(self) -> None:
        """NaN samples should go right when nan_direction is 'right'."""
        sample_indices = _int_array([0, 1, 2, 3])
        nan_bin = 2
        # Sample 0: bin 0, Sample 1: NaN bin, Sample 2: bin 1, Sample 3: NaN bin
        sample_bins = _int_array([0, nan_bin, 1, nan_bin])

        left, right = partition_by_bin(
            sample_indices=sample_indices,
            sample_bins=sample_bins,
            split_bin=0,
            nan_bin=nan_bin,
            nan_direction="right",
        )

        # Sample 0 (bin 0) goes left
        assert np.array_equal(left, _int_array([0]))
        # Samples 1, 3 (NaN) and sample 2 (bin 1 > 0) go right
        assert np.array_equal(right, _int_array([1, 2, 3]))


class TestHistogramSplit:
    """Tests for HistogramSplit namedtuple."""

    def test_create_histogram_split(self) -> None:
        """Should create HistogramSplit with all fields."""
        split = HistogramSplit(
            feature_index=1,
            bin_index=2,
            threshold=0.5,
            gain=10.0,
            nan_direction="left",
        )

        assert split.feature_index == 1
        assert split.bin_index == 2
        assert split.threshold == 0.5
        assert split.gain == 10.0
        assert split.nan_direction == "left"


class TestHistogramBuffer:
    """Tests for HistogramBuffer class via from_tuples."""

    def test_create_histogram_buffer(self) -> None:
        """Should create HistogramBuffer with all fields."""
        hist = HistogramBuffer.from_tuples(
            gradient_sums=(1.0, 2.0, 3.0),
            hessian_sums=(0.1, 0.2, 0.3),
            counts=(10, 20, 30),
        )

        assert hist.gradient_sums_tuple() == (1.0, 2.0, 3.0)
        assert hist.hessian_sums_tuple() == (0.1, 0.2, 0.3)
        assert hist.counts_tuple() == (10, 20, 30)


class TestFeatureBins:
    """Tests for FeatureBins namedtuple."""

    def test_create_feature_bins(self) -> None:
        """Should create FeatureBins with all fields."""
        sample_bins = _int_matrix([[0, 1], [1, 0], [0, 1], [1, 0]])
        bins = FeatureBins(
            bin_edges=(BinEdges(edges=(0.5,)), BinEdges(edges=(0.5,))),
            sample_bins=sample_bins,
        )

        assert len(bins.bin_edges) == 2
        n_samples: int = int(bins.sample_bins.shape[0])
        n_features: int = int(bins.sample_bins.shape[1])
        assert n_samples == 4
        assert n_features == 2
        # Access 2D array elements using flat indexing via item()
        # Row 0: [0, 1], so flat index 0 = 0, flat index 1 = 1
        row0 = bins.sample_bins[0, :]
        assert row0.item(0) == 0
        assert row0.item(1) == 1
