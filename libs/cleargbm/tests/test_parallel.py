"""Tests for cleargbm.parallel module.

Parallel histogram building with batched workers.
"""

from __future__ import annotations

from collections.abc import Callable

import cleargbm.parallel as parallel_module
from cleargbm.histogram import (
    BinEdges,
    FeatureBins,
    Histogram,
    build_histogram,
    precompute_feature_bins,
)
from cleargbm.parallel import (
    _build_histogram_worker_batched,
    _find_best_histogram_split,
    _find_best_histogram_split_parallel,
    _find_best_histogram_split_sequential,
    _find_best_histogram_split_with_cache,
    _resolve_n_jobs,
    _worker_initializer,
)
from cleargbm.tree import build_tree
from cleargbm.types import FloatArray

from .conftest import make_config


class _FakeSequentialPool:
    """Fake pool that processes batched items sequentially for testing.

    Conforms to WorkerPoolProtocol without multiprocessing overhead.
    Actually executes functions - not a mock.

    Sets up _WORKER_FEATURE_BINS global before calling workers, mimicking
    the pool initializer behavior of the real multiprocessing pool.
    """

    def __init__(
        self,
        bin_edges: tuple[tuple[float, ...], ...],
        sample_bins: tuple[tuple[int, ...], ...],
    ) -> None:
        """Initialize with feature bins data.

        Args:
            bin_edges: Bin edges for each feature.
            sample_bins: Per-sample bin assignments for each feature.
        """
        self._bin_edges = bin_edges
        self._sample_bins = sample_bins

    def map_batched(
        self,
        func: Callable[
            [
                tuple[
                    tuple[int, ...],
                    tuple[int, ...],
                    str,
                    str,
                    int,
                    tuple[int, ...],
                ]
            ],
            list[tuple[int, Histogram]],
        ],
        args_list: list[
            tuple[
                tuple[int, ...],
                tuple[int, ...],
                str,
                str,
                int,
                tuple[int, ...],
            ]
        ],
    ) -> list[list[tuple[int, Histogram]]]:
        """Process batched items sequentially.

        Sets up the global _WORKER_FEATURE_BINS before calling workers,
        mimicking the pool initializer behavior.

        Args:
            func: Batched function to apply.
            args_list: Batched arguments with shared memory names.

        Returns:
            List of lists of results.
        """
        # Set up global like the real pool initializer does
        parallel_module._WORKER_FEATURE_BINS = FeatureBins(
            bin_edges=tuple(BinEdges(edges=edges) for edges in self._bin_edges),
            sample_bins=self._sample_bins,
        )
        try:
            return [func(args) for args in args_list]
        finally:
            # Clean up global after use
            parallel_module._WORKER_FEATURE_BINS = None

    def close(self) -> None:
        """No-op for fake pool."""

    def join(self) -> None:
        """No-op for fake pool."""


class TestResolveNJobs:
    """Tests for _resolve_n_jobs."""

    def test_returns_value_for_positive(self) -> None:
        """Should return the value for positive n_jobs."""
        assert _resolve_n_jobs(1) == 1
        assert _resolve_n_jobs(4) == 4
        assert _resolve_n_jobs(8) == 8

    def test_returns_cpu_count_for_minus_one(self) -> None:
        """Should return cpu_count for n_jobs=-1."""
        import os

        expected = os.cpu_count() or 1
        assert _resolve_n_jobs(-1) == expected


class TestWorkerInitializer:
    """Tests for _worker_initializer."""

    def test_sets_global_feature_bins(self) -> None:
        """Should set _WORKER_FEATURE_BINS global from raw tuples."""
        bin_edges = ((0.5, 1.5), (0.25, 0.75))
        sample_bins = ((0, 1, 2, 0), (0, 1, 0, 1))

        # Ensure global is None before
        parallel_module._WORKER_FEATURE_BINS = None

        try:
            _worker_initializer(bin_edges, sample_bins)

            # Verify global was set
            fb = parallel_module._WORKER_FEATURE_BINS
            if fb is None:
                raise AssertionError("Expected _WORKER_FEATURE_BINS to be set")

            # Verify bin_edges were reconstructed correctly
            assert len(fb.bin_edges) == 2
            assert fb.bin_edges[0].edges == (0.5, 1.5)
            assert fb.bin_edges[1].edges == (0.25, 0.75)

            # Verify sample_bins were set correctly
            assert fb.sample_bins == sample_bins
        finally:
            parallel_module._WORKER_FEATURE_BINS = None

    def test_reconstructs_bin_edges_namedtuple(self) -> None:
        """Should reconstruct BinEdges NamedTuples from raw tuples."""
        bin_edges = ((1.0,),)
        sample_bins = ((0, 1),)

        parallel_module._WORKER_FEATURE_BINS = None
        try:
            _worker_initializer(bin_edges, sample_bins)

            fb = parallel_module._WORKER_FEATURE_BINS
            if fb is None:
                raise AssertionError("Expected _WORKER_FEATURE_BINS to be set")

            # Verify it's a proper BinEdges NamedTuple by accessing the edges field
            # This will raise AttributeError if not a proper NamedTuple
            first_bin_edge = fb.bin_edges[0]
            assert first_bin_edge.edges == (1.0,)
        finally:
            parallel_module._WORKER_FEATURE_BINS = None


class TestBuildHistogramWorkerBatched:
    """Tests for _build_histogram_worker_batched."""

    def test_builds_histograms_for_batch(self) -> None:
        """Should build histograms for a batch of features."""
        import struct
        from multiprocessing import shared_memory

        gradients: FloatArray = (-1.0, -1.0, 1.0, 1.0)
        hessians: FloatArray = (0.25, 0.25, 0.25, 0.25)
        sample_indices = (0, 1, 2, 3)
        sample_bins_f0 = (0, 0, 1, 1)  # 2 bins for feature 0
        sample_bins_f1 = (0, 1, 0, 1)  # 2 bins for feature 1
        n_bins_f0 = 2
        n_bins_f1 = 2
        n_samples = len(gradients)

        # Create shared memory for gradients/hessians
        shm_grad = shared_memory.SharedMemory(create=True, size=n_samples * 8)
        shm_hess = shared_memory.SharedMemory(create=True, size=n_samples * 8)
        try:
            grad_buf = shm_grad.buf
            hess_buf = shm_hess.buf
            if grad_buf is None or hess_buf is None:
                raise RuntimeError("Buffer not available")
            for i in range(n_samples):
                struct.pack_into("d", grad_buf, i * 8, gradients[i])
                struct.pack_into("d", hess_buf, i * 8, hessians[i])

            # Set up global feature_bins (mimics pool initializer)
            parallel_module._WORKER_FEATURE_BINS = FeatureBins(
                bin_edges=(BinEdges(edges=(0.5,)), BinEdges(edges=(0.5,))),
                sample_bins=(sample_bins_f0, sample_bins_f1),
            )
            try:
                args = (
                    (0, 1),  # feature indices
                    sample_indices,
                    shm_grad.name,
                    shm_hess.name,
                    n_samples,
                    (n_bins_f0, n_bins_f1),  # n_bins per feature
                )
                results = _build_histogram_worker_batched(args)

                assert len(results) == 2
                feat_idx_0, histogram_0 = results[0]
                feat_idx_1, histogram_1 = results[1]

                assert feat_idx_0 == 0
                assert feat_idx_1 == 1
                assert len(histogram_0.gradient_sums) == 2
                assert len(histogram_1.gradient_sums) == 2
            finally:
                parallel_module._WORKER_FEATURE_BINS = None
        finally:
            shm_grad.close()
            shm_hess.close()
            shm_grad.unlink()
            shm_hess.unlink()

    def test_builds_single_feature_batch(self) -> None:
        """Should handle batch with single feature."""
        import struct
        from multiprocessing import shared_memory

        gradients: FloatArray = (-1.0, 1.0)
        hessians: FloatArray = (0.5, 0.5)
        sample_indices = (0, 1)
        sample_bins = (0, 1)
        n_bins = 2
        n_samples = len(gradients)

        # Create shared memory for gradients/hessians
        shm_grad = shared_memory.SharedMemory(create=True, size=n_samples * 8)
        shm_hess = shared_memory.SharedMemory(create=True, size=n_samples * 8)
        try:
            grad_buf = shm_grad.buf
            hess_buf = shm_hess.buf
            if grad_buf is None or hess_buf is None:
                raise RuntimeError("Buffer not available")
            for i in range(n_samples):
                struct.pack_into("d", grad_buf, i * 8, gradients[i])
                struct.pack_into("d", hess_buf, i * 8, hessians[i])

            # Set up global feature_bins (mimics pool initializer)
            parallel_module._WORKER_FEATURE_BINS = FeatureBins(
                bin_edges=(BinEdges(edges=(0.5,)),),
                sample_bins=(sample_bins,),
            )
            try:
                args = (
                    (0,),  # single feature
                    sample_indices,
                    shm_grad.name,
                    shm_hess.name,
                    n_samples,
                    (n_bins,),  # n_bins per feature
                )
                results = _build_histogram_worker_batched(args)

                assert len(results) == 1
                feat_idx, histogram = results[0]
                assert feat_idx == 0
                assert len(histogram.gradient_sums) == 2
            finally:
                parallel_module._WORKER_FEATURE_BINS = None
        finally:
            shm_grad.close()
            shm_hess.close()
            shm_grad.unlink()
            shm_hess.unlink()

    def test_raises_when_global_not_initialized(self) -> None:
        """Should raise RuntimeError when _WORKER_FEATURE_BINS is None."""
        import struct
        from multiprocessing import shared_memory

        # Ensure global is None
        parallel_module._WORKER_FEATURE_BINS = None

        gradients: FloatArray = (0.0, 0.0)
        hessians: FloatArray = (0.5, 0.5)
        n_samples = len(gradients)

        shm_grad = shared_memory.SharedMemory(create=True, size=n_samples * 8)
        shm_hess = shared_memory.SharedMemory(create=True, size=n_samples * 8)
        try:
            grad_buf = shm_grad.buf
            hess_buf = shm_hess.buf
            if grad_buf is None or hess_buf is None:
                raise RuntimeError("Buffer not available")
            for i in range(n_samples):
                struct.pack_into("d", grad_buf, i * 8, gradients[i])
                struct.pack_into("d", hess_buf, i * 8, hessians[i])

            args = (
                (0,),
                (0, 1),
                shm_grad.name,
                shm_hess.name,
                n_samples,
                (2,),
            )

            raised = False
            try:
                _build_histogram_worker_batched(args)
            except RuntimeError as e:
                raised = True
                assert "Worker not initialized" in str(e)
            if not raised:
                raise AssertionError("Expected RuntimeError")
        finally:
            shm_grad.close()
            shm_hess.close()
            shm_grad.unlink()
            shm_hess.unlink()


class TestFindBestHistogramSplit:
    """Tests for _find_best_histogram_split."""

    def test_finds_best_split(self) -> None:
        """Should find best split using histogram approach."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (0.0,),
            (1.0,),
            (1.0,),
        )
        gradients = (-1.0, -1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)
        config = make_config(max_depth=2, min_samples_leaf=1)

        feature_bins = precompute_feature_bins(x, config["max_bins"])
        sample_indices = (0, 1, 2, 3)

        split = _find_best_histogram_split(
            sample_indices=sample_indices,
            gradients=gradients,
            hessians=hessians,
            feature_indices=(0,),
            config=config,
            feature_bins=feature_bins,
            feature_names=("f0",),
        )

        # Should find a valid split - verify split is SplitCandidate by accessing typed fields
        if split is None:
            raise AssertionError("Expected to find a split")
        assert split["gain"] > 0
        assert split["feature_index"] == 0

    def test_returns_none_when_no_valid_split(self) -> None:
        """Should return None when no valid split exists."""
        x: tuple[tuple[float, ...], ...] = (
            (1.0,),
            (1.0,),
        )
        gradients = (-1.0, 1.0)
        hessians = (0.25, 0.25)
        config = make_config(min_samples_leaf=2)

        feature_bins = precompute_feature_bins(x, config["max_bins"])
        sample_indices = (0, 1)

        split = _find_best_histogram_split(
            sample_indices=sample_indices,
            gradients=gradients,
            hessians=hessians,
            feature_indices=(0,),
            config=config,
            feature_bins=feature_bins,
            feature_names=("f0",),
        )

        # No split should be found
        assert split is None


class TestFindBestHistogramSplitWithCache:
    """Tests for _find_best_histogram_split_with_cache."""

    def test_uses_cached_histogram(self) -> None:
        """Should use cached histogram when available."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0,),
            (0.0,),
            (1.0,),
            (1.0,),
        )
        gradients = (-1.0, -1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)
        config = make_config(max_depth=2, min_samples_leaf=1)

        feature_bins = precompute_feature_bins(x, config["max_bins"])
        sample_indices = (0, 1, 2, 3)

        # First call to get histograms
        _, histograms = _find_best_histogram_split_with_cache(
            sample_indices=sample_indices,
            gradients=gradients,
            hessians=hessians,
            feature_indices=(0,),
            config=config,
            feature_bins=feature_bins,
            cached_histograms=None,
        )

        # Histograms should be populated - verify by accessing specific fields
        assert 0 in histograms
        # Verify histogram has bins (number depends on unique quantiles in data)
        n_bins = len(histograms[0].gradient_sums)
        assert n_bins >= 2  # At least 2 bins for this data

        # Second call with cached histograms should still work
        split, _ = _find_best_histogram_split_with_cache(
            sample_indices=sample_indices,
            gradients=gradients,
            hessians=hessians,
            feature_indices=(0,),
            config=config,
            feature_bins=feature_bins,
            cached_histograms=histograms,
        )

        if split is None:
            raise AssertionError("Expected to find a split with cached histograms")
        assert split["gain"] > 0

    def test_builds_histogram_for_uncached_feature(self) -> None:
        """Should build histogram for features not in cache."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0, 0.5),
            (0.0, 0.5),
            (1.0, 0.5),
            (1.0, 0.5),
        )
        gradients = (-1.0, -1.0, 1.0, 1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)
        config = make_config(max_depth=2, min_samples_leaf=1)

        feature_bins = precompute_feature_bins(x, config["max_bins"])
        sample_indices = (0, 1, 2, 3)

        # Create cache with only feature 0
        cached = {
            0: Histogram(
                gradient_sums=(0.0,) * 64,
                hessian_sums=(0.0,) * 64,
                counts=(0,) * 64,
            ),
        }

        # Search both features - feature 1 not in cache
        _, histograms = _find_best_histogram_split_with_cache(
            sample_indices=sample_indices,
            gradients=gradients,
            hessians=hessians,
            feature_indices=(0, 1),
            config=config,
            feature_bins=feature_bins,
            cached_histograms=cached,
        )

        # Both features should be in returned histograms
        assert 0 in histograms
        assert 1 in histograms


class TestParallelHistogramSplit:
    """Tests for parallel histogram split finding."""

    def test_parallel_matches_sequential(self) -> None:
        """Parallel and sequential should give identical results."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        )
        gradients: FloatArray = (-1.0, -0.5, 0.5, 1.0)
        hessians: FloatArray = (0.25, 0.25, 0.25, 0.25)
        sample_indices = (0, 1, 2, 3)

        config_seq = make_config(min_samples_leaf=1, n_jobs=1)
        config_par = make_config(min_samples_leaf=1, n_jobs=2)

        feature_bins = precompute_feature_bins(x, config_seq["max_bins"])
        feature_indices = (0, 1)

        # Sequential
        split_seq, _histograms_seq = _find_best_histogram_split_sequential(
            sample_indices,
            gradients,
            hessians,
            feature_indices,
            config_seq,
            feature_bins,
            None,
        )

        # Parallel - create pool with feature_bins data
        bin_edges_raw = tuple(be.edges for be in feature_bins.bin_edges)
        pool = _FakeSequentialPool(bin_edges_raw, feature_bins.sample_bins)
        split_par, _histograms_par = _find_best_histogram_split_parallel(
            sample_indices,
            gradients,
            hessians,
            feature_indices,
            config_par,
            feature_bins,
            None,
            pool,
        )

        # Both should find a split or both should be None
        if split_seq is None:
            assert split_par is None
        else:
            # Verify parallel found same split as sequential
            if split_par is None:
                raise AssertionError("Expected parallel to find a split")
            assert split_seq["feature_index"] == split_par["feature_index"]
            assert abs(split_seq["gain"] - split_par["gain"]) < 1e-10
            assert abs(split_seq["threshold"] - split_par["threshold"]) < 1e-10

    def test_parallel_uses_cached_histograms(self) -> None:
        """Parallel should use cached histograms from sibling subtraction."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        )
        gradients: FloatArray = (-1.0, -0.5, 0.5, 1.0)
        hessians: FloatArray = (0.25, 0.25, 0.25, 0.25)
        sample_indices = (0, 1, 2, 3)

        config = make_config(min_samples_leaf=1, n_jobs=2)
        feature_bins = precompute_feature_bins(x, config["max_bins"])

        # Pre-build cached histograms
        cached_hist_0 = build_histogram(
            sample_indices,
            gradients,
            hessians,
            feature_bins.sample_bins[0],
            len(feature_bins.bin_edges[0].edges) + 1,
        )
        cached_histograms = {0: cached_hist_0}

        # Create pool with feature_bins data
        bin_edges_raw = tuple(be.edges for be in feature_bins.bin_edges)
        pool = _FakeSequentialPool(bin_edges_raw, feature_bins.sample_bins)

        # With cached, should still work
        _split, histograms = _find_best_histogram_split_parallel(
            sample_indices,
            gradients,
            hessians,
            (0, 1),
            config,
            feature_bins,
            cached_histograms,
            pool,
        )

        # Should have histograms for both features
        assert 0 in histograms
        assert 1 in histograms
        # Feature 0 should be the cached histogram
        assert histograms[0] is cached_hist_0

    def test_parallel_with_no_valid_splits(self) -> None:
        """Parallel should return None when no valid splits exist."""
        # All same values - no valid splits
        x: tuple[tuple[float, ...], ...] = (
            (0.5, 0.5),
            (0.5, 0.5),
            (0.5, 0.5),
            (0.5, 0.5),
        )
        gradients: FloatArray = (0.0, 0.0, 0.0, 0.0)
        hessians: FloatArray = (0.25, 0.25, 0.25, 0.25)
        sample_indices = (0, 1, 2, 3)

        config = make_config(min_samples_leaf=1, n_jobs=2)
        feature_bins = precompute_feature_bins(x, config["max_bins"])

        # Create pool with feature_bins data
        bin_edges_raw = tuple(be.edges for be in feature_bins.bin_edges)
        pool = _FakeSequentialPool(bin_edges_raw, feature_bins.sample_bins)

        split, _histograms = _find_best_histogram_split_parallel(
            sample_indices,
            gradients,
            hessians,
            (0, 1),
            config,
            feature_bins,
            None,
            pool,
        )

        assert split is None

    def test_dispatch_uses_parallel_when_pool_provided(self) -> None:
        """_find_best_histogram_split_with_cache should use parallel when pool provided."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        )
        gradients: FloatArray = (-1.0, -0.5, 0.5, 1.0)
        hessians: FloatArray = (0.25, 0.25, 0.25, 0.25)
        sample_indices = (0, 1, 2, 3)

        config = make_config(min_samples_leaf=1, n_jobs=2)
        feature_bins = precompute_feature_bins(x, config["max_bins"])

        # Create pool with feature_bins data
        bin_edges_raw = tuple(be.edges for be in feature_bins.bin_edges)
        pool = _FakeSequentialPool(bin_edges_raw, feature_bins.sample_bins)

        # This should dispatch to parallel path when pool is provided
        split, histograms = _find_best_histogram_split_with_cache(
            sample_indices,
            gradients,
            hessians,
            (0, 1),
            config,
            feature_bins,
            None,
            pool,
        )

        # Should find a valid split
        if split is not None:
            assert split["gain"] > 0
        # Should have histograms for both features
        assert 0 in histograms
        assert 1 in histograms

    def test_dispatch_uses_sequential_when_no_pool(self) -> None:
        """_find_best_histogram_split_with_cache uses sequential when no pool provided."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        )
        gradients: FloatArray = (-1.0, -0.5, 0.5, 1.0)
        hessians: FloatArray = (0.25, 0.25, 0.25, 0.25)
        sample_indices = (0, 1, 2, 3)

        config = make_config(min_samples_leaf=1, n_jobs=2)
        feature_bins = precompute_feature_bins(x, config["max_bins"])

        # Without pool, should use sequential path even with n_jobs > 1
        split, histograms = _find_best_histogram_split_with_cache(
            sample_indices,
            gradients,
            hessians,
            (0, 1),
            config,
            feature_bins,
            None,
            None,  # No pool provided
        )

        # Should still find a valid split
        if split is not None:
            assert split["gain"] > 0
        # Should have histograms for both features
        assert 0 in histograms
        assert 1 in histograms


class TestBuildTreeWithNJobs:
    """Tests for build_tree with n_jobs."""

    def test_build_tree_with_parallel(self) -> None:
        """Should build tree with parallel histogram computation."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0, 0.0),
            (0.0, 1.0),
            (0.0, 2.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (1.0, 2.0),
        )
        gradients = (-1.0, -0.5, -0.3, 0.3, 0.5, 1.0)
        hessians = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
        config = make_config(max_depth=2, min_samples_leaf=1, n_jobs=2)

        tree = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config,
            feature_names=("f0", "f1"),
        )

        assert len(tree["nodes"]) >= 3
        assert tree["n_leaves"] >= 2

    def test_build_tree_deterministic_with_parallel(self) -> None:
        """Should produce same tree with n_jobs=1 and n_jobs=2."""
        x: tuple[tuple[float, ...], ...] = (
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        )
        gradients = (-1.0, -0.5, 0.5, 1.0)
        hessians = (0.25, 0.25, 0.25, 0.25)

        config_seq = make_config(max_depth=2, min_samples_leaf=1, n_jobs=1, random_state=42)
        config_par = make_config(max_depth=2, min_samples_leaf=1, n_jobs=2, random_state=42)

        tree_seq = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config_seq,
            feature_names=("f0", "f1"),
        )

        tree_par = build_tree(
            x=x,
            gradients=gradients,
            hessians=hessians,
            config=config_par,
            feature_names=("f0", "f1"),
        )

        # Trees should have same structure
        assert len(tree_seq["nodes"]) == len(tree_par["nodes"])
        assert tree_seq["n_leaves"] == tree_par["n_leaves"]
        assert tree_seq["max_depth"] == tree_par["max_depth"]

        # Nodes should have same values
        for node_seq, node_par in zip(tree_seq["nodes"], tree_par["nodes"], strict=True):
            assert node_seq["is_leaf"] == node_par["is_leaf"]
            if not node_seq["is_leaf"]:
                assert node_seq["feature_index"] == node_par["feature_index"]
                thresh_seq = node_seq["threshold"]
                thresh_par = node_par["threshold"]
                if thresh_seq is None or thresh_par is None:
                    raise AssertionError("Internal node should have threshold")
                assert abs(thresh_seq - thresh_par) < 1e-10
