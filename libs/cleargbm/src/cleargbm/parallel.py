"""Parallel histogram building for gradient boosting trees.

Contains multiprocessing workers and parallel split finding logic.
Uses batched workers with pool initializer to minimize IPC overhead.

Architecture:
- feature_bins is set once via pool initializer (not sent per-batch)
- Workers access global _WORKER_FEATURE_BINS instead of receiving bin data
- Batches send feature indices, sample indices, gradients, hessians, and n_bins
"""

from __future__ import annotations

import os
import struct
from multiprocessing import shared_memory

from cleargbm._test_hooks import WorkerPoolProtocol
from cleargbm.histogram import (
    NAN_BIN_OFFSET,
    FeatureBins,
    Histogram,
    build_histogram,
    find_best_split_from_histogram,
    partition_by_bin,
)
from cleargbm.types import FloatArray, GradientBoostingConfig, SplitCandidate

# =============================================================================
# Worker Global State (set via pool initializer)
# =============================================================================

# Module-level storage for feature bins in worker processes.
# Set once by _worker_initializer when pool is created.
# Workers access this instead of receiving bin data via IPC.
_WORKER_FEATURE_BINS: FeatureBins | None = None


def _worker_initializer(
    bin_edges: tuple[tuple[float, ...], ...],
    sample_bins: tuple[tuple[int, ...], ...],
) -> None:
    """Initialize worker process with feature bins.

    Called once per worker when pool is created. Sets module-level
    global that workers access during histogram building.

    Args:
        bin_edges: Bin edges for each feature (as raw tuples for pickling).
        sample_bins: Per-sample bin assignments for each feature.
    """
    global _WORKER_FEATURE_BINS
    from cleargbm.histogram import BinEdges, FeatureBins

    # Reconstruct FeatureBins from raw tuples
    _WORKER_FEATURE_BINS = FeatureBins(
        bin_edges=tuple(BinEdges(edges=edges) for edges in bin_edges),
        sample_bins=sample_bins,
    )


def _resolve_n_jobs(n_jobs: int) -> int:
    """Resolve n_jobs to actual worker count.

    Args:
        n_jobs: Number of jobs (-1 = all cores, 1 = sequential, n = n workers).

    Returns:
        Actual number of workers to use.
    """
    if n_jobs == -1:
        return os.cpu_count() or 1
    return n_jobs


def _unpack_double(buf: memoryview, offset: int) -> float:
    """Unpack a double from buffer at offset.

    Uses bytes slicing and struct.unpack to avoid mypy issues with
    struct.unpack_from returning tuple[Any, ...].

    Args:
        buf: Buffer to read from.
        offset: Byte offset.

    Returns:
        The unpacked float value.
    """
    # Extract 8 bytes and unpack as double
    data = bytes(buf[offset : offset + 8])
    unpacked: tuple[float] = struct.unpack("d", data)
    return unpacked[0]


def _read_floats_from_shm(shm_name: str, n: int) -> FloatArray:
    """Read n floats from shared memory by name.

    Args:
        shm_name: Name of the shared memory block.
        n: Number of floats to read.

    Returns:
        Tuple of floats.

    Raises:
        RuntimeError: If shared memory buffer is not available.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        buf = shm.buf
        # buf is always valid after successful SharedMemory creation
        assert buf is not None, "Shared memory buffer is not available"
        result: list[float] = []
        for i in range(n):
            value = _unpack_double(buf, i * 8)
            result.append(value)
        return tuple(result)
    finally:
        shm.close()


def _build_histogram_worker_batched(
    args: tuple[
        tuple[int, ...],
        tuple[int, ...],
        str,
        str,
        int,
        tuple[int, ...],
    ],
) -> list[tuple[int, Histogram]]:
    """Build histograms for a batch of features using global feature_bins.

    Accesses _WORKER_FEATURE_BINS from pool initializer. Reads gradients
    and hessians from shared memory by name.

    Args:
        args: Tuple of (feature_indices, sample_indices, grad_shm_name,
              hess_shm_name, n_samples, n_bins_per_feature).

    Returns:
        List of (feature_index, histogram) tuples.

    Raises:
        RuntimeError: If _WORKER_FEATURE_BINS not initialized.
    """
    feat_indices, sample_indices, grad_shm_name, hess_shm_name, n_samples, batch_n_bins = args

    if _WORKER_FEATURE_BINS is None:
        raise RuntimeError("Worker not initialized: _WORKER_FEATURE_BINS is None")

    # Read gradients/hessians from shared memory
    gradients = _read_floats_from_shm(grad_shm_name, n_samples)
    hessians = _read_floats_from_shm(hess_shm_name, n_samples)

    results: list[tuple[int, Histogram]] = []
    for i, feat_idx in enumerate(feat_indices):
        histogram = build_histogram(
            sample_indices,
            gradients,
            hessians,
            _WORKER_FEATURE_BINS.sample_bins[feat_idx],
            batch_n_bins[i],
        )
        results.append((feat_idx, histogram))
    return results


def _find_best_histogram_split_with_cache(
    sample_indices: tuple[int, ...],
    gradients: FloatArray,
    hessians: FloatArray,
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    cached_histograms: dict[int, Histogram] | None,
    pool: WorkerPoolProtocol | None = None,
) -> tuple[SplitCandidate | None, dict[int, Histogram]]:
    """Find best split using histogram-based search with optional cache.

    Uses cached histograms from parent's sibling subtraction when available,
    otherwise builds histograms from scratch. Supports parallel histogram
    building via n_jobs configuration when a pool is provided.

    Args:
        sample_indices: Indices of samples in this node.
        gradients: Gradient for each sample.
        hessians: Hessian for each sample.
        feature_indices: Features to consider.
        config: Configuration.
        feature_bins: Precomputed feature bins.
        cached_histograms: Precomputed histograms from parent (sibling subtraction).
        pool: Optional worker pool for parallel processing.

    Returns:
        Tuple of (best split candidate or None, histograms built for all features).
    """
    n_jobs = _resolve_n_jobs(config["n_jobs"])

    # Sequential path (n_jobs=1, no pool, or few features)
    if n_jobs <= 1 or pool is None or len(feature_indices) < 2:
        return _find_best_histogram_split_sequential(
            sample_indices,
            gradients,
            hessians,
            feature_indices,
            config,
            feature_bins,
            cached_histograms,
        )

    # Parallel path (pool provided)
    return _find_best_histogram_split_parallel(
        sample_indices,
        gradients,
        hessians,
        feature_indices,
        config,
        feature_bins,
        cached_histograms,
        pool,
    )


def _find_best_histogram_split_sequential(
    sample_indices: tuple[int, ...],
    gradients: FloatArray,
    hessians: FloatArray,
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    cached_histograms: dict[int, Histogram] | None,
) -> tuple[SplitCandidate | None, dict[int, Histogram]]:
    """Sequential histogram split finding (n_jobs=1).

    Args:
        sample_indices: Indices of samples in this node.
        gradients: Gradient for each sample.
        hessians: Hessian for each sample.
        feature_indices: Features to consider.
        config: Configuration.
        feature_bins: Precomputed feature bins.
        cached_histograms: Precomputed histograms from parent.

    Returns:
        Tuple of (best split candidate or None, histograms built for all features).
    """
    constraints = config["monotonic_constraints"]
    best_split: SplitCandidate | None = None
    histograms: dict[int, Histogram] = {}

    for feat_idx in feature_indices:
        # n_bins includes NaN bin: regular bins + 1 for NaN
        n_edges = len(feature_bins.bin_edges[feat_idx].edges)
        n_regular_bins = n_edges + 1
        n_bins = n_regular_bins + NAN_BIN_OFFSET  # +1 for NaN bin
        nan_bin = n_regular_bins  # NaN bin is at index n_regular_bins

        # Use cached histogram if available, otherwise build from scratch
        if cached_histograms is not None and feat_idx in cached_histograms:
            histogram = cached_histograms[feat_idx]
        else:
            histogram = build_histogram(
                sample_indices, gradients, hessians, feature_bins.sample_bins[feat_idx], n_bins
            )

        # Store histogram for use in sibling subtraction
        histograms[feat_idx] = histogram

        # Find best split from histogram
        constraint = 0 if constraints is None else constraints[feat_idx]
        split_result = find_best_split_from_histogram(
            histogram,
            feature_bins.bin_edges[feat_idx],
            feat_idx,
            config["min_samples_leaf"],
            constraint,
        )

        # Deterministic tie-breaking: prefer higher gain, then lower feature_index
        if split_result is None or split_result.gain <= 0:
            continue
        if best_split is not None and split_result.gain <= best_split["gain"]:
            continue
        left_indices, right_indices = partition_by_bin(
            sample_indices,
            feature_bins.sample_bins[feat_idx],
            split_result.bin_index,
            nan_bin,
            split_result.nan_direction,
        )
        best_split = SplitCandidate(
            feature_index=split_result.feature_index,
            threshold=split_result.threshold,
            gain=split_result.gain,
            left_indices=left_indices,
            right_indices=right_indices,
            nan_direction=split_result.nan_direction,
        )

    return best_split, histograms


def _build_batched_args(
    uncached_features: list[int],
    sample_indices: tuple[int, ...],
    grad_shm_name: str,
    hess_shm_name: str,
    n_samples: int,
    feature_bins: FeatureBins,
    n_jobs: int,
) -> list[
    tuple[
        tuple[int, ...],
        tuple[int, ...],
        str,
        str,
        int,
        tuple[int, ...],
    ]
]:
    """Build batched arguments for parallel histogram workers.

    Does NOT include sample_bins - workers access that via global
    _WORKER_FEATURE_BINS set by pool initializer.

    Args:
        uncached_features: Feature indices to process.
        sample_indices: Indices of samples in this node.
        grad_shm_name: Shared memory name for gradients.
        hess_shm_name: Shared memory name for hessians.
        n_samples: Number of samples (for reading shared memory).
        feature_bins: Precomputed feature bins (for n_bins calculation only).
        n_jobs: Number of workers.

    Returns:
        List of batched argument tuples.
    """
    # Compute n_bins for each uncached feature
    n_bins_list: list[int] = []
    for feat_idx in uncached_features:
        n_edges = len(feature_bins.bin_edges[feat_idx].edges)
        n_bins = n_edges + 1 + NAN_BIN_OFFSET
        n_bins_list.append(n_bins)

    # Split features into batches (one batch per worker)
    n_features = len(uncached_features)
    batch_size = max(1, (n_features + n_jobs - 1) // n_jobs)

    batched_args: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            str,
            str,
            int,
            tuple[int, ...],
        ]
    ] = []

    for batch_start in range(0, n_features, batch_size):
        batch_end = min(batch_start + batch_size, n_features)
        batch_feat_indices = tuple(uncached_features[batch_start:batch_end])
        batch_n_bins = tuple(n_bins_list[batch_start:batch_end])

        # Pass shared memory names instead of arrays
        batch_args = (
            batch_feat_indices,
            sample_indices,
            grad_shm_name,
            hess_shm_name,
            n_samples,
            batch_n_bins,
        )
        batched_args.append(batch_args)

    return batched_args


def _find_best_histogram_split_parallel(
    sample_indices: tuple[int, ...],
    gradients: FloatArray,
    hessians: FloatArray,
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    cached_histograms: dict[int, Histogram] | None,
    pool: WorkerPoolProtocol,
) -> tuple[SplitCandidate | None, dict[int, Histogram]]:
    """Parallel histogram split finding using batched workers.

    Uses batched workers with shared memory: gradients/hessians are written
    to shared memory once, and workers read by name. This avoids sending
    large arrays via IPC for each batch.

    Args:
        sample_indices: Indices of samples in this node.
        gradients: Gradient for each sample.
        hessians: Hessian for each sample.
        feature_indices: Features to consider.
        config: Configuration.
        feature_bins: Precomputed feature bins.
        cached_histograms: Precomputed histograms from parent.
        pool: Worker pool for parallel processing.

    Returns:
        Tuple of (best split candidate or None, histograms built for all features).
    """
    histograms: dict[int, Histogram] = {}
    n_jobs = _resolve_n_jobs(config["n_jobs"])

    # Separate features into cached and uncached
    uncached_features: list[int] = []
    for feat_idx in feature_indices:
        if cached_histograms is not None and feat_idx in cached_histograms:
            histograms[feat_idx] = cached_histograms[feat_idx]
        else:
            uncached_features.append(feat_idx)

    # Build histograms for uncached features using batched workers
    if uncached_features:
        n_samples = len(gradients)
        # Create shared memory for gradients and hessians
        shm_grad = shared_memory.SharedMemory(create=True, size=n_samples * 8)
        shm_hess = shared_memory.SharedMemory(create=True, size=n_samples * 8)
        try:
            # Write floats to shared memory using struct.pack_into
            grad_buf = shm_grad.buf
            hess_buf = shm_hess.buf
            # Buffers are always valid after successful SharedMemory creation
            assert grad_buf is not None and hess_buf is not None
            for i in range(n_samples):
                struct.pack_into("d", grad_buf, i * 8, gradients[i])
                struct.pack_into("d", hess_buf, i * 8, hessians[i])

            batched_args = _build_batched_args(
                uncached_features,
                sample_indices,
                shm_grad.name,
                shm_hess.name,
                n_samples,
                feature_bins,
                n_jobs,
            )
            batch_results = pool.map_batched(_build_histogram_worker_batched, batched_args)

            # Flatten results
            for batch_result in batch_results:
                for feat_idx, histogram in batch_result:
                    histograms[feat_idx] = histogram
        finally:
            # Clean up shared memory
            shm_grad.close()
            shm_hess.close()
            shm_grad.unlink()
            shm_hess.unlink()

    # Find best split from all histograms
    best_split = _select_best_split(
        sample_indices, feature_indices, config, feature_bins, histograms
    )

    return best_split, histograms


def _select_best_split(
    sample_indices: tuple[int, ...],
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    histograms: dict[int, Histogram],
) -> SplitCandidate | None:
    """Select best split from histograms with deterministic tie-breaking.

    Args:
        sample_indices: Indices of samples in this node.
        feature_indices: Features to consider.
        config: Configuration.
        feature_bins: Precomputed feature bins.
        histograms: Histograms for each feature.

    Returns:
        Best split candidate or None.
    """
    constraints = config["monotonic_constraints"]
    sorted_features = sorted(feature_indices)
    best_split: SplitCandidate | None = None

    for feat_idx in sorted_features:
        n_edges = len(feature_bins.bin_edges[feat_idx].edges)
        nan_bin = n_edges + 1

        histogram = histograms[feat_idx]
        constraint = 0 if constraints is None else constraints[feat_idx]
        split_result = find_best_split_from_histogram(
            histogram,
            feature_bins.bin_edges[feat_idx],
            feat_idx,
            config["min_samples_leaf"],
            constraint,
        )

        if split_result is None or split_result.gain <= 0:
            continue
        if best_split is not None and split_result.gain <= best_split["gain"]:
            continue

        left_indices, right_indices = partition_by_bin(
            sample_indices,
            feature_bins.sample_bins[feat_idx],
            split_result.bin_index,
            nan_bin,
            split_result.nan_direction,
        )
        best_split = SplitCandidate(
            feature_index=split_result.feature_index,
            threshold=split_result.threshold,
            gain=split_result.gain,
            left_indices=left_indices,
            right_indices=right_indices,
            nan_direction=split_result.nan_direction,
        )

    return best_split


def _find_best_histogram_split(
    sample_indices: tuple[int, ...],
    gradients: FloatArray,
    hessians: FloatArray,
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    feature_names: tuple[str, ...],
) -> SplitCandidate | None:
    """Find best split using histogram-based search.

    Args:
        sample_indices: Indices of samples in this node.
        gradients: Gradient for each sample.
        hessians: Hessian for each sample.
        feature_indices: Features to consider.
        config: Configuration.
        feature_bins: Precomputed feature bins.
        feature_names: Feature names (unused, kept for compatibility).

    Returns:
        Best split candidate or None.
    """
    split, _ = _find_best_histogram_split_with_cache(
        sample_indices,
        gradients,
        hessians,
        feature_indices,
        config,
        feature_bins,
        None,
        None,
    )
    return split


__all__ = [
    "_build_batched_args",
    "_build_histogram_worker_batched",
    "_find_best_histogram_split",
    "_find_best_histogram_split_parallel",
    "_find_best_histogram_split_sequential",
    "_find_best_histogram_split_with_cache",
    "_read_floats_from_shm",
    "_resolve_n_jobs",
    "_select_best_split",
    "_worker_initializer",
]
