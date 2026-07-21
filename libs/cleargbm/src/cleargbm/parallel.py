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
from multiprocessing import shared_memory

import numpy as np
from numpy.typing import NDArray

from cleargbm._hooks_infra import WorkerPoolProtocol
from cleargbm.buffers import HistogramBuffer
from cleargbm.histogram import (
    NAN_BIN_OFFSET,
    build_histogram,
    find_best_split_from_histogram,
    partition_by_bin,
)
from cleargbm.types import FeatureBins, GradientBoostingConfig, SplitCandidate

# =============================================================================
# Worker Global State (set via pool initializer)
# =============================================================================

# Module-level storage for feature bins in worker processes.
# Set once by _worker_initializer when pool is created.
# Workers access this instead of receiving bin data via IPC.
_WORKER_FEATURE_BINS: FeatureBins | None = None


def _set_worker_feature_bins_for_test(bins: FeatureBins) -> None:
    """Set the worker-global FeatureBins from a test.

    Complements ``_worker_initializer`` — the real code path sets this global
    inside a fresh worker process. Tests that exercise the worker functions in
    the parent process (via a fake pool) need to seed the global first, then
    reset it in a ``finally`` block. This function is the DI-friendly seam so
    tests do not need to mutate the module attribute directly.
    """
    global _WORKER_FEATURE_BINS
    _WORKER_FEATURE_BINS = bins


def _reset_worker_feature_bins_for_test() -> None:
    """Clear the worker-global FeatureBins after a test.

    Pairs with ``_set_worker_feature_bins_for_test`` — call in a ``finally``
    block so a failing assertion never leaks worker-global state into the next
    test.
    """
    global _WORKER_FEATURE_BINS
    _WORKER_FEATURE_BINS = None


def _worker_initializer(
    bin_edges: tuple[tuple[float, ...], ...],
    sample_bins_flat: bytes,
    n_samples: int,
    n_features: int,
) -> None:
    """Initialize worker process with feature bins.

    Called once per worker when pool is created. Sets module-level
    global that workers access during histogram building.

    Args:
        bin_edges: Bin edges for each feature (as raw tuples for pickling).
        sample_bins_flat: Flattened sample bins as bytes (for efficient IPC).
        n_samples: Number of samples.
        n_features: Number of features.
    """
    global _WORKER_FEATURE_BINS
    from cleargbm.types import BinEdges, FeatureBins

    # Reconstruct sample_bins as 2D numpy array from bytes
    sample_bins_arr: NDArray[np.int64] = np.frombuffer(sample_bins_flat, dtype=np.int64).reshape(
        (n_samples, n_features)
    )

    # Reconstruct FeatureBins from raw tuples
    _WORKER_FEATURE_BINS = FeatureBins(
        bin_edges=tuple(BinEdges(edges=edges) for edges in bin_edges),
        sample_bins=sample_bins_arr.copy(),  # Make a copy to own the data
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


def _read_floats_from_shm(shm_name: str, n: int) -> NDArray[np.float64]:
    """Read n floats from shared memory by name.

    Args:
        shm_name: Name of the shared memory block.
        n: Number of floats to read.

    Returns:
        Numpy array of floats.

    Raises:
        RuntimeError: If shared memory buffer is not available.
    """
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        buf = shm.buf
        # buf is always valid after successful SharedMemory creation
        assert buf is not None, "Shared memory buffer is not available"
        # Read directly into numpy array using frombuffer
        arr: NDArray[np.float64] = np.frombuffer(
            bytes(buf[: n * 8]), dtype=np.float64
        ).copy()  # Copy to own the data
        return arr
    finally:
        shm.close()


def _build_histogram_worker_batched(
    args: tuple[
        tuple[int, ...],
        bytes,  # sample_indices as bytes (int64)
        int,  # n_indices
        str,
        str,
        int,
        tuple[int, ...],
    ],
) -> list[tuple[int, HistogramBuffer]]:
    """Build histograms for a batch of features using global feature_bins.

    Accesses _WORKER_FEATURE_BINS from pool initializer. Reads gradients
    and hessians from shared memory by name.

    Args:
        args: Tuple of (feature_indices, sample_indices_bytes, n_indices,
              grad_shm_name, hess_shm_name, n_samples, n_bins_per_feature).

    Returns:
        List of (feature_index, HistogramBuffer) tuples.

    Raises:
        RuntimeError: If _WORKER_FEATURE_BINS not initialized.
    """
    (
        feat_indices,
        sample_indices_bytes,
        _n_indices,
        grad_shm_name,
        hess_shm_name,
        n_samples,
        batch_n_bins,
    ) = args

    if _WORKER_FEATURE_BINS is None:
        raise RuntimeError("Worker not initialized: _WORKER_FEATURE_BINS is None")

    # Reconstruct sample_indices as numpy array
    sample_indices: NDArray[np.int64] = np.frombuffer(sample_indices_bytes, dtype=np.int64).copy()

    # Read gradients/hessians from shared memory
    gradients = _read_floats_from_shm(grad_shm_name, n_samples)
    hessians = _read_floats_from_shm(hess_shm_name, n_samples)

    results: list[tuple[int, HistogramBuffer]] = []
    for i, feat_idx in enumerate(feat_indices):
        # Access sample_bins column for this feature
        feat_bins: NDArray[np.int64] = _WORKER_FEATURE_BINS.sample_bins[:, feat_idx]
        histogram = build_histogram(
            sample_indices,
            gradients,
            hessians,
            feat_bins,
            batch_n_bins[i],
        )
        results.append((feat_idx, histogram))
    return results


def _find_best_histogram_split_with_cache(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    cached_histograms: dict[int, HistogramBuffer] | None,
    pool: WorkerPoolProtocol | None = None,
) -> tuple[SplitCandidate | None, dict[int, HistogramBuffer]]:
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
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    cached_histograms: dict[int, HistogramBuffer] | None,
) -> tuple[SplitCandidate | None, dict[int, HistogramBuffer]]:
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
    histograms: dict[int, HistogramBuffer] = {}

    for feat_idx in feature_indices:
        # n_bins includes NaN bin: regular bins + 1 for NaN
        n_edges = len(feature_bins.bin_edges[feat_idx].edges)
        n_regular_bins = n_edges + 1
        n_bins = n_regular_bins + NAN_BIN_OFFSET  # +1 for NaN bin
        nan_bin = n_regular_bins  # NaN bin is at index n_regular_bins

        # Get sample_bins column for this feature
        feat_bins: NDArray[np.int64] = feature_bins.sample_bins[:, feat_idx]

        # Use cached histogram if available, otherwise build from scratch
        if cached_histograms is not None and feat_idx in cached_histograms:
            histogram = cached_histograms[feat_idx]
        else:
            histogram = build_histogram(sample_indices, gradients, hessians, feat_bins, n_bins)

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
            config["reg_lambda"],
        )

        # Deterministic tie-breaking: prefer higher gain, then lower feature_index
        if split_result is None or split_result.gain <= 0:
            continue
        if best_split is not None and split_result.gain <= best_split["gain"]:
            continue
        left_indices, right_indices = partition_by_bin(
            sample_indices,
            feat_bins,
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
    sample_indices: NDArray[np.int64],
    grad_shm_name: str,
    hess_shm_name: str,
    n_samples: int,
    feature_bins: FeatureBins,
    n_jobs: int,
) -> list[
    tuple[
        tuple[int, ...],
        bytes,  # sample_indices as bytes
        int,  # n_indices
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
    # Convert sample_indices to bytes for IPC
    sample_indices_bytes: bytes = sample_indices.tobytes()
    n_indices = sample_indices.shape[0]

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
            bytes,
            int,
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

        # Pass shared memory names and sample indices as bytes
        batch_args = (
            batch_feat_indices,
            sample_indices_bytes,
            n_indices,
            grad_shm_name,
            hess_shm_name,
            n_samples,
            batch_n_bins,
        )
        batched_args.append(batch_args)

    return batched_args


def _find_best_histogram_split_parallel(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    cached_histograms: dict[int, HistogramBuffer] | None,
    pool: WorkerPoolProtocol,
) -> tuple[SplitCandidate | None, dict[int, HistogramBuffer]]:
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
    histograms: dict[int, HistogramBuffer] = {}
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
        n_samples: int = gradients.shape[0]
        shm_size: int = n_samples * 8
        # Create shared memory for gradients and hessians
        shm_grad = shared_memory.SharedMemory(create=True, size=shm_size)
        shm_hess = shared_memory.SharedMemory(create=True, size=shm_size)
        try:
            # Write numpy arrays to shared memory efficiently
            grad_buf = shm_grad.buf
            hess_buf = shm_hess.buf
            # Buffers are always valid after successful SharedMemory creation
            assert grad_buf is not None and hess_buf is not None
            # Copy numpy array bytes directly into shared memory
            grad_bytes: bytes = gradients.tobytes()
            hess_bytes: bytes = hessians.tobytes()
            grad_buf[:shm_size] = grad_bytes
            hess_buf[:shm_size] = hess_bytes

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
    sample_indices: NDArray[np.int64],
    feature_indices: tuple[int, ...],
    config: GradientBoostingConfig,
    feature_bins: FeatureBins,
    histograms: dict[int, HistogramBuffer],
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

        # Get sample_bins column for this feature
        feat_bins: NDArray[np.int64] = feature_bins.sample_bins[:, feat_idx]

        histogram = histograms[feat_idx]
        constraint = 0 if constraints is None else constraints[feat_idx]
        split_result = find_best_split_from_histogram(
            histogram,
            feature_bins.bin_edges[feat_idx],
            feat_idx,
            config["min_samples_leaf"],
            constraint,
            config["reg_lambda"],
        )

        if split_result is None or split_result.gain <= 0:
            continue
        if best_split is not None and split_result.gain <= best_split["gain"]:
            continue

        left_indices, right_indices = partition_by_bin(
            sample_indices,
            feat_bins,
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
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
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
