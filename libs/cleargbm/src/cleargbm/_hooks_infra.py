"""Infrastructure hooks for cleargbm.

Random state, worker pool, and buffer factory hooks. Tests inject fakes,
production uses real implementations.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

import multiprocessing
import random
from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm.buffers import FloatBuffer, HistogramBuffer, IntBuffer


class RandomStateProtocol(Protocol):
    """Protocol for random number generation."""

    def permutation(self, n: int) -> tuple[int, ...]:
        """Return random permutation of integers 0 to n-1.

        Args:
            n: Number of integers.

        Returns:
            Random permutation as tuple.
        """
        ...

    def choice(
        self,
        n: int,
        size: int,
        replace: bool,
    ) -> tuple[int, ...]:
        """Return random sample of integers.

        Args:
            n: Range of integers (0 to n-1).
            size: Number of samples.
            replace: Whether to sample with replacement.

        Returns:
            Random sample as tuple.
        """
        ...

    def rand_1d(self, size: int) -> tuple[float, ...]:
        """Return 1D tuple of random floats in [0, 1).

        Args:
            size: Number of random floats.

        Returns:
            Random floats as tuple.
        """
        ...

    def rand_2d(self, rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
        """Return 2D tuple of random floats in [0, 1).

        Args:
            rows: Number of rows.
            cols: Number of columns.

        Returns:
            Random floats as tuple of tuples.
        """
        ...


class _PythonRandomStateWrapper:
    """Wrapper for Python stdlib random that conforms to RandomStateProtocol."""

    def __init__(self, seed: int) -> None:
        """Initialize with seed.

        Args:
            seed: Random seed.
        """
        self._rng = random.Random(seed)

    def permutation(self, n: int) -> tuple[int, ...]:
        """Return random permutation of integers 0 to n-1.

        Args:
            n: Number of integers.

        Returns:
            Random permutation as tuple.
        """
        indices = list(range(n))
        self._rng.shuffle(indices)
        return tuple(indices)

    def choice(
        self,
        n: int,
        size: int,
        replace: bool,
    ) -> tuple[int, ...]:
        """Return random sample of integers.

        Args:
            n: Range of integers (0 to n-1).
            size: Number of samples.
            replace: Whether to sample with replacement.

        Returns:
            Random sample as tuple.
        """
        population = range(n)
        if replace:
            result = [self._rng.choice(population) for _ in range(size)]
        else:
            result = self._rng.sample(population, size)
        return tuple(result)

    def rand_1d(self, size: int) -> tuple[float, ...]:
        """Return 1D tuple of random floats in [0, 1).

        Args:
            size: Number of random floats.

        Returns:
            Random floats as tuple.
        """
        return tuple(self._rng.random() for _ in range(size))

    def rand_2d(self, rows: int, cols: int) -> tuple[tuple[float, ...], ...]:
        """Return 2D tuple of random floats in [0, 1).

        Args:
            rows: Number of rows.
            cols: Number of columns.

        Returns:
            Random floats as tuple of tuples.
        """
        return tuple(tuple(self._rng.random() for _ in range(cols)) for _ in range(rows))


def _default_random_state_factory(seed: int) -> RandomStateProtocol:
    """Production implementation - creates Python stdlib random wrapper.

    Args:
        seed: Random seed.

    Returns:
        RandomState instance.
    """
    return _PythonRandomStateWrapper(seed)


# Module-level hook for random state factory.
# Tests can override to provide deterministic behavior.
_random_state_factory: Callable[[int], RandomStateProtocol] = _default_random_state_factory


def get_random_state(seed: int) -> RandomStateProtocol:
    """Get random state instance.

    Args:
        seed: Random seed.

    Returns:
        RandomState instance.
    """
    return _random_state_factory(seed)


# =============================================================================
# Worker Pool Protocol and Hooks
# =============================================================================


class WorkerPoolProtocol(Protocol):
    """Protocol for worker pool with initialized feature bins.

    The pool is constructed with a feature-bins initializer so workers can read
    sample bin assignments from a module-global without receiving them via IPC.
    Gradients and hessians are passed via shared memory names in args.
    """

    def map_batched(
        self,
        func: Callable[
            [
                tuple[
                    tuple[int, ...],  # feature_indices (batch)
                    bytes,  # sample_indices as bytes
                    int,  # n_indices
                    str,  # grad_shm_name
                    str,  # hess_shm_name
                    int,  # n_samples
                    tuple[int, ...],  # n_bins per feature
                ]
            ],
            list[tuple[int, HistogramBuffer]],
        ],
        args_list: list[
            tuple[
                tuple[int, ...],
                bytes,
                int,
                str,
                str,
                int,
                tuple[int, ...],
            ]
        ],
    ) -> list[list[tuple[int, HistogramBuffer]]]:
        """Apply batched histogram worker to each batch.

        Args:
            func: Batched worker that accesses feature_bins from global.
            args_list: Batched args with shared memory names for gradients/hessians.

        Returns:
            List of lists of (feature_idx, HistogramBuffer) tuples.
        """
        ...

    def close(self) -> None:
        """Prevent any more tasks from being submitted."""
        ...

    def join(self) -> None:
        """Wait for worker processes to exit."""
        ...


class _MultiprocessingPoolWrapper:
    """Wrapper around multiprocessing.Pool with feature-bins initializer."""

    def __init__(
        self,
        n_workers: int,
        bin_edges: tuple[tuple[float, ...], ...],
        sample_bins: NDArray[np.int64],
    ) -> None:
        """Initialize pool with feature_bins set in workers.

        Args:
            n_workers: Number of worker processes.
            bin_edges: Bin edges for each feature (raw tuples for pickle).
            sample_bins: Per-sample bin assignments (n_samples, n_features).
        """
        from cleargbm.parallel import _worker_initializer

        # Convert 2D numpy array to bytes for IPC
        n_samples, n_features = sample_bins.shape
        sample_bins_flat: bytes = sample_bins.tobytes()

        self._pool: multiprocessing.pool.Pool = multiprocessing.Pool(
            n_workers,
            initializer=_worker_initializer,
            initargs=(bin_edges, sample_bins_flat, n_samples, n_features),
        )
        self._n_workers = n_workers

    def map_batched(
        self,
        func: Callable[
            [
                tuple[
                    tuple[int, ...],
                    bytes,  # sample_indices as bytes
                    int,  # n_indices
                    str,
                    str,
                    int,
                    tuple[int, ...],
                ]
            ],
            list[tuple[int, HistogramBuffer]],
        ],
        args_list: list[
            tuple[
                tuple[int, ...],
                bytes,  # sample_indices as bytes
                int,  # n_indices
                str,
                str,
                int,
                tuple[int, ...],
            ]
        ],
    ) -> list[list[tuple[int, HistogramBuffer]]]:
        """Apply batched histogram worker to each batch.

        Args:
            func: Batched worker function.
            args_list: Batched args with shared memory names.

        Returns:
            List of lists of (feature_idx, HistogramBuffer) tuples.
        """
        return self._pool.map(func, args_list)

    def close(self) -> None:
        """Prevent any more tasks from being submitted."""
        self._pool.close()

    def join(self) -> None:
        """Wait for worker processes to exit."""
        self._pool.join()


def _default_pool_factory(
    n_workers: int,
    bin_edges: tuple[tuple[float, ...], ...],
    sample_bins: NDArray[np.int64],
) -> WorkerPoolProtocol:
    """Production implementation - creates pool with initializer.

    Args:
        n_workers: Number of worker processes.
        bin_edges: Bin edges for each feature.
        sample_bins: Per-sample bin assignments (n_samples, n_features).

    Returns:
        WorkerPool instance with feature_bins initialized.
    """
    return _MultiprocessingPoolWrapper(n_workers, bin_edges, sample_bins)


# Module-level hook for pool factory.
# Tests can override to provide sequential or fake behavior.
_pool_factory: Callable[
    [int, tuple[tuple[float, ...], ...], NDArray[np.int64]],
    WorkerPoolProtocol,
] = _default_pool_factory


def create_worker_pool(
    n_workers: int,
    bin_edges: tuple[tuple[float, ...], ...],
    sample_bins: NDArray[np.int64],
) -> WorkerPoolProtocol:
    """Create a worker pool with feature_bins initialized in workers.

    Args:
        n_workers: Number of worker processes.
        bin_edges: Bin edges for each feature.
        sample_bins: Per-sample bin assignments (n_samples, n_features).

    Returns:
        WorkerPool instance with initialized feature_bins.
    """
    return _pool_factory(n_workers, bin_edges, sample_bins)


# =============================================================================
# Buffer Factory Hooks
# =============================================================================


def _default_float_buffer_factory(size: int) -> FloatBuffer:
    """Production implementation - creates FloatBuffer.

    Args:
        size: Number of elements in buffer.

    Returns:
        FloatBuffer instance.
    """
    return FloatBuffer(size)


def _default_int_buffer_factory(size: int) -> IntBuffer:
    """Production implementation - creates IntBuffer.

    Args:
        size: Number of elements in buffer.

    Returns:
        IntBuffer instance.
    """
    return IntBuffer(size)


def _default_histogram_buffer_factory(n_bins: int) -> HistogramBuffer:
    """Production implementation - creates HistogramBuffer.

    Args:
        n_bins: Number of bins in histogram.

    Returns:
        HistogramBuffer instance.
    """
    return HistogramBuffer(n_bins)


# Module-level hooks for buffer factories.
# Tests can override to provide instrumented or fake buffers.
_float_buffer_factory: Callable[[int], FloatBuffer] = _default_float_buffer_factory
_int_buffer_factory: Callable[[int], IntBuffer] = _default_int_buffer_factory
_histogram_buffer_factory: Callable[[int], HistogramBuffer] = _default_histogram_buffer_factory


def create_float_buffer(size: int) -> FloatBuffer:
    """Create a float buffer.

    Args:
        size: Number of elements in buffer.

    Returns:
        FloatBuffer instance.
    """
    return _float_buffer_factory(size)


def create_int_buffer(size: int) -> IntBuffer:
    """Create an int buffer.

    Args:
        size: Number of elements in buffer.

    Returns:
        IntBuffer instance.
    """
    return _int_buffer_factory(size)


def create_histogram_buffer(n_bins: int) -> HistogramBuffer:
    """Create a histogram buffer.

    Args:
        n_bins: Number of bins in histogram.

    Returns:
        HistogramBuffer instance.
    """
    return _histogram_buffer_factory(n_bins)


__all__ = [
    "RandomStateProtocol",
    "WorkerPoolProtocol",
    "create_float_buffer",
    "create_histogram_buffer",
    "create_int_buffer",
    "create_worker_pool",
    "get_random_state",
]
