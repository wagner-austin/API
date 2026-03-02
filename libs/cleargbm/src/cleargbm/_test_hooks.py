"""Internal test hooks for cleargbm.

These hooks allow testing without mocking. Tests inject fakes,
production uses real implementations.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

import math
import multiprocessing
import random
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm.buffers import FloatBuffer, HistogramBuffer, IntBuffer
from cleargbm.types import DecisionTree, TreeNode


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


# =============================================================================
# Histogram Backend Hooks
# =============================================================================


class BuildHistogramBackend(Protocol):
    """Protocol for histogram building backend."""

    def __call__(
        self,
        sample_indices: NDArray[np.int64],
        gradients: NDArray[np.float64],
        hessians: NDArray[np.float64],
        sample_bins: NDArray[np.int64],
        n_bins: int,
    ) -> HistogramBuffer:
        """Build gradient/hessian histogram for one feature in a node.

        Args:
            sample_indices: Indices of samples in this node.
            gradients: Gradient for each sample (full dataset).
            hessians: Hessian for each sample (full dataset).
            sample_bins: Bin ID for each sample on this feature (1D array).
            n_bins: Number of bins.

        Returns:
            HistogramBuffer with gradient/hessian sums per bin.
        """
        ...


class SubtractHistogramBackend(Protocol):
    """Protocol for histogram subtraction backend."""

    def __call__(
        self,
        parent: HistogramBuffer,
        child: HistogramBuffer,
    ) -> HistogramBuffer:
        """Compute sibling histogram by subtraction (parent - child).

        Args:
            parent: Parent node histogram buffer.
            child: One child's histogram buffer.

        Returns:
            Other child's histogram buffer (sibling = parent - child).
        """
        ...


def _default_build_histogram(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    sample_bins: NDArray[np.int64],
    n_bins: int,
) -> HistogramBuffer:
    """Python histogram building implementation.

    Accumulates gradient/hessian statistics into bins using vectorized
    numpy operations.

    Args:
        sample_indices: Indices of samples in this node.
        gradients: Gradient for each sample (full dataset).
        hessians: Hessian for each sample (full dataset).
        sample_bins: Bin ID for each sample on this feature (1D array).
        n_bins: Number of bins.

    Returns:
        HistogramBuffer with gradient/hessian sums per bin.
    """
    buf = _histogram_buffer_factory(n_bins)
    bins_for_node: NDArray[np.int64] = sample_bins[sample_indices]
    grads_for_node: NDArray[np.float64] = gradients[sample_indices]
    hess_for_node: NDArray[np.float64] = hessians[sample_indices]
    buf.accumulate_batch(bins_for_node, grads_for_node, hess_for_node)
    return buf


def _default_subtract_histogram(
    parent: HistogramBuffer,
    child: HistogramBuffer,
) -> HistogramBuffer:
    """Python histogram subtraction implementation.

    Computes sibling = parent - child using numpy subtraction.

    Args:
        parent: Parent node histogram buffer.
        child: One child's histogram buffer.

    Returns:
        Other child's histogram buffer (sibling = parent - child).
    """
    sibling = _histogram_buffer_factory(parent.n_bins)
    sibling.subtract_into(parent, child)
    return sibling


# Module-level hooks for histogram backend.
# Production sets these to Rust implementations at startup.
# Tests override to provide Python fakes.
_build_histogram_backend: BuildHistogramBackend = _default_build_histogram
_subtract_histogram_backend: SubtractHistogramBackend = _default_subtract_histogram


def build_histogram(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    sample_bins: NDArray[np.int64],
    n_bins: int,
) -> HistogramBuffer:
    """Build gradient/hessian histogram for one feature in a node.

    Delegates to the active backend hook.

    Args:
        sample_indices: Indices of samples in this node.
        gradients: Gradient for each sample (full dataset).
        hessians: Hessian for each sample (full dataset).
        sample_bins: Bin ID for each sample on this feature (1D array).
        n_bins: Number of bins.

    Returns:
        HistogramBuffer with gradient/hessian sums per bin.
    """
    return _build_histogram_backend(sample_indices, gradients, hessians, sample_bins, n_bins)


def subtract_histogram(
    parent: HistogramBuffer,
    child: HistogramBuffer,
) -> HistogramBuffer:
    """Compute sibling histogram by subtraction (parent - child).

    Delegates to the active backend hook.

    Args:
        parent: Parent node histogram buffer.
        child: One child's histogram buffer.

    Returns:
        Other child's histogram buffer (sibling = parent - child).
    """
    return _subtract_histogram_backend(parent, child)


# =============================================================================
# Prediction Backend Hooks
# =============================================================================


class PredictTreeBackend(Protocol):
    """Protocol for tree prediction backend."""

    def __call__(
        self,
        tree: DecisionTree,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Get predictions from tree for all samples.

        Args:
            tree: Trained decision tree.
            x: Feature matrix (n_samples, n_features).

        Returns:
            Prediction array for each sample.
        """
        ...


def _traverse_tree_single(
    nodes: tuple[TreeNode, ...],
    x_single: NDArray[np.float64],
) -> float:
    """Traverse decision tree for a single sample.

    Args:
        nodes: All nodes in the tree.
        x_single: Single sample feature vector (1D array).

    Returns:
        Prediction value.
    """
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


def _default_predict_tree(
    tree: DecisionTree,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Python tree prediction implementation.

    Loops over all samples and traverses the tree for each.

    Args:
        tree: Trained decision tree.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Prediction array for each sample.
    """
    n_samples: int = int(x.shape[0])
    predictions: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    nodes = tree["nodes"]
    for i in range(n_samples):
        x_row: NDArray[np.float64] = x[i, :]
        predictions[i] = _traverse_tree_single(nodes, x_row)
    return predictions


# Module-level hook for tree prediction backend.
# Production sets this to Rust implementation at startup.
# Tests override to provide Python fakes.
_predict_tree_backend: PredictTreeBackend = _default_predict_tree


def predict_tree(
    tree: DecisionTree,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Get predictions from tree for all samples.

    Delegates to the active backend hook.

    Args:
        tree: Trained decision tree.
        x: Feature matrix (n_samples, n_features).

    Returns:
        Prediction array for each sample.
    """
    return _predict_tree_backend(tree, x)


# =============================================================================
# Sigmoid Backend Hooks
# =============================================================================


class SigmoidBackend(Protocol):
    """Protocol for scalar sigmoid backend."""

    def __call__(self, x: float) -> float:
        """Compute sigmoid function.

        Args:
            x: Input value (log-odds).

        Returns:
            Probability in [0, 1].
        """
        ...


class SigmoidArrayBackend(Protocol):
    """Protocol for vectorized sigmoid backend."""

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute sigmoid function for array.

        Args:
            x: Input array (log-odds).

        Returns:
            Probabilities in [0, 1].
        """
        ...


def _default_sigmoid(x: float) -> float:
    """Python scalar sigmoid implementation.

    Clips input to [-500, 500] to prevent overflow.

    Args:
        x: Input value (log-odds).

    Returns:
        Probability in [0, 1].
    """
    x_clipped = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x_clipped))


def _default_sigmoid_array(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Python vectorized sigmoid implementation.

    Uses numpy for efficient array operations. Clips to [-500, 500].

    Args:
        x: Input array (log-odds).

    Returns:
        Probabilities in [0, 1].
    """
    x_clipped: NDArray[np.float64] = np.clip(x, -500.0, 500.0)
    result: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-x_clipped))
    return result


# Module-level hooks for sigmoid backend.
# Production sets these to Rust implementations at startup.
# Tests override to provide Python fakes.
_sigmoid_backend: SigmoidBackend = _default_sigmoid
_sigmoid_array_backend: SigmoidArrayBackend = _default_sigmoid_array


def sigmoid(x: float) -> float:
    """Compute sigmoid function.

    Delegates to the active backend hook.

    Args:
        x: Input value (log-odds).

    Returns:
        Probability in [0, 1].
    """
    return _sigmoid_backend(x)


def sigmoid_array(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute sigmoid function for array.

    Delegates to the active backend hook.

    Args:
        x: Input array (log-odds).

    Returns:
        Probabilities in [0, 1].
    """
    return _sigmoid_array_backend(x)


# =============================================================================
# Guard Script Hooks
# =============================================================================


class FindMonorepoRootProto(Protocol):
    """Protocol for _find_monorepo_root hook."""

    def __call__(self, start: Path) -> Path:
        """Find monorepo root starting from given path.

        Args:
            start: Starting path to search from.

        Returns:
            Path to monorepo root.
        """
        ...


class RunForProjectProto(Protocol):
    """Protocol for run_for_project hook."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guard checks for a project.

        Args:
            monorepo_root: Path to monorepo root.
            project_root: Path to project root.

        Returns:
            Exit code from guard checks.
        """
        ...


class LoadOrchestratorProto(Protocol):
    """Protocol for _load_orchestrator hook."""

    def __call__(self, monorepo_root: Path) -> RunForProjectProto:
        """Load the guard orchestrator.

        Args:
            monorepo_root: Path to monorepo root.

        Returns:
            run_for_project function.
        """
        ...


# Guard hooks - None means use default behavior (production implementation)
guard_find_monorepo_root: FindMonorepoRootProto | None = None
guard_load_orchestrator: LoadOrchestratorProto | None = None


__all__ = [
    "BuildHistogramBackend",
    "FindMonorepoRootProto",
    "FloatBuffer",
    "HistogramBuffer",
    "IntBuffer",
    "LoadOrchestratorProto",
    "PredictTreeBackend",
    "RandomStateProtocol",
    "RunForProjectProto",
    "SigmoidArrayBackend",
    "SigmoidBackend",
    "SubtractHistogramBackend",
    "WorkerPoolProtocol",
    "build_histogram",
    "create_float_buffer",
    "create_histogram_buffer",
    "create_int_buffer",
    "create_worker_pool",
    "get_random_state",
    "guard_find_monorepo_root",
    "guard_load_orchestrator",
    "predict_tree",
    "sigmoid",
    "sigmoid_array",
    "subtract_histogram",
]
