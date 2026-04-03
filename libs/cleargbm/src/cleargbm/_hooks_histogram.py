"""Histogram backend hooks for cleargbm.

Build and subtract gradient/hessian histograms for split finding.
Tests inject fakes, production uses real implementations.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm._hooks_infra import create_histogram_buffer
from cleargbm.buffers import HistogramBuffer


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
    buf = create_histogram_buffer(n_bins)
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
    sibling = create_histogram_buffer(parent.n_bins)
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


__all__ = [
    "BuildHistogramBackend",
    "SubtractHistogramBackend",
    "build_histogram",
    "subtract_histogram",
]
