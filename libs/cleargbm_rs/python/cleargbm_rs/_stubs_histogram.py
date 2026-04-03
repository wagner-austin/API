"""Histogram function stubs for cleargbm_rs.

Mirrors ``pyo3_module/histogram_fns.rs``.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm_rs._constants import NOT_BUILT_MSG


def build_histogram_rs(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    bins: NDArray[np.int64],
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint64]]:
    """Build histogram from sample gradients and hessians.

    Args:
        sample_indices: Indices of samples at this node.
        gradients: Gradient values for all samples.
        hessians: Hessian values for all samples.
        bins: Pre-computed bin assignments.
        n_bins: Number of histogram bins.

    Returns:
        Tuple of (gradient_sums, hessian_sums, counts) as numpy arrays.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def subtract_histogram_rs(
    parent_grads: NDArray[np.float64],
    parent_hess: NDArray[np.float64],
    parent_counts: NDArray[np.uint64],
    child_grads: NDArray[np.float64],
    child_hess: NDArray[np.float64],
    child_counts: NDArray[np.uint64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint64]]:
    """Compute sibling histogram by subtraction (parent - child).

    Args:
        parent_grads: Parent gradient sums per bin.
        parent_hess: Parent hessian sums per bin.
        parent_counts: Parent sample counts per bin.
        child_grads: Child gradient sums per bin.
        child_hess: Child hessian sums per bin.
        child_counts: Child sample counts per bin.

    Returns:
        Tuple of (gradient_sums, hessian_sums, counts) for sibling.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


__all__ = [
    "build_histogram_rs",
    "subtract_histogram_rs",
]
