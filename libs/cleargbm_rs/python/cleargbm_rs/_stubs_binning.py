"""Binning function stubs for cleargbm_rs.

Mirrors ``pyo3_module/binning_fns.rs``.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm_rs._constants import NOT_BUILT_MSG


def precompute_feature_bins_rs(
    features: NDArray[np.float64],
    max_bins: int,
) -> tuple[list[list[float]], NDArray[np.int64], int]:
    """Precompute feature bins from a 2D feature matrix.

    Args:
        features: 2D feature matrix (n_samples, n_features).
        max_bins: Maximum number of bins per feature.

    Returns:
        Tuple of (bin_thresholds, sample_bins, n_regular_bins).

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def compute_bin_edges_rs(
    features: NDArray[np.float64],
    max_bins: int,
) -> list[list[float]]:
    """Compute bin edges for each feature.

    Args:
        features: 2D feature matrix (n_samples, n_features).
        max_bins: Maximum number of bins per feature.

    Returns:
        List of lists of edge thresholds per feature.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def bin_samples_rs(
    features: NDArray[np.float64],
    bin_edges: list[list[float]],
    n_regular_bins: int,
) -> NDArray[np.int64]:
    """Assign bin indices to samples given precomputed bin edges.

    Args:
        features: 2D feature matrix (n_samples, n_features).
        bin_edges: List of lists of edge thresholds per feature.
        n_regular_bins: Number of regular bins (NaN bin is at this index).

    Returns:
        2D int64 array of shape (n_samples, n_features) with bin indices.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


__all__ = [
    "bin_samples_rs",
    "compute_bin_edges_rs",
    "precompute_feature_bins_rs",
]
