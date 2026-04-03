"""Feature binning backend hooks for cleargbm.

Precompute feature bin assignments for histogram-based split finding.
Tests inject fakes, production uses real implementations.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm.types import FeatureBins


class PrecomputeFeatureBinsBackend(Protocol):
    """Protocol for feature binning backend."""

    def __call__(
        self,
        x: NDArray[np.float64],
        max_bins: int,
    ) -> FeatureBins:
        """Precompute all bin assignments for the dataset.

        Args:
            x: Feature matrix (n_samples, n_features).
            max_bins: Maximum number of bins per feature.

        Returns:
            FeatureBins containing edges and per-sample bin assignments.
        """
        ...


def _default_precompute_feature_bins(
    x: NDArray[np.float64],
    max_bins: int,
) -> FeatureBins:
    """Python feature binning implementation.

    Uses lazy import to avoid circular dependency with histogram module.

    Args:
        x: Feature matrix (n_samples, n_features).
        max_bins: Maximum number of bins per feature.

    Returns:
        FeatureBins containing edges and per-sample bin assignments.
    """
    from cleargbm.histogram import bin_samples, compute_bin_edges

    n_features = x.shape[1] if x.size > 0 else 0
    edges = compute_bin_edges(x, n_features, max_bins)
    sample_bins_arr = bin_samples(x, edges)
    return FeatureBins(bin_edges=edges, sample_bins=sample_bins_arr)


# Module-level hook for feature binning backend.
# Production sets this to Rust implementation at startup.
_precompute_feature_bins_backend: PrecomputeFeatureBinsBackend = _default_precompute_feature_bins


def precompute_feature_bins(
    x: NDArray[np.float64],
    max_bins: int,
) -> FeatureBins:
    """Precompute all bin assignments for the dataset.

    Delegates to the active backend hook.

    Args:
        x: Feature matrix (n_samples, n_features).
        max_bins: Maximum number of bins per feature.

    Returns:
        FeatureBins containing edges and per-sample bin assignments.
    """
    return _precompute_feature_bins_backend(x, max_bins)


__all__ = [
    "PrecomputeFeatureBinsBackend",
    "precompute_feature_bins",
]
