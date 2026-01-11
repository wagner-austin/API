"""ClearGBM Rust core Python bindings.

This module provides Python bindings to the high-performance Rust
implementation of ClearGBM's core algorithms.

When the Rust extension is not available, functions raise ImportError
with a helpful message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    import numpy as np

__version__ = "0.1.0"


def _raise_not_built() -> None:
    """Raise ImportError indicating Rust extension not built."""
    msg = (
        "cleargbm_rs Rust extension not built. "
        "Run 'maturin develop' to build the extension."
    )
    raise ImportError(msg)


def build_histogram_rs(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    bins: NDArray[np.int64],
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.uint64]]:
    """Build histogram from sample data using Rust implementation.

    Args:
        sample_indices: Indices of samples at this node.
        gradients: Gradient values for all samples.
        hessians: Hessian values for all samples.
        bins: Pre-computed bin assignments.
        n_bins: Number of histogram bins.

    Returns:
        Tuple of (gradient_sums, hessian_sums, counts) as numpy arrays.

    Raises:
        ImportError: If Rust extension not built.
        ValueError: If input arrays have invalid shapes.
    """
    _raise_not_built()
    # This line is never reached but satisfies type checker
    raise AssertionError("unreachable")


__all__ = [
    "__version__",
    "build_histogram_rs",
]
