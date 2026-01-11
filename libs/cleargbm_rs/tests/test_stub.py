"""Tests for cleargbm_rs Python stub."""

from __future__ import annotations

import numpy as np
import pytest

from cleargbm_rs import __version__, build_histogram_rs


def test_version_is_string() -> None:
    """Version should be a string."""
    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"


def test_build_histogram_rs_raises_import_error() -> None:
    """build_histogram_rs should raise ImportError when Rust extension not built."""
    sample_indices = np.array([0, 1, 2], dtype=np.int64)
    gradients = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    hessians = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    bins = np.array([0, 1, 0], dtype=np.int64)
    n_bins = 3

    with pytest.raises(ImportError, match="Rust extension not built"):
        build_histogram_rs(sample_indices, gradients, hessians, bins, n_bins)
