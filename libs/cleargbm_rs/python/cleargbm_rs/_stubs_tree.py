"""Tree function stubs for cleargbm_rs.

Mirrors ``pyo3_module/tree_fns.rs``.

This module is private (underscore prefix) — not for external use.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cleargbm_rs._constants import NOT_BUILT_MSG


class PyTree:
    """Opaque wrapper around a Rust Tree.

    Avoids JSON serialization overhead by keeping the tree in Rust memory.
    Created by ``build_tree_rs`` and consumed by prediction functions.

    Raises:
        ImportError: Always, when native extension is not built.
    """

    def __init__(self) -> None:
        """Create a PyTree.

        Raises:
            ImportError: Always, when native extension is not built.
        """
        raise ImportError(NOT_BUILT_MSG)


def build_tree_rs(
    sample_indices: NDArray[np.int64],
    gradients: NDArray[np.float64],
    hessians: NDArray[np.float64],
    bins: NDArray[np.int64],
    n_bins: int,
    thresholds: list[list[float]],
    config_json: str,
) -> PyTree:
    """Build a decision tree using histogram-based split finding.

    Args:
        sample_indices: Indices of samples at this node.
        gradients: Gradient values for all samples.
        hessians: Hessian values for all samples.
        bins: Pre-computed bin assignments (2D).
        n_bins: Number of histogram bins.
        thresholds: Bin edge thresholds per feature.
        config_json: JSON string with tree build configuration.

    Returns:
        PyTree wrapping the built tree.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def py_tree_from_json_rs(json_str: str) -> PyTree:
    """Deserialize a PyTree from a JSON string.

    Args:
        json_str: JSON string in Rust Tree serde format.

    Returns:
        PyTree instance.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def py_tree_to_json_rs(tree: PyTree) -> str:
    """Serialize a PyTree to a JSON string.

    Args:
        tree: PyTree to serialize.

    Returns:
        JSON string representation.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def py_tree_max_depth_rs(tree: PyTree) -> int:
    """Return the maximum depth of a PyTree.

    Args:
        tree: PyTree to inspect.

    Returns:
        Maximum depth of the tree.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def py_tree_n_leaves_rs(tree: PyTree) -> int:
    """Return the number of leaf nodes in a PyTree.

    Args:
        tree: PyTree to inspect.

    Returns:
        Number of leaf nodes.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def py_tree_n_nodes_rs(tree: PyTree) -> int:
    """Return the total number of nodes in a PyTree.

    Args:
        tree: PyTree to inspect.

    Returns:
        Total number of nodes.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


def py_tree_repr_rs(tree: PyTree) -> str:
    """Return a string representation of a PyTree for debugging.

    Args:
        tree: PyTree to represent.

    Returns:
        Debug string representation.

    Raises:
        ImportError: When native extension is not built.
    """
    raise ImportError(NOT_BUILT_MSG)


__all__ = [
    "PyTree",
    "build_tree_rs",
    "py_tree_from_json_rs",
    "py_tree_max_depth_rs",
    "py_tree_n_leaves_rs",
    "py_tree_n_nodes_rs",
    "py_tree_repr_rs",
    "py_tree_to_json_rs",
]
