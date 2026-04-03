"""Tree prediction backend hooks for cleargbm.

Single-tree prediction traversal. Tests inject fakes, production uses
real implementations.

This module is private (underscore prefix) - not for external use.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from cleargbm.types import DecisionTree, TreeNode


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


__all__ = [
    "PredictTreeBackend",
    "predict_tree",
]
