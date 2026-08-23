"""Shared fixtures and helpers for test_cleargbm_shap splits."""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from cleargbm.types import DecisionTree, GradientBoostingConfig, GradientBoostingModel, TreeNode
from numpy.typing import NDArray


class _NativePyGbmModelProto(Protocol):
    """Opaque native model handle produced by the Rust training loop."""

    ...


def _make_x_1x2() -> NDArray[np.float64]:
    """Create 1x2 test array."""
    data: NDArray[np.float64] = np.zeros((1, 2), dtype=np.float64)
    data[0, 0] = 0.3
    data[0, 1] = 0.5
    return data


def _make_x_2x2() -> NDArray[np.float64]:
    """Create 2x2 test array."""
    data: NDArray[np.float64] = np.zeros((2, 2), dtype=np.float64)
    data[0, 0] = 0.1
    data[0, 1] = 0.2
    data[1, 0] = 0.6
    data[1, 1] = 0.7
    return data


def _make_x_3x2() -> NDArray[np.float64]:
    """Create 3x2 test array."""
    data: NDArray[np.float64] = np.zeros((3, 2), dtype=np.float64)
    data[0, 0] = 0.2
    data[0, 1] = 0.4
    data[1, 0] = 0.7
    data[1, 1] = 0.3
    data[2, 0] = 0.5
    data[2, 1] = 0.5
    return data


def _make_x_1x3() -> NDArray[np.float64]:
    """Create 1x3 test array."""
    data: NDArray[np.float64] = np.zeros((1, 3), dtype=np.float64)
    data[0, 0] = 0.1
    data[0, 1] = 0.2
    data[0, 2] = 0.3
    return data


def _make_config() -> GradientBoostingConfig:
    """Create minimal GradientBoostingConfig for testing."""
    config: GradientBoostingConfig = GradientBoostingConfig(
        n_estimators=2,
        max_depth=2,
        learning_rate=0.1,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        colsample_bytree=None,
        max_bins=64,
        subsample=1.0,
        random_state=42,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=1,
        early_stopping_rounds=10,
        growth_strategy="depth_wise",
        num_leaves=None,
        objective="binary_log_loss",
        scale_pos_weight=1.0,
    )
    return config


def _make_leaf_node(
    node_id: int,
    value: float,
    n_samples: int,
) -> TreeNode:
    """Create a leaf TreeNode."""
    return TreeNode(
        node_id=node_id,
        is_leaf=True,
        feature_index=None,
        feature_name=None,
        threshold=None,
        nan_direction=None,
        value=value,
        n_samples=n_samples,
        left_child=None,
        right_child=None,
    )


def _make_split_node(
    node_id: int,
    feature_index: int,
    threshold: float,
    left_child: int,
    right_child: int,
    n_samples: int,
    nan_direction: Literal["left", "right"] | None = None,
    value: float = 0.0,
) -> TreeNode:
    """Create an internal (split) TreeNode."""
    return TreeNode(
        node_id=node_id,
        is_leaf=False,
        feature_index=feature_index,
        feature_name=f"feature_{feature_index}",
        threshold=threshold,
        nan_direction=nan_direction,
        value=value,
        n_samples=n_samples,
        left_child=left_child,
        right_child=right_child,
    )


def _make_simple_tree() -> DecisionTree:
    """Create a simple tree with one split and two leaves.

    Structure:
        [0] root (split on feature 0 at 0.5)
         ├── [1] left leaf (value=-0.1)
         └── [2] right leaf (value=0.1)
    """
    root = _make_split_node(
        node_id=0,
        feature_index=0,
        threshold=0.5,
        left_child=1,
        right_child=2,
        n_samples=100,
        nan_direction="left",
    )
    left_leaf = _make_leaf_node(node_id=1, value=-0.1, n_samples=40)
    right_leaf = _make_leaf_node(node_id=2, value=0.1, n_samples=60)

    return DecisionTree(
        nodes=(root, left_leaf, right_leaf),
        max_depth=1,
        n_leaves=2,
        feature_names=("feature_0", "feature_1"),
    )


def _make_deeper_tree() -> DecisionTree:
    """Create a tree with depth 2 (two levels of splits).

    Structure:
        [0] root (split on feature 0 at 0.5)
         ├── [1] left split (feature 1 at 0.3)
         │    ├── [3] left-left leaf
         │    └── [4] left-right leaf
         └── [2] right leaf
    """
    root = _make_split_node(
        node_id=0,
        feature_index=0,
        threshold=0.5,
        left_child=1,
        right_child=2,
        n_samples=100,
        nan_direction="right",
    )
    left_split = _make_split_node(
        node_id=1,
        feature_index=1,
        threshold=0.3,
        left_child=3,
        right_child=4,
        n_samples=40,
        nan_direction=None,
    )
    right_leaf = _make_leaf_node(node_id=2, value=0.2, n_samples=60)
    left_left_leaf = _make_leaf_node(node_id=3, value=-0.2, n_samples=15)
    left_right_leaf = _make_leaf_node(node_id=4, value=0.0, n_samples=25)

    return DecisionTree(
        nodes=(root, left_split, right_leaf, left_left_leaf, left_right_leaf),
        max_depth=2,
        n_leaves=3,
        feature_names=("feature_0", "feature_1"),
    )


def _make_model(trees: list[DecisionTree]) -> GradientBoostingModel:
    """Create a GradientBoostingModel with given trees."""
    return GradientBoostingModel(
        trees=tuple(trees),
        base_prediction=0.0,
        learning_rate=0.1,
        feature_names=("feature_0", "feature_1"),
        config=_make_config(),
    )
