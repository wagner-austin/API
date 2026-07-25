"""Type guard functions for search space type narrowing.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Provides TypeGuard functions for narrowing SearchSpace union types.
"""

from __future__ import annotations

from typing import TypeGuard

from .types import (
    ClearGBMSearchSpace,
    LightGBMSearchSpace,
    LogRegSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    RandomForestSearchSpace,
    SearchSpace,
    XGBoostSearchSpace,
)


def is_xgboost_search_space(space: SearchSpace) -> TypeGuard[XGBoostSearchSpace]:
    """Check if search space is XGBoostSearchSpace.

    Args:
        space: The search space to check.

    Both keys are required, because neither alone identifies the space:
    ClearGBM and RandomForest also carry max_depth, and LightGBM also carries
    colsample_bytree. Matching on max_depth alone sent every RandomForest and
    ClearGBM space into the XGBoost sampler, which then failed on a
    learning_rate RandomForest never samples.

    Returns:
        True if space is XGBoostSearchSpace, False otherwise.
    """
    return "max_depth" in space and "colsample_bytree" in space


def is_mlp_search_space(space: SearchSpace) -> TypeGuard[MLPSearchSpace]:
    """Check if search space is MLPSearchSpace.

    Args:
        space: The search space to check.

    Returns:
        True if space is MLPSearchSpace, False otherwise.
    """
    return "n_layers" in space


def is_lstm_search_space(space: SearchSpace) -> TypeGuard[LSTMSearchSpace]:
    """Check if search space is LSTMSearchSpace.

    Args:
        space: The search space to check.

    Returns:
        True if space is LSTMSearchSpace, False otherwise.
    """
    return "num_layers" in space


def is_cleargbm_search_space(space: SearchSpace) -> TypeGuard[ClearGBMSearchSpace]:
    """Check if search space is ClearGBMSearchSpace.

    Args:
        space: The search space to check.

    Returns:
        True if space is ClearGBMSearchSpace, False otherwise.
    """
    return "max_bins" in space


def is_random_forest_search_space(space: SearchSpace) -> TypeGuard[RandomForestSearchSpace]:
    """Check if search space is RandomForestSearchSpace.

    Args:
        space: The search space to check.

    Returns:
        True if space is RandomForestSearchSpace, False otherwise.
    """
    return "max_features" in space


def is_logreg_search_space(space: SearchSpace) -> TypeGuard[LogRegSearchSpace]:
    """Check if search space is LogRegSearchSpace.

    Args:
        space: The search space to check.

    Returns:
        True if space is LogRegSearchSpace, False otherwise.
    """
    return "C" in space


def is_lightgbm_search_space(space: SearchSpace) -> TypeGuard[LightGBMSearchSpace]:
    """Check if search space is LightGBMSearchSpace.

    Args:
        space: The search space to check.

    Returns:
        True if space is LightGBMSearchSpace, False otherwise.
    """
    return "num_leaves" in space


__all__ = [
    "is_cleargbm_search_space",
    "is_lightgbm_search_space",
    "is_logreg_search_space",
    "is_lstm_search_space",
    "is_mlp_search_space",
    "is_random_forest_search_space",
    "is_xgboost_search_space",
]
