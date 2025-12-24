"""Type guard functions for search space type narrowing.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Provides TypeGuard functions for narrowing SearchSpace union types.
"""

from __future__ import annotations

from typing import TypeGuard

from .types import (
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    SearchSpace,
    XGBoostSearchSpace,
)


def is_xgboost_search_space(space: SearchSpace) -> TypeGuard[XGBoostSearchSpace]:
    """Check if search space is XGBoostSearchSpace.

    Args:
        space: The search space to check.

    Returns:
        True if space is XGBoostSearchSpace, False otherwise.
    """
    return "max_depth" in space


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


def is_lightgbm_search_space(space: SearchSpace) -> TypeGuard[LightGBMSearchSpace]:
    """Check if search space is LightGBMSearchSpace.

    Args:
        space: The search space to check.

    Returns:
        True if space is LightGBMSearchSpace, False otherwise.
    """
    return "num_leaves" in space


__all__ = [
    "is_lightgbm_search_space",
    "is_lstm_search_space",
    "is_mlp_search_space",
    "is_xgboost_search_space",
]
