"""ClearGBM search space definitions for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from ..types import (
    CategoricalIntSpec,
    ClearGBMSearchSpace,
    FloatRangeSpec,
    IntRangeSpec,
)


def make_cleargbm_default_space() -> ClearGBMSearchSpace:
    """Create default ClearGBM search space for bankruptcy prediction.

    ClearGBM is a numpy-based gradient boosting implementation with built-in
    interpretability features. The search space includes:
    - n_estimators 50-300 (number of boosting rounds)
    - max_depth 3-10 (tree depth control)
    - learning_rate 0.01-0.3 in log scale
    - min_samples_split 5-50 (minimum samples to split a node)
    - min_samples_leaf 2-20 (minimum samples in a leaf)
    - max_bins 32-128 (histogram bins for split finding)
    - subsample 0.6-1.0 (row subsampling ratio)

    Returns:
        ClearGBMSearchSpace with sensible default ranges.
    """
    n_estimators_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 50,
        "high": 300,
        "log_scale": False,
    }
    max_depth_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 3,
        "high": 10,
        "log_scale": False,
    }
    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.01,
        "high": 0.3,
        "log_scale": True,
    }
    min_samples_split_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 5,
        "high": 50,
        "log_scale": False,
    }
    min_samples_leaf_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 2,
        "high": 20,
        "log_scale": False,
    }
    max_bins_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (32, 64, 128),
    }
    subsample_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.6,
        "high": 1.0,
        "log_scale": False,
    }

    space: ClearGBMSearchSpace = {
        "n_estimators": n_estimators_spec,
        "max_depth": max_depth_spec,
        "learning_rate": learning_rate_spec,
        "min_samples_split": min_samples_split_spec,
        "min_samples_leaf": min_samples_leaf_spec,
        "max_bins": max_bins_spec,
        "subsample": subsample_spec,
    }
    return space


def make_cleargbm_focused_space(
    *,
    best_max_depth: int,
    best_learning_rate: float,
) -> ClearGBMSearchSpace:
    """Create focused ClearGBM search space around known good values.

    Use after initial optimization to fine-tune near the best region.

    Args:
        best_max_depth: Best max_depth from initial search.
        best_learning_rate: Best learning_rate from initial search.

    Returns:
        ClearGBMSearchSpace with narrowed ranges around best values.
    """
    n_estimators_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 75,
        "high": 200,
        "log_scale": False,
    }

    depth_low = max(2, best_max_depth - 2)
    depth_high = min(15, best_max_depth + 2)

    max_depth_spec: IntRangeSpec = {
        "param_type": "int",
        "low": depth_low,
        "high": depth_high,
        "log_scale": False,
    }

    lr_low = max(0.001, best_learning_rate * 0.5)
    lr_high = min(0.5, best_learning_rate * 2.0)

    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": lr_low,
        "high": lr_high,
        "log_scale": True,
    }
    min_samples_split_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 5,
        "high": 30,
        "log_scale": False,
    }
    min_samples_leaf_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 2,
        "high": 15,
        "log_scale": False,
    }
    max_bins_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (64, 128),
    }
    subsample_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.7,
        "high": 1.0,
        "log_scale": False,
    }

    space: ClearGBMSearchSpace = {
        "n_estimators": n_estimators_spec,
        "max_depth": max_depth_spec,
        "learning_rate": learning_rate_spec,
        "min_samples_split": min_samples_split_spec,
        "min_samples_leaf": min_samples_leaf_spec,
        "max_bins": max_bins_spec,
        "subsample": subsample_spec,
    }
    return space


__all__ = [
    "make_cleargbm_default_space",
    "make_cleargbm_focused_space",
]
