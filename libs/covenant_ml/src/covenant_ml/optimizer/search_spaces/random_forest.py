"""Random Forest search space definitions for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from ..types import (
    CategoricalStringSpec,
    IntRangeSpec,
    RandomForestSearchSpace,
)


def make_random_forest_default_space() -> RandomForestSearchSpace:
    """Create default Random Forest search space for classification.

    Based on empirical testing for tabular data:
    - n_estimators 50-500 (number of trees in the forest)
    - max_depth 3-20 (tree depth)
    - min_samples_split 2-20 (minimum samples to split a node)
    - min_samples_leaf 1-10 (minimum samples in a leaf)
    - max_features sqrt or log2 (feature selection strategy)

    Returns:
        RandomForestSearchSpace with sensible default ranges.
    """
    n_estimators_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 50,
        "high": 500,
        "log_scale": False,
    }
    max_depth_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 3,
        "high": 20,
        "log_scale": False,
    }
    min_samples_split_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 2,
        "high": 20,
        "log_scale": False,
    }
    min_samples_leaf_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 1,
        "high": 10,
        "log_scale": False,
    }
    max_features_spec: CategoricalStringSpec = {
        "param_type": "categorical_str",
        "choices": ("sqrt", "log2"),
    }

    space: RandomForestSearchSpace = {
        "n_estimators": n_estimators_spec,
        "max_depth": max_depth_spec,
        "min_samples_split": min_samples_split_spec,
        "min_samples_leaf": min_samples_leaf_spec,
        "max_features": max_features_spec,
    }
    return space


def make_random_forest_focused_space(
    *,
    best_max_depth: int,
    best_n_estimators: int,
) -> RandomForestSearchSpace:
    """Create focused Random Forest search space around known good values.

    Args:
        best_max_depth: Best max_depth from initial search.
        best_n_estimators: Best n_estimators from initial search.

    Returns:
        RandomForestSearchSpace with narrowed ranges around best values.
    """
    depth_low = max(2, best_max_depth - 3)
    depth_high = min(25, best_max_depth + 3)

    est_low = max(50, best_n_estimators - 100)
    est_high = min(800, best_n_estimators + 100)

    n_estimators_spec: IntRangeSpec = {
        "param_type": "int",
        "low": est_low,
        "high": est_high,
        "log_scale": False,
    }
    max_depth_spec: IntRangeSpec = {
        "param_type": "int",
        "low": depth_low,
        "high": depth_high,
        "log_scale": False,
    }
    min_samples_split_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 2,
        "high": 15,
        "log_scale": False,
    }
    min_samples_leaf_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 1,
        "high": 8,
        "log_scale": False,
    }
    max_features_spec: CategoricalStringSpec = {
        "param_type": "categorical_str",
        "choices": ("sqrt", "log2"),
    }

    space: RandomForestSearchSpace = {
        "n_estimators": n_estimators_spec,
        "max_depth": max_depth_spec,
        "min_samples_split": min_samples_split_spec,
        "min_samples_leaf": min_samples_leaf_spec,
        "max_features": max_features_spec,
    }
    return space


__all__ = [
    "make_random_forest_default_space",
    "make_random_forest_focused_space",
]
