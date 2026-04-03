"""LightGBM search space definitions for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from ..types import (
    CategoricalStringSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
)


def make_lightgbm_default_space() -> LightGBMSearchSpace:
    """Create default LightGBM search space for bankruptcy prediction.

    Based on empirical testing:
    - n_estimators 50-500 (more trees for larger datasets)
    - num_leaves 20-100 (primary tree complexity control)
    - learning_rate 0.01-0.3 in log scale
    - Regularization helps prevent overfitting
    - DART boosting included for dropout regularization exploration

    Note: max_depth is intentionally excluded. LightGBM uses leaf-wise growth
    where num_leaves is the primary complexity control. The optimizer uses
    max_depth=-1 (unlimited) to avoid constraint conflicts when
    num_leaves > 2^max_depth, which can cause training failures.

    Returns:
        LightGBMSearchSpace with sensible default ranges including DART support.
    """
    n_estimators_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 50,
        "high": 500,
        "log_scale": False,
    }
    num_leaves_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 20,
        "high": 100,
        "log_scale": False,
    }
    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.01,
        "high": 0.3,
        "log_scale": True,
    }
    subsample_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.6,
        "high": 1.0,
        "log_scale": False,
    }
    colsample_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.6,
        "high": 1.0,
        "log_scale": False,
    }
    reg_alpha_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 10.0,
        "log_scale": False,
    }
    reg_lambda_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.1,
        "high": 50.0,
        "log_scale": True,
    }
    boosting_type_spec: CategoricalStringSpec = {
        "param_type": "categorical_str",
        "choices": ("gbdt", "dart"),
    }
    drop_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.5,
        "log_scale": False,
    }
    skip_drop_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.5,
        "log_scale": False,
    }
    feature_fraction_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.02,
        "high": 0.1,
        "log_scale": False,
    }

    space: LightGBMSearchSpace = {
        "n_estimators": n_estimators_spec,
        "num_leaves": num_leaves_spec,
        "learning_rate": learning_rate_spec,
        "subsample": subsample_spec,
        "colsample_bytree": colsample_spec,
        "reg_alpha": reg_alpha_spec,
        "reg_lambda": reg_lambda_spec,
        "boosting_type": boosting_type_spec,
        "drop_rate": drop_rate_spec,
        "skip_drop": skip_drop_spec,
        "feature_fraction": feature_fraction_spec,
    }
    return space


def make_lightgbm_focused_space(
    *,
    best_num_leaves: int,
    best_learning_rate: float,
) -> LightGBMSearchSpace:
    """Create focused LightGBM search space around known good values.

    Args:
        best_num_leaves: Best num_leaves from initial search.
        best_learning_rate: Best learning_rate from initial search.

    Returns:
        LightGBMSearchSpace with narrowed ranges around best values.
    """
    n_estimators_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 100,
        "high": 300,
        "log_scale": False,
    }

    leaves_low = max(10, best_num_leaves - 20)
    leaves_high = min(150, best_num_leaves + 20)

    num_leaves_spec: IntRangeSpec = {
        "param_type": "int",
        "low": leaves_low,
        "high": leaves_high,
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
    subsample_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.7,
        "high": 1.0,
        "log_scale": False,
    }
    colsample_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.7,
        "high": 1.0,
        "log_scale": False,
    }
    reg_alpha_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 5.0,
        "log_scale": False,
    }
    reg_lambda_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.5,
        "high": 5.0,
        "log_scale": False,
    }

    space: LightGBMSearchSpace = {
        "n_estimators": n_estimators_spec,
        "num_leaves": num_leaves_spec,
        "learning_rate": learning_rate_spec,
        "subsample": subsample_spec,
        "colsample_bytree": colsample_spec,
        "reg_alpha": reg_alpha_spec,
        "reg_lambda": reg_lambda_spec,
    }
    return space


__all__ = [
    "make_lightgbm_default_space",
    "make_lightgbm_focused_space",
]
