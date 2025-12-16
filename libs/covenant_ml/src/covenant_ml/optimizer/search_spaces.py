"""Predefined search spaces for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
Provides sensible default search ranges for XGBoost and MLP.
"""

from __future__ import annotations

from .types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    OptimizationConfig,
    XGBoostSearchSpace,
)


def make_xgboost_default_space() -> XGBoostSearchSpace:
    """Create default XGBoost search space for bankruptcy prediction.

    Based on empirical testing:
    - max_depth 3-10 covers shallow to moderately deep trees
    - n_estimators 50-300 balances training time and performance
    - learning_rate 0.01-0.3 in log scale (most important hyperparameter)
    - Regularization helps prevent overfitting on tabular data

    Returns:
        XGBoostSearchSpace with sensible default ranges
    """
    max_depth_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 3,
        "high": 10,
        "log_scale": False,
    }
    n_estimators_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 50,
        "high": 300,
        "log_scale": False,
    }
    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.01,
        "high": 0.3,
        "log_scale": True,
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
        "high": 10.0,
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

    space: XGBoostSearchSpace = {
        "max_depth": max_depth_spec,
        "n_estimators": n_estimators_spec,
        "learning_rate": learning_rate_spec,
        "reg_alpha": reg_alpha_spec,
        "reg_lambda": reg_lambda_spec,
        "subsample": subsample_spec,
        "colsample_bytree": colsample_spec,
    }
    return space


def make_xgboost_focused_space(
    *,
    best_max_depth: int,
    best_learning_rate: float,
) -> XGBoostSearchSpace:
    """Create focused XGBoost search space around known good values.

    Use after initial optimization to fine-tune near the best region.

    Args:
        best_max_depth: Best max_depth from initial search
        best_learning_rate: Best learning_rate from initial search

    Returns:
        XGBoostSearchSpace with narrowed ranges around best values
    """
    # Narrow depth range: +/- 2 from best, clamped to valid range
    depth_low = max(2, best_max_depth - 2)
    depth_high = min(15, best_max_depth + 2)

    max_depth_spec: IntRangeSpec = {
        "param_type": "int",
        "low": depth_low,
        "high": depth_high,
        "log_scale": False,
    }

    # Narrow learning rate: half to double the best
    lr_low = max(0.001, best_learning_rate * 0.5)
    lr_high = min(0.5, best_learning_rate * 2.0)

    n_estimators_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 75,
        "high": 200,
        "log_scale": False,
    }
    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": lr_low,
        "high": lr_high,
        "log_scale": True,
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

    space: XGBoostSearchSpace = {
        "max_depth": max_depth_spec,
        "n_estimators": n_estimators_spec,
        "learning_rate": learning_rate_spec,
        "reg_alpha": reg_alpha_spec,
        "reg_lambda": reg_lambda_spec,
        "subsample": subsample_spec,
        "colsample_bytree": colsample_spec,
    }
    return space


def make_xgboost_categorical_space() -> XGBoostSearchSpace:
    """Create XGBoost search space using categorical choices.

    Uses discrete values instead of continuous ranges.
    Useful when you want to test specific configurations.

    Returns:
        XGBoostSearchSpace with categorical parameter choices
    """
    max_depth_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (3, 4, 5, 6, 7, 8),
    }
    n_estimators_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (50, 75, 100, 150, 200),
    }
    learning_rate_spec: CategoricalFloatSpec = {
        "param_type": "categorical_float",
        "choices": (0.01, 0.05, 0.1, 0.2, 0.3),
    }
    reg_alpha_spec: CategoricalFloatSpec = {
        "param_type": "categorical_float",
        "choices": (0.0, 0.5, 1.0, 2.0, 5.0),
    }
    reg_lambda_spec: CategoricalFloatSpec = {
        "param_type": "categorical_float",
        "choices": (0.5, 1.0, 2.0, 5.0),
    }
    subsample_spec: CategoricalFloatSpec = {
        "param_type": "categorical_float",
        "choices": (0.7, 0.8, 0.9, 1.0),
    }
    colsample_spec: CategoricalFloatSpec = {
        "param_type": "categorical_float",
        "choices": (0.7, 0.8, 0.9, 1.0),
    }

    space: XGBoostSearchSpace = {
        "max_depth": max_depth_spec,
        "n_estimators": n_estimators_spec,
        "learning_rate": learning_rate_spec,
        "reg_alpha": reg_alpha_spec,
        "reg_lambda": reg_lambda_spec,
        "subsample": subsample_spec,
        "colsample_bytree": colsample_spec,
    }
    return space


def make_default_optimization_config(
    *,
    n_trials: int = 100,
    timeout_seconds: int | None = None,
    random_state: int = 42,
) -> OptimizationConfig:
    """Create default optimization configuration.

    Args:
        n_trials: Number of trials to run (default 100)
        timeout_seconds: Optional timeout in seconds (None = no timeout)
        random_state: Random seed for reproducibility

    Returns:
        OptimizationConfig with sensible defaults
    """
    config: OptimizationConfig = {
        "n_trials": n_trials,
        "timeout_seconds": timeout_seconds,
        "n_startup_trials": 10,  # Random exploration before TPE
        "random_state": random_state,
        "direction": "maximize",  # Maximize validation AUC
        "pruning_enabled": True,  # Early-stop bad trials
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }
    return config


__all__ = [
    "make_default_optimization_config",
    "make_xgboost_categorical_space",
    "make_xgboost_default_space",
    "make_xgboost_focused_space",
]
