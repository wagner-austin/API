"""Predefined search spaces for hyperparameter optimization.

Strict typing only: no Any, no casts, no stubs.
Provides sensible default search ranges for XGBoost, MLP, LSTM, and LightGBM.
"""

from __future__ import annotations

from .types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    XGBoostSearchSpace,
)

# =============================================================================
# XGBoost Search Spaces
# =============================================================================


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


# =============================================================================
# MLP Search Spaces
# =============================================================================


def make_mlp_default_space() -> MLPSearchSpace:
    """Create default MLP search space for bankruptcy prediction.

    Based on empirical testing for tabular data:
    - n_layers 1-4 (deeper not usually better for tabular)
    - hidden_size 64-512 (common hidden layer sizes)
    - learning_rate 1e-5 to 1e-2 in log scale
    - dropout 0.0-0.5 for regularization
    - batch_size 32-256 for stable gradients

    Returns:
        MLPSearchSpace with sensible default ranges
    """
    n_layers_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 1,
        "high": 4,
        "log_scale": False,
    }
    hidden_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (64, 128, 256, 512),
    }
    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "log_scale": True,
    }
    dropout_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.5,
        "log_scale": False,
    }
    batch_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (32, 64, 128, 256),
    }

    space: MLPSearchSpace = {
        "n_layers": n_layers_spec,
        "hidden_size": hidden_size_spec,
        "learning_rate": learning_rate_spec,
        "dropout": dropout_spec,
        "batch_size": batch_size_spec,
    }
    return space


def make_mlp_focused_space(
    *,
    best_n_layers: int,
    best_hidden_size: int,
    best_learning_rate: float,
) -> MLPSearchSpace:
    """Create focused MLP search space around known good values.

    Args:
        best_n_layers: Best n_layers from initial search
        best_hidden_size: Best hidden_size from initial search
        best_learning_rate: Best learning_rate from initial search

    Returns:
        MLPSearchSpace with narrowed ranges around best values
    """
    layers_low = max(1, best_n_layers - 1)
    layers_high = min(5, best_n_layers + 1)

    n_layers_spec: IntRangeSpec = {
        "param_type": "int",
        "low": layers_low,
        "high": layers_high,
        "log_scale": False,
    }

    hidden_choices: list[int] = []
    for size in [32, 64, 128, 256, 512]:
        if abs(size - best_hidden_size) <= best_hidden_size:
            hidden_choices.append(size)
    if not hidden_choices:
        hidden_choices = [best_hidden_size]

    hidden_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": tuple(hidden_choices),
    }

    lr_low = max(1e-6, best_learning_rate * 0.1)
    lr_high = min(1e-1, best_learning_rate * 10.0)

    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": lr_low,
        "high": lr_high,
        "log_scale": True,
    }
    dropout_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.4,
        "log_scale": False,
    }
    batch_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (32, 64, 128),
    }

    space: MLPSearchSpace = {
        "n_layers": n_layers_spec,
        "hidden_size": hidden_size_spec,
        "learning_rate": learning_rate_spec,
        "dropout": dropout_spec,
        "batch_size": batch_size_spec,
    }
    return space


# =============================================================================
# LSTM Search Spaces
# =============================================================================


def make_lstm_default_space() -> LSTMSearchSpace:
    """Create default LSTM search space for temporal bankruptcy prediction.

    Based on empirical testing for sequential financial data:
    - hidden_size 64-256 (smaller than NLP tasks)
    - num_layers 1-3 (deeper LSTMs often overfit on financial data)
    - learning_rate 1e-5 to 1e-2 in log scale
    - dropout 0.0-0.5 for regularization
    - batch_size 16-64 (smaller batches for sequential data)

    Returns:
        LSTMSearchSpace with sensible default ranges
    """
    hidden_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (64, 128, 256),
    }
    num_layers_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 1,
        "high": 3,
        "log_scale": False,
    }
    dropout_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.5,
        "log_scale": False,
    }
    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "log_scale": True,
    }
    batch_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (16, 32, 64),
    }

    space: LSTMSearchSpace = {
        "hidden_size": hidden_size_spec,
        "num_layers": num_layers_spec,
        "dropout": dropout_spec,
        "learning_rate": learning_rate_spec,
        "batch_size": batch_size_spec,
    }
    return space


def make_lstm_focused_space(
    *,
    best_hidden_size: int,
    best_num_layers: int,
    best_learning_rate: float,
) -> LSTMSearchSpace:
    """Create focused LSTM search space around known good values.

    Args:
        best_hidden_size: Best hidden_size from initial search
        best_num_layers: Best num_layers from initial search
        best_learning_rate: Best learning_rate from initial search

    Returns:
        LSTMSearchSpace with narrowed ranges around best values
    """
    hidden_choices: list[int] = []
    for size in [32, 64, 128, 256, 512]:
        if abs(size - best_hidden_size) <= best_hidden_size:
            hidden_choices.append(size)
    if not hidden_choices:
        hidden_choices = [best_hidden_size]

    hidden_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": tuple(hidden_choices),
    }

    layers_low = max(1, best_num_layers - 1)
    layers_high = min(4, best_num_layers + 1)

    num_layers_spec: IntRangeSpec = {
        "param_type": "int",
        "low": layers_low,
        "high": layers_high,
        "log_scale": False,
    }
    dropout_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 0.4,
        "log_scale": False,
    }

    lr_low = max(1e-6, best_learning_rate * 0.1)
    lr_high = min(1e-1, best_learning_rate * 10.0)

    learning_rate_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": lr_low,
        "high": lr_high,
        "log_scale": True,
    }
    batch_size_spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (16, 32),
    }

    space: LSTMSearchSpace = {
        "hidden_size": hidden_size_spec,
        "num_layers": num_layers_spec,
        "dropout": dropout_spec,
        "learning_rate": learning_rate_spec,
        "batch_size": batch_size_spec,
    }
    return space


# =============================================================================
# LightGBM Search Spaces
# =============================================================================


def make_lightgbm_default_space() -> LightGBMSearchSpace:
    """Create default LightGBM search space for bankruptcy prediction.

    Based on empirical testing:
    - max_depth 3-12 (LightGBM uses leaf-wise growth)
    - n_estimators 50-500 (more trees for larger datasets)
    - num_leaves 20-100 (key LightGBM parameter)
    - learning_rate 0.01-0.3 in log scale
    - Regularization helps prevent overfitting

    Returns:
        LightGBMSearchSpace with sensible default ranges
    """
    max_depth_spec: IntRangeSpec = {
        "param_type": "int",
        "low": 3,
        "high": 12,
        "log_scale": False,
    }
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
        "high": 10.0,
        "log_scale": True,
    }

    space: LightGBMSearchSpace = {
        "max_depth": max_depth_spec,
        "n_estimators": n_estimators_spec,
        "num_leaves": num_leaves_spec,
        "learning_rate": learning_rate_spec,
        "subsample": subsample_spec,
        "colsample_bytree": colsample_spec,
        "reg_alpha": reg_alpha_spec,
        "reg_lambda": reg_lambda_spec,
    }
    return space


def make_lightgbm_focused_space(
    *,
    best_max_depth: int,
    best_num_leaves: int,
    best_learning_rate: float,
) -> LightGBMSearchSpace:
    """Create focused LightGBM search space around known good values.

    Args:
        best_max_depth: Best max_depth from initial search
        best_num_leaves: Best num_leaves from initial search
        best_learning_rate: Best learning_rate from initial search

    Returns:
        LightGBMSearchSpace with narrowed ranges around best values
    """
    depth_low = max(2, best_max_depth - 2)
    depth_high = min(15, best_max_depth + 2)

    max_depth_spec: IntRangeSpec = {
        "param_type": "int",
        "low": depth_low,
        "high": depth_high,
        "log_scale": False,
    }
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
        "max_depth": max_depth_spec,
        "n_estimators": n_estimators_spec,
        "num_leaves": num_leaves_spec,
        "learning_rate": learning_rate_spec,
        "subsample": subsample_spec,
        "colsample_bytree": colsample_spec,
        "reg_alpha": reg_alpha_spec,
        "reg_lambda": reg_lambda_spec,
    }
    return space


# =============================================================================
# Optimization Configuration
# =============================================================================


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
    "make_lightgbm_default_space",
    "make_lightgbm_focused_space",
    "make_lstm_default_space",
    "make_lstm_focused_space",
    "make_mlp_default_space",
    "make_mlp_focused_space",
    "make_xgboost_categorical_space",
    "make_xgboost_default_space",
    "make_xgboost_focused_space",
]
