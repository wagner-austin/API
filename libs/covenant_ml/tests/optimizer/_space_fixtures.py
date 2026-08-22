"""Search-space and optimization-config factories for optimizer tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    CategoricalStringSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    XGBoostSearchSpace,
)


def make_features(n_samples: int, n_features: int) -> NDArray[np.float64]:
    """Create feature matrix."""
    rng = np.random.default_rng(42)
    result: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    return result


def make_labels(n_samples: int) -> NDArray[np.int64]:
    """Create binary label array."""
    rng = np.random.default_rng(42)
    result: NDArray[np.int64] = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    return result


def make_xgboost_search_space() -> XGBoostSearchSpace:
    """Create a simple XGBoost search space."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=6, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=100, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.1, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
    )


def make_mlp_search_space() -> MLPSearchSpace:
    """Create a simple MLP search space."""
    return MLPSearchSpace(
        n_layers=IntRangeSpec(param_type="int", low=1, high=3, log_scale=False),
        hidden_size=IntRangeSpec(param_type="int", low=32, high=128, log_scale=False),
        batch_size=IntRangeSpec(param_type="int", low=16, high=64, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.0001, high=0.01, log_scale=True),
        dropout=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
    )


def make_lstm_search_space() -> LSTMSearchSpace:
    """Create a simple LSTM search space."""
    return LSTMSearchSpace(
        num_layers=IntRangeSpec(param_type="int", low=1, high=2, log_scale=False),
        hidden_size=IntRangeSpec(param_type="int", low=32, high=64, log_scale=False),
        batch_size=IntRangeSpec(param_type="int", low=16, high=32, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.0001, high=0.01, log_scale=True),
        dropout=FloatRangeSpec(param_type="float", low=0.0, high=0.3, log_scale=False),
    )


def make_lightgbm_search_space() -> LightGBMSearchSpace:
    """Create a simple LightGBM search space."""
    return LightGBMSearchSpace(
        n_estimators=IntRangeSpec(param_type="int", low=50, high=100, log_scale=False),
        num_leaves=IntRangeSpec(param_type="int", low=20, high=50, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.1, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
    )


def make_xgboost_dart_search_space() -> XGBoostSearchSpace:
    """Create XGBoost search space with DART booster."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=6, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=100, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.1, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        booster=CategoricalStringSpec(param_type="categorical_str", choices=("gbtree", "dart")),
        rate_drop=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
        skip_drop=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
    )


def make_lightgbm_dart_search_space() -> LightGBMSearchSpace:
    """Create LightGBM search space with DART boosting."""
    return LightGBMSearchSpace(
        n_estimators=IntRangeSpec(param_type="int", low=50, high=100, log_scale=False),
        num_leaves=IntRangeSpec(param_type="int", low=20, high=50, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.1, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        boosting_type=CategoricalStringSpec(param_type="categorical_str", choices=("gbdt", "dart")),
        drop_rate=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
        skip_drop=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
        feature_fraction=FloatRangeSpec(param_type="float", low=0.02, high=0.2, log_scale=False),
    )


def make_xgboost_categorical_space() -> XGBoostSearchSpace:
    """Create XGBoost search space with categorical int parameters."""
    return XGBoostSearchSpace(
        max_depth=CategoricalIntSpec(param_type="categorical_int", choices=(3, 5, 7)),
        n_estimators=CategoricalIntSpec(param_type="categorical_int", choices=(50, 100, 150)),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.1, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
    )


def make_xgboost_log_scale_space() -> XGBoostSearchSpace:
    """Create XGBoost search space with log-scale int parameters."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=2, high=10, log_scale=True),
        n_estimators=IntRangeSpec(param_type="int", low=10, high=500, log_scale=True),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.1, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
    )


def make_xgboost_categorical_float_space() -> XGBoostSearchSpace:
    """Create XGBoost space with categorical float params."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        learning_rate=CategoricalFloatSpec(
            param_type="categorical_float", choices=(0.01, 0.05, 0.1)
        ),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=CategoricalFloatSpec(
            param_type="categorical_float", choices=(0.7, 0.8, 0.9, 1.0)
        ),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.6, high=1.0, log_scale=False),
    )


def make_xgboost_dart_no_params_space() -> XGBoostSearchSpace:
    """Create XGBoost DART space without rate_drop/skip_drop."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        booster=CategoricalStringSpec(param_type="categorical_str", choices=("dart",)),
    )


def make_lightgbm_dart_no_params_space() -> LightGBMSearchSpace:
    """Create LightGBM DART space without drop_rate/skip_drop/feature_fraction."""
    return LightGBMSearchSpace(
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        num_leaves=IntRangeSpec(param_type="int", low=10, high=100, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        boosting_type=CategoricalStringSpec(param_type="categorical_str", choices=("dart",)),
    )


def make_xgboost_gbtree_space() -> XGBoostSearchSpace:
    """Create XGBoost space with gbtree booster (non-DART)."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        booster=CategoricalStringSpec(param_type="categorical_str", choices=("gbtree",)),
    )


def make_lightgbm_gbdt_space() -> LightGBMSearchSpace:
    """Create LightGBM space with gbdt boosting (non-DART)."""
    return LightGBMSearchSpace(
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        num_leaves=IntRangeSpec(param_type="int", low=10, high=100, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        boosting_type=CategoricalStringSpec(param_type="categorical_str", choices=("gbdt",)),
    )


def make_xgboost_narrow_range_space() -> XGBoostSearchSpace:
    """Create XGBoost space where high won't be in int_values initially."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=8, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=1.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
    )


def make_optimization_config(n_trials: int = 5) -> OptimizationConfig:
    """Create optimization config."""
    return OptimizationConfig(
        n_trials=n_trials,
        timeout_seconds=None,
        n_startup_trials=2,
        random_state=42,
        direction="maximize",
        pruning_enabled=False,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )


def make_timeout_config(
    n_trials: int,
    timeout_seconds: float,
    direction: Literal["maximize", "minimize"] = "maximize",
) -> OptimizationConfig:
    """Create config with timeout for timeout tests."""
    return OptimizationConfig(
        n_trials=n_trials,
        timeout_seconds=timeout_seconds,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
        direction=direction,
        n_startup_trials=2,
        pruning_enabled=False,
    )
