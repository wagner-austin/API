"""Shared fixtures and helpers for test_space_narrowing splits."""

from __future__ import annotations

from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    XGBoostSearchSpace,
)


def _make_xgboost_space() -> XGBoostSearchSpace:
    """Create a standard XGBoost search space."""
    return XGBoostSearchSpace(
        max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=10.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=10.0, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
    )


def _make_mlp_space() -> MLPSearchSpace:
    """Create a standard MLP search space."""
    return MLPSearchSpace(
        n_layers=IntRangeSpec(param_type="int", low=1, high=5, log_scale=False),
        hidden_size=IntRangeSpec(param_type="int", low=32, high=256, log_scale=False),
        batch_size=IntRangeSpec(param_type="int", low=16, high=128, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.0001, high=0.01, log_scale=True),
        dropout=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
    )


def _make_lstm_space() -> LSTMSearchSpace:
    """Create a standard LSTM search space."""
    return LSTMSearchSpace(
        hidden_size=IntRangeSpec(param_type="int", low=32, high=256, log_scale=False),
        num_layers=IntRangeSpec(param_type="int", low=1, high=4, log_scale=False),
        batch_size=IntRangeSpec(param_type="int", low=16, high=128, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.0001, high=0.01, log_scale=True),
        dropout=FloatRangeSpec(param_type="float", low=0.0, high=0.5, log_scale=False),
    )


def _make_lightgbm_space() -> LightGBMSearchSpace:
    """Create a standard LightGBM search space."""
    return LightGBMSearchSpace(
        n_estimators=IntRangeSpec(param_type="int", low=50, high=200, log_scale=False),
        num_leaves=IntRangeSpec(param_type="int", low=15, high=127, log_scale=False),
        learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
        subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        reg_alpha=FloatRangeSpec(param_type="float", low=0.001, high=10.0, log_scale=True),
        reg_lambda=FloatRangeSpec(param_type="float", low=0.001, high=10.0, log_scale=True),
    )
