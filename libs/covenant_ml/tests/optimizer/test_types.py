"""Tests for optimizer type definitions."""

from __future__ import annotations

from covenant_ml.optimizer.types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    SearchSpace,
    TrialResult,
    XGBoostSearchSpace,
)


def test_float_range_spec_construction() -> None:
    """FloatRangeSpec can be constructed with required fields."""
    spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.01,
        "high": 0.3,
        "log_scale": True,
    }
    assert spec["param_type"] == "float"
    assert spec["low"] == 0.01
    assert spec["high"] == 0.3
    assert spec["log_scale"] is True


def test_int_range_spec_construction() -> None:
    """IntRangeSpec can be constructed with required fields."""
    spec: IntRangeSpec = {
        "param_type": "int",
        "low": 3,
        "high": 10,
        "log_scale": False,
    }
    assert spec["param_type"] == "int"
    assert spec["low"] == 3
    assert spec["high"] == 10
    assert spec["log_scale"] is False


def test_categorical_float_spec_construction() -> None:
    """CategoricalFloatSpec can be constructed with choices."""
    spec: CategoricalFloatSpec = {
        "param_type": "categorical_float",
        "choices": (0.01, 0.05, 0.1, 0.2),
    }
    assert spec["param_type"] == "categorical_float"
    assert spec["choices"] == (0.01, 0.05, 0.1, 0.2)


def test_categorical_int_spec_construction() -> None:
    """CategoricalIntSpec can be constructed with choices."""
    spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (3, 5, 7, 10),
    }
    assert spec["param_type"] == "categorical_int"
    assert spec["choices"] == (3, 5, 7, 10)


def test_xgboost_search_space_construction() -> None:
    """XGBoostSearchSpace can be constructed with all required fields."""
    lr_spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.01,
        "high": 0.3,
        "log_scale": True,
    }
    space: XGBoostSearchSpace = {
        "max_depth": {"param_type": "int", "low": 3, "high": 10, "log_scale": False},
        "n_estimators": {"param_type": "int", "low": 50, "high": 300, "log_scale": False},
        "learning_rate": lr_spec,
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
    }
    assert space["max_depth"]["param_type"] == "int"
    assert lr_spec["log_scale"] is True


def test_xgboost_search_space_with_categorical() -> None:
    """XGBoostSearchSpace accepts categorical specs."""
    space: XGBoostSearchSpace = {
        "max_depth": {"param_type": "categorical_int", "choices": (3, 5, 7)},
        "n_estimators": {"param_type": "categorical_int", "choices": (50, 100, 200)},
        "learning_rate": {"param_type": "categorical_float", "choices": (0.01, 0.1, 0.3)},
        "reg_alpha": {"param_type": "categorical_float", "choices": (0.0, 1.0, 5.0)},
        "reg_lambda": {"param_type": "categorical_float", "choices": (0.5, 1.0, 5.0)},
        "subsample": {"param_type": "categorical_float", "choices": (0.7, 0.8, 1.0)},
        "colsample_bytree": {"param_type": "categorical_float", "choices": (0.7, 0.8, 1.0)},
    }
    assert space["max_depth"]["param_type"] == "categorical_int"
    assert space["learning_rate"]["param_type"] == "categorical_float"


def test_mlp_search_space_construction() -> None:
    """MLPSearchSpace can be constructed with all required fields."""
    space: MLPSearchSpace = {
        "n_layers": {"param_type": "int", "low": 1, "high": 4, "log_scale": False},
        "hidden_size": {"param_type": "int", "low": 32, "high": 256, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.0001, "high": 0.01, "log_scale": True},
        "dropout": {"param_type": "float", "low": 0.0, "high": 0.5, "log_scale": False},
        "batch_size": {"param_type": "categorical_int", "choices": (16, 32, 64, 128)},
    }
    assert space["n_layers"]["param_type"] == "int"
    assert space["batch_size"]["param_type"] == "categorical_int"


def test_lstm_search_space_construction() -> None:
    """LSTMSearchSpace can be constructed with all required fields."""
    space: LSTMSearchSpace = {
        "hidden_size": {"param_type": "categorical_int", "choices": (64, 128, 256)},
        "num_layers": {"param_type": "int", "low": 1, "high": 3, "log_scale": False},
        "dropout": {"param_type": "float", "low": 0.0, "high": 0.5, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 1e-5, "high": 1e-2, "log_scale": True},
        "batch_size": {"param_type": "categorical_int", "choices": (16, 32, 64)},
    }
    assert space["hidden_size"]["param_type"] == "categorical_int"
    assert space["num_layers"]["param_type"] == "int"


def test_lightgbm_search_space_construction() -> None:
    """LightGBMSearchSpace can be constructed with all required fields.

    Note: max_depth is intentionally excluded. LightGBM uses leaf-wise growth
    where num_leaves is the primary complexity control. Using max_depth=-1
    (unlimited) avoids constraint conflicts when num_leaves > 2^max_depth.
    """
    space: LightGBMSearchSpace = {
        "n_estimators": {"param_type": "int", "low": 50, "high": 500, "log_scale": False},
        "num_leaves": {"param_type": "int", "low": 20, "high": 100, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
    }
    assert space["num_leaves"]["param_type"] == "int"
    assert space["n_estimators"]["param_type"] == "int"


def test_sampled_int_params_construction() -> None:
    """SampledIntParams can be constructed with optional fields."""
    params: SampledIntParams = {
        "max_depth": 5,
        "n_estimators": 100,
    }
    assert params["max_depth"] == 5
    assert params["n_estimators"] == 100


def test_sampled_float_params_construction() -> None:
    """SampledFloatParams can be constructed with optional fields."""
    params: SampledFloatParams = {
        "learning_rate": 0.1,
        "dropout": 0.2,
        "subsample": 0.8,
    }
    assert params["learning_rate"] == 0.1
    assert params["dropout"] == 0.2
    assert params["subsample"] == 0.8


def test_trial_result_construction() -> None:
    """TrialResult can be constructed with all required fields."""
    int_params: SampledIntParams = {
        "max_depth": 5,
        "n_estimators": 100,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.1,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
    }
    string_params: SampledStringParams = {}
    result: TrialResult = {
        "trial_number": 0,
        "int_params": int_params,
        "float_params": float_params,
        "string_params": string_params,
        "value": 0.85,
        "state": "complete",
        "duration_seconds": 1.5,
    }
    assert result["trial_number"] == 0
    assert result["value"] == 0.85
    assert result["state"] == "complete"
    assert result["int_params"]["max_depth"] == 5


def test_trial_result_complete_state() -> None:
    """TrialResult accepts complete state."""
    result: TrialResult = {
        "trial_number": 0,
        "int_params": {"max_depth": 3},
        "float_params": {"learning_rate": 0.1},
        "string_params": {},
        "value": 0.5,
        "state": "complete",
        "duration_seconds": 1.0,
    }
    assert result["state"] == "complete"


def test_trial_result_pruned_state() -> None:
    """TrialResult accepts pruned state."""
    result: TrialResult = {
        "trial_number": 1,
        "int_params": {"max_depth": 3},
        "float_params": {"learning_rate": 0.1},
        "string_params": {},
        "value": 0.4,
        "state": "pruned",
        "duration_seconds": 0.5,
    }
    assert result["state"] == "pruned"


def test_trial_result_failed_state() -> None:
    """TrialResult accepts failed state."""
    result: TrialResult = {
        "trial_number": 2,
        "int_params": {"max_depth": 3},
        "float_params": {"learning_rate": 0.1},
        "string_params": {},
        "value": 0.0,
        "state": "failed",
        "duration_seconds": 0.1,
    }
    assert result["state"] == "failed"


def test_trial_result_running_state() -> None:
    """TrialResult accepts running state."""
    result: TrialResult = {
        "trial_number": 3,
        "int_params": {"max_depth": 3},
        "float_params": {"learning_rate": 0.1},
        "string_params": {},
        "value": 0.0,
        "state": "running",
        "duration_seconds": 0.0,
    }
    assert result["state"] == "running"


def test_optimization_summary_construction() -> None:
    """OptimizationSummary can be constructed with all required fields."""
    best_int_params: SampledIntParams = {
        "max_depth": 6,
        "n_estimators": 150,
    }
    best_float_params: SampledFloatParams = {
        "learning_rate": 0.08,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
    }
    best_string_params: SampledStringParams = {}
    summary: OptimizationSummary = {
        "best_trial_number": 5,
        "best_value": 0.92,
        "best_int_params": best_int_params,
        "best_float_params": best_float_params,
        "best_string_params": best_string_params,
        "n_trials_total": 50,
        "n_trials_complete": 48,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "total_duration_seconds": 120.5,
    }
    assert summary["best_trial_number"] == 5
    assert summary["best_value"] == 0.92
    assert summary["n_trials_total"] == 50
    assert summary["n_trials_complete"] == 48
    assert summary["best_int_params"]["max_depth"] == 6


def test_optimization_config_construction() -> None:
    """OptimizationConfig can be constructed with all required fields."""
    config: OptimizationConfig = {
        "n_trials": 100,
        "timeout_seconds": 3600,
        "n_startup_trials": 10,
        "random_state": 42,
        "direction": "maximize",
        "pruning_enabled": True,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }
    assert config["n_trials"] == 100
    assert config["timeout_seconds"] == 3600
    assert config["direction"] == "maximize"


def test_optimization_config_null_timeout() -> None:
    """OptimizationConfig accepts None for timeout_seconds."""
    config: OptimizationConfig = {
        "n_trials": 50,
        "timeout_seconds": None,
        "n_startup_trials": 5,
        "random_state": 123,
        "direction": "minimize",
        "pruning_enabled": False,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
    }
    assert config["timeout_seconds"] is None
    assert config["direction"] == "minimize"
    assert config["pruning_enabled"] is False


def test_search_space_union_accepts_xgboost() -> None:
    """SearchSpace union accepts XGBoostSearchSpace."""
    xgb_space: XGBoostSearchSpace = {
        "max_depth": {"param_type": "int", "low": 3, "high": 10, "log_scale": False},
        "n_estimators": {"param_type": "int", "low": 50, "high": 300, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
    }
    space: SearchSpace = xgb_space
    assert "max_depth" in space
    assert "n_estimators" in space


def test_search_space_union_accepts_mlp() -> None:
    """SearchSpace union accepts MLPSearchSpace."""
    mlp_space: MLPSearchSpace = {
        "n_layers": {"param_type": "int", "low": 1, "high": 4, "log_scale": False},
        "hidden_size": {"param_type": "int", "low": 32, "high": 256, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.0001, "high": 0.01, "log_scale": True},
        "dropout": {"param_type": "float", "low": 0.0, "high": 0.5, "log_scale": False},
        "batch_size": {"param_type": "categorical_int", "choices": (16, 32, 64, 128)},
    }
    space: SearchSpace = mlp_space
    assert "n_layers" in space
    assert "hidden_size" in space


def test_search_space_union_accepts_lstm() -> None:
    """SearchSpace union accepts LSTMSearchSpace."""
    lstm_space: LSTMSearchSpace = {
        "hidden_size": {"param_type": "categorical_int", "choices": (64, 128, 256)},
        "num_layers": {"param_type": "int", "low": 1, "high": 3, "log_scale": False},
        "dropout": {"param_type": "float", "low": 0.0, "high": 0.5, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 1e-5, "high": 1e-2, "log_scale": True},
        "batch_size": {"param_type": "categorical_int", "choices": (16, 32, 64)},
    }
    space: SearchSpace = lstm_space
    assert "num_layers" in space
    assert "hidden_size" in space


def test_search_space_union_accepts_lightgbm() -> None:
    """SearchSpace union accepts LightGBMSearchSpace.

    Note: max_depth is intentionally excluded from LightGBM search space.
    """
    lgbm_space: LightGBMSearchSpace = {
        "n_estimators": {"param_type": "int", "low": 50, "high": 500, "log_scale": False},
        "num_leaves": {"param_type": "int", "low": 20, "high": 100, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
    }
    space: SearchSpace = lgbm_space
    assert "num_leaves" in space
    assert "n_estimators" in space
