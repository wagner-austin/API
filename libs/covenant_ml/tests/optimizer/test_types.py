"""Tests for optimizer type definitions."""

from __future__ import annotations

from covenant_ml.optimizer.types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    MLPSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    ParamSpec,
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


def test_param_spec_union_accepts_all_types() -> None:
    """ParamSpec union accepts all parameter specification types."""
    float_spec: ParamSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 1.0,
        "log_scale": False,
    }
    int_spec: ParamSpec = {
        "param_type": "int",
        "low": 1,
        "high": 10,
        "log_scale": False,
    }
    cat_float: ParamSpec = {
        "param_type": "categorical_float",
        "choices": (0.1, 0.2),
    }
    cat_int: ParamSpec = {
        "param_type": "categorical_int",
        "choices": (1, 2),
    }

    # Verify all are valid ParamSpec
    assert float_spec["param_type"] == "float"
    assert int_spec["param_type"] == "int"
    assert cat_float["param_type"] == "categorical_float"
    assert cat_int["param_type"] == "categorical_int"


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


def test_trial_result_construction() -> None:
    """TrialResult can be constructed with all required fields."""
    result: TrialResult = {
        "trial_number": 0,
        "params_max_depth": 5,
        "params_n_estimators": 100,
        "params_learning_rate": 0.1,
        "params_reg_alpha": 1.0,
        "params_reg_lambda": 2.0,
        "params_subsample": 0.8,
        "params_colsample_bytree": 0.9,
        "value": 0.85,
        "state": "complete",
        "duration_seconds": 1.5,
    }
    assert result["trial_number"] == 0
    assert result["value"] == 0.85
    assert result["state"] == "complete"


def test_trial_result_complete_state() -> None:
    """TrialResult accepts complete state."""
    result: TrialResult = {
        "trial_number": 0,
        "params_max_depth": 3,
        "params_n_estimators": 50,
        "params_learning_rate": 0.1,
        "params_reg_alpha": 0.0,
        "params_reg_lambda": 1.0,
        "params_subsample": 0.8,
        "params_colsample_bytree": 0.8,
        "value": 0.5,
        "state": "complete",
        "duration_seconds": 1.0,
    }
    assert result["state"] == "complete"


def test_trial_result_pruned_state() -> None:
    """TrialResult accepts pruned state."""
    result: TrialResult = {
        "trial_number": 1,
        "params_max_depth": 3,
        "params_n_estimators": 50,
        "params_learning_rate": 0.1,
        "params_reg_alpha": 0.0,
        "params_reg_lambda": 1.0,
        "params_subsample": 0.8,
        "params_colsample_bytree": 0.8,
        "value": 0.4,
        "state": "pruned",
        "duration_seconds": 0.5,
    }
    assert result["state"] == "pruned"


def test_trial_result_failed_state() -> None:
    """TrialResult accepts failed state."""
    result: TrialResult = {
        "trial_number": 2,
        "params_max_depth": 3,
        "params_n_estimators": 50,
        "params_learning_rate": 0.1,
        "params_reg_alpha": 0.0,
        "params_reg_lambda": 1.0,
        "params_subsample": 0.8,
        "params_colsample_bytree": 0.8,
        "value": 0.0,
        "state": "failed",
        "duration_seconds": 0.1,
    }
    assert result["state"] == "failed"


def test_trial_result_running_state() -> None:
    """TrialResult accepts running state."""
    result: TrialResult = {
        "trial_number": 3,
        "params_max_depth": 3,
        "params_n_estimators": 50,
        "params_learning_rate": 0.1,
        "params_reg_alpha": 0.0,
        "params_reg_lambda": 1.0,
        "params_subsample": 0.8,
        "params_colsample_bytree": 0.8,
        "value": 0.0,
        "state": "running",
        "duration_seconds": 0.0,
    }
    assert result["state"] == "running"


def test_optimization_summary_construction() -> None:
    """OptimizationSummary can be constructed with all required fields."""
    summary: OptimizationSummary = {
        "best_trial_number": 5,
        "best_value": 0.92,
        "best_max_depth": 6,
        "best_n_estimators": 150,
        "best_learning_rate": 0.08,
        "best_reg_alpha": 0.5,
        "best_reg_lambda": 2.0,
        "best_subsample": 0.85,
        "best_colsample_bytree": 0.9,
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
