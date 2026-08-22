"""Tests for optimizer protocol definitions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.protocol import (
    LightGBMOptimizerProtocol,
    LSTMOptimizerProtocol,
    MLPOptimizerProtocol,
    ObjectiveProtocol,
    TrialCallbackProtocol,
)
from covenant_ml.optimizer.types import (
    LightGBMSearchSpace,
    LSTMSearchSpace,
    MLPSearchSpace,
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from tests.optimizer._protocol_fixtures import (
    _ConcreteObjective,
)


class _ConcreteMLPOptimizer:
    """Concrete implementation of MLPOptimizerProtocol."""

    def __init__(self) -> None:
        self.optimize_called = False

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: MLPSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        _ = (
            x_features,
            y_labels,
            feature_names,
            search_space,
            config,
            objective,
            trial_callback,
        )
        self.optimize_called = True

        best_int_params: SampledIntParams = {
            "n_layers": 2,
            "hidden_size": 128,
            "batch_size": 64,
        }
        best_float_params: SampledFloatParams = {
            "learning_rate": 0.001,
            "dropout": 0.2,
        }
        best_string_params: SampledStringParams = {}
        summary: OptimizationSummary = {
            "best_trial_number": 0,
            "best_value": 0.88,
            "best_int_params": best_int_params,
            "best_float_params": best_float_params,
            "best_string_params": best_string_params,
            "n_trials_total": 1,
            "n_trials_complete": 1,
            "n_trials_pruned": 0,
            "n_trials_failed": 0,
            "total_duration_seconds": 2.0,
        }
        return summary


def test_mlp_optimizer_protocol_implementation() -> None:
    """Concrete class can implement MLPOptimizerProtocol."""
    concrete_optimizer = _ConcreteMLPOptimizer()
    optimizer: MLPOptimizerProtocol = concrete_optimizer

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)
    names = ["f0", "f1", "f2", "f3"]

    space: MLPSearchSpace = {
        "n_layers": {"param_type": "int", "low": 1, "high": 4, "log_scale": False},
        "hidden_size": {"param_type": "int", "low": 32, "high": 256, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.0001, "high": 0.01, "log_scale": True},
        "dropout": {"param_type": "float", "low": 0.0, "high": 0.5, "log_scale": False},
        "batch_size": {"param_type": "categorical_int", "choices": (16, 32, 64, 128)},
    }

    config: OptimizationConfig = {
        "n_trials": 1,
        "timeout_seconds": None,
        "n_startup_trials": 1,
        "random_state": 42,
        "direction": "maximize",
        "pruning_enabled": False,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }

    objective = _ConcreteObjective()

    summary = optimizer.optimize(
        x_features=x,
        y_labels=y,
        feature_names=names,
        search_space=space,
        config=config,
        objective=objective,
    )

    assert summary["best_value"] == 0.88
    assert concrete_optimizer.optimize_called is True


class _ConcreteLSTMOptimizer:
    """Concrete implementation of LSTMOptimizerProtocol."""

    def __init__(self) -> None:
        self.optimize_called = False

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: LSTMSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        _ = (
            x_features,
            y_labels,
            feature_names,
            search_space,
            config,
            objective,
            trial_callback,
        )
        self.optimize_called = True

        best_int_params: SampledIntParams = {
            "hidden_size": 128,
            "num_layers": 2,
            "batch_size": 32,
        }
        best_float_params: SampledFloatParams = {
            "learning_rate": 0.001,
            "dropout": 0.3,
        }
        best_string_params: SampledStringParams = {}
        summary: OptimizationSummary = {
            "best_trial_number": 0,
            "best_value": 0.91,
            "best_int_params": best_int_params,
            "best_float_params": best_float_params,
            "best_string_params": best_string_params,
            "n_trials_total": 1,
            "n_trials_complete": 1,
            "n_trials_pruned": 0,
            "n_trials_failed": 0,
            "total_duration_seconds": 5.0,
        }
        return summary


def test_lstm_optimizer_protocol_implementation() -> None:
    """Concrete class can implement LSTMOptimizerProtocol."""
    concrete_optimizer = _ConcreteLSTMOptimizer()
    optimizer: LSTMOptimizerProtocol = concrete_optimizer

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)
    names = ["f0", "f1", "f2", "f3"]

    space: LSTMSearchSpace = {
        "hidden_size": {"param_type": "categorical_int", "choices": (64, 128, 256)},
        "num_layers": {"param_type": "int", "low": 1, "high": 3, "log_scale": False},
        "dropout": {"param_type": "float", "low": 0.0, "high": 0.5, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 1e-5, "high": 1e-2, "log_scale": True},
        "batch_size": {"param_type": "categorical_int", "choices": (16, 32, 64)},
    }

    config: OptimizationConfig = {
        "n_trials": 1,
        "timeout_seconds": None,
        "n_startup_trials": 1,
        "random_state": 42,
        "direction": "maximize",
        "pruning_enabled": False,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }

    objective = _ConcreteObjective()

    summary = optimizer.optimize(
        x_features=x,
        y_labels=y,
        feature_names=names,
        search_space=space,
        config=config,
        objective=objective,
    )

    assert summary["best_value"] == 0.91
    assert concrete_optimizer.optimize_called is True


class _ConcreteLightGBMOptimizer:
    """Concrete implementation of LightGBMOptimizerProtocol."""

    def __init__(self) -> None:
        self.optimize_called = False

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: LightGBMSearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        _ = (
            x_features,
            y_labels,
            feature_names,
            search_space,
            config,
            objective,
            trial_callback,
        )
        self.optimize_called = True

        best_int_params: SampledIntParams = {
            "max_depth": 6,
            "n_estimators": 200,
            "num_leaves": 50,
        }
        best_float_params: SampledFloatParams = {
            "learning_rate": 0.05,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
        }
        best_string_params: SampledStringParams = {}
        summary: OptimizationSummary = {
            "best_trial_number": 0,
            "best_value": 0.93,
            "best_int_params": best_int_params,
            "best_float_params": best_float_params,
            "best_string_params": best_string_params,
            "n_trials_total": 1,
            "n_trials_complete": 1,
            "n_trials_pruned": 0,
            "n_trials_failed": 0,
            "total_duration_seconds": 3.0,
        }
        return summary


def test_lightgbm_optimizer_protocol_implementation() -> None:
    """Concrete class can implement LightGBMOptimizerProtocol."""
    concrete_optimizer = _ConcreteLightGBMOptimizer()
    optimizer: LightGBMOptimizerProtocol = concrete_optimizer

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)
    names = ["f0", "f1", "f2", "f3"]

    # Note: max_depth is intentionally excluded - num_leaves controls complexity
    space: LightGBMSearchSpace = {
        "n_estimators": {"param_type": "int", "low": 50, "high": 500, "log_scale": False},
        "num_leaves": {"param_type": "int", "low": 20, "high": 100, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
    }

    config: OptimizationConfig = {
        "n_trials": 1,
        "timeout_seconds": None,
        "n_startup_trials": 1,
        "random_state": 42,
        "direction": "maximize",
        "pruning_enabled": False,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }

    objective = _ConcreteObjective()

    summary = optimizer.optimize(
        x_features=x,
        y_labels=y,
        feature_names=names,
        search_space=space,
        config=config,
        objective=objective,
    )

    assert summary["best_value"] == 0.93
    assert concrete_optimizer.optimize_called is True


def _standalone_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
    string_params: SampledStringParams,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> float:
    """Standalone function that matches ObjectiveProtocol signature."""
    _ = (
        x_features,
        y_labels,
        feature_names,
        int_params,
        float_params,
        string_params,
        train_ratio,
        val_ratio,
        test_ratio,
        random_state,
    )
    return 0.88


def test_standalone_function_matches_objective_protocol() -> None:
    """Standalone function can be used as ObjectiveProtocol."""
    objective: ObjectiveProtocol = _standalone_objective

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)
    names = ["f0", "f1", "f2", "f3"]

    int_params: SampledIntParams = {"max_depth": 5}
    float_params: SampledFloatParams = {"learning_rate": 0.1}
    string_params: SampledStringParams = {}

    result = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert result == 0.88
