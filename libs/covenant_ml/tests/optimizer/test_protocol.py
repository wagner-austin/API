"""Tests for optimizer protocol definitions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.protocol import (
    TrialCallbackProtocol,
    XGBoostObjectiveCallable,
    XGBoostObjectiveProtocol,
    XGBoostOptimizerProtocol,
)
from covenant_ml.optimizer.types import (
    OptimizationConfig,
    OptimizationSummary,
    TrialResult,
    XGBoostSearchSpace,
)

# =============================================================================
# Protocol Implementation Tests
# =============================================================================


class _ConcreteTrialCallback:
    """Concrete implementation of TrialCallbackProtocol."""

    def __init__(self) -> None:
        self.results: list[TrialResult] = []

    def __call__(self, result: TrialResult) -> None:
        self.results.append(result)


def test_trial_callback_protocol_implementation() -> None:
    """Concrete class can implement TrialCallbackProtocol."""
    concrete_callback = _ConcreteTrialCallback()
    callback: TrialCallbackProtocol = concrete_callback

    result: TrialResult = {
        "trial_number": 0,
        "params_max_depth": 5,
        "params_n_estimators": 100,
        "params_learning_rate": 0.1,
        "params_reg_alpha": 0.0,
        "params_reg_lambda": 1.0,
        "params_subsample": 0.8,
        "params_colsample_bytree": 0.8,
        "value": 0.85,
        "state": "complete",
        "duration_seconds": 1.5,
    }

    callback(result)

    # Verify the concrete implementation received the call
    assert len(concrete_callback.results) == 1
    assert concrete_callback.results[0]["trial_number"] == 0


class _ConcreteXGBoostObjective:
    """Concrete implementation of XGBoostObjectiveProtocol."""

    def __init__(self, return_value: float = 0.85) -> None:
        self._return_value = return_value
        self.call_count = 0

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        max_depth: int,
        n_estimators: int,
        learning_rate: float,
        reg_alpha: float,
        reg_lambda: float,
        subsample: float,
        colsample_bytree: float,
        random_state: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> float:
        _ = (
            x_features,
            y_labels,
            feature_names,
            max_depth,
            n_estimators,
            learning_rate,
            reg_alpha,
            reg_lambda,
            subsample,
            colsample_bytree,
            random_state,
            train_ratio,
            val_ratio,
            test_ratio,
        )
        self.call_count += 1
        return self._return_value


def test_xgboost_objective_protocol_implementation() -> None:
    """Concrete class can implement XGBoostObjectiveProtocol."""
    concrete_objective = _ConcreteXGBoostObjective(return_value=0.92)
    objective: XGBoostObjectiveProtocol = concrete_objective

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)
    names = ["f0", "f1", "f2", "f3"]

    result = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        reg_alpha=1.0,
        reg_lambda=2.0,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    assert result == 0.92
    assert concrete_objective.call_count == 1


class _ConcreteXGBoostOptimizer:
    """Concrete implementation of XGBoostOptimizerProtocol."""

    def __init__(self) -> None:
        self.optimize_called = False

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: XGBoostSearchSpace,
        config: OptimizationConfig,
        objective: XGBoostObjectiveProtocol,
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

        summary: OptimizationSummary = {
            "best_trial_number": 0,
            "best_value": 0.9,
            "best_max_depth": 5,
            "best_n_estimators": 100,
            "best_learning_rate": 0.1,
            "best_reg_alpha": 1.0,
            "best_reg_lambda": 2.0,
            "best_subsample": 0.8,
            "best_colsample_bytree": 0.9,
            "n_trials_total": 1,
            "n_trials_complete": 1,
            "n_trials_pruned": 0,
            "n_trials_failed": 0,
            "total_duration_seconds": 1.0,
        }
        return summary


def test_xgboost_optimizer_protocol_implementation() -> None:
    """Concrete class can implement XGBoostOptimizerProtocol."""
    concrete_optimizer = _ConcreteXGBoostOptimizer()
    optimizer: XGBoostOptimizerProtocol = concrete_optimizer

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)
    names = ["f0", "f1", "f2", "f3"]

    space: XGBoostSearchSpace = {
        "max_depth": {"param_type": "int", "low": 3, "high": 10, "log_scale": False},
        "n_estimators": {"param_type": "int", "low": 50, "high": 300, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
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

    objective = _ConcreteXGBoostObjective()

    summary = optimizer.optimize(
        x_features=x,
        y_labels=y,
        feature_names=names,
        search_space=space,
        config=config,
        objective=objective,
    )

    assert summary["best_value"] == 0.9
    assert concrete_optimizer.optimize_called is True


def test_xgboost_optimizer_protocol_with_callback() -> None:
    """XGBoostOptimizerProtocol accepts trial callback."""
    concrete_optimizer = _ConcreteXGBoostOptimizer()
    optimizer: XGBoostOptimizerProtocol = concrete_optimizer
    callback = _ConcreteTrialCallback()

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)
    names = ["f0", "f1", "f2", "f3"]

    space: XGBoostSearchSpace = {
        "max_depth": {"param_type": "int", "low": 3, "high": 10, "log_scale": False},
        "n_estimators": {"param_type": "int", "low": 50, "high": 300, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
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

    objective = _ConcreteXGBoostObjective()

    _ = optimizer.optimize(
        x_features=x,
        y_labels=y,
        feature_names=names,
        search_space=space,
        config=config,
        objective=objective,
        trial_callback=callback,
    )

    # Verify optimizer.optimize was called with callback
    assert concrete_optimizer.optimize_called is True


# =============================================================================
# Callable Type Alias Tests
# =============================================================================


def _standalone_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    reg_alpha: float,
    reg_lambda: float,
    subsample: float,
    colsample_bytree: float,
    random_state: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> float:
    """Standalone function that matches XGBoostObjectiveCallable signature."""
    _ = (
        x_features,
        y_labels,
        feature_names,
        max_depth,
        n_estimators,
        learning_rate,
        reg_alpha,
        reg_lambda,
        subsample,
        colsample_bytree,
        random_state,
        train_ratio,
        val_ratio,
        test_ratio,
    )
    return 0.88


def test_xgboost_objective_callable_type_alias() -> None:
    """Standalone function can be typed as XGBoostObjectiveCallable."""
    objective: XGBoostObjectiveCallable = _standalone_objective

    x = np.zeros((10, 4), dtype=np.float64)
    y = np.zeros(10, dtype=np.int64)
    names = ["f0", "f1", "f2", "f3"]

    result = objective(
        x,
        y,
        names,
        5,
        100,
        0.1,
        1.0,
        2.0,
        0.8,
        0.9,
        42,
        0.7,
        0.15,
        0.15,
    )

    assert result == 0.88
