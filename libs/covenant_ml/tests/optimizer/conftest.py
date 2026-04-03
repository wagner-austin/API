"""Shared fixtures and helpers for optimizer tests.

Provides common test utilities for strategy tests, Optuna backend tests,
and per-backend optimizer tests.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.optuna_backend._protocols import (
    OptunaCreateStudyProtocol,
    OptunaMedianPrunerProtocol,
    OptunaPrunerProtocol,
    OptunaSamplerProtocol,
    OptunaStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaTrialProtocol,
)
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
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    XGBoostSearchSpace,
)

# =============================================================================
# Data Helpers
# =============================================================================


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


# =============================================================================
# Search Space Helpers
# =============================================================================


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


# =============================================================================
# Config Helpers
# =============================================================================


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


# =============================================================================
# Objective Helpers
# =============================================================================


def dummy_objective(
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
    """Dummy objective that returns a random value."""
    rng = np.random.default_rng(random_state + int_params.get("max_depth", 0))
    return float(rng.uniform(0.5, 0.9))


def mlp_objective(
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
    """Dummy objective for MLP."""
    return 0.75


def lstm_objective(
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
    """Dummy objective for LSTM."""
    return 0.70


def lightgbm_objective(
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
    """Dummy objective for LightGBM."""
    return 0.80


def xgboost_dart_objective(
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
    """Dummy objective for XGBoost DART."""
    return 0.82


def lightgbm_dart_objective(
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
    """Dummy objective for LightGBM DART."""
    return 0.78


def xgboost_dart_no_params_objective(
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
    """Dummy objective for XGBoost DART without extra params."""
    return 0.85


def lightgbm_dart_no_params_objective(
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
    """Dummy objective for LightGBM DART without extra params."""
    return 0.82


def slow_objective(
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
    """Slow objective that sleeps to trigger timeout."""
    import time

    time.sleep(0.2)
    return 0.85


# =============================================================================
# Fake Optuna Implementation
# =============================================================================


class FakeTrial:
    """Fake Optuna trial that returns deterministic values."""

    def __init__(self, trial_number: int) -> None:
        self._number = trial_number
        self._suggestions: dict[str, float | int | str] = {}

    @property
    def number(self) -> int:
        """Return trial number."""
        return self._number

    @property
    def suggestions(self) -> dict[str, float | int | str]:
        """Return all suggestions made during this trial."""
        return self._suggestions

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        """Suggest an integer parameter."""
        _ = log
        value = low + (self._number % (high - low + 1))
        self._suggestions[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """Suggest a float parameter."""
        _ = log
        ratio = (self._number % 10) / 10.0
        value = low + ratio * (high - low)
        self._suggestions[name] = value
        return value

    def suggest_categorical(
        self, name: str, choices: tuple[float, ...] | tuple[int, ...] | tuple[str, ...]
    ) -> float | int | str:
        """Suggest a categorical parameter."""
        index = self._number % len(choices)
        value = choices[index]
        self._suggestions[name] = value
        return value

    def report(self, value: float, step: int) -> None:
        """Report intermediate value (no-op)."""
        _ = value, step

    def should_prune(self) -> bool:
        """Check if trial should be pruned (always False)."""
        return False


class FakeSampler:
    """Fake Optuna sampler."""

    def __init__(self, *, seed: int, n_startup_trials: int) -> None:
        self.seed = seed
        self.n_startup_trials = n_startup_trials


class FakePruner:
    """Fake Optuna pruner."""

    def __init__(self, *, n_startup_trials: int, n_warmup_steps: int) -> None:
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps


class FakeStudy:
    """Fake Optuna study that runs trials deterministically."""

    def __init__(
        self,
        *,
        direction: str,
        sampler: OptunaSamplerProtocol,
        pruner: OptunaPrunerProtocol | None,
    ) -> None:
        self._direction = direction
        self._sampler = sampler
        self._pruner = pruner
        self._trials: list[FakeTrial] = []
        self._values: list[float] = []
        self._best_idx = 0

    @property
    def best_trial(self) -> OptunaTrialProtocol:
        """Return the best trial."""
        return self._trials[self._best_idx]

    @property
    def best_value(self) -> float:
        """Return the best value."""
        return self._values[self._best_idx]

    @property
    def best_params(self) -> dict[str, float | int | str]:
        """Return the best trial's parameters."""
        return self._trials[self._best_idx].suggestions

    def optimize(
        self,
        func: Callable[[OptunaTrialProtocol], float],
        n_trials: int,
        timeout: float | None = None,
        callbacks: list[Callable[[OptunaStudyProtocol, OptunaTrialProtocol], None]] | None = None,
    ) -> None:
        """Run optimization by calling func for each trial."""
        _ = timeout, callbacks
        for i in range(n_trials):
            trial = FakeTrial(i)
            value = func(trial)
            self._trials.append(trial)
            self._values.append(value)
            if self._direction == "maximize":
                if value > self._values[self._best_idx]:
                    self._best_idx = i
            else:
                if value < self._values[self._best_idx]:
                    self._best_idx = i

    def get_trials(
        self, deepcopy: bool = True, states: tuple[str, ...] | None = None
    ) -> list[OptunaTrialProtocol]:
        """Return all trials."""
        _ = deepcopy, states
        return list(self._trials)


def fake_create_study(
    *, direction: str, sampler: OptunaSamplerProtocol, pruner: OptunaPrunerProtocol | None = None
) -> OptunaStudyProtocol:
    """Create a fake Optuna study."""
    return FakeStudy(direction=direction, sampler=sampler, pruner=pruner)


def fake_tpe_sampler(*, seed: int, n_startup_trials: int) -> OptunaSamplerProtocol:
    """Create a fake TPE sampler."""
    return FakeSampler(seed=seed, n_startup_trials=n_startup_trials)


def fake_median_pruner(*, n_startup_trials: int, n_warmup_steps: int) -> OptunaPrunerProtocol:
    """Create a fake median pruner."""
    return FakePruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)


def get_fake_optuna_factories() -> tuple[
    OptunaCreateStudyProtocol, OptunaTPESamplerProtocol, OptunaMedianPrunerProtocol
]:
    """Return fake Optuna factory functions for testing."""
    return fake_create_study, fake_tpe_sampler, fake_median_pruner


# =============================================================================
# Optuna Backend Test Helpers
# =============================================================================


def make_optuna_test_data(
    n_samples: int = 50, n_features: int = 4, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create test dataset for Optuna backend optimizer tests.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of features.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (features, labels, feature_names).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_positive = n_samples // 3
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_positive] = 1
    rng.shuffle(y)
    return x, y, [f"feat_{i}" for i in range(n_features)]


def make_optuna_config(
    n_trials: int = 5, pruning_enabled: bool = True, timeout_seconds: int | None = None
) -> OptimizationConfig:
    """Create test optimization config for Optuna backend tests.

    Args:
        n_trials: Number of trials to run.
        pruning_enabled: Whether to enable pruning.
        timeout_seconds: Optional timeout in seconds.

    Returns:
        OptimizationConfig with test defaults.
    """
    return {
        "n_trials": n_trials,
        "timeout_seconds": timeout_seconds,
        "n_startup_trials": 10,
        "random_state": 42,
        "direction": "maximize",
        "pruning_enabled": pruning_enabled,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    }


class FakeObjective:
    """Generic fake objective that returns deterministic values based on params.

    Returns values in [0.5, 1.0] based on learning_rate distance from 0.1.
    """

    def __init__(self, base_auc: float = 0.75) -> None:
        self._base_auc = base_auc
        self.call_count = 0

    def __call__(
        self,
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
        """Return deterministic AUC value."""
        _ = x_features, y_labels, feature_names, string_params
        _ = train_ratio, val_ratio, test_ratio, random_state
        self.call_count += 1
        lr = float_params.get("learning_rate", 0.1)
        return max(0.5, min(1.0, self._base_auc - abs(lr - 0.1) * 0.5))
