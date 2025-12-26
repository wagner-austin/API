"""Tests for Optuna backend optimizer implementation.

Uses fake implementations of Optuna protocols for testing without mocks.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.optimizer.optuna_backend import (
    OptunaCreateStudyProtocol,
    OptunaMedianPrunerProtocol,
    OptunaPrunerProtocol,
    OptunaSamplerProtocol,
    OptunaStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaTrialProtocol,
    _extract_lightgbm_dart_best_params,
    _extract_xgboost_dart_best_params,
    _sample_lightgbm_dart_params,
    _sample_param_float,
    _sample_param_int,
    _sample_param_str,
    _sample_xgboost_dart_params,
    create_cleargbm_optimizer,
    create_lightgbm_optimizer,
    create_lstm_optimizer,
    create_mlp_optimizer,
    create_xgboost_optimizer,
    set_optuna_module_hook,
    use_real_optuna,
)
from covenant_ml.optimizer.search_spaces import (
    make_cleargbm_default_space,
    make_lightgbm_default_space,
    make_lstm_default_space,
    make_mlp_default_space,
    make_xgboost_categorical_space,
    make_xgboost_default_space,
)
from covenant_ml.optimizer.types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    CategoricalStringSpec,
    FloatRangeSpec,
    IntRangeSpec,
    LightGBMSearchSpace,
    OptimizationConfig,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
    XGBoostSearchSpace,
)

# =============================================================================
# Fake Optuna Implementation
# =============================================================================


class _FakeTrial:
    """Fake Optuna trial that returns deterministic values."""

    def __init__(self, trial_number: int) -> None:
        self._number = trial_number
        self._suggestions: dict[str, float | int | str] = {}

    @property
    def number(self) -> int:
        return self._number

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        _ = log
        value = low + (self._number % (high - low + 1))
        self._suggestions[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        _ = log
        ratio = (self._number % 10) / 10.0
        value = low + ratio * (high - low)
        self._suggestions[name] = value
        return value

    def suggest_categorical(
        self, name: str, choices: tuple[float, ...] | tuple[int, ...] | tuple[str, ...]
    ) -> float | int | str:
        index = self._number % len(choices)
        value = choices[index]
        self._suggestions[name] = value
        return value

    def report(self, value: float, step: int) -> None:
        _ = value, step

    def should_prune(self) -> bool:
        return False


class _FakeSampler:
    def __init__(self, *, seed: int, n_startup_trials: int) -> None:
        self.seed = seed
        self.n_startup_trials = n_startup_trials


class _FakePruner:
    def __init__(self, *, n_startup_trials: int, n_warmup_steps: int) -> None:
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps


class _FakeStudy:
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
        self._trials: list[_FakeTrial] = []
        self._values: list[float] = []
        self._best_idx = 0

    @property
    def best_trial(self) -> OptunaTrialProtocol:
        return self._trials[self._best_idx]

    @property
    def best_value(self) -> float:
        return self._values[self._best_idx]

    @property
    def best_params(self) -> dict[str, float | int | str]:
        return self._trials[self._best_idx]._suggestions

    def optimize(
        self,
        func: Callable[[OptunaTrialProtocol], float],
        n_trials: int,
        timeout: float | None = None,
        callbacks: list[Callable[[OptunaStudyProtocol, OptunaTrialProtocol], None]] | None = None,
    ) -> None:
        _ = timeout, callbacks
        for i in range(n_trials):
            trial = _FakeTrial(i)
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
        _ = deepcopy, states
        return list(self._trials)


def _fake_create_study(
    *, direction: str, sampler: OptunaSamplerProtocol, pruner: OptunaPrunerProtocol | None = None
) -> OptunaStudyProtocol:
    return _FakeStudy(direction=direction, sampler=sampler, pruner=pruner)


def _fake_tpe_sampler(*, seed: int, n_startup_trials: int) -> OptunaSamplerProtocol:
    return _FakeSampler(seed=seed, n_startup_trials=n_startup_trials)


def _fake_median_pruner(*, n_startup_trials: int, n_warmup_steps: int) -> OptunaPrunerProtocol:
    return _FakePruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)


def _get_fake_optuna_factories() -> tuple[
    OptunaCreateStudyProtocol, OptunaTPESamplerProtocol, OptunaMedianPrunerProtocol
]:
    return _fake_create_study, _fake_tpe_sampler, _fake_median_pruner


# =============================================================================
# Test Data Helpers
# =============================================================================


def _make_test_data(
    n_samples: int = 50, n_features: int = 4, seed: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create test dataset for optimization."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_positive = n_samples // 3
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_positive] = 1
    rng.shuffle(y)
    return x, y, [f"feat_{i}" for i in range(n_features)]


def _make_config(
    n_trials: int = 5, pruning_enabled: bool = True, timeout_seconds: int | None = None
) -> OptimizationConfig:
    """Create test optimization config."""
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


class _FakeObjective:
    """Generic fake objective that returns deterministic values based on params."""

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
        _ = x_features, y_labels, feature_names, string_params
        _ = train_ratio, val_ratio, test_ratio, random_state
        self.call_count += 1
        lr = float_params.get("learning_rate", 0.1)
        return max(0.5, min(1.0, self._base_auc - abs(lr - 0.1) * 0.5))


# =============================================================================
# Tests: Parameter Sampling Functions
# =============================================================================


def test_sample_param_int_range_spec() -> None:
    """_sample_param_int handles IntRangeSpec correctly."""
    trial = _FakeTrial(0)
    spec: IntRangeSpec = {"param_type": "int", "low": 3, "high": 10, "log_scale": False}
    result = _sample_param_int(trial, "max_depth", spec)
    assert 3 <= result <= 10


def test_sample_param_int_categorical_spec() -> None:
    """_sample_param_int handles CategoricalIntSpec correctly."""
    trial = _FakeTrial(0)
    spec: CategoricalIntSpec = {"param_type": "categorical_int", "choices": (3, 5, 7, 10)}
    result = _sample_param_int(trial, "max_depth", spec)
    assert result in (3, 5, 7, 10)


def test_sample_param_int_varies_by_trial() -> None:
    """_sample_param_int returns different values for different trials."""
    spec: IntRangeSpec = {"param_type": "int", "low": 1, "high": 100, "log_scale": False}
    values = [_sample_param_int(_FakeTrial(i), "x", spec) for i in range(10)]
    assert len(set(values)) > 1


def test_sample_param_float_range_spec() -> None:
    """_sample_param_float handles FloatRangeSpec correctly."""
    trial = _FakeTrial(0)
    spec: FloatRangeSpec = {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True}
    result = _sample_param_float(trial, "learning_rate", spec)
    assert 0.01 <= result <= 0.3


def test_sample_param_float_categorical_spec() -> None:
    """_sample_param_float handles CategoricalFloatSpec correctly."""
    trial = _FakeTrial(0)
    spec: CategoricalFloatSpec = {"param_type": "categorical_float", "choices": (0.01, 0.1, 0.3)}
    result = _sample_param_float(trial, "learning_rate", spec)
    assert result in (0.01, 0.1, 0.3)


def test_sample_param_float_varies_by_trial() -> None:
    """_sample_param_float returns different values for different trials."""
    spec: FloatRangeSpec = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    values = [_sample_param_float(_FakeTrial(i), "x", spec) for i in range(10)]
    assert len(set(values)) > 1


# =============================================================================
# Tests: String Parameter Sampling
# =============================================================================


def test_sample_param_str_returns_string() -> None:
    """_sample_param_str returns string value from choices."""
    spec: CategoricalStringSpec = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    trial = _FakeTrial(0)
    result = _sample_param_str(trial, "boosting_type", spec)
    assert result in ("gbdt", "dart")


def test_sample_param_str_varies_by_trial() -> None:
    """_sample_param_str returns different values for different trials."""
    spec: CategoricalStringSpec = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    trial0 = _FakeTrial(0)
    trial1 = _FakeTrial(1)
    result0 = _sample_param_str(trial0, "boosting_type", spec)
    result1 = _sample_param_str(trial1, "boosting_type", spec)
    # trial 0 selects index 0 (gbdt), trial 1 selects index 1 (dart)
    assert result0 == "gbdt"
    assert result1 == "dart"


# =============================================================================
# Tests: XGBoost DART Parameter Sampling
# =============================================================================


def _make_xgboost_space_with_dart() -> XGBoostSearchSpace:
    """Create XGBoost search space with DART params.

    Note: The default space now includes DART, so this uses the default.
    """
    return make_xgboost_default_space()


def _make_xgboost_space_without_dart() -> XGBoostSearchSpace:
    """Create XGBoost search space WITHOUT DART params.

    Used for testing edge cases where DART is not in the search space.
    """
    space: XGBoostSearchSpace = {
        "max_depth": {"param_type": "int", "low": 3, "high": 10, "log_scale": False},
        "n_estimators": {"param_type": "int", "low": 50, "high": 300, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
    }
    return space


def _make_lightgbm_space_with_dart() -> LightGBMSearchSpace:
    """Create LightGBM search space with DART params.

    Note: The default space now includes DART with feature_fraction, so this uses the default.
    """
    return make_lightgbm_default_space()


def _make_lightgbm_space_without_dart() -> LightGBMSearchSpace:
    """Create LightGBM search space WITHOUT DART params.

    Used for testing edge cases where DART is not in the search space.
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
    return space


def test_sample_xgboost_dart_params_no_booster_in_space() -> None:
    """_sample_xgboost_dart_params does nothing when booster not in space."""
    trial = _FakeTrial(0)
    space = _make_xgboost_space_without_dart()  # No booster key
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    # Nothing added since booster not in space
    assert "booster" not in string_params
    assert "rate_drop" not in float_params


def test_sample_xgboost_dart_params_with_dart() -> None:
    """_sample_xgboost_dart_params adds DART params when booster is dart."""
    trial = _FakeTrial(1)  # Trial 1 selects dart (index 1)
    space = _make_xgboost_space_with_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    # DART params added
    assert string_params["booster"] == "dart"
    assert "rate_drop" in float_params
    assert "skip_drop" in float_params


def test_sample_xgboost_dart_params_with_dart_partial() -> None:
    """_sample_xgboost_dart_params handles partial DART params in search space."""
    trial = _FakeTrial(1)  # Trial 1 selects dart (index 1)
    # Only rate_drop, no skip_drop - start from space without DART
    space = _make_xgboost_space_without_dart()
    space["booster"] = {"param_type": "categorical_str", "choices": ("gbtree", "dart")}
    space["rate_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    assert string_params["booster"] == "dart"
    assert "rate_drop" in float_params
    assert "skip_drop" not in float_params


def test_sample_xgboost_dart_params_with_dart_skip_drop_only() -> None:
    """_sample_xgboost_dart_params handles skip_drop only (no rate_drop) in search space."""
    trial = _FakeTrial(1)  # Trial 1 selects dart (index 1)
    # Only skip_drop, no rate_drop - start from space without DART
    space = _make_xgboost_space_without_dart()
    space["booster"] = {"param_type": "categorical_str", "choices": ("gbtree", "dart")}
    space["skip_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    assert string_params["booster"] == "dart"
    assert "rate_drop" not in float_params
    assert "skip_drop" in float_params


def test_sample_xgboost_dart_params_with_gbtree() -> None:
    """_sample_xgboost_dart_params does not add DART params when booster is gbtree."""
    trial = _FakeTrial(0)  # Trial 0 selects gbtree (index 0)
    space = _make_xgboost_space_with_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    # Booster set but no DART params
    assert string_params["booster"] == "gbtree"
    assert "rate_drop" not in float_params
    assert "skip_drop" not in float_params


# =============================================================================
# Tests: LightGBM DART Parameter Sampling
# =============================================================================


def test_sample_lightgbm_dart_params_no_boosting_type_in_space() -> None:
    """_sample_lightgbm_dart_params does nothing when boosting_type not in space."""
    trial = _FakeTrial(0)
    space = _make_lightgbm_space_without_dart()  # No boosting_type key
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    # Nothing added since boosting_type not in space
    assert "boosting_type" not in string_params
    assert "drop_rate" not in float_params


def test_sample_lightgbm_dart_params_with_dart() -> None:
    """_sample_lightgbm_dart_params adds DART params when boosting_type is dart."""
    trial = _FakeTrial(1)  # Trial 1 selects dart (index 1)
    space = _make_lightgbm_space_with_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    # DART params added including feature_fraction (Phase 6)
    assert string_params["boosting_type"] == "dart"
    assert "drop_rate" in float_params
    assert "skip_drop" in float_params
    assert "feature_fraction" in float_params


def test_sample_lightgbm_dart_params_with_dart_partial() -> None:
    """_sample_lightgbm_dart_params handles partial DART params in search space."""
    trial = _FakeTrial(1)  # Trial 1 selects dart (index 1)
    # Only drop_rate, no skip_drop - start from space without DART
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    space["drop_rate"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    assert string_params["boosting_type"] == "dart"
    assert "drop_rate" in float_params
    assert "skip_drop" not in float_params


def test_sample_lightgbm_dart_params_with_dart_skip_drop_only() -> None:
    """_sample_lightgbm_dart_params handles skip_drop only (no drop_rate) in search space."""
    trial = _FakeTrial(1)  # Trial 1 selects dart (index 1)
    # Only skip_drop, no drop_rate - start from space without DART
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    space["skip_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    assert string_params["boosting_type"] == "dart"
    assert "drop_rate" not in float_params
    assert "skip_drop" in float_params


def test_sample_lightgbm_dart_params_with_dart_feature_fraction_only() -> None:
    """_sample_lightgbm_dart_params handles feature_fraction only in search space."""
    trial = _FakeTrial(1)  # Trial 1 selects dart (index 1)
    # Only feature_fraction, no drop_rate or skip_drop - start from space without DART
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    ff_spec: FloatRangeSpec = {"param_type": "float", "low": 0.02, "high": 0.1, "log_scale": False}
    space["feature_fraction"] = ff_spec
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    assert string_params["boosting_type"] == "dart"
    assert "drop_rate" not in float_params
    assert "skip_drop" not in float_params
    assert "feature_fraction" in float_params


def test_sample_lightgbm_dart_params_with_gbdt() -> None:
    """_sample_lightgbm_dart_params does not add DART params when boosting_type is gbdt."""
    trial = _FakeTrial(0)  # Trial 0 selects gbdt (index 0)
    space = _make_lightgbm_space_with_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    # Boosting type set but no DART params
    assert string_params["boosting_type"] == "gbdt"
    assert "drop_rate" not in float_params
    assert "skip_drop" not in float_params
    assert "feature_fraction" not in float_params


# =============================================================================
# Tests: XGBoost DART Best Params Extraction
# =============================================================================


def test_extract_xgboost_dart_best_params_no_booster_in_space() -> None:
    """_extract_xgboost_dart_best_params does nothing when booster not in space."""
    space = _make_xgboost_space_without_dart()  # No booster key
    best_params: dict[str, float | int | str] = {"max_depth": 5}
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    # Nothing added since booster not in space
    assert "booster" not in best_string_params


def test_extract_xgboost_dart_best_params_with_dart() -> None:
    """_extract_xgboost_dart_best_params extracts DART params when booster is dart."""
    space = _make_xgboost_space_with_dart()
    best_params: dict[str, float | int | str] = {
        "max_depth": 5,
        "booster": "dart",
        "rate_drop": 0.15,
        "skip_drop": 0.5,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    # DART params extracted
    assert best_string_params["booster"] == "dart"
    assert best_float_params["rate_drop"] == 0.15
    assert best_float_params["skip_drop"] == 0.5


def test_extract_xgboost_dart_best_params_with_dart_partial() -> None:
    """_extract_xgboost_dart_best_params handles partial DART params in search space."""
    # Only rate_drop in space, no skip_drop - start from space without DART
    space = _make_xgboost_space_without_dart()
    space["booster"] = {"param_type": "categorical_str", "choices": ("gbtree", "dart")}
    space["rate_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    best_params: dict[str, float | int | str] = {
        "max_depth": 5,
        "booster": "dart",
        "rate_drop": 0.15,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["booster"] == "dart"
    assert best_float_params["rate_drop"] == 0.15
    assert "skip_drop" not in best_float_params


def test_extract_xgboost_dart_best_params_with_dart_skip_drop_only() -> None:
    """_extract_xgboost_dart_best_params handles skip_drop only (no rate_drop) in search space."""
    # Only skip_drop in space, no rate_drop - start from space without DART
    space = _make_xgboost_space_without_dart()
    space["booster"] = {"param_type": "categorical_str", "choices": ("gbtree", "dart")}
    space["skip_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    best_params: dict[str, float | int | str] = {
        "max_depth": 5,
        "booster": "dart",
        "skip_drop": 0.5,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["booster"] == "dart"
    assert "rate_drop" not in best_float_params
    assert best_float_params["skip_drop"] == 0.5


def test_extract_xgboost_dart_best_params_with_gbtree() -> None:
    """_extract_xgboost_dart_best_params skips DART params when booster is gbtree."""
    space = _make_xgboost_space_with_dart()
    best_params: dict[str, float | int | str] = {
        "max_depth": 5,
        "booster": "gbtree",
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    # Booster extracted but no DART params
    assert best_string_params["booster"] == "gbtree"
    assert "rate_drop" not in best_float_params
    assert "skip_drop" not in best_float_params


# =============================================================================
# Tests: LightGBM DART Best Params Extraction
# =============================================================================


def test_extract_lightgbm_dart_best_params_no_boosting_type_in_space() -> None:
    """_extract_lightgbm_dart_best_params does nothing when boosting_type not in space."""
    space = _make_lightgbm_space_without_dart()  # No boosting_type key
    best_params: dict[str, float | int | str] = {"num_leaves": 50}
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    # Nothing added since boosting_type not in space
    assert "boosting_type" not in best_string_params


def test_extract_lightgbm_dart_best_params_with_dart() -> None:
    """_extract_lightgbm_dart_best_params extracts DART params when boosting_type is dart."""
    space = _make_lightgbm_space_with_dart()
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "dart",
        "drop_rate": 0.1,
        "skip_drop": 0.6,
        "feature_fraction": 0.05,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    # DART params extracted
    assert best_string_params["boosting_type"] == "dart"
    assert best_float_params["drop_rate"] == 0.1
    assert best_float_params["skip_drop"] == 0.6
    assert best_float_params["feature_fraction"] == 0.05


def test_extract_lightgbm_dart_best_params_with_dart_partial() -> None:
    """_extract_lightgbm_dart_best_params handles partial DART params in search space."""
    # Only drop_rate in space, no skip_drop - start from space without DART
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    space["drop_rate"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "dart",
        "drop_rate": 0.1,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["boosting_type"] == "dart"
    assert best_float_params["drop_rate"] == 0.1
    assert "skip_drop" not in best_float_params


def test_extract_lightgbm_dart_best_params_with_dart_skip_drop_only() -> None:
    """_extract_lightgbm_dart_best_params handles skip_drop only (no drop_rate) in space."""
    # Only skip_drop in space, no drop_rate - start from space without DART
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    space["skip_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "dart",
        "skip_drop": 0.6,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["boosting_type"] == "dart"
    assert "drop_rate" not in best_float_params
    assert best_float_params["skip_drop"] == 0.6


def test_extract_lightgbm_dart_best_params_with_dart_feature_fraction_only() -> None:
    """_extract_lightgbm_dart_best_params handles feature_fraction only in search space."""
    # Only feature_fraction in space, no drop_rate or skip_drop
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    ff_spec: FloatRangeSpec = {"param_type": "float", "low": 0.02, "high": 0.1, "log_scale": False}
    space["feature_fraction"] = ff_spec
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "dart",
        "feature_fraction": 0.05,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["boosting_type"] == "dart"
    assert "drop_rate" not in best_float_params
    assert "skip_drop" not in best_float_params
    assert best_float_params["feature_fraction"] == 0.05


def test_extract_lightgbm_dart_best_params_with_gbdt() -> None:
    """_extract_lightgbm_dart_best_params skips DART params when boosting_type is gbdt."""
    space = _make_lightgbm_space_with_dart()
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "gbdt",
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    # Boosting type extracted but no DART params
    assert best_string_params["boosting_type"] == "gbdt"
    assert "drop_rate" not in best_float_params
    assert "skip_drop" not in best_float_params
    assert "feature_fraction" not in best_float_params


# =============================================================================
# Tests: Hook Management
# =============================================================================


def test_set_optuna_module_hook_can_be_cleared() -> None:
    """Hook can be set to None to clear it."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    set_optuna_module_hook(None)
    optimizer = create_xgboost_optimizer()
    x, y, names = _make_test_data(n_samples=20)
    with pytest.raises(RuntimeError, match="hook not set"):
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=_make_config(n_trials=1),
            objective=_FakeObjective(),
        )


def test_optimizer_raises_when_hook_not_set() -> None:
    """Optimizer raises RuntimeError when hook is not set."""
    set_optuna_module_hook(None)
    optimizer = create_xgboost_optimizer()
    x, y, names = _make_test_data(n_samples=20)
    with pytest.raises(RuntimeError, match="hook not set"):
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=_make_config(n_trials=1),
            objective=_FakeObjective(),
        )


def test_use_real_optuna_sets_hook() -> None:
    """use_real_optuna() sets the hook to use real Optuna."""
    set_optuna_module_hook(None)
    use_real_optuna()

    def simple_objective(
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
        _ = x_features, y_labels, feature_names, int_params, float_params, string_params
        _ = train_ratio, val_ratio, test_ratio, random_state
        return 0.75

    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data(n_samples=30)
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=_make_config(n_trials=1),
            objective=simple_objective,
        )
        assert summary["n_trials_complete"] == 1
        assert summary["best_value"] == 0.75
    finally:
        set_optuna_module_hook(None)


# =============================================================================
# Tests: XGBoost Optimizer
# =============================================================================


def test_xgboost_optimizer_runs_trials() -> None:
    """XGBoost optimizer runs all trials and returns summary."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        objective = _FakeObjective()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=_make_config(n_trials=5),
            objective=objective,
        )
        assert summary["n_trials_total"] == 5
        assert summary["n_trials_complete"] == 5
        assert objective.call_count == 5
        assert 0.0 <= summary["best_value"] <= 1.0
        assert "learning_rate" in summary["best_float_params"]
    finally:
        set_optuna_module_hook(None)


def test_xgboost_optimizer_with_callback() -> None:
    """XGBoost optimizer calls trial callback after each trial."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=_make_config(n_trials=3),
            objective=_FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for i, result in enumerate(callbacks):
            assert result["trial_number"] == i
            assert result["state"] == "complete"
    finally:
        set_optuna_module_hook(None)


def test_xgboost_optimizer_with_categorical_space() -> None:
    """XGBoost optimizer works with categorical search space."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_categorical_space(),
            config=_make_config(n_trials=5),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 5
        assert summary["best_int_params"]["max_depth"] in (3, 4, 5, 6, 7, 8)
    finally:
        set_optuna_module_hook(None)


def test_xgboost_optimizer_with_timeout() -> None:
    """XGBoost optimizer accepts timeout parameter."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=_make_config(n_trials=3, timeout_seconds=60),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


def test_xgboost_optimizer_with_pruning_disabled() -> None:
    """XGBoost optimizer works with pruning disabled."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=_make_config(n_trials=3, pruning_enabled=False),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


# =============================================================================
# Tests: MLP Optimizer
# =============================================================================


def test_mlp_optimizer_runs_trials() -> None:
    """MLP optimizer runs all trials and returns summary."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_mlp_optimizer()
        x, y, names = _make_test_data()
        objective = _FakeObjective()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_mlp_default_space(),
            config=_make_config(n_trials=5),
            objective=objective,
        )
        assert summary["n_trials_total"] == 5
        assert summary["n_trials_complete"] == 5
        assert objective.call_count == 5
        assert "learning_rate" in summary["best_float_params"]
    finally:
        set_optuna_module_hook(None)


def test_mlp_optimizer_with_pruning_disabled() -> None:
    """MLP optimizer works with pruning disabled."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_mlp_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_mlp_default_space(),
            config=_make_config(n_trials=3, pruning_enabled=False),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


def test_mlp_optimizer_with_trial_callback() -> None:
    """MLP optimizer calls trial_callback for each trial."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_mlp_optimizer()
        x, y, names = _make_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_mlp_default_space(),
            config=_make_config(n_trials=3),
            objective=_FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for result in callbacks:
            assert result["state"] == "complete"
    finally:
        set_optuna_module_hook(None)


def test_mlp_optimizer_with_timeout() -> None:
    """MLP optimizer accepts timeout_seconds."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_mlp_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_mlp_default_space(),
            config=_make_config(n_trials=3, timeout_seconds=60),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


# =============================================================================
# Tests: LSTM Optimizer
# =============================================================================


def test_lstm_optimizer_runs_trials() -> None:
    """LSTM optimizer runs all trials and returns summary."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_lstm_optimizer()
        x, y, names = _make_test_data()
        objective = _FakeObjective()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lstm_default_space(),
            config=_make_config(n_trials=5),
            objective=objective,
        )
        assert summary["n_trials_total"] == 5
        assert summary["n_trials_complete"] == 5
        assert objective.call_count == 5
        assert "learning_rate" in summary["best_float_params"]
    finally:
        set_optuna_module_hook(None)


def test_lstm_optimizer_with_pruning_disabled() -> None:
    """LSTM optimizer works with pruning disabled."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_lstm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lstm_default_space(),
            config=_make_config(n_trials=3, pruning_enabled=False),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


def test_lstm_optimizer_with_trial_callback() -> None:
    """LSTM optimizer calls trial_callback for each trial."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_lstm_optimizer()
        x, y, names = _make_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lstm_default_space(),
            config=_make_config(n_trials=3),
            objective=_FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for result in callbacks:
            assert result["state"] == "complete"
    finally:
        set_optuna_module_hook(None)


def test_lstm_optimizer_with_timeout() -> None:
    """LSTM optimizer accepts timeout_seconds."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_lstm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lstm_default_space(),
            config=_make_config(n_trials=3, timeout_seconds=60),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


# =============================================================================
# Tests: LightGBM Optimizer
# =============================================================================


def test_lightgbm_optimizer_runs_trials() -> None:
    """LightGBM optimizer runs all trials and returns summary."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_lightgbm_optimizer()
        x, y, names = _make_test_data()
        objective = _FakeObjective()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lightgbm_default_space(),
            config=_make_config(n_trials=5),
            objective=objective,
        )
        assert summary["n_trials_total"] == 5
        assert summary["n_trials_complete"] == 5
        assert objective.call_count == 5
        assert "learning_rate" in summary["best_float_params"]
    finally:
        set_optuna_module_hook(None)


def test_lightgbm_optimizer_with_pruning_disabled() -> None:
    """LightGBM optimizer works with pruning disabled."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_lightgbm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lightgbm_default_space(),
            config=_make_config(n_trials=3, pruning_enabled=False),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


def test_lightgbm_optimizer_with_trial_callback() -> None:
    """LightGBM optimizer calls trial_callback for each trial."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_lightgbm_optimizer()
        x, y, names = _make_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lightgbm_default_space(),
            config=_make_config(n_trials=3),
            objective=_FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for result in callbacks:
            assert result["state"] == "complete"
    finally:
        set_optuna_module_hook(None)


def test_lightgbm_optimizer_with_timeout() -> None:
    """LightGBM optimizer accepts timeout_seconds."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_lightgbm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lightgbm_default_space(),
            config=_make_config(n_trials=3, timeout_seconds=60),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


# =============================================================================
# ClearGBM Optimizer Tests
# =============================================================================


def test_cleargbm_optimizer_completes_trials() -> None:
    """ClearGBM optimizer completes configured number of trials."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=_make_config(n_trials=3),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


def test_cleargbm_optimizer_returns_best_params() -> None:
    """ClearGBM optimizer returns best parameters in summary."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=_make_config(n_trials=3),
            objective=_FakeObjective(),
        )
        # ClearGBM has 5 int params: n_estimators, max_depth, min_samples_split,
        # min_samples_leaf, max_bins
        assert len(summary["best_int_params"]) == 5
        # ClearGBM has 2 float params: learning_rate, subsample
        assert len(summary["best_float_params"]) == 2
        # ClearGBM has no string params
        assert len(summary["best_string_params"]) == 0
    finally:
        set_optuna_module_hook(None)


def test_cleargbm_optimizer_best_value_in_summary() -> None:
    """ClearGBM optimizer returns best value in summary."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=_make_config(n_trials=3),
            objective=_FakeObjective(),
        )
        # _FakeObjective returns values in [0.5, 1.0] based on learning_rate
        assert 0.5 <= summary["best_value"] <= 1.0
    finally:
        set_optuna_module_hook(None)


def test_cleargbm_optimizer_records_duration() -> None:
    """ClearGBM optimizer records total duration."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=_make_config(n_trials=3),
            objective=_FakeObjective(),
        )
        assert summary["total_duration_seconds"] >= 0.0
    finally:
        set_optuna_module_hook(None)


def test_cleargbm_optimizer_without_pruning() -> None:
    """ClearGBM optimizer works without pruning."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=_make_config(n_trials=3, pruning_enabled=False),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


def test_cleargbm_optimizer_with_trial_callback() -> None:
    """ClearGBM optimizer calls trial_callback for each trial."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_cleargbm_optimizer()
        x, y, names = _make_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=_make_config(n_trials=3),
            objective=_FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for result in callbacks:
            assert result["state"] == "complete"
    finally:
        set_optuna_module_hook(None)


def test_cleargbm_optimizer_with_timeout() -> None:
    """ClearGBM optimizer accepts timeout_seconds."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = _make_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=_make_config(n_trials=3, timeout_seconds=60),
            objective=_FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)
