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
    _sample_param_float,
    _sample_param_int,
    create_lightgbm_optimizer,
    create_lstm_optimizer,
    create_mlp_optimizer,
    create_xgboost_optimizer,
    set_optuna_module_hook,
    use_real_optuna,
)
from covenant_ml.optimizer.search_spaces import (
    make_lightgbm_default_space,
    make_lstm_default_space,
    make_mlp_default_space,
    make_xgboost_categorical_space,
    make_xgboost_default_space,
)
from covenant_ml.optimizer.types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    OptimizationConfig,
    SampledFloatParams,
    SampledIntParams,
    TrialResult,
)

# =============================================================================
# Fake Optuna Implementation
# =============================================================================


class _FakeTrial:
    """Fake Optuna trial that returns deterministic values."""

    def __init__(self, trial_number: int) -> None:
        self._number = trial_number
        self._suggestions: dict[str, float | int] = {}

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
        self, name: str, choices: tuple[float, ...] | tuple[int, ...]
    ) -> float | int:
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
    def best_params(self) -> dict[str, float | int]:
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
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        _ = x_features, y_labels, feature_names, train_ratio, val_ratio, test_ratio, random_state
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
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        _ = x_features, y_labels, feature_names, int_params, float_params
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
