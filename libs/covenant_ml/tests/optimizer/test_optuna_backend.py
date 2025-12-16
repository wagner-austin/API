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
    create_xgboost_optimizer,
    set_optuna_module_hook,
    use_real_optuna,
)
from covenant_ml.optimizer.search_spaces import (
    make_default_optimization_config,
    make_xgboost_categorical_space,
    make_xgboost_default_space,
)
from covenant_ml.optimizer.types import (
    CategoricalFloatSpec,
    CategoricalIntSpec,
    FloatRangeSpec,
    IntRangeSpec,
    OptimizationConfig,
    TrialResult,
)

# =============================================================================
# Fake Optuna Implementation for Testing
# =============================================================================


class _FakeTrial:
    """Fake Optuna trial that returns deterministic values."""

    def __init__(self, trial_number: int) -> None:
        self._number = trial_number
        self._suggestions: dict[str, float | int] = {}

    @property
    def number(self) -> int:
        return self._number

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        log: bool = False,
    ) -> int:
        """Return deterministic int based on trial number."""
        _ = log  # Unused in fake
        # Return value that varies by trial but stays in range
        value = low + (self._number % (high - low + 1))
        self._suggestions[name] = value
        return value

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        """Return deterministic float based on trial number."""
        _ = log  # Unused in fake
        # Return value that varies by trial but stays in range
        ratio = (self._number % 10) / 10.0
        value = low + ratio * (high - low)
        self._suggestions[name] = value
        return value

    def suggest_categorical(
        self,
        name: str,
        choices: tuple[float, ...] | tuple[int, ...],
    ) -> float | int:
        """Return deterministic choice based on trial number."""
        index = self._number % len(choices)
        value = choices[index]
        self._suggestions[name] = value
        return value

    def report(self, value: float, step: int) -> None:
        """No-op for fake trial."""
        _ = value, step

    def should_prune(self) -> bool:
        """Never prune in fake trial."""
        return False


class _FakeSampler:
    """Fake TPE sampler."""

    def __init__(self, *, seed: int, n_startup_trials: int) -> None:
        self.seed = seed
        self.n_startup_trials = n_startup_trials


class _FakePruner:
    """Fake median pruner."""

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
        callbacks: (list[Callable[[OptunaStudyProtocol, OptunaTrialProtocol], None]] | None) = None,
    ) -> None:
        """Run fake optimization."""
        _ = timeout, callbacks  # Unused in fake

        for i in range(n_trials):
            trial = _FakeTrial(i)
            value = func(trial)
            self._trials.append(trial)
            self._values.append(value)

            # Update best index based on direction
            if self._direction == "maximize":
                if value > self._values[self._best_idx]:
                    self._best_idx = i
            else:
                if value < self._values[self._best_idx]:
                    self._best_idx = i

    def get_trials(
        self,
        deepcopy: bool = True,
        states: tuple[str, ...] | None = None,
    ) -> list[OptunaTrialProtocol]:
        """Return all trials."""
        _ = deepcopy, states  # Unused in fake
        return list(self._trials)


def _fake_create_study(
    *,
    direction: str,
    sampler: OptunaSamplerProtocol,
    pruner: OptunaPrunerProtocol | None = None,
) -> OptunaStudyProtocol:
    """Factory for fake study."""
    return _FakeStudy(direction=direction, sampler=sampler, pruner=pruner)


def _fake_tpe_sampler(
    *,
    seed: int,
    n_startup_trials: int,
) -> OptunaSamplerProtocol:
    """Factory for fake TPE sampler."""
    return _FakeSampler(seed=seed, n_startup_trials=n_startup_trials)


def _fake_median_pruner(
    *,
    n_startup_trials: int,
    n_warmup_steps: int,
) -> OptunaPrunerProtocol:
    """Factory for fake median pruner."""
    return _FakePruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)


def _get_fake_optuna_factories() -> tuple[
    OptunaCreateStudyProtocol,
    OptunaTPESamplerProtocol,
    OptunaMedianPrunerProtocol,
]:
    """Return fake Optuna factories for testing."""
    return _fake_create_study, _fake_tpe_sampler, _fake_median_pruner


# =============================================================================
# Test Data Helpers
# =============================================================================


def _make_test_data(
    n_samples: int = 50,
    n_features: int = 4,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Create test dataset for optimization."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    n_positive = n_samples // 3
    y = np.zeros(n_samples, dtype=np.int64)
    y[:n_positive] = 1
    rng.shuffle(y)
    names = [f"feat_{i}" for i in range(n_features)]
    return x, y, names


class _FakeObjective:
    """Fake XGBoost objective that returns deterministic values."""

    def __init__(self, base_auc: float = 0.7) -> None:
        self._base_auc = base_auc
        self._call_count = 0

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
        """Return AUC that varies slightly with hyperparameters."""
        _ = (
            x_features,
            y_labels,
            feature_names,
            random_state,
            train_ratio,
            val_ratio,
            test_ratio,
        )
        self._call_count += 1

        # Simple formula that makes different hyperparams give different AUC
        # Optimal around max_depth=5, learning_rate=0.1
        depth_penalty = abs(max_depth - 5) * 0.01
        lr_penalty = abs(learning_rate - 0.1) * 0.1
        n_est_bonus = min(n_estimators / 200.0, 0.1)
        reg_bonus = (reg_alpha + reg_lambda) * 0.001
        sample_bonus = (subsample + colsample_bytree) * 0.01

        auc = self._base_auc - depth_penalty - lr_penalty + n_est_bonus + reg_bonus + sample_bonus
        return max(0.5, min(1.0, auc))  # Clamp to valid AUC range


# =============================================================================
# Test _sample_param_int
# =============================================================================


def test_sample_param_int_range_spec() -> None:
    """_sample_param_int samples from IntRangeSpec."""
    trial = _FakeTrial(0)
    spec: IntRangeSpec = {
        "param_type": "int",
        "low": 3,
        "high": 10,
        "log_scale": False,
    }
    result = _sample_param_int(trial, "max_depth", spec)
    # Verify result is in valid range (type guaranteed by function signature)
    assert 3 <= result <= 10


def test_sample_param_int_categorical_spec() -> None:
    """_sample_param_int samples from CategoricalIntSpec."""
    trial = _FakeTrial(0)
    spec: CategoricalIntSpec = {
        "param_type": "categorical_int",
        "choices": (3, 5, 7, 10),
    }
    result = _sample_param_int(trial, "max_depth", spec)
    # Verify result is one of the choices (type guaranteed by function signature)
    assert result in (3, 5, 7, 10)


def test_sample_param_int_varies_by_trial() -> None:
    """_sample_param_int returns different values for different trials."""
    spec: IntRangeSpec = {
        "param_type": "int",
        "low": 1,
        "high": 100,
        "log_scale": False,
    }
    results = [_sample_param_int(_FakeTrial(i), "n", spec) for i in range(10)]
    # Not all values should be the same
    assert len(set(results)) > 1


# =============================================================================
# Test _sample_param_float
# =============================================================================


def test_sample_param_float_range_spec() -> None:
    """_sample_param_float samples from FloatRangeSpec."""
    trial = _FakeTrial(0)
    spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.01,
        "high": 0.3,
        "log_scale": True,
    }
    result = _sample_param_float(trial, "learning_rate", spec)
    # Verify result is in valid range (type guaranteed by function signature)
    assert 0.01 <= result <= 0.3


def test_sample_param_float_categorical_spec() -> None:
    """_sample_param_float samples from CategoricalFloatSpec."""
    trial = _FakeTrial(0)
    spec: CategoricalFloatSpec = {
        "param_type": "categorical_float",
        "choices": (0.01, 0.05, 0.1, 0.2),
    }
    result = _sample_param_float(trial, "learning_rate", spec)
    # Verify result is one of the choices (type guaranteed by function signature)
    assert result in (0.01, 0.05, 0.1, 0.2)


def test_sample_param_float_varies_by_trial() -> None:
    """_sample_param_float returns different values for different trials."""
    spec: FloatRangeSpec = {
        "param_type": "float",
        "low": 0.0,
        "high": 1.0,
        "log_scale": False,
    }
    results = [_sample_param_float(_FakeTrial(i), "x", spec) for i in range(10)]
    # Not all values should be the same
    assert len(set(results)) > 1


# =============================================================================
# Test OptunaXGBoostOptimizer
# =============================================================================


def test_create_xgboost_optimizer_can_be_called() -> None:
    """create_xgboost_optimizer returns a working optimizer that can run."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data(n_samples=20)
        space = make_xgboost_default_space()
        config = make_default_optimization_config(n_trials=1)
        objective = _FakeObjective()

        # Actually call optimize to verify it works
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )
        assert summary["n_trials_complete"] == 1
    finally:
        set_optuna_module_hook(None)


def test_optimizer_runs_all_trials() -> None:
    """Optimizer runs the specified number of trials."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()
        config = make_default_optimization_config(n_trials=5)
        objective = _FakeObjective()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )

        assert summary["n_trials_total"] == 5
        assert summary["n_trials_complete"] == 5
        assert objective._call_count == 5
    finally:
        set_optuna_module_hook(None)


def test_optimizer_returns_best_parameters() -> None:
    """Optimizer returns the best hyperparameters found."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()
        config = make_default_optimization_config(n_trials=10)
        objective = _FakeObjective()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )

        # Best parameters should be within search space bounds
        assert 3 <= summary["best_max_depth"] <= 10
        assert 50 <= summary["best_n_estimators"] <= 300
        assert 0.01 <= summary["best_learning_rate"] <= 0.3
        assert 0.0 <= summary["best_reg_alpha"] <= 10.0
        assert 0.1 <= summary["best_reg_lambda"] <= 10.0
        assert 0.6 <= summary["best_subsample"] <= 1.0
        assert 0.6 <= summary["best_colsample_bytree"] <= 1.0
    finally:
        set_optuna_module_hook(None)


def test_optimizer_returns_best_value() -> None:
    """Optimizer returns the best objective value found."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()
        config = make_default_optimization_config(n_trials=10)
        objective = _FakeObjective(base_auc=0.75)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )

        # Best value should be in valid AUC range
        assert 0.5 <= summary["best_value"] <= 1.0
        # Should be reasonably close to base_auc
        assert summary["best_value"] > 0.65
    finally:
        set_optuna_module_hook(None)


def test_optimizer_tracks_duration() -> None:
    """Optimizer tracks total optimization duration."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()
        config = make_default_optimization_config(n_trials=3)
        objective = _FakeObjective()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )

        # Duration should be positive
        assert summary["total_duration_seconds"] > 0.0
    finally:
        set_optuna_module_hook(None)


def test_optimizer_calls_trial_callback() -> None:
    """Optimizer calls trial callback after each trial."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()
        config = make_default_optimization_config(n_trials=5)
        objective = _FakeObjective()

        callbacks_received: list[TrialResult] = []

        def trial_callback(result: TrialResult) -> None:
            callbacks_received.append(result)

        _ = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
            trial_callback=trial_callback,
        )

        # Should have received 5 callbacks
        assert len(callbacks_received) == 5

        # Each callback should have valid trial result
        for i, result in enumerate(callbacks_received):
            assert result["trial_number"] == i
            assert result["state"] == "complete"
            assert result["duration_seconds"] > 0.0
            assert 0.5 <= result["value"] <= 1.0
    finally:
        set_optuna_module_hook(None)


def test_optimizer_with_categorical_space() -> None:
    """Optimizer works with categorical search space."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_categorical_space()
        config = make_default_optimization_config(n_trials=5)
        objective = _FakeObjective()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )

        # Best parameters should be from the categorical choices
        assert summary["best_max_depth"] in (3, 4, 5, 6, 7, 8)
        assert summary["best_n_estimators"] in (50, 75, 100, 150, 200)
        assert summary["best_learning_rate"] in (0.01, 0.05, 0.1, 0.2, 0.3)
    finally:
        set_optuna_module_hook(None)


def test_optimizer_with_timeout() -> None:
    """Optimizer passes timeout to study.optimize."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()
        config = make_default_optimization_config(n_trials=3, timeout_seconds=60)
        objective = _FakeObjective()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )

        # Should still complete (fake doesn't enforce timeout)
        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


def test_optimizer_with_pruning_disabled() -> None:
    """Optimizer works with pruning disabled."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()

        config: OptimizationConfig = {
            "n_trials": 5,
            "timeout_seconds": None,
            "n_startup_trials": 2,
            "random_state": 42,
            "direction": "maximize",
            "pruning_enabled": False,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        }
        objective = _FakeObjective()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )

        assert summary["n_trials_complete"] == 5
        assert summary["n_trials_pruned"] == 0
    finally:
        set_optuna_module_hook(None)


def test_optimizer_minimize_direction() -> None:
    """Optimizer can minimize instead of maximize."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()

        config: OptimizationConfig = {
            "n_trials": 5,
            "timeout_seconds": None,
            "n_startup_trials": 2,
            "random_state": 42,
            "direction": "minimize",
            "pruning_enabled": True,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
        }
        objective = _FakeObjective()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )

        # Should complete successfully
        assert summary["n_trials_complete"] == 5
    finally:
        set_optuna_module_hook(None)


def test_optimizer_resets_counters_each_run() -> None:
    """Optimizer resets trial counters for each optimize() call."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()
        objective = _FakeObjective()

        # First run with 3 trials
        config1 = make_default_optimization_config(n_trials=3)
        summary1 = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config1,
            objective=objective,
        )
        assert summary1["n_trials_complete"] == 3

        # Second run with 5 trials - counters should be reset
        config2 = make_default_optimization_config(n_trials=5)
        summary2 = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config2,
            objective=objective,
        )
        assert summary2["n_trials_complete"] == 5
        assert summary2["n_trials_total"] == 5
    finally:
        set_optuna_module_hook(None)


# =============================================================================
# Test set_optuna_module_hook
# =============================================================================


def test_set_optuna_module_hook_can_be_cleared() -> None:
    """set_optuna_module_hook can be cleared by passing None."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    set_optuna_module_hook(None)
    # No exception means success


def test_optimizer_without_callback() -> None:
    """Optimizer works without trial callback."""
    set_optuna_module_hook(_get_fake_optuna_factories)
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = _make_test_data()
        space = make_xgboost_default_space()
        config = make_default_optimization_config(n_trials=3)
        objective = _FakeObjective()

        # Pass None for callback
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
            trial_callback=None,
        )

        assert summary["n_trials_complete"] == 3
    finally:
        set_optuna_module_hook(None)


# =============================================================================
# Test Protocol Compliance
# =============================================================================


def test_fake_trial_protocol_compliance() -> None:
    """_FakeTrial implements OptunaTrialProtocol correctly."""
    trial: OptunaTrialProtocol = _FakeTrial(5)

    # Test number property
    assert trial.number == 5

    # Test suggest_int returns value in range
    int_val = trial.suggest_int("x", 1, 10)
    assert 1 <= int_val <= 10

    # Test suggest_float returns value in range
    float_val = trial.suggest_float("y", 0.0, 1.0)
    assert 0.0 <= float_val <= 1.0

    # Test suggest_categorical returns one of the choices
    cat_val = trial.suggest_categorical("z", (0.1, 0.2, 0.3))
    assert cat_val in (0.1, 0.2, 0.3)

    # Test report and should_prune (should not raise)
    trial.report(0.5, 1)
    assert trial.should_prune() is False


def test_fake_study_protocol_compliance() -> None:
    """_FakeStudy implements OptunaStudyProtocol correctly."""
    sampler = _FakeSampler(seed=42, n_startup_trials=5)
    study: OptunaStudyProtocol = _FakeStudy(direction="maximize", sampler=sampler, pruner=None)

    # Run a simple optimization
    def simple_objective(trial: OptunaTrialProtocol) -> float:
        return float(trial.number) * 0.1

    study.optimize(simple_objective, n_trials=3)

    # Test properties - best_trial should be trial 2 (highest value in maximize)
    assert study.best_trial.number == 2
    assert study.best_value == 0.2  # 2 * 0.1

    # Test best_params is accessible - verify it's a dict with expected type
    best_params = study.best_params
    # Verify best_params is dict-like by checking operations work
    _ = len(best_params)  # Should not raise

    # Test get_trials returns all trials
    trials = study.get_trials()
    assert len(trials) == 3


# =============================================================================
# Test Hook Requirement
# =============================================================================


def test_optimizer_raises_when_hook_not_set() -> None:
    """Optimizer raises RuntimeError when hook is not set."""
    # Clear the hook
    set_optuna_module_hook(None)

    optimizer = create_xgboost_optimizer()
    x, y, names = _make_test_data(n_samples=20)
    space = make_xgboost_default_space()
    config = make_default_optimization_config(n_trials=1)
    objective = _FakeObjective()

    with pytest.raises(RuntimeError, match="Optuna module hook not set"):
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=objective,
        )


def test_use_real_optuna_sets_hook() -> None:
    """use_real_optuna() sets the hook to use real Optuna."""
    # Clear the hook first
    set_optuna_module_hook(None)

    # Set real Optuna
    use_real_optuna()

    # Now optimizer should work (run minimal test with real Optuna)
    optimizer = create_xgboost_optimizer()
    x, y, names = _make_test_data(n_samples=30)
    space = make_xgboost_default_space()
    config = make_default_optimization_config(n_trials=1)

    # Use a simple objective that returns a fixed value
    def simple_objective(
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
        return 0.75

    try:
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=space,
            config=config,
            objective=simple_objective,
        )
        assert summary["n_trials_complete"] == 1
        assert summary["best_value"] == 0.75
    finally:
        # Clean up - reset hook for other tests
        set_optuna_module_hook(None)
