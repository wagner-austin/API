"""Tests for OptunaTpeOptimizer.

Tests cover:
- Strategy name and capabilities
- Optimization with fake Optuna hook
- Various search spaces
- DART configurations
- Pruning support
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from covenant_ml.optimizer.strategies import _hooks as _tpe_hooks
from covenant_ml.optimizer.strategies.optuna_tpe import (
    OptunaCreateStudyProtocol,
    OptunaMedianPrunerProtocol,
    OptunaPrunerProtocol,
    OptunaSamplerProtocol,
    OptunaStudyProtocol,
    OptunaTpeOptimizer,
    OptunaTPESamplerProtocol,
    OptunaTrialProtocol,
    create_optuna_tpe_optimizer,
)
from covenant_ml.optimizer.types import OptimizationConfig

from .conftest import (
    dummy_objective,
    lightgbm_dart_objective,
    lightgbm_objective,
    lstm_objective,
    make_features,
    make_labels,
    make_lightgbm_dart_search_space,
    make_lightgbm_search_space,
    make_lstm_search_space,
    make_mlp_search_space,
    make_optimization_config,
    make_xgboost_dart_search_space,
    make_xgboost_search_space,
    mlp_objective,
    xgboost_dart_objective,
)

# =============================================================================
# Fake Optuna Implementation
# =============================================================================


class _FakeSampler:
    """Fake TPE sampler for testing."""

    def __init__(self, seed: int, n_startup_trials: int) -> None:
        self._seed = seed
        self._n_startup_trials = n_startup_trials


class _FakePruner:
    """Fake pruner for testing."""

    def __init__(self, n_startup_trials: int, n_warmup_steps: int) -> None:
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps


class _FakeTrial:
    """Fake Optuna trial for testing."""

    def __init__(self, trial_number: int, rng_seed: int) -> None:
        self._number = trial_number
        self._rng = np.random.default_rng(rng_seed + trial_number)
        self._params: dict[str, float | int | str] = {}

    @property
    def number(self) -> int:
        return self._number

    def get_params(self) -> dict[str, float | int | str]:
        """Get all suggested params."""
        return dict(self._params)

    def suggest_int(self, name: str, low: int, high: int, *, log: bool = False) -> int:
        """Suggest integer parameter."""
        value = int(self._rng.integers(low, high + 1))
        self._params[name] = value
        return value

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        """Suggest float parameter."""
        if log:
            import math

            log_low = math.log(low)
            log_high = math.log(high)
            uniform_val: float = float(self._rng.uniform(log_low, log_high))
            value = math.exp(uniform_val)
        else:
            value = float(self._rng.uniform(low, high))
        self._params[name] = value
        return value

    def suggest_categorical(
        self,
        name: str,
        choices: tuple[float, ...] | tuple[int, ...] | tuple[str, ...],
    ) -> float | int | str:
        """Suggest categorical parameter."""
        idx = int(self._rng.integers(0, len(choices)))
        value = choices[idx]
        self._params[name] = value
        return value


class _FakeStudy:
    """Fake Optuna study for testing."""

    def __init__(
        self,
        direction: str,
        sampler: OptunaSamplerProtocol,
        pruner: OptunaPrunerProtocol | None = None,
    ) -> None:
        self._direction = direction
        self._sampler = sampler
        self._pruner = pruner
        self._best_trial: OptunaTrialProtocol | None = None
        self._best_value = float("-inf") if direction == "maximize" else float("inf")
        self._best_params: dict[str, float | int | str] = {}
        self._rng_seed = 42

    @property
    def best_trial(self) -> OptunaTrialProtocol:
        if self._best_trial is None:
            return _FakeTrial(0, self._rng_seed)
        return self._best_trial

    @property
    def best_value(self) -> float:
        return self._best_value

    @property
    def best_params(self) -> dict[str, float | int | str]:
        return self._best_params

    def optimize(
        self,
        func: Callable[[OptunaTrialProtocol], float],
        n_trials: int,
        timeout: float | None = None,
        callbacks: list[Callable[[OptunaStudyProtocol, OptunaTrialProtocol], None]] | None = None,
    ) -> None:
        """Run optimization."""
        for i in range(n_trials):
            fake_trial = _FakeTrial(i, self._rng_seed)
            trial: OptunaTrialProtocol = fake_trial
            value = func(trial)

            is_better = (self._direction == "maximize" and value > self._best_value) or (
                self._direction == "minimize" and value < self._best_value
            )
            if is_better:
                self._best_value = value
                self._best_trial = trial
                self._best_params = fake_trial.get_params()


def _make_fake_optuna_hook() -> Callable[
    [],
    tuple[
        OptunaCreateStudyProtocol,
        OptunaTPESamplerProtocol,
        OptunaMedianPrunerProtocol,
    ],
]:
    """Create fake Optuna hook for testing."""

    def create_study(
        *,
        direction: str,
        sampler: OptunaSamplerProtocol,
        pruner: OptunaPrunerProtocol | None = None,
    ) -> OptunaStudyProtocol:
        return _FakeStudy(direction=direction, sampler=sampler, pruner=pruner)

    def tpe_sampler(*, seed: int, n_startup_trials: int) -> OptunaSamplerProtocol:
        return _FakeSampler(seed=seed, n_startup_trials=n_startup_trials)

    def median_pruner(*, n_startup_trials: int, n_warmup_steps: int) -> OptunaPrunerProtocol:
        return _FakePruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps)

    def hook() -> tuple[
        OptunaCreateStudyProtocol,
        OptunaTPESamplerProtocol,
        OptunaMedianPrunerProtocol,
    ]:
        cs: OptunaCreateStudyProtocol = create_study
        ts: OptunaTPESamplerProtocol = tpe_sampler
        mp: OptunaMedianPrunerProtocol = median_pruner
        return cs, ts, mp

    return hook


# =============================================================================
# Core Tests
# =============================================================================


class TestOptunaTpeOptimizer:
    """Tests for OptunaTpeOptimizer core functionality."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        optimizer = OptunaTpeOptimizer()
        assert optimizer.strategy_name() == "optuna_tpe"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        optimizer = OptunaTpeOptimizer()
        caps = optimizer.capabilities()

        assert caps["supports_pruning"] is True
        assert caps["supports_parallel"] is True
        assert caps["is_deterministic"] is False
        assert caps["requires_bounds"] is True

    def test_optimize_with_fake_hook(self) -> None:
        """Optimize runs with fake Optuna hook."""
        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_search_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=dummy_objective,
            )

            assert summary["n_trials_complete"] == 3
            assert summary["best_value"] > 0
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories

    def test_optimize_with_pruning_enabled(self) -> None:
        """Optimize runs with pruning enabled."""
        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_search_space()
            config = OptimizationConfig(
                n_trials=3,
                timeout_seconds=None,
                n_startup_trials=2,
                random_state=42,
                direction="maximize",
                pruning_enabled=True,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=dummy_objective,
            )

            assert summary["n_trials_complete"] == 3
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


class TestOptunaTpeFactory:
    """Tests for create_optuna_tpe_optimizer factory."""

    def test_factory_creates_optimizer(self) -> None:
        """Factory creates optimizer."""
        optimizer = create_optuna_tpe_optimizer()
        assert optimizer.strategy_name() == "optuna_tpe"


# =============================================================================
# Backend-Specific Tests
# =============================================================================


class TestOptunaTpeWithMLP:
    """Tests for OptunaTpeOptimizer with MLP search space."""

    def test_optimize_with_mlp_space(self) -> None:
        """Optimize runs with MLP search space."""
        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_mlp_search_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=mlp_objective,
            )

            assert summary["n_trials_complete"] == 3
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


class TestOptunaTpeWithLSTM:
    """Tests for OptunaTpeOptimizer with LSTM search space."""

    def test_optimize_with_lstm_space(self) -> None:
        """Optimize runs with LSTM search space."""
        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_lstm_search_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=lstm_objective,
            )

            assert summary["n_trials_complete"] == 3
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


class TestOptunaTpeWithLightGBM:
    """Tests for OptunaTpeOptimizer with LightGBM search space."""

    def test_optimize_with_lightgbm_space(self) -> None:
        """Optimize runs with LightGBM search space."""
        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_lightgbm_search_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=lightgbm_objective,
            )

            assert summary["n_trials_complete"] == 3
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


# =============================================================================
# DART Tests
# =============================================================================


class TestOptunaTpeWithDART:
    """Tests for OptunaTpeOptimizer with DART search spaces."""

    def test_optimize_with_xgboost_dart(self) -> None:
        """Optimize runs with XGBoost DART search space."""
        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_dart_search_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=xgboost_dart_objective,
            )

            assert summary["n_trials_complete"] == 3
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories

    def test_optimize_with_lightgbm_dart(self) -> None:
        """Optimize runs with LightGBM DART search space."""
        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_lightgbm_dart_search_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=lightgbm_dart_objective,
            )

            assert summary["n_trials_complete"] == 3
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


# =============================================================================
# Hook and Factory Edge Case Tests
# =============================================================================


class TestOptunaFactoriesBinding:
    """Tests for the optuna factories the strategy ships bound to."""

    def test_the_hook_is_bound_to_real_optuna(self) -> None:
        """The strategy binds real optuna, so nothing has to be wired first."""
        assert _tpe_hooks.optuna_factories is _tpe_hooks._real_optuna_factories

        factories = _tpe_hooks.optuna_factories()
        assert len(factories) == 3

        # Verify each factory is callable
        create_study, tpe_sampler, median_pruner = factories
        assert callable(create_study)
        assert callable(tpe_sampler)
        assert callable(median_pruner)

        # Reset hook
        _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


# =============================================================================
# DART Without Optional Params Tests
# =============================================================================


class TestOptunaTpeWithDARTNoParams:
    """Tests for OptunaTpeOptimizer with DART but no DART params."""

    def test_optimize_with_xgboost_dart_no_params(self) -> None:
        """Optimize runs with XGBoost DART without rate_drop/skip_drop."""
        from .conftest import make_xgboost_dart_no_params_space, xgboost_dart_no_params_objective

        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_dart_no_params_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=xgboost_dart_no_params_objective,
            )

            assert summary["n_trials_complete"] == 3
            assert summary["best_string_params"].get("booster") == "dart"
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories

    def test_optimize_with_lightgbm_dart_no_params(self) -> None:
        """Optimize runs with LightGBM DART without drop_rate/skip_drop."""
        from .conftest import (
            lightgbm_dart_no_params_objective,
            make_lightgbm_dart_no_params_space,
        )

        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_lightgbm_dart_no_params_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=lightgbm_dart_no_params_objective,
            )

            assert summary["n_trials_complete"] == 3
            assert summary["best_string_params"].get("boosting_type") == "dart"
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


# =============================================================================
# Categorical Parameter Tests
# =============================================================================


class TestOptunaTpeWithCategoricalParams:
    """Tests for OptunaTpeOptimizer with categorical parameters."""

    def test_optimize_with_categorical_int(self) -> None:
        """Optimize runs with categorical int parameters."""
        from .conftest import make_xgboost_categorical_space

        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_categorical_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=dummy_objective,
            )

            assert summary["n_trials_complete"] == 3
            best_max_depth = summary["best_int_params"].get("max_depth")
            assert best_max_depth in (3, 5, 7)
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories

    def test_optimize_with_categorical_float(self) -> None:
        """Optimize runs with categorical float parameters."""
        from .conftest import make_xgboost_categorical_float_space

        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_categorical_float_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=dummy_objective,
            )

            assert summary["n_trials_complete"] == 3
            best_lr = summary["best_float_params"].get("learning_rate")
            assert best_lr in (0.01, 0.05, 0.1)
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


# =============================================================================
# Timeout Tests
# =============================================================================


class TestOptunaTpeWithTimeout:
    """Tests for OptunaTpeOptimizer timeout behavior."""

    def test_optimize_with_timeout(self) -> None:
        """Optimize runs with timeout configured."""
        from .conftest import make_timeout_config

        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_search_space()
            config = make_timeout_config(n_trials=5, timeout_seconds=10.0)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=dummy_objective,
            )

            # With fake implementation, all trials complete before timeout
            assert summary["n_trials_complete"] == 5
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


# =============================================================================
# Non-DART Booster Tests
# =============================================================================


class TestOptunaTpeWithNonDARTBoosters:
    """Tests for OptunaTpeOptimizer with non-DART boosters (gbtree, gbdt)."""

    def test_optimize_with_xgboost_gbtree(self) -> None:
        """Optimize runs with XGBoost gbtree booster (non-DART)."""
        from .conftest import make_xgboost_gbtree_space

        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_gbtree_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=dummy_objective,
            )

            assert summary["n_trials_complete"] == 3
            assert summary["best_string_params"].get("booster") == "gbtree"
            # No DART params should be present
            assert "rate_drop" not in summary["best_float_params"]
            assert "skip_drop" not in summary["best_float_params"]
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories

    def test_optimize_with_lightgbm_gbdt(self) -> None:
        """Optimize runs with LightGBM gbdt boosting (non-DART)."""
        from .conftest import make_lightgbm_gbdt_space

        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_lightgbm_gbdt_space()
            config = make_optimization_config(n_trials=3)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=lightgbm_objective,
            )

            assert summary["n_trials_complete"] == 3
            assert summary["best_string_params"].get("boosting_type") == "gbdt"
            # No DART params should be present
            assert "drop_rate" not in summary["best_float_params"]
            assert "skip_drop" not in summary["best_float_params"]
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories


# =============================================================================
# Trial Callback Tests
# =============================================================================


class TestOptunaTpeWithTrialCallback:
    """Tests for OptunaTpeOptimizer with trial callback."""

    def test_trial_callback_called(self) -> None:
        """Trial callback is called for each completed trial."""
        from covenant_ml.optimizer.types import TrialResult

        _tpe_hooks.optuna_factories = _make_fake_optuna_hook()
        try:
            optimizer = OptunaTpeOptimizer()
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_search_space()
            config = make_optimization_config(n_trials=3)

            callback_results: list[TrialResult] = []

            def capture_callback(result: TrialResult) -> None:
                callback_results.append(result)

            optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=dummy_objective,
                trial_callback=capture_callback,
            )

            assert len(callback_results) == 3
            for result in callback_results:
                assert result["state"] == "complete"
                assert result["value"] > 0
        finally:
            _tpe_hooks.optuna_factories = _tpe_hooks._real_optuna_factories
