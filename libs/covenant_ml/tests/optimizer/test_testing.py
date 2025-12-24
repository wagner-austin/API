"""Tests for optimizer testing utilities.

Tests cover:
- FakeHyperparameterOptimizer
- Factory functions
- Test registry creation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from covenant_ml.optimizer.testing import (
    FakeHyperparameterOptimizer,
    make_fake_optimizer,
    make_test_optimizer_registry,
)
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
    OptimizationConfig,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    XGBoostSearchSpace,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _make_features(n_samples: int, n_features: int) -> NDArray[np.float64]:
    """Create feature matrix."""
    rng = np.random.default_rng(42)
    result: NDArray[np.float64] = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    return result


def _make_labels(n_samples: int) -> NDArray[np.int64]:
    """Create binary label array."""
    rng = np.random.default_rng(42)
    result: NDArray[np.int64] = rng.integers(0, 2, size=n_samples, dtype=np.int64)
    return result


def _make_search_space() -> XGBoostSearchSpace:
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


def _make_config() -> OptimizationConfig:
    """Create optimization config."""
    return OptimizationConfig(
        n_trials=5,
        timeout_seconds=None,
        n_startup_trials=2,
        random_state=42,
        direction="maximize",
        pruning_enabled=False,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )


def _dummy_objective(
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
    """Dummy objective."""
    return 0.85


# =============================================================================
# FakeHyperparameterOptimizer Tests
# =============================================================================


class TestFakeHyperparameterOptimizer:
    """Tests for FakeHyperparameterOptimizer."""

    def test_default_strategy_name(self) -> None:
        """Default strategy name is optuna_tpe."""
        optimizer = FakeHyperparameterOptimizer()
        assert optimizer.strategy_name() == "optuna_tpe"

    def test_custom_strategy_name(self) -> None:
        """Can set custom strategy name."""
        optimizer = FakeHyperparameterOptimizer(name="grid_search")
        assert optimizer.strategy_name() == "grid_search"

    def test_default_capabilities(self) -> None:
        """Default capabilities are correct."""
        optimizer = FakeHyperparameterOptimizer()
        caps = optimizer.capabilities()

        assert caps["supports_pruning"] is True
        assert caps["supports_parallel"] is True
        assert caps["is_deterministic"] is False
        assert caps["requires_bounds"] is True

    def test_custom_capabilities(self) -> None:
        """Can set custom capabilities."""
        from covenant_ml.optimizer.strategy_protocol import OptimizerStrategyCapabilities

        custom_caps = OptimizerStrategyCapabilities(
            supports_pruning=True,
            supports_parallel=False,
            is_deterministic=False,
            requires_bounds=False,
        )
        optimizer = FakeHyperparameterOptimizer(capabilities=custom_caps)
        caps = optimizer.capabilities()

        assert caps["supports_pruning"] is True
        assert caps["supports_parallel"] is False

    def test_optimize_returns_generated_result(self) -> None:
        """Optimize returns generated fake result."""
        optimizer = FakeHyperparameterOptimizer()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert summary["n_trials_complete"] == config["n_trials"]
        assert summary["best_value"] == 0.85
        assert "max_depth" in summary["best_int_params"]

    def test_optimize_call_count(self) -> None:
        """Tracks number of optimize calls."""
        optimizer = FakeHyperparameterOptimizer()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        assert optimizer.optimize_call_count == 0

        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )
        assert optimizer.optimize_call_count == 1

        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )
        assert optimizer.optimize_call_count == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestMakeFakeOptimizer:
    """Tests for make_fake_optimizer factory."""

    def test_default_factory(self) -> None:
        """Factory creates optimizer with defaults."""
        optimizer = make_fake_optimizer()
        assert optimizer.strategy_name() == "optuna_tpe"
        caps = optimizer.capabilities()
        assert caps["is_deterministic"] is False

    def test_factory_with_custom_name(self) -> None:
        """Factory creates optimizer with custom name."""
        optimizer = make_fake_optimizer(name="optuna_tpe")
        assert optimizer.strategy_name() == "optuna_tpe"

    def test_factory_with_custom_value(self) -> None:
        """Factory creates optimizer with custom best value."""
        optimizer = make_fake_optimizer(best_value=0.95)
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert summary["best_value"] == 0.95


# =============================================================================
# Test Registry Tests
# =============================================================================


class TestMakeTestOptimizerRegistry:
    """Tests for make_test_optimizer_registry factory."""

    def test_registry_has_all_strategies(self) -> None:
        """Test registry has expected fake strategies."""
        registry = make_test_optimizer_registry()
        strategies = registry.list_strategies()

        assert "random_search" in strategies
        assert "grid_search" in strategies
        assert "optuna_tpe" in strategies

    def test_strategies_are_fake(self) -> None:
        """All strategies are FakeHyperparameterOptimizer instances."""
        registry = make_test_optimizer_registry()

        optimizer = registry.get("random_search")
        assert optimizer.strategy_name() == "random_search"

        optimizer2 = registry.get("optuna_tpe")
        assert optimizer2.strategy_name() == "optuna_tpe"

    def test_strategies_work(self) -> None:
        """Fake strategies produce valid results."""
        registry = make_test_optimizer_registry()
        optimizer = registry.get("random_search")

        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert summary["n_trials_complete"] >= 1
        assert summary["best_value"] > 0

    def test_grid_search_strategy(self) -> None:
        """Grid search strategy is accessible and works."""
        registry = make_test_optimizer_registry()
        optimizer = registry.get("grid_search")

        assert optimizer.strategy_name() == "grid_search"
        caps = optimizer.capabilities()
        assert caps["is_deterministic"] is True


# =============================================================================
# FakeObjective Tests
# =============================================================================


class TestFakeObjective:
    """Tests for FakeObjective."""

    def test_default_return_value(self) -> None:
        """Default return value is 0.85."""
        from covenant_ml.optimizer.testing import FakeObjective

        objective = FakeObjective()
        x = _make_features(100, 10)
        y = _make_labels(100)

        result = objective(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            int_params=SampledIntParams(max_depth=5, n_estimators=100),
            float_params=SampledFloatParams(learning_rate=0.1),
            string_params=SampledStringParams(),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert result == 0.85

    def test_custom_return_value(self) -> None:
        """Can set custom return value."""
        from covenant_ml.optimizer.testing import FakeObjective

        objective = FakeObjective(return_value=0.92)
        x = _make_features(100, 10)
        y = _make_labels(100)

        result = objective(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            int_params=SampledIntParams(max_depth=5, n_estimators=100),
            float_params=SampledFloatParams(learning_rate=0.1),
            string_params=SampledStringParams(),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert result == 0.92

    def test_call_count(self) -> None:
        """Tracks number of calls."""
        from covenant_ml.optimizer.testing import FakeObjective

        objective = FakeObjective()
        x = _make_features(100, 10)
        y = _make_labels(100)

        assert objective.call_count == 0

        objective(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            int_params=SampledIntParams(max_depth=5, n_estimators=100),
            float_params=SampledFloatParams(learning_rate=0.1),
            string_params=SampledStringParams(),
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert objective.call_count == 1

    def test_records_calls(self) -> None:
        """Records parameter tuples from calls."""
        from covenant_ml.optimizer.testing import FakeObjective

        objective = FakeObjective()
        x = _make_features(100, 10)
        y = _make_labels(100)

        int_params = SampledIntParams(max_depth=5, n_estimators=100)
        float_params = SampledFloatParams(learning_rate=0.1)
        string_params = SampledStringParams()

        objective(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            int_params=int_params,
            float_params=float_params,
            string_params=string_params,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        assert len(objective.calls) == 1
        recorded_int, recorded_float, _recorded_string = objective.calls[0]
        assert recorded_int["max_depth"] == 5
        assert recorded_float["learning_rate"] == 0.1


# =============================================================================
# FakeTrialCallback Tests
# =============================================================================


class TestFakeTrialCallback:
    """Tests for FakeTrialCallback."""

    def test_starts_empty(self) -> None:
        """Callback starts with no results."""
        from covenant_ml.optimizer.testing import FakeTrialCallback

        callback = FakeTrialCallback()
        assert callback.results == []

    def test_records_trial_results(self) -> None:
        """Callback records trial results."""
        from covenant_ml.optimizer.testing import FakeTrialCallback
        from covenant_ml.optimizer.types import TrialResult

        callback = FakeTrialCallback()

        result = TrialResult(
            trial_number=0,
            value=0.85,
            int_params=SampledIntParams(max_depth=5, n_estimators=100),
            float_params=SampledFloatParams(learning_rate=0.1),
            string_params=SampledStringParams(),
            duration_seconds=1.0,
            state="complete",
        )

        callback(result)

        assert len(callback.results) == 1
        assert callback.results[0]["trial_number"] == 0
        assert callback.results[0]["value"] == 0.85


# =============================================================================
# make_test_optimization_config Tests
# =============================================================================


class TestMakeTestOptimizationConfig:
    """Tests for make_test_optimization_config factory."""

    def test_default_config(self) -> None:
        """Factory creates config with defaults."""
        from covenant_ml.optimizer.testing import make_test_optimization_config

        config = make_test_optimization_config()

        assert config["n_trials"] == 10
        assert config["random_state"] == 42
        assert config["direction"] == "maximize"

    def test_custom_n_trials(self) -> None:
        """Factory creates config with custom n_trials."""
        from covenant_ml.optimizer.testing import make_test_optimization_config

        config = make_test_optimization_config(n_trials=20)
        assert config["n_trials"] == 20

    def test_custom_random_state(self) -> None:
        """Factory creates config with custom random_state."""
        from covenant_ml.optimizer.testing import make_test_optimization_config

        config = make_test_optimization_config(random_state=123)
        assert config["random_state"] == 123


# =============================================================================
# FakeHyperparameterOptimizer Additional Tests
# =============================================================================


class TestFakeHyperparameterOptimizerProperties:
    """Tests for FakeHyperparameterOptimizer additional properties."""

    def test_last_search_space_starts_none(self) -> None:
        """last_search_space is None before any optimize call."""
        optimizer = FakeHyperparameterOptimizer()
        assert optimizer.last_search_space is None

    def test_last_config_starts_none(self) -> None:
        """last_config is None before any optimize call."""
        optimizer = FakeHyperparameterOptimizer()
        assert optimizer.last_config is None

    def test_last_search_space_after_optimize(self) -> None:
        """last_search_space is set after optimize call."""
        optimizer = FakeHyperparameterOptimizer()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        # Verify space was captured and matches input
        last_space = optimizer.last_search_space
        assert last_space == space

    def test_last_config_after_optimize(self) -> None:
        """last_config is set after optimize call."""
        optimizer = FakeHyperparameterOptimizer()
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        # Verify config was captured and matches input
        last_cfg = optimizer.last_config
        assert last_cfg == config

    def test_custom_result(self) -> None:
        """Can provide custom result to return."""
        from covenant_ml.optimizer.types import OptimizationSummary

        custom_result = OptimizationSummary(
            best_trial_number=3,
            best_value=0.99,
            best_int_params=SampledIntParams(max_depth=10, n_estimators=500),
            best_float_params=SampledFloatParams(learning_rate=0.01),
            best_string_params=SampledStringParams(),
            n_trials_total=20,
            n_trials_complete=18,
            n_trials_pruned=2,
            n_trials_failed=0,
            total_duration_seconds=100.0,
        )

        optimizer = FakeHyperparameterOptimizer(result=custom_result)
        x = _make_features(100, 10)
        y = _make_labels(100)
        space = _make_search_space()
        config = _make_config()

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=_dummy_objective,
        )

        assert summary["best_trial_number"] == 3
        assert summary["best_value"] == 0.99
        assert summary["n_trials_pruned"] == 2
