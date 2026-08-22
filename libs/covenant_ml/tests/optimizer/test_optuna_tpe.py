"""Tests for OptunaTpeOptimizer.

Tests cover:
- Strategy name and capabilities
- Optimization with fake Optuna hook
- Various search spaces
- DART configurations
- Pruning support
"""

from __future__ import annotations

from covenant_ml.optimizer.strategies import _hooks as _tpe_hooks
from covenant_ml.optimizer.strategies.optuna_tpe import (
    OptunaTpeOptimizer,
    create_optuna_tpe_optimizer,
)
from covenant_ml.optimizer.types import OptimizationConfig
from tests.optimizer._objective_fixtures import (
    dummy_objective,
    lightgbm_objective,
    lstm_objective,
    mlp_objective,
)
from tests.optimizer._optuna_fixtures import (
    _make_fake_optuna_hook,
)
from tests.optimizer._space_fixtures import (
    make_features,
    make_labels,
    make_lightgbm_search_space,
    make_lstm_search_space,
    make_mlp_search_space,
    make_optimization_config,
    make_xgboost_search_space,
)


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
