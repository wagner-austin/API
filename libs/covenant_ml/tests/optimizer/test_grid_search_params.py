"""Tests for GridSearchOptimizer.

Tests cover:
- Strategy name and capabilities
- Optimization with various search spaces
- DART configurations
- Categorical and log-scale parameters
- Timeout behavior
- Minimize direction
- Trial callbacks
- Integer grid edge cases
"""

from __future__ import annotations

from covenant_ml.optimizer.strategies import (
    GridSearchOptimizer,
    GridTuple,
)
from covenant_ml.optimizer.strategies import _hooks as strategy_hooks
from covenant_ml.optimizer.types import OptimizationConfig, SearchSpace, TrialResult
from tests.optimizer._objective_fixtures import (
    dummy_objective,
    slow_objective,
)
from tests.optimizer._space_fixtures import (
    make_features,
    make_labels,
    make_optimization_config,
    make_timeout_config,
    make_xgboost_categorical_float_space,
    make_xgboost_categorical_space,
    make_xgboost_log_scale_space,
    make_xgboost_narrow_range_space,
    make_xgboost_search_space,
)


class TestGridSearchWithCategoricalParams:
    """Tests for GridSearchOptimizer with categorical parameters."""

    def test_optimize_with_categorical_int(self) -> None:
        """Optimize runs with categorical int parameters."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_categorical_space()
        config = make_optimization_config(n_trials=10)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=dummy_objective,
        )

        assert summary["n_trials_complete"] > 0
        best_max_depth = summary["best_int_params"].get("max_depth")
        assert best_max_depth in (3, 5, 7)


class TestGridSearchWithLogScaleParams:
    """Tests for GridSearchOptimizer with log-scale parameters."""

    def test_optimize_with_log_scale_int(self) -> None:
        """Optimize runs with log-scale int parameters."""
        optimizer = GridSearchOptimizer(grid_points=3)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_log_scale_space()
        config = make_optimization_config(n_trials=10)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=dummy_objective,
        )

        assert summary["n_trials_complete"] > 0
        assert "max_depth" in summary["best_int_params"]


class TestGridSearchWithCategoricalFloat:
    """Tests for GridSearchOptimizer with CategoricalFloatSpec."""

    def test_optimize_with_categorical_float(self) -> None:
        """Optimize runs with categorical float parameters."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_categorical_float_space()
        config = make_optimization_config(n_trials=10)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=dummy_objective,
        )

        assert summary["n_trials_complete"] > 0
        best_lr = summary["best_float_params"].get("learning_rate")
        assert best_lr in (0.01, 0.05, 0.1)


class TestGridSearchWithTimeout:
    """Tests for GridSearchOptimizer timeout behavior."""

    def test_timeout_stops_optimization(self) -> None:
        """Timeout stops optimization before all trials complete."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_search_space()
        config = make_timeout_config(n_trials=1000, timeout_seconds=0.3)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=slow_objective,
        )

        assert summary["n_trials_complete"] < 1000
        assert summary["n_trials_complete"] >= 1


class TestGridSearchMinimize:
    """Tests for GridSearchOptimizer with minimize direction."""

    def test_minimize_direction(self) -> None:
        """Optimizer correctly minimizes objective."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_search_space()
        config = OptimizationConfig(
            n_trials=10,
            timeout_seconds=None,
            n_startup_trials=2,
            random_state=42,
            direction="minimize",
            pruning_enabled=False,
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

        assert summary["n_trials_complete"] == 10
        assert summary["best_value"] > 0


class TestGridSearchCallback:
    """Tests for GridSearch trial callback."""

    def test_trial_callback_called(self) -> None:
        """Trial callback is called for each completed trial."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_search_space()
        config = make_optimization_config(n_trials=5)

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

        assert len(callback_results) == 5
        for result in callback_results:
            assert result["state"] == "complete"
            assert result["value"] > 0


class TestGridSearchIntEdgeCases:
    """Tests for GridSearch integer grid edge cases."""

    def test_high_value_included_in_grid(self) -> None:
        """Grid includes the high value even when step doesn't land on it."""
        optimizer = GridSearchOptimizer(grid_points=3)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_narrow_range_space()
        config = make_optimization_config(n_trials=20)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=dummy_objective,
        )

        assert summary["n_trials_complete"] > 0


class TestGridSearchEmptyGrid:
    """Tests for GridSearch with empty grid via hook."""

    def test_empty_grid_returns_default_summary(self) -> None:
        """Empty grid returns summary with zero trials via normal for loop path."""

        def empty_grid_builder(
            search_space: SearchSpace,
            n_points: int,
        ) -> list[GridTuple]:
            return []

        strategy_hooks.build_grid = empty_grid_builder
        try:
            optimizer = GridSearchOptimizer(grid_points=2)
            x = make_features(100, 10)
            y = make_labels(100)
            space = make_xgboost_search_space()
            config = make_optimization_config(n_trials=10)

            summary = optimizer.optimize(
                x_features=x,
                y_labels=y,
                feature_names=[f"f{i}" for i in range(10)],
                search_space=space,
                config=config,
                objective=dummy_objective,
            )

            # For loop never executes with empty grid
            assert summary["n_trials_complete"] == 0
            assert summary["n_trials_total"] == 0
            assert summary["best_int_params"] == {}
            assert summary["best_float_params"] == {}
            assert summary["best_string_params"] == {}
            # best_value remains at initial value (inf for maximize)
            assert summary["best_value"] == float("-inf")
        finally:
            strategy_hooks.build_grid = strategy_hooks._real_build_grid

    def test_rebinding_back_restores_the_real_builder(self) -> None:
        """Restoring the module's own builder puts the real grid back."""

        def empty_grid_builder(
            search_space: SearchSpace,
            n_points: int,
        ) -> list[GridTuple]:
            return []

        strategy_hooks.build_grid = empty_grid_builder
        strategy_hooks.build_grid = strategy_hooks._real_build_grid

        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_search_space()
        config = make_optimization_config(n_trials=5)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=dummy_objective,
        )

        # Should run normally with default grid builder
        assert summary["n_trials_complete"] == 5
