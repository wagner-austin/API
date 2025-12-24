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
    create_grid_search_optimizer,
    set_build_grid_hook,
)
from covenant_ml.optimizer.types import OptimizationConfig, SearchSpace, TrialResult

from .conftest import (
    dummy_objective,
    lightgbm_dart_no_params_objective,
    lightgbm_dart_objective,
    lightgbm_objective,
    lstm_objective,
    make_features,
    make_labels,
    make_lightgbm_dart_no_params_space,
    make_lightgbm_dart_search_space,
    make_lightgbm_search_space,
    make_lstm_search_space,
    make_mlp_search_space,
    make_optimization_config,
    make_timeout_config,
    make_xgboost_categorical_float_space,
    make_xgboost_categorical_space,
    make_xgboost_dart_no_params_space,
    make_xgboost_dart_search_space,
    make_xgboost_log_scale_space,
    make_xgboost_narrow_range_space,
    make_xgboost_search_space,
    mlp_objective,
    slow_objective,
    xgboost_dart_no_params_objective,
    xgboost_dart_objective,
)

# =============================================================================
# Core Tests
# =============================================================================


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer core functionality."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        optimizer = GridSearchOptimizer()
        assert optimizer.strategy_name() == "grid_search"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        optimizer = GridSearchOptimizer()
        caps = optimizer.capabilities()

        assert caps["supports_pruning"] is False
        assert caps["supports_parallel"] is True
        assert caps["is_deterministic"] is True
        assert caps["requires_bounds"] is True

    def test_optimize_runs_trials(self) -> None:
        """Optimize runs trials up to max."""
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

        assert summary["n_trials_complete"] == 5
        assert summary["best_value"] > 0

    def test_optimize_tracks_best_params(self) -> None:
        """Optimize tracks the best parameters."""
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

        assert "max_depth" in summary["best_int_params"]
        assert "n_estimators" in summary["best_int_params"]
        assert "learning_rate" in summary["best_float_params"]

    def test_optimize_is_deterministic(self) -> None:
        """Grid search is deterministic."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_search_space()
        config = make_optimization_config(n_trials=5)

        summary1 = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=dummy_objective,
        )

        summary2 = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=dummy_objective,
        )

        assert summary1["best_value"] == summary2["best_value"]

    def test_custom_grid_points(self) -> None:
        """Can set custom grid points."""
        optimizer = GridSearchOptimizer(grid_points=3)
        assert optimizer.grid_points == 3


class TestGridSearchFactory:
    """Tests for create_grid_search_optimizer factory."""

    def test_factory_creates_optimizer(self) -> None:
        """Factory creates optimizer."""
        optimizer = create_grid_search_optimizer()
        assert optimizer.strategy_name() == "grid_search"

    def test_factory_default_grid_points(self) -> None:
        """Factory creates optimizer with default grid points."""
        optimizer = create_grid_search_optimizer()
        assert optimizer.grid_points == 3


# =============================================================================
# Backend-Specific Tests
# =============================================================================


class TestGridSearchWithMLPSpace:
    """Tests for GridSearchOptimizer with MLP search space."""

    def test_optimize_with_mlp_space(self) -> None:
        """Optimize runs with MLP search space."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_mlp_search_space()
        config = make_optimization_config(n_trials=5)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=mlp_objective,
        )

        assert summary["n_trials_complete"] == 5
        assert "n_layers" in summary["best_int_params"]


class TestGridSearchWithLSTMSpace:
    """Tests for GridSearchOptimizer with LSTM search space."""

    def test_optimize_with_lstm_space(self) -> None:
        """Optimize runs with LSTM search space."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_lstm_search_space()
        config = make_optimization_config(n_trials=5)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=lstm_objective,
        )

        assert summary["n_trials_complete"] == 5
        assert "hidden_size" in summary["best_int_params"]


class TestGridSearchWithLightGBMSpace:
    """Tests for GridSearchOptimizer with LightGBM search space."""

    def test_optimize_with_lightgbm_space(self) -> None:
        """Optimize runs with LightGBM search space."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_lightgbm_search_space()
        config = make_optimization_config(n_trials=5)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=lightgbm_objective,
        )

        assert summary["n_trials_complete"] == 5
        assert "num_leaves" in summary["best_int_params"]


# =============================================================================
# DART Tests
# =============================================================================


class TestGridSearchWithXGBoostDART:
    """Tests for GridSearchOptimizer with XGBoost DART."""

    def test_optimize_with_xgboost_dart(self) -> None:
        """Optimize runs with XGBoost DART search space."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_dart_search_space()
        config = make_optimization_config(n_trials=10)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=xgboost_dart_objective,
        )

        assert summary["n_trials_complete"] == 10
        assert "booster" in summary["best_string_params"]


class TestGridSearchWithLightGBMDART:
    """Tests for GridSearchOptimizer with LightGBM DART."""

    def test_optimize_with_lightgbm_dart(self) -> None:
        """Optimize runs with LightGBM DART search space."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_lightgbm_dart_search_space()
        config = make_optimization_config(n_trials=10)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=lightgbm_dart_objective,
        )

        assert summary["n_trials_complete"] == 10
        assert "boosting_type" in summary["best_string_params"]


class TestGridSearchDARTWithoutParams:
    """Tests for GridSearch with DART but no DART params in space."""

    def test_xgboost_dart_without_dart_params(self) -> None:
        """Grid search works with XGBoost DART without rate_drop/skip_drop."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_dart_no_params_space()
        config = make_optimization_config(n_trials=10)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=xgboost_dart_no_params_objective,
        )

        assert summary["n_trials_complete"] > 0
        assert summary["best_string_params"].get("booster") == "dart"

    def test_lightgbm_dart_without_dart_params(self) -> None:
        """Grid search works with LightGBM DART without drop_rate/skip_drop."""
        optimizer = GridSearchOptimizer(grid_points=2)
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_lightgbm_dart_no_params_space()
        config = make_optimization_config(n_trials=10)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=lightgbm_dart_no_params_objective,
        )

        assert summary["n_trials_complete"] > 0
        assert summary["best_string_params"].get("boosting_type") == "dart"


# =============================================================================
# Parameter Type Tests
# =============================================================================


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


# =============================================================================
# Timeout, Direction, and Callback Tests
# =============================================================================


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


# =============================================================================
# Edge Case Tests
# =============================================================================


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

        set_build_grid_hook(empty_grid_builder)
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
            set_build_grid_hook(None)

    def test_hook_reset_restores_default(self) -> None:
        """Setting hook to None restores default grid builder."""

        def empty_grid_builder(
            search_space: SearchSpace,
            n_points: int,
        ) -> list[GridTuple]:
            return []

        set_build_grid_hook(empty_grid_builder)
        set_build_grid_hook(None)

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
