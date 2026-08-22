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
    create_grid_search_optimizer,
)
from tests.optimizer._objective_fixtures import (
    dummy_objective,
    lightgbm_dart_no_params_objective,
    lightgbm_dart_objective,
    lightgbm_objective,
    lstm_objective,
    mlp_objective,
    xgboost_dart_no_params_objective,
    xgboost_dart_objective,
)
from tests.optimizer._space_fixtures import (
    make_features,
    make_labels,
    make_lightgbm_dart_no_params_space,
    make_lightgbm_dart_search_space,
    make_lightgbm_search_space,
    make_lstm_search_space,
    make_mlp_search_space,
    make_optimization_config,
    make_xgboost_dart_no_params_space,
    make_xgboost_dart_search_space,
    make_xgboost_search_space,
)


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
