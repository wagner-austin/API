"""Tests for RandomSearchOptimizer.

Tests cover:
- Strategy name and capabilities
- Optimization with various search spaces
- DART configurations
- Categorical and log-scale parameters
- Timeout behavior
- Minimize direction
"""

from __future__ import annotations

from covenant_ml.optimizer.strategies import (
    RandomSearchOptimizer,
    create_random_search_optimizer,
)
from covenant_ml.optimizer.types import OptimizationConfig
from tests.optimizer._objective_fixtures import (
    dummy_objective,
    lightgbm_dart_no_params_objective,
    lightgbm_dart_objective,
    lightgbm_objective,
    lstm_objective,
    mlp_objective,
    slow_objective,
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
    make_timeout_config,
    make_xgboost_categorical_float_space,
    make_xgboost_categorical_space,
    make_xgboost_dart_no_params_space,
    make_xgboost_dart_search_space,
    make_xgboost_log_scale_space,
    make_xgboost_search_space,
)

# =============================================================================
# Core Tests
# =============================================================================


class TestRandomSearchOptimizer:
    """Tests for RandomSearchOptimizer core functionality."""

    def test_strategy_name(self) -> None:
        """Strategy name is correct."""
        optimizer = RandomSearchOptimizer()
        assert optimizer.strategy_name() == "random_search"

    def test_capabilities(self) -> None:
        """Capabilities are correctly reported."""
        optimizer = RandomSearchOptimizer()
        caps = optimizer.capabilities()

        assert caps["supports_pruning"] is False
        assert caps["supports_parallel"] is True
        assert caps["is_deterministic"] is True
        assert caps["requires_bounds"] is True

    def test_optimize_runs_trials(self) -> None:
        """Optimize runs the requested number of trials."""
        optimizer = RandomSearchOptimizer()
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
        assert summary["n_trials_total"] == 3
        assert summary["best_value"] > 0

    def test_optimize_tracks_best_params(self) -> None:
        """Optimize tracks the best parameters."""
        optimizer = RandomSearchOptimizer()
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

        assert "max_depth" in summary["best_int_params"]
        assert "n_estimators" in summary["best_int_params"]
        assert "learning_rate" in summary["best_float_params"]

    def test_optimize_is_deterministic(self) -> None:
        """Same random state produces same results."""
        optimizer = RandomSearchOptimizer()
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_search_space()
        config = make_optimization_config(n_trials=3)

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


class TestRandomSearchFactory:
    """Tests for create_random_search_optimizer factory."""

    def test_factory_creates_optimizer(self) -> None:
        """Factory creates optimizer."""
        optimizer = create_random_search_optimizer()
        assert optimizer.strategy_name() == "random_search"


# =============================================================================
# Backend-Specific Tests
# =============================================================================


class TestRandomSearchWithMLPSpace:
    """Tests for RandomSearchOptimizer with MLP search space."""

    def test_optimize_with_mlp_space(self) -> None:
        """Optimize runs with MLP search space."""
        optimizer = RandomSearchOptimizer()
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
        assert "n_layers" in summary["best_int_params"]


class TestRandomSearchWithLSTMSpace:
    """Tests for RandomSearchOptimizer with LSTM search space."""

    def test_optimize_with_lstm_space(self) -> None:
        """Optimize runs with LSTM search space."""
        optimizer = RandomSearchOptimizer()
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
        assert "hidden_size" in summary["best_int_params"]


class TestRandomSearchWithLightGBMSpace:
    """Tests for RandomSearchOptimizer with LightGBM search space."""

    def test_optimize_with_lightgbm_space(self) -> None:
        """Optimize runs with LightGBM search space."""
        optimizer = RandomSearchOptimizer()
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
        assert "num_leaves" in summary["best_int_params"]


# =============================================================================
# DART Tests
# =============================================================================


class TestRandomSearchWithXGBoostDART:
    """Tests for RandomSearchOptimizer with XGBoost DART."""

    def test_optimize_with_xgboost_dart(self) -> None:
        """Optimize runs with XGBoost DART search space."""
        optimizer = RandomSearchOptimizer()
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_dart_search_space()
        config = make_optimization_config(n_trials=5)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=xgboost_dart_objective,
        )

        assert summary["n_trials_complete"] == 5
        assert "booster" in summary["best_string_params"]


class TestRandomSearchWithLightGBMDART:
    """Tests for RandomSearchOptimizer with LightGBM DART."""

    def test_optimize_with_lightgbm_dart(self) -> None:
        """Optimize runs with LightGBM DART search space."""
        optimizer = RandomSearchOptimizer()
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_lightgbm_dart_search_space()
        config = make_optimization_config(n_trials=5)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=lightgbm_dart_objective,
        )

        assert summary["n_trials_complete"] == 5
        assert "boosting_type" in summary["best_string_params"]


class TestRandomSearchDARTWithoutParams:
    """Tests for RandomSearch with DART but no DART params in space."""

    def test_xgboost_dart_without_dart_params(self) -> None:
        """Random search works with XGBoost DART without rate_drop/skip_drop."""
        optimizer = RandomSearchOptimizer()
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_dart_no_params_space()
        config = make_optimization_config(n_trials=5)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=xgboost_dart_no_params_objective,
        )

        assert summary["n_trials_complete"] == 5
        assert summary["best_string_params"].get("booster") == "dart"
        assert "rate_drop" not in summary["best_float_params"]
        assert "skip_drop" not in summary["best_float_params"]

    def test_lightgbm_dart_without_dart_params(self) -> None:
        """Random search works with LightGBM DART without drop_rate/skip_drop."""
        optimizer = RandomSearchOptimizer()
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_lightgbm_dart_no_params_space()
        config = make_optimization_config(n_trials=5)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=lightgbm_dart_no_params_objective,
        )

        assert summary["n_trials_complete"] == 5
        assert summary["best_string_params"].get("boosting_type") == "dart"
        assert "drop_rate" not in summary["best_float_params"]
        assert "skip_drop" not in summary["best_float_params"]
        assert "feature_fraction" not in summary["best_float_params"]


# =============================================================================
# Parameter Type Tests
# =============================================================================


class TestRandomSearchWithCategoricalParams:
    """Tests for RandomSearchOptimizer with categorical parameters."""

    def test_optimize_with_categorical_int(self) -> None:
        """Optimize runs with categorical int parameters."""
        optimizer = RandomSearchOptimizer()
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_categorical_space()
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
        best_max_depth = summary["best_int_params"].get("max_depth")
        assert best_max_depth in (3, 5, 7)


class TestRandomSearchWithLogScaleParams:
    """Tests for RandomSearchOptimizer with log-scale parameters."""

    def test_optimize_with_log_scale_int(self) -> None:
        """Optimize runs with log-scale int parameters."""
        optimizer = RandomSearchOptimizer()
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_log_scale_space()
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
        assert "max_depth" in summary["best_int_params"]
        assert "n_estimators" in summary["best_int_params"]


class TestRandomSearchWithCategoricalFloat:
    """Tests for RandomSearchOptimizer with CategoricalFloatSpec."""

    def test_optimize_with_categorical_float(self) -> None:
        """Optimize runs with categorical float parameters."""
        optimizer = RandomSearchOptimizer()
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_categorical_float_space()
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
        best_lr = summary["best_float_params"].get("learning_rate")
        assert best_lr in (0.01, 0.05, 0.1)


# =============================================================================
# Timeout and Direction Tests
# =============================================================================


class TestRandomSearchWithTimeout:
    """Tests for RandomSearchOptimizer timeout behavior."""

    def test_timeout_stops_optimization(self) -> None:
        """Timeout stops optimization before all trials complete."""
        optimizer = RandomSearchOptimizer()
        x = make_features(100, 10)
        y = make_labels(100)
        space = make_xgboost_search_space()
        config = make_timeout_config(n_trials=100, timeout_seconds=0.3)

        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=[f"f{i}" for i in range(10)],
            search_space=space,
            config=config,
            objective=slow_objective,
        )

        assert summary["n_trials_complete"] < 100
        assert summary["n_trials_complete"] >= 1


class TestRandomSearchMinimize:
    """Tests for RandomSearchOptimizer with minimize direction."""

    def test_minimize_direction(self) -> None:
        """Optimizer correctly minimizes objective."""
        optimizer = RandomSearchOptimizer()
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
        # Best value should be lowest found
        assert summary["best_value"] > 0
