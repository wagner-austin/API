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
)
from tests.optimizer._objective_fixtures import (
    dummy_objective,
    lightgbm_dart_objective,
    lightgbm_objective,
    xgboost_dart_objective,
)
from tests.optimizer._optuna_fixtures import (
    _make_fake_optuna_hook,
)
from tests.optimizer._space_fixtures import (
    make_features,
    make_labels,
    make_lightgbm_dart_search_space,
    make_optimization_config,
    make_xgboost_dart_search_space,
    make_xgboost_search_space,
)


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


class TestOptunaTpeWithDARTNoParams:
    """Tests for OptunaTpeOptimizer with DART but no DART params."""

    def test_optimize_with_xgboost_dart_no_params(self) -> None:
        """Optimize runs with XGBoost DART without rate_drop/skip_drop."""
        from tests.optimizer._objective_fixtures import xgboost_dart_no_params_objective
        from tests.optimizer._space_fixtures import make_xgboost_dart_no_params_space

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
        from tests.optimizer._objective_fixtures import lightgbm_dart_no_params_objective
        from tests.optimizer._space_fixtures import make_lightgbm_dart_no_params_space

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


class TestOptunaTpeWithCategoricalParams:
    """Tests for OptunaTpeOptimizer with categorical parameters."""

    def test_optimize_with_categorical_int(self) -> None:
        """Optimize runs with categorical int parameters."""
        from tests.optimizer._space_fixtures import make_xgboost_categorical_space

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
        from tests.optimizer._space_fixtures import make_xgboost_categorical_float_space

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


class TestOptunaTpeWithTimeout:
    """Tests for OptunaTpeOptimizer timeout behavior."""

    def test_optimize_with_timeout(self) -> None:
        """Optimize runs with timeout configured."""
        from tests.optimizer._space_fixtures import make_timeout_config

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


class TestOptunaTpeWithNonDARTBoosters:
    """Tests for OptunaTpeOptimizer with non-DART boosters (gbtree, gbdt)."""

    def test_optimize_with_xgboost_gbtree(self) -> None:
        """Optimize runs with XGBoost gbtree booster (non-DART)."""
        from tests.optimizer._space_fixtures import make_xgboost_gbtree_space

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
        from tests.optimizer._space_fixtures import make_lightgbm_gbdt_space

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
