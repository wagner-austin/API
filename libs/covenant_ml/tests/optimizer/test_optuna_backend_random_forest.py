"""Tests for Random Forest Optuna backend optimizer.

Uses fake Optuna implementation for testing without mocks.
"""

from __future__ import annotations

from covenant_ml.optimizer.optuna_backend import _hooks as _backend_hooks
from covenant_ml.optimizer.optuna_backend import (
    create_random_forest_optimizer,
)
from covenant_ml.optimizer.search_spaces import make_random_forest_default_space
from covenant_ml.optimizer.types import TrialResult

from .conftest import (
    FakeObjective,
    get_fake_optuna_factories,
    make_optuna_config,
    make_optuna_test_data,
)

# =============================================================================
# Tests: Random Forest Optimizer
# =============================================================================


def test_random_forest_optimizer_completes_trials() -> None:
    """Random Forest optimizer completes configured number of trials."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_random_forest_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_random_forest_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_random_forest_optimizer_returns_best_params() -> None:
    """Random Forest optimizer returns best parameters in summary."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_random_forest_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_random_forest_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert "n_estimators" in summary["best_int_params"]
        assert "max_depth" in summary["best_int_params"]
        assert "min_samples_split" in summary["best_int_params"]
        assert "min_samples_leaf" in summary["best_int_params"]
        assert len(summary["best_float_params"]) == 0
        assert "max_features" in summary["best_string_params"]
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_random_forest_optimizer_best_value_in_summary() -> None:
    """Random Forest optimizer returns best value in summary."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_random_forest_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_random_forest_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert 0.5 <= summary["best_value"] <= 1.0
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_random_forest_optimizer_records_duration() -> None:
    """Random Forest optimizer records total duration."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_random_forest_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_random_forest_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert summary["total_duration_seconds"] >= 0.0
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_random_forest_optimizer_without_pruning() -> None:
    """Random Forest optimizer works without pruning."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_random_forest_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_random_forest_default_space(),
            config=make_optuna_config(n_trials=3, pruning_enabled=False),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_random_forest_optimizer_with_trial_callback() -> None:
    """Random Forest optimizer calls trial_callback for each trial."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_random_forest_optimizer()
        x, y, names = make_optuna_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_random_forest_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for result in callbacks:
            assert result["state"] == "complete"
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_random_forest_optimizer_with_timeout() -> None:
    """Random Forest optimizer accepts timeout_seconds."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_random_forest_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_random_forest_default_space(),
            config=make_optuna_config(n_trials=3, timeout_seconds=60),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories
