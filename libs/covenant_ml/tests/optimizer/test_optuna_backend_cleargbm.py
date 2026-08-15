"""Tests for ClearGBM Optuna backend optimizer.

Uses fake Optuna implementation for testing without mocks.
"""

from __future__ import annotations

from covenant_ml.optimizer.optuna_backend import _hooks as _backend_hooks
from covenant_ml.optimizer.optuna_backend import (
    create_cleargbm_optimizer,
)
from covenant_ml.optimizer.search_spaces import make_cleargbm_default_space
from covenant_ml.optimizer.types import TrialResult

from .conftest import (
    FakeObjective,
    get_fake_optuna_factories,
    make_optuna_config,
    make_optuna_test_data,
)

# =============================================================================
# Tests: ClearGBM Optimizer
# =============================================================================


def test_cleargbm_optimizer_completes_trials() -> None:
    """ClearGBM optimizer completes configured number of trials."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_cleargbm_optimizer_returns_best_params() -> None:
    """ClearGBM optimizer returns best parameters in summary."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        # ClearGBM has 5 int params
        assert len(summary["best_int_params"]) == 5
        # ClearGBM has 2 float params
        assert len(summary["best_float_params"]) == 2
        # ClearGBM has no string params
        assert len(summary["best_string_params"]) == 0
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_cleargbm_optimizer_best_value_in_summary() -> None:
    """ClearGBM optimizer returns best value in summary."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert 0.5 <= summary["best_value"] <= 1.0
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_cleargbm_optimizer_records_duration() -> None:
    """ClearGBM optimizer records total duration."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert summary["total_duration_seconds"] >= 0.0
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_cleargbm_optimizer_without_pruning() -> None:
    """ClearGBM optimizer works without pruning."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=make_optuna_config(n_trials=3, pruning_enabled=False),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_cleargbm_optimizer_with_trial_callback() -> None:
    """ClearGBM optimizer calls trial_callback for each trial."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_cleargbm_optimizer()
        x, y, names = make_optuna_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for result in callbacks:
            assert result["state"] == "complete"
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_cleargbm_optimizer_with_timeout() -> None:
    """ClearGBM optimizer accepts timeout_seconds."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_cleargbm_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_cleargbm_default_space(),
            config=make_optuna_config(n_trials=3, timeout_seconds=60),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories
