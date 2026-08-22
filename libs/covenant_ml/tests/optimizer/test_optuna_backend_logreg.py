"""Tests for LogReg Optuna backend optimizer and optional parameter handling.

Uses fake Optuna implementation for testing without mocks.
"""

from __future__ import annotations

from covenant_ml.optimizer.optuna_backend import _hooks as _backend_hooks
from covenant_ml.optimizer.optuna_backend import (
    create_logreg_optimizer,
)
from covenant_ml.optimizer.optuna_backend.logreg import _sample_logreg_optional_params
from covenant_ml.optimizer.search_spaces import (
    make_logreg_default_space,
    make_logreg_focused_space,
)
from covenant_ml.optimizer.types import (
    LogRegSearchSpace,
    SampledFloatParams,
    SampledStringParams,
    TrialResult,
)
from tests.optimizer._fake_optuna import (
    FakeTrial,
    get_fake_optuna_factories,
    make_optuna_config,
    make_optuna_test_data,
)
from tests.optimizer._objective_fixtures import FakeObjective

# =============================================================================
# Tests: _sample_logreg_optional_params
# =============================================================================


def test_sample_logreg_optional_params_all_present() -> None:
    """_sample_logreg_optional_params samples penalty, solver, l1_ratio when present."""
    trial = FakeTrial(0)
    space = make_logreg_default_space()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_logreg_optional_params(trial, space, float_params, string_params)
    assert "penalty" in string_params
    assert "solver" in string_params
    assert "l1_ratio" in float_params


def test_sample_logreg_optional_params_none_present() -> None:
    """_sample_logreg_optional_params skips when optional keys absent."""
    trial = FakeTrial(0)
    space = make_logreg_focused_space(best_c=1.0, best_tol=1e-4)
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_logreg_optional_params(trial, space, float_params, string_params)
    assert "penalty" not in string_params
    assert "solver" not in string_params
    assert "l1_ratio" not in float_params


def test_sample_logreg_optional_params_partial() -> None:
    """_sample_logreg_optional_params handles partial optional keys."""
    trial = FakeTrial(0)
    space: LogRegSearchSpace = {
        "C": {"param_type": "float", "low": 0.01, "high": 10.0, "log_scale": True},
        "max_iter": {"param_type": "int", "low": 100, "high": 500, "log_scale": False},
        "tol": {"param_type": "float", "low": 1e-5, "high": 1e-3, "log_scale": True},
        "penalty": {"param_type": "categorical_str", "choices": ("l2",)},
    }
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_logreg_optional_params(trial, space, float_params, string_params)
    assert "penalty" in string_params
    assert "solver" not in string_params
    assert "l1_ratio" not in float_params


# =============================================================================
# Tests: LogReg Optimizer
# =============================================================================


def test_logreg_optimizer_completes_trials() -> None:
    """LogReg optimizer completes configured number of trials."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_logreg_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_logreg_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_logreg_optimizer_returns_best_params() -> None:
    """LogReg optimizer returns best parameters in summary."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_logreg_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_logreg_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert "max_iter" in summary["best_int_params"]
        assert "C" in summary["best_float_params"]
        assert "tol" in summary["best_float_params"]
        assert "penalty" in summary["best_string_params"]
        assert "solver" in summary["best_string_params"]
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_logreg_optimizer_best_value_in_summary() -> None:
    """LogReg optimizer returns best value in summary."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_logreg_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_logreg_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert 0.5 <= summary["best_value"] <= 1.0
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_logreg_optimizer_records_duration() -> None:
    """LogReg optimizer records total duration."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_logreg_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_logreg_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert summary["total_duration_seconds"] >= 0.0
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_logreg_optimizer_without_pruning() -> None:
    """LogReg optimizer works without pruning."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_logreg_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_logreg_default_space(),
            config=make_optuna_config(n_trials=3, pruning_enabled=False),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_logreg_optimizer_with_trial_callback() -> None:
    """LogReg optimizer calls trial_callback for each trial."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_logreg_optimizer()
        x, y, names = make_optuna_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_logreg_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for result in callbacks:
            assert result["state"] == "complete"
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_logreg_optimizer_with_timeout() -> None:
    """LogReg optimizer accepts timeout_seconds."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_logreg_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_logreg_default_space(),
            config=make_optuna_config(n_trials=3, timeout_seconds=60),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_logreg_optimizer_with_focused_space() -> None:
    """LogReg optimizer works with focused space (no optional params)."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_logreg_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_logreg_focused_space(best_c=1.0, best_tol=1e-4),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
        assert len(summary["best_string_params"]) == 0
        assert "l1_ratio" not in summary["best_float_params"]
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories
