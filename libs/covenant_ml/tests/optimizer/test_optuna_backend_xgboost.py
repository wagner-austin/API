"""Tests for XGBoost Optuna backend optimizer and DART parameter handling.

Uses fake Optuna implementation for testing without mocks.
"""

from __future__ import annotations

from covenant_ml.optimizer.optuna_backend import _hooks as _backend_hooks
from covenant_ml.optimizer.optuna_backend import (
    create_xgboost_optimizer,
)
from covenant_ml.optimizer.optuna_backend.xgboost import (
    _extract_xgboost_dart_best_params,
    _sample_xgboost_dart_params,
)
from covenant_ml.optimizer.search_spaces import (
    make_xgboost_categorical_space,
    make_xgboost_default_space,
)
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledStringParams,
    TrialResult,
    XGBoostSearchSpace,
)
from tests.optimizer._fake_optuna import (
    FakeTrial,
    get_fake_optuna_factories,
    make_optuna_config,
    make_optuna_test_data,
)
from tests.optimizer._objective_fixtures import FakeObjective

# =============================================================================
# Helpers
# =============================================================================


def _make_xgboost_space_with_dart() -> XGBoostSearchSpace:
    """Create XGBoost search space with DART params."""
    return make_xgboost_default_space()


def _make_xgboost_space_without_dart() -> XGBoostSearchSpace:
    """Create XGBoost search space WITHOUT DART params."""
    space: XGBoostSearchSpace = {
        "max_depth": {"param_type": "int", "low": 3, "high": 10, "log_scale": False},
        "n_estimators": {"param_type": "int", "low": 50, "high": 300, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
    }
    return space


# =============================================================================
# Tests: DART Parameter Sampling
# =============================================================================


def test_sample_xgboost_dart_params_no_booster_in_space() -> None:
    """_sample_xgboost_dart_params does nothing when booster not in space."""
    trial = FakeTrial(0)
    space = _make_xgboost_space_without_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    assert "booster" not in string_params
    assert "rate_drop" not in float_params


def test_sample_xgboost_dart_params_with_dart() -> None:
    """_sample_xgboost_dart_params adds DART params when booster is dart."""
    trial = FakeTrial(1)  # Trial 1 selects dart (index 1)
    space = _make_xgboost_space_with_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    assert string_params["booster"] == "dart"
    assert "rate_drop" in float_params
    assert "skip_drop" in float_params


def test_sample_xgboost_dart_params_with_dart_partial() -> None:
    """_sample_xgboost_dart_params handles partial DART params in search space."""
    trial = FakeTrial(1)
    space = _make_xgboost_space_without_dart()
    space["booster"] = {"param_type": "categorical_str", "choices": ("gbtree", "dart")}
    space["rate_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    assert string_params["booster"] == "dart"
    assert "rate_drop" in float_params
    assert "skip_drop" not in float_params


def test_sample_xgboost_dart_params_with_dart_skip_drop_only() -> None:
    """_sample_xgboost_dart_params handles skip_drop only (no rate_drop) in search space."""
    trial = FakeTrial(1)
    space = _make_xgboost_space_without_dart()
    space["booster"] = {"param_type": "categorical_str", "choices": ("gbtree", "dart")}
    space["skip_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    assert string_params["booster"] == "dart"
    assert "rate_drop" not in float_params
    assert "skip_drop" in float_params


def test_sample_xgboost_dart_params_with_gbtree() -> None:
    """_sample_xgboost_dart_params does not add DART params when booster is gbtree."""
    trial = FakeTrial(0)  # Trial 0 selects gbtree (index 0)
    space = _make_xgboost_space_with_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_xgboost_dart_params(trial, space, float_params, string_params)
    assert string_params["booster"] == "gbtree"
    assert "rate_drop" not in float_params
    assert "skip_drop" not in float_params


# =============================================================================
# Tests: DART Best Params Extraction
# =============================================================================


def test_extract_xgboost_dart_best_params_no_booster_in_space() -> None:
    """_extract_xgboost_dart_best_params does nothing when booster not in space."""
    space = _make_xgboost_space_without_dart()
    best_params: dict[str, float | int | str] = {"max_depth": 5}
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert "booster" not in best_string_params


def test_extract_xgboost_dart_best_params_with_dart() -> None:
    """_extract_xgboost_dart_best_params extracts DART params when booster is dart."""
    space = _make_xgboost_space_with_dart()
    best_params: dict[str, float | int | str] = {
        "max_depth": 5,
        "booster": "dart",
        "rate_drop": 0.15,
        "skip_drop": 0.5,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["booster"] == "dart"
    assert best_float_params["rate_drop"] == 0.15
    assert best_float_params["skip_drop"] == 0.5


def test_extract_xgboost_dart_best_params_with_dart_partial() -> None:
    """_extract_xgboost_dart_best_params handles partial DART params in search space."""
    space = _make_xgboost_space_without_dart()
    space["booster"] = {"param_type": "categorical_str", "choices": ("gbtree", "dart")}
    space["rate_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    best_params: dict[str, float | int | str] = {
        "max_depth": 5,
        "booster": "dart",
        "rate_drop": 0.15,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["booster"] == "dart"
    assert best_float_params["rate_drop"] == 0.15
    assert "skip_drop" not in best_float_params


def test_extract_xgboost_dart_best_params_with_dart_skip_drop_only() -> None:
    """_extract_xgboost_dart_best_params handles skip_drop only (no rate_drop) in search space."""
    space = _make_xgboost_space_without_dart()
    space["booster"] = {"param_type": "categorical_str", "choices": ("gbtree", "dart")}
    space["skip_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    best_params: dict[str, float | int | str] = {
        "max_depth": 5,
        "booster": "dart",
        "skip_drop": 0.5,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["booster"] == "dart"
    assert "rate_drop" not in best_float_params
    assert best_float_params["skip_drop"] == 0.5


def test_extract_xgboost_dart_best_params_with_gbtree() -> None:
    """_extract_xgboost_dart_best_params skips DART params when booster is gbtree."""
    space = _make_xgboost_space_with_dart()
    best_params: dict[str, float | int | str] = {
        "max_depth": 5,
        "booster": "gbtree",
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_xgboost_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["booster"] == "gbtree"
    assert "rate_drop" not in best_float_params
    assert "skip_drop" not in best_float_params


# =============================================================================
# Tests: XGBoost Optimizer
# =============================================================================


def test_xgboost_optimizer_runs_trials() -> None:
    """XGBoost optimizer runs all trials and returns summary."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = make_optuna_test_data()
        objective = FakeObjective()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=make_optuna_config(n_trials=5),
            objective=objective,
        )
        assert summary["n_trials_total"] == 5
        assert summary["n_trials_complete"] == 5
        assert objective.call_count == 5
        assert 0.0 <= summary["best_value"] <= 1.0
        assert "learning_rate" in summary["best_float_params"]
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_xgboost_optimizer_with_callback() -> None:
    """XGBoost optimizer calls trial callback after each trial."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_xgboost_optimizer()
        x, y, names = make_optuna_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for i, result in enumerate(callbacks):
            assert result["trial_number"] == i
            assert result["state"] == "complete"
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_xgboost_optimizer_with_categorical_space() -> None:
    """XGBoost optimizer works with categorical search space."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_categorical_space(),
            config=make_optuna_config(n_trials=5),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 5
        assert summary["best_int_params"]["max_depth"] in (3, 4, 5, 6, 7, 8)
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_xgboost_optimizer_with_timeout() -> None:
    """XGBoost optimizer accepts timeout parameter."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=make_optuna_config(n_trials=3, timeout_seconds=60),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_xgboost_optimizer_with_pruning_disabled() -> None:
    """XGBoost optimizer works with pruning disabled."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_xgboost_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_xgboost_default_space(),
            config=make_optuna_config(n_trials=3, pruning_enabled=False),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories
