"""Tests for LightGBM Optuna backend optimizer and DART parameter handling.

Uses fake Optuna implementation for testing without mocks.
"""

from __future__ import annotations

from covenant_ml.optimizer.optuna_backend import _hooks as _backend_hooks
from covenant_ml.optimizer.optuna_backend import (
    create_lightgbm_optimizer,
)
from covenant_ml.optimizer.optuna_backend.lightgbm import (
    _extract_lightgbm_dart_best_params,
    _sample_lightgbm_dart_params,
)
from covenant_ml.optimizer.search_spaces import make_lightgbm_default_space
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    LightGBMSearchSpace,
    SampledFloatParams,
    SampledStringParams,
    TrialResult,
)

from .conftest import (
    FakeObjective,
    FakeTrial,
    get_fake_optuna_factories,
    make_optuna_config,
    make_optuna_test_data,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_lightgbm_space_with_dart() -> LightGBMSearchSpace:
    """Create LightGBM search space with DART params."""
    return make_lightgbm_default_space()


def _make_lightgbm_space_without_dart() -> LightGBMSearchSpace:
    """Create LightGBM search space WITHOUT DART params."""
    space: LightGBMSearchSpace = {
        "n_estimators": {"param_type": "int", "low": 50, "high": 500, "log_scale": False},
        "num_leaves": {"param_type": "int", "low": 20, "high": 100, "log_scale": False},
        "learning_rate": {"param_type": "float", "low": 0.01, "high": 0.3, "log_scale": True},
        "subsample": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "colsample_bytree": {"param_type": "float", "low": 0.6, "high": 1.0, "log_scale": False},
        "reg_alpha": {"param_type": "float", "low": 0.0, "high": 10.0, "log_scale": False},
        "reg_lambda": {"param_type": "float", "low": 0.1, "high": 10.0, "log_scale": True},
    }
    return space


# =============================================================================
# Tests: DART Parameter Sampling
# =============================================================================


def test_sample_lightgbm_dart_params_no_boosting_type_in_space() -> None:
    """_sample_lightgbm_dart_params does nothing when boosting_type not in space."""
    trial = FakeTrial(0)
    space = _make_lightgbm_space_without_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    assert "boosting_type" not in string_params
    assert "drop_rate" not in float_params


def test_sample_lightgbm_dart_params_with_dart() -> None:
    """_sample_lightgbm_dart_params adds DART params when boosting_type is dart."""
    trial = FakeTrial(1)
    space = _make_lightgbm_space_with_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    assert string_params["boosting_type"] == "dart"
    assert "drop_rate" in float_params
    assert "skip_drop" in float_params
    assert "feature_fraction" in float_params


def test_sample_lightgbm_dart_params_with_dart_partial() -> None:
    """_sample_lightgbm_dart_params handles partial DART params in search space."""
    trial = FakeTrial(1)
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    space["drop_rate"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    assert string_params["boosting_type"] == "dart"
    assert "drop_rate" in float_params
    assert "skip_drop" not in float_params


def test_sample_lightgbm_dart_params_with_dart_skip_drop_only() -> None:
    """_sample_lightgbm_dart_params handles skip_drop only (no drop_rate) in search space."""
    trial = FakeTrial(1)
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    space["skip_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    assert string_params["boosting_type"] == "dart"
    assert "drop_rate" not in float_params
    assert "skip_drop" in float_params


def test_sample_lightgbm_dart_params_with_dart_feature_fraction_only() -> None:
    """_sample_lightgbm_dart_params handles feature_fraction only in search space."""
    trial = FakeTrial(1)
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    ff_spec: FloatRangeSpec = {"param_type": "float", "low": 0.02, "high": 0.1, "log_scale": False}
    space["feature_fraction"] = ff_spec
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    assert string_params["boosting_type"] == "dart"
    assert "drop_rate" not in float_params
    assert "skip_drop" not in float_params
    assert "feature_fraction" in float_params


def test_sample_lightgbm_dart_params_with_gbdt() -> None:
    """_sample_lightgbm_dart_params does not add DART params when boosting_type is gbdt."""
    trial = FakeTrial(0)
    space = _make_lightgbm_space_with_dart()
    float_params: SampledFloatParams = {}
    string_params: SampledStringParams = {}
    _sample_lightgbm_dart_params(trial, space, float_params, string_params)
    assert string_params["boosting_type"] == "gbdt"
    assert "drop_rate" not in float_params
    assert "skip_drop" not in float_params
    assert "feature_fraction" not in float_params


# =============================================================================
# Tests: DART Best Params Extraction
# =============================================================================


def test_extract_lightgbm_dart_best_params_no_boosting_type_in_space() -> None:
    """_extract_lightgbm_dart_best_params does nothing when boosting_type not in space."""
    space = _make_lightgbm_space_without_dart()
    best_params: dict[str, float | int | str] = {"num_leaves": 50}
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert "boosting_type" not in best_string_params


def test_extract_lightgbm_dart_best_params_with_dart() -> None:
    """_extract_lightgbm_dart_best_params extracts DART params when boosting_type is dart."""
    space = _make_lightgbm_space_with_dart()
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "dart",
        "drop_rate": 0.1,
        "skip_drop": 0.6,
        "feature_fraction": 0.05,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["boosting_type"] == "dart"
    assert best_float_params["drop_rate"] == 0.1
    assert best_float_params["skip_drop"] == 0.6
    assert best_float_params["feature_fraction"] == 0.05


def test_extract_lightgbm_dart_best_params_with_dart_partial() -> None:
    """_extract_lightgbm_dart_best_params handles partial DART params in search space."""
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    space["drop_rate"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "dart",
        "drop_rate": 0.1,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["boosting_type"] == "dart"
    assert best_float_params["drop_rate"] == 0.1
    assert "skip_drop" not in best_float_params


def test_extract_lightgbm_dart_best_params_with_dart_skip_drop_only() -> None:
    """_extract_lightgbm_dart_best_params handles skip_drop only (no drop_rate) in space."""
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    space["skip_drop"] = {"param_type": "float", "low": 0.0, "high": 1.0, "log_scale": False}
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "dart",
        "skip_drop": 0.6,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["boosting_type"] == "dart"
    assert "drop_rate" not in best_float_params
    assert best_float_params["skip_drop"] == 0.6


def test_extract_lightgbm_dart_best_params_with_dart_feature_fraction_only() -> None:
    """_extract_lightgbm_dart_best_params handles feature_fraction only in search space."""
    space = _make_lightgbm_space_without_dart()
    space["boosting_type"] = {"param_type": "categorical_str", "choices": ("gbdt", "dart")}
    ff_spec: FloatRangeSpec = {"param_type": "float", "low": 0.02, "high": 0.1, "log_scale": False}
    space["feature_fraction"] = ff_spec
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "dart",
        "feature_fraction": 0.05,
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["boosting_type"] == "dart"
    assert "drop_rate" not in best_float_params
    assert "skip_drop" not in best_float_params
    assert best_float_params["feature_fraction"] == 0.05


def test_extract_lightgbm_dart_best_params_with_gbdt() -> None:
    """_extract_lightgbm_dart_best_params skips DART params when boosting_type is gbdt."""
    space = _make_lightgbm_space_with_dart()
    best_params: dict[str, float | int | str] = {
        "num_leaves": 50,
        "boosting_type": "gbdt",
    }
    best_float_params: SampledFloatParams = {}
    best_string_params: SampledStringParams = {}
    _extract_lightgbm_dart_best_params(space, best_params, best_float_params, best_string_params)
    assert best_string_params["boosting_type"] == "gbdt"
    assert "drop_rate" not in best_float_params
    assert "skip_drop" not in best_float_params
    assert "feature_fraction" not in best_float_params


# =============================================================================
# Tests: LightGBM Optimizer
# =============================================================================


def test_lightgbm_optimizer_runs_trials() -> None:
    """LightGBM optimizer runs all trials and returns summary."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_lightgbm_optimizer()
        x, y, names = make_optuna_test_data()
        objective = FakeObjective()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lightgbm_default_space(),
            config=make_optuna_config(n_trials=5),
            objective=objective,
        )
        assert summary["n_trials_total"] == 5
        assert summary["n_trials_complete"] == 5
        assert objective.call_count == 5
        assert "learning_rate" in summary["best_float_params"]
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_lightgbm_optimizer_with_pruning_disabled() -> None:
    """LightGBM optimizer works with pruning disabled."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_lightgbm_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lightgbm_default_space(),
            config=make_optuna_config(n_trials=3, pruning_enabled=False),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_lightgbm_optimizer_with_trial_callback() -> None:
    """LightGBM optimizer calls trial_callback for each trial."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        callbacks: list[TrialResult] = []

        def on_trial(result: TrialResult) -> None:
            callbacks.append(result)

        optimizer = create_lightgbm_optimizer()
        x, y, names = make_optuna_test_data()
        optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lightgbm_default_space(),
            config=make_optuna_config(n_trials=3),
            objective=FakeObjective(),
            trial_callback=on_trial,
        )
        assert len(callbacks) == 3
        for result in callbacks:
            assert result["state"] == "complete"
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories


def test_lightgbm_optimizer_with_timeout() -> None:
    """LightGBM optimizer accepts timeout_seconds."""
    _backend_hooks.optuna_factories = get_fake_optuna_factories
    try:
        optimizer = create_lightgbm_optimizer()
        x, y, names = make_optuna_test_data()
        summary = optimizer.optimize(
            x_features=x,
            y_labels=y,
            feature_names=names,
            search_space=make_lightgbm_default_space(),
            config=make_optuna_config(n_trials=3, timeout_seconds=60),
            objective=FakeObjective(),
        )
        assert summary["n_trials_complete"] == 3
    finally:
        _backend_hooks.optuna_factories = _backend_hooks._real_optuna_factories
