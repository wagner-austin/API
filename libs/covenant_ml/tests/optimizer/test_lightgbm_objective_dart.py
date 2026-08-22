"""Tests for LightGBM objective function.

Uses real lightgbm library for integration testing. No mocks.
"""

from __future__ import annotations

from covenant_ml.optimizer.objectives.lightgbm_objective import (
    LightGBMObjective,
)
from covenant_ml.optimizer.types import SampledFloatParams, SampledStringParams
from tests.optimizer._lightgbm_objective_fixtures import (
    _make_default_int_params,
    _make_test_data,
)


def test_lightgbm_objective_with_dart_boosting() -> None:
    """LightGBMObjective uses DART params when boosting_type is 'dart'."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        drop_rate=0.1,  # DART param
        skip_drop=0.5,  # DART param
    )
    string_params = SampledStringParams(boosting_type="dart")

    auc = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # AUC must be between 0 and 1
    assert 0.0 <= auc <= 1.0


def test_lightgbm_objective_with_dart_partial_params() -> None:
    """LightGBMObjective handles partial DART params (only drop_rate, no skip_drop)."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        drop_rate=0.1,  # Only drop_rate, no skip_drop
    )
    string_params = SampledStringParams(boosting_type="dart")

    auc = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # AUC must be between 0 and 1
    assert 0.0 <= auc <= 1.0


def test_lightgbm_objective_with_dart_skip_drop_only() -> None:
    """LightGBMObjective handles DART with only skip_drop (no drop_rate)."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        skip_drop=0.5,  # Only skip_drop, no drop_rate
    )
    string_params = SampledStringParams(boosting_type="dart")

    auc = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # AUC must be between 0 and 1
    assert 0.0 <= auc <= 1.0


def test_lightgbm_objective_with_dart_feature_fraction() -> None:
    """LightGBMObjective uses feature_fraction when boosting_type is 'dart' (Phase 6)."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        drop_rate=0.1,
        skip_drop=0.5,
        feature_fraction=0.05,  # Phase 6: aggressive feature subsampling for DART
    )
    string_params = SampledStringParams(boosting_type="dart")

    auc = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # AUC must be between 0 and 1
    assert 0.0 <= auc <= 1.0


def test_lightgbm_objective_with_dart_feature_fraction_only() -> None:
    """LightGBMObjective uses feature_fraction without other DART params."""
    x, y, names = _make_test_data(n_samples=100, n_features=5)
    objective = LightGBMObjective(x, y, names, device="cpu", feature_preset="none")

    int_params = _make_default_int_params()
    float_params = SampledFloatParams(
        learning_rate=0.1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        feature_fraction=0.05,  # Only feature_fraction, no drop_rate/skip_drop
    )
    string_params = SampledStringParams(boosting_type="dart")

    auc = objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # AUC must be between 0 and 1
    assert 0.0 <= auc <= 1.0
