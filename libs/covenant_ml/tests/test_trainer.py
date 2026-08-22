"""Tests for covenant_ml trainer module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml import save_model, train_model
from covenant_ml.testing import make_train_config, reset_cuda_hook, set_cuda_hook
from covenant_ml.trainer_fit import (
    _resolve_device,
    _XGBModuleProto,
    train_model_with_validation,
)
from tests._trainer_fixtures import (
    _FakeXGBModule,
    _make_imbalanced_data,
    _make_training_data,
)


def test_make_train_config_includes_scale_weight() -> None:
    """Helper adds optional scale_pos_weight when provided."""
    config = make_train_config(scale_pos_weight=2.5)
    assert config["scale_pos_weight"] == 2.5


def test_train_model_returns_fitted_model() -> None:
    """train_model returns a model that can predict."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    model = train_model(x_features, y_labels, config)
    proba = model.predict_proba(x_features)
    assert proba.shape == (20, 2)


def test_train_model_sets_cpu_parallel_params() -> None:
    """XGBoost model uses multi-core histogram training."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    model = train_model(x_features, y_labels, config)
    params = model.get_xgb_params()

    expected_jobs = max(1, int(os.cpu_count() or 1))
    assert params["n_jobs"] == expected_jobs
    assert params["tree_method"] == "hist"
    assert params["device"] == "cpu"


def test_resolve_device_prefers_cuda_when_supported() -> None:
    """_resolve_device chooses cuda when xgboost reports support."""
    fake_module: _XGBModuleProto = _FakeXGBModule(True)
    resolved = _resolve_device("auto", fake_module)
    assert resolved == "cuda"


def test_resolve_device_rejects_cuda_when_unsupported() -> None:
    """_resolve_device raises if cuda requested without support."""
    fake_module: _XGBModuleProto = _FakeXGBModule(False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        _resolve_device("cuda", fake_module)


def test_cuda_hook_forces_cpu_when_disabled() -> None:
    """Hook path executes and can force CPU resolution."""
    fake_module: _XGBModuleProto = _FakeXGBModule(True)
    set_cuda_hook(lambda: False)
    try:
        resolved = _resolve_device("auto", fake_module)
        assert resolved == "cpu"
    finally:
        reset_cuda_hook()


def test_cuda_hook_allows_cuda_request_when_supported() -> None:
    """Hook path allows explicit cuda when supported."""
    fake_module: _XGBModuleProto = _FakeXGBModule(True)
    set_cuda_hook(lambda: True)
    try:
        resolved = _resolve_device("cuda", fake_module)
        assert resolved == "cuda"
    finally:
        reset_cuda_hook()


def test_train_model_produces_valid_probabilities() -> None:
    """Predicted probabilities are in valid range."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    model = train_model(x_features, y_labels, config)
    proba = model.predict_proba(x_features)

    proba_list: list[list[float]] = proba.tolist()
    for row in proba_list:
        for p in row:
            assert 0.0 <= p <= 1.0


def test_train_model_deterministic_with_same_seed() -> None:
    """Training with same random_state produces identical models."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(random_state=123, reg_alpha=1.0, reg_lambda=5.0)

    model1 = train_model(x_features, y_labels, config)
    model2 = train_model(x_features, y_labels, config)

    proba1 = model1.predict_proba(x_features)
    proba2 = model2.predict_proba(x_features)

    proba1_list: list[list[float]] = proba1.tolist()
    proba2_list: list[list[float]] = proba2.tolist()
    for p1_row, p2_row in zip(proba1_list, proba2_list, strict=True):
        assert p1_row[1] == p2_row[1]


def test_save_model_creates_file() -> None:
    """save_model creates a file at the specified path."""
    x_features, y_labels = _make_training_data()
    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    model = train_model(x_features, y_labels, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.json")
        save_model(model, model_path)
        assert Path(model_path).exists()
        assert Path(model_path).stat().st_size > 0


def test_train_model_auto_calculates_scale_pos_weight() -> None:
    """train_model auto-calculates scale_pos_weight when not provided."""
    # Create imbalanced data: 10% positive, 90% negative
    x_features, y_labels = _make_imbalanced_data(100, positive_ratio=0.1)
    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    # No scale_pos_weight in config
    assert "scale_pos_weight" not in config

    # Training should succeed (auto-calculation happens internally)
    model = train_model(x_features, y_labels, config)
    proba = model.predict_proba(x_features)
    assert proba.shape == (100, 2)


def test_train_model_uses_provided_scale_pos_weight() -> None:
    """train_model uses provided scale_pos_weight when given."""
    x_features, y_labels = _make_imbalanced_data(100, positive_ratio=0.1)
    config = make_train_config(scale_pos_weight=5.0, reg_alpha=1.0, reg_lambda=5.0)

    assert config["scale_pos_weight"] == 5.0

    model = train_model(x_features, y_labels, config)
    proba = model.predict_proba(x_features)
    assert proba.shape == (100, 2)


def test_train_model_with_validation_returns_computed_scale_pos_weight() -> None:
    """train_model_with_validation includes auto-calculated scale_pos_weight in outcome."""
    # Create imbalanced data: 10% positive, 90% negative
    x_features, y_labels = _make_imbalanced_data(100, positive_ratio=0.1)
    config = make_train_config(
        n_estimators=3,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    # No scale_pos_weight in config
    assert "scale_pos_weight" not in config

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )

        # Should have computed scale_pos_weight
        computed = outcome["scale_pos_weight_computed"]
        assert computed > 0.0

        # Expected ratio: ~90 negative / ~7 positive in training set (70% of 100)
        # Training set has 70 samples, with 10% positive ratio = 7 positive, 63 negative
        # Expected scale_pos_weight = 63 / 7 = 9.0
        assert 7.0 <= computed <= 11.0  # Allow some variance from stratified split


def test_train_model_with_validation_uses_provided_scale_pos_weight() -> None:
    """train_model_with_validation uses provided scale_pos_weight."""
    x_features, y_labels = _make_imbalanced_data(100, positive_ratio=0.1)
    config = make_train_config(
        n_estimators=3,
        scale_pos_weight=15.0,  # Explicit value
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )

        # Should use provided value exactly
        assert outcome["scale_pos_weight_computed"] == 15.0


def test_train_model_raises_on_no_positive_samples() -> None:
    """train_model raises ValueError when no positive samples exist."""
    x_features: NDArray[np.float64] = np.zeros((100, 8), dtype=np.float64)
    y_labels: NDArray[np.int64] = np.zeros(100, dtype=np.int64)  # All zeros

    config = make_train_config(reg_alpha=1.0, reg_lambda=5.0)

    with pytest.raises(ValueError, match=r"no positive samples"):
        train_model(x_features, y_labels, config)


def test_train_model_with_validation_raises_on_no_positive_samples() -> None:
    """train_model_with_validation raises ValueError when no positive samples exist."""
    x_features: NDArray[np.float64] = np.zeros((100, 8), dtype=np.float64)
    y_labels: NDArray[np.int64] = np.zeros(100, dtype=np.int64)  # All zeros

    config = make_train_config(
        n_estimators=3,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        pytest.raises(ValueError, match=r"no positive samples"),
    ):
        train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )
