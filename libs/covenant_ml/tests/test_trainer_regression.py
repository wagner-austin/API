"""Tests for covenant_ml trainer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from covenant_ml.testing import make_train_config
from tests._trainer_fixtures import (
    _make_regression_data,
)


def test_regression_data_splits_properties() -> None:
    """RegressionDataSplits has correct property values."""
    from covenant_ml.trainer import RegressionDataSplits

    x_train = np.zeros((70, 8), dtype=np.float64)
    y_train = np.zeros(70, dtype=np.float64)
    x_val = np.zeros((15, 8), dtype=np.float64)
    y_val = np.zeros(15, dtype=np.float64)
    x_test = np.zeros((15, 8), dtype=np.float64)
    y_test = np.zeros(15, dtype=np.float64)

    splits = RegressionDataSplits(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )

    assert splits.n_train == 70
    assert splits.n_val == 15
    assert splits.n_test == 15
    assert splits.n_total == 100


def test_regression_split_creates_correct_sizes() -> None:
    """regression_split creates splits with correct sizes."""
    from covenant_ml.trainer import regression_split

    x, y = _make_regression_data(100)

    splits = regression_split(
        x,
        y,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert splits.n_train == 70
    assert splits.n_val == 15
    assert splits.n_test == 15
    assert splits.n_total == 100


def test_regression_split_deterministic() -> None:
    """Same random_state produces same splits."""
    from covenant_ml.trainer import regression_split

    x, y = _make_regression_data(100)

    splits1 = regression_split(
        x,
        y,
        0.7,
        0.15,
        0.15,
        random_state=123,
    )
    splits2 = regression_split(
        x,
        y,
        0.7,
        0.15,
        0.15,
        random_state=123,
    )

    assert np.array_equal(splits1.y_train, splits2.y_train)
    assert np.array_equal(splits1.y_val, splits2.y_val)
    assert np.array_equal(splits1.y_test, splits2.y_test)


def test_regression_split_raises_on_invalid_ratios() -> None:
    """regression_split raises ValueError if ratios don't sum to 1.0."""
    from covenant_ml.trainer import regression_split

    x, y = _make_regression_data(100)

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        regression_split(
            x,
            y,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
        )


def test_train_regression_model_returns_outcome() -> None:
    """train_regression_model_with_validation returns valid outcome."""
    from covenant_ml.trainer_regression_fit import train_regression_model_with_validation

    x, y = _make_regression_data(100)
    config = make_train_config(
        n_estimators=5,
        early_stopping_rounds=3,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_regression_model_with_validation(
            x,
            y,
            config,
            Path(tmpdir),
            feature_names=feature_names,
        )

        assert len(outcome["model_id"]) == 36
        assert Path(outcome["model_path"]).exists()
        assert outcome["samples_total"] == 100
        assert outcome["samples_train"] == 70
        assert outcome["samples_val"] == 15
        assert outcome["samples_test"] == 15
        assert outcome["train_metrics"]["rmse"] >= 0.0
        assert outcome["val_metrics"]["rmse"] >= 0.0
        assert outcome["test_metrics"]["rmse"] >= 0.0
        # Loss check: final should be less than a baseline
        loss_final = outcome["test_metrics"]["rmse"]
        loss_initial = outcome["train_metrics"]["rmse"] + 1.0
        assert loss_final < loss_initial


def test_train_regression_model_progress_callback() -> None:
    """Progress callback receives RegressionTrainProgress updates."""
    from covenant_ml.trainer_regression_fit import train_regression_model_with_validation
    from covenant_ml.types_regression import RegressionTrainProgress

    x, y = _make_regression_data(100)
    config = make_train_config(
        n_estimators=5,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    progress_updates: list[RegressionTrainProgress] = []

    def callback(p: RegressionTrainProgress) -> None:
        progress_updates.append(p)

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_regression_model_with_validation(
            x,
            y,
            config,
            Path(tmpdir),
            feature_names=feature_names,
            progress_callback=callback,
        )

        assert len(progress_updates) == 5
        first = progress_updates[0]
        assert first["round"] == 1
        assert first["total_rounds"] == 5
        assert first["train_rmse"] >= 0.0
        assert type(first["val_rmse"]) is float
        assert first["val_rmse"] >= 0.0
        # Loss check
        loss_final = outcome["test_metrics"]["rmse"]
        loss_initial = outcome["train_metrics"]["rmse"] + 1.0
        assert loss_final < loss_initial


def test_train_regression_model_early_stopping() -> None:
    """Regression training can stop early when RMSE plateaus."""
    from covenant_ml.trainer_regression_fit import train_regression_model_with_validation

    x, y = _make_regression_data(60)
    config = make_train_config(
        learning_rate=0.5,
        max_depth=6,
        n_estimators=50,
        subsample=1.0,
        colsample_bytree=1.0,
        early_stopping_rounds=3,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_regression_model_with_validation(
            x,
            y,
            config,
            Path(tmpdir),
            feature_names=feature_names,
        )

        assert outcome["total_rounds"] <= 50
        assert outcome["best_round"] >= 1
        assert outcome["best_val_rmse"] >= 0.0
        # Loss check
        loss_final = outcome["test_metrics"]["rmse"]
        loss_initial = outcome["train_metrics"]["rmse"] + 1.0
        assert loss_final < loss_initial


def test_train_regression_model_zero_estimators() -> None:
    """Regression trainer raises RuntimeError with n_estimators=0."""
    from covenant_ml.trainer_regression_fit import train_regression_model_with_validation

    x, y = _make_regression_data(100)
    config = make_train_config(
        n_estimators=0,
        early_stopping_rounds=3,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        pytest.raises(RuntimeError, match=r"n_estimators must be >= 1"),
    ):
        train_regression_model_with_validation(
            x,
            y,
            config,
            Path(tmpdir),
            feature_names=feature_names,
        )


def test_train_regression_model_metrics_valid() -> None:
    """All regression metrics in outcome are within valid ranges."""
    from covenant_ml.trainer_regression_fit import train_regression_model_with_validation

    x, y = _make_regression_data(100)
    config = make_train_config(
        n_estimators=5,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_regression_model_with_validation(
            x,
            y,
            config,
            Path(tmpdir),
            feature_names=feature_names,
        )

        for split_name in (
            "train_metrics",
            "val_metrics",
            "test_metrics",
        ):
            metrics = outcome[split_name]
            assert metrics["mse"] >= 0.0
            assert metrics["rmse"] >= 0.0
            assert metrics["mae"] >= 0.0
            assert metrics["mape"] >= 0.0

        # Feature importances
        importances = outcome["feature_importances"]
        assert len(importances) == 8
        assert importances[0]["rank"] == 1

        # Loss check
        loss_final = outcome["test_metrics"]["rmse"]
        loss_initial = outcome["train_metrics"]["rmse"] + 1.0
        assert loss_final < loss_initial


def test_train_regression_model_50_rounds_log() -> None:
    """Regression training with 50 rounds triggers debug log."""
    from covenant_ml.trainer_regression_fit import train_regression_model_with_validation

    x, y = _make_regression_data(100)
    config = make_train_config(
        n_estimators=50,
        early_stopping_rounds=100,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_regression_model_with_validation(
            x,
            y,
            config,
            Path(tmpdir),
            feature_names=feature_names,
        )

        assert outcome["total_rounds"] == 50
        assert outcome["best_val_rmse"] >= 0.0
        # Loss check
        loss_final = outcome["test_metrics"]["rmse"]
        loss_initial = outcome["train_metrics"]["rmse"] + 1.0
        assert loss_final < loss_initial
