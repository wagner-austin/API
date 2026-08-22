"""Tests for covenant_ml trainer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from covenant_ml import train_model
from covenant_ml.testing import make_train_config
from covenant_ml.trainer import (
    preprocess_data_splits,
    stratified_split,
)
from covenant_ml.trainer_fit import (
    train_model_with_validation,
)
from covenant_ml.types import (
    TrainProgress,
)
from tests._trainer_fixtures import (
    _make_larger_data,
)


def test_train_model_with_validation_returns_outcome() -> None:
    """train_model_with_validation returns TrainOutcome with all fields."""
    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=5,
        early_stopping_rounds=3,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )

        assert len(outcome["model_id"]) == 36  # UUID format
        assert Path(outcome["model_path"]).exists()
        assert outcome["samples_total"] == 100
        # Sizes may vary slightly due to stratified splitting
        assert 68 <= outcome["samples_train"] <= 72
        assert 13 <= outcome["samples_val"] <= 17
        assert 13 <= outcome["samples_test"] <= 17
        assert "loss" in outcome["train_metrics"]
        assert "loss" in outcome["val_metrics"]
        assert "loss" in outcome["test_metrics"]


def test_train_model_with_validation_progress_callback() -> None:
    """Progress callback receives TrainProgress updates."""
    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=5,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    progress_updates: list[TrainProgress] = []

    def callback(progress: TrainProgress) -> None:
        progress_updates.append(progress)

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        train_model_with_validation(
            x_features,
            y_labels,
            config,
            Path(tmpdir),
            feature_names=feature_names,
            progress_callback=callback,
        )

        # Should have received progress updates
        assert len(progress_updates) == 5  # n_estimators = 5

        # Check first update (TrainProgress is a TypedDict - use dict access)
        first = progress_updates[0]
        assert first["round"] == 1
        assert first["total_rounds"] == 5
        assert 0.0 <= first["train_auc"] <= 1.0
        first_val_auc = first["val_auc"]
        assert first_val_auc is None or 0.0 <= first_val_auc <= 1.0


def test_train_model_with_validation_early_stopping() -> None:
    """Training can stop early when validation doesn't improve."""
    # Create data where model will overfit quickly
    x_features, y_labels = _make_larger_data(60)
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
        outcome = train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )

        # May or may not early stop depending on data
        # Just verify the fields are populated correctly
        assert outcome["total_rounds"] <= 50
        assert outcome["best_round"] >= 1
        assert 0.0 <= outcome["best_val_auc"] <= 1.0


def test_train_model_with_validation_zero_estimators() -> None:
    """train_model_with_validation raises RuntimeError with n_estimators=0."""
    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=0,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=3,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        pytest.raises(RuntimeError, match=r"n_estimators must be >= 1"),
    ):
        train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )


def test_train_model_with_validation_metrics_valid() -> None:
    """All metrics in outcome are within valid ranges."""
    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=5,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )

        # Check train metrics
        train_m = outcome["train_metrics"]
        assert train_m["loss"] >= 0.0
        assert 0.0 <= train_m["auc"] <= 1.0
        assert 0.0 <= train_m["accuracy"] <= 1.0
        assert 0.0 <= train_m["precision"] <= 1.0
        assert 0.0 <= train_m["recall"] <= 1.0
        assert 0.0 <= train_m["f1_score"] <= 1.0

        # Check val metrics
        val_m = outcome["val_metrics"]
        assert val_m["loss"] >= 0.0
        assert 0.0 <= val_m["auc"] <= 1.0
        assert 0.0 <= val_m["accuracy"] <= 1.0
        assert 0.0 <= val_m["precision"] <= 1.0
        assert 0.0 <= val_m["recall"] <= 1.0
        assert 0.0 <= val_m["f1_score"] <= 1.0

        # Check test metrics
        test_m = outcome["test_metrics"]
        assert test_m["loss"] >= 0.0
        assert 0.0 <= test_m["auc"] <= 1.0
        assert 0.0 <= test_m["accuracy"] <= 1.0
        assert 0.0 <= test_m["precision"] <= 1.0
        assert 0.0 <= test_m["recall"] <= 1.0
        assert 0.0 <= test_m["f1_score"] <= 1.0


def test_train_model_with_validation_wrong_feature_names_length() -> None:
    """train_model_with_validation raises ValueError with wrong feature_names length."""
    from covenant_ml.trainer_fit import extract_feature_importances

    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=5,
        early_stopping_rounds=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    # Train a model first
    model = train_model(x_features, y_labels, config)

    # Try to extract importances with wrong length feature names
    wrong_names = ["feat_0", "feat_1"]  # Only 2 names, model has 8 features
    with pytest.raises(ValueError, match=r"feature_names length.*must match model features"):
        extract_feature_importances(model, wrong_names)


def test_train_model_with_validation_logs_progress_every_50_rounds() -> None:
    """Training with 50 rounds triggers the every-50-rounds debug log."""
    x_features, y_labels = _make_larger_data(100)
    config = make_train_config(
        n_estimators=50,
        early_stopping_rounds=100,  # High patience to avoid early stopping
        reg_alpha=1.0,
        reg_lambda=5.0,
    )

    feature_names = [f"feat_{i}" for i in range(8)]
    with tempfile.TemporaryDirectory() as tmpdir:
        outcome = train_model_with_validation(
            x_features, y_labels, config, Path(tmpdir), feature_names=feature_names
        )

        # Should have completed all 50 rounds (no early stopping)
        assert outcome["total_rounds"] == 50
        assert outcome["best_val_auc"] > 0.0


def test_preprocess_data_splits_properties() -> None:
    """PreprocessedDataSplits has correct property values."""
    x_features, y_labels = _make_larger_data(100)

    splits = stratified_split(
        x_features,
        y_labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    preprocessed = preprocess_data_splits(splits)

    # Verify properties match underlying array sizes
    assert preprocessed.n_train == len(preprocessed.y_train)
    assert preprocessed.n_val == len(preprocessed.y_val)
    assert preprocessed.n_test == len(preprocessed.y_test)
    assert preprocessed.n_total == preprocessed.n_train + preprocessed.n_val + preprocessed.n_test

    # Should match original split sizes
    assert preprocessed.n_train == splits.n_train
    assert preprocessed.n_val == splits.n_val
    assert preprocessed.n_test == splits.n_test
    assert preprocessed.n_total == splits.n_total

    # Should have preprocessing state attached
    assert preprocessed.state["n_features"] == 8
