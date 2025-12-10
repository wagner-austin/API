"""Tests for covenant_ml trainer module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from covenant_ml import TrainConfig, save_model, train_model


def _make_training_data(
    n_samples: int = 20,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Create simple training data for binary classification."""
    x_features: NDArray[np.float64] = np.zeros((n_samples, 8), dtype=np.float64)
    y_labels: NDArray[np.int64] = np.zeros(n_samples, dtype=np.int64)

    # Deterministic data generation based on index
    for i in range(n_samples):
        first_feat = ((i + seed) % 100) / 100.0
        for j in range(8):
            x_features[i, j] = (first_feat + j * 0.1) % 1.0
        # Label based on first feature (computed before assignment)
        y_labels[i] = 1 if first_feat > 0.5 else 0

    return x_features, y_labels


def test_train_model_returns_fitted_model() -> None:
    """train_model returns a model that can predict."""
    x_features, y_labels = _make_training_data()
    config: TrainConfig = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    model = train_model(x_features, y_labels, config)
    proba = model.predict_proba(x_features)
    assert proba.shape == (20, 2)


def test_train_model_produces_valid_probabilities() -> None:
    """Predicted probabilities are in valid range."""
    x_features, y_labels = _make_training_data()
    config: TrainConfig = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    model = train_model(x_features, y_labels, config)
    proba = model.predict_proba(x_features)

    for i in range(proba.shape[0]):
        for j in range(proba.shape[1]):
            p = float(proba[i, j])
            assert 0.0 <= p <= 1.0


def test_train_model_deterministic_with_same_seed() -> None:
    """Training with same random_state produces identical models."""
    x_features, y_labels = _make_training_data()
    config: TrainConfig = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 123,
    }

    model1 = train_model(x_features, y_labels, config)
    model2 = train_model(x_features, y_labels, config)

    proba1 = model1.predict_proba(x_features)
    proba2 = model2.predict_proba(x_features)

    for i in range(proba1.shape[0]):
        assert float(proba1[i, 1]) == float(proba2[i, 1])


def test_save_model_creates_file() -> None:
    """save_model creates a file at the specified path."""
    x_features, y_labels = _make_training_data()
    config: TrainConfig = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    model = train_model(x_features, y_labels, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.json")
        save_model(model, model_path)
        assert Path(model_path).exists()
        assert Path(model_path).stat().st_size > 0
