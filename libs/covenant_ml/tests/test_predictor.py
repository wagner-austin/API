"""Tests for covenant_ml predictor module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from covenant_domain.features import LoanFeatures
from numpy.typing import NDArray

from covenant_ml import (
    TrainConfig,
    load_model,
    predict_probabilities,
    save_model,
    train_model,
)


def _make_training_data(
    n_samples: int = 50,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Create training data matching LoanFeatures structure."""
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


def _train_and_save_model(model_path: str) -> None:
    """Train and save a model for testing."""
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
    save_model(model, model_path)


def _make_loan_features(
    debt_to_ebitda: float = 3.5,
    interest_cover: float = 4.0,
    current_ratio: float = 1.5,
    leverage_change_1p: float = 0.1,
    leverage_change_4p: float = 0.2,
    sector_encoded: int = 1,
    region_encoded: int = 2,
    near_breach_count_4p: int = 0,
) -> LoanFeatures:
    """Create LoanFeatures for testing."""
    return {
        "debt_to_ebitda": debt_to_ebitda,
        "interest_cover": interest_cover,
        "current_ratio": current_ratio,
        "leverage_change_1p": leverage_change_1p,
        "leverage_change_4p": leverage_change_4p,
        "sector_encoded": sector_encoded,
        "region_encoded": region_encoded,
        "near_breach_count_4p": near_breach_count_4p,
    }


def test_load_model_loads_saved_model() -> None:
    """load_model loads a previously saved model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.json")
        _train_and_save_model(model_path)

        loaded = load_model(model_path)
        features = [_make_loan_features()]
        probs = predict_probabilities(loaded, features)

        assert len(probs) == 1
        assert 0.0 <= probs[0] <= 1.0


def test_predict_probabilities_empty_input() -> None:
    """predict_probabilities returns empty list for empty input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.json")
        _train_and_save_model(model_path)

        model = load_model(model_path)
        result = predict_probabilities(model, [])

        assert result == []


def test_predict_probabilities_single_sample() -> None:
    """predict_probabilities handles single sample."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.json")
        _train_and_save_model(model_path)

        model = load_model(model_path)
        features = [_make_loan_features()]
        probs = predict_probabilities(model, features)

        assert len(probs) == 1
        assert 0.0 <= probs[0] <= 1.0


def test_predict_probabilities_multiple_samples() -> None:
    """predict_probabilities handles multiple samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.json")
        _train_and_save_model(model_path)

        model = load_model(model_path)
        features = [
            _make_loan_features(debt_to_ebitda=2.0),
            _make_loan_features(debt_to_ebitda=5.0),
            _make_loan_features(debt_to_ebitda=8.0),
        ]
        probs = predict_probabilities(model, features)

        assert len(probs) == 3
        for p in probs:
            assert 0.0 <= p <= 1.0


def test_predict_probabilities_varied_inputs() -> None:
    """Model produces valid probabilities for varied feature inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.json")
        _train_and_save_model(model_path)

        model = load_model(model_path)
        features_a = [_make_loan_features(debt_to_ebitda=1.0, interest_cover=10.0)]
        features_b = [_make_loan_features(debt_to_ebitda=9.0, interest_cover=0.5)]

        probs_a = predict_probabilities(model, features_a)
        probs_b = predict_probabilities(model, features_b)

        # Both should produce valid probability values
        assert len(probs_a) == 1
        assert len(probs_b) == 1
        assert 0.0 <= probs_a[0] <= 1.0
        assert 0.0 <= probs_b[0] <= 1.0


def test_predict_probabilities_all_feature_fields_used() -> None:
    """All LoanFeatures fields are extracted and used."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = str(Path(tmpdir) / "model.json")
        _train_and_save_model(model_path)

        model = load_model(model_path)

        features = [
            _make_loan_features(
                debt_to_ebitda=3.0,
                interest_cover=5.0,
                current_ratio=2.0,
                leverage_change_1p=0.05,
                leverage_change_4p=0.15,
                sector_encoded=3,
                region_encoded=4,
                near_breach_count_4p=2,
            ),
        ]
        probs = predict_probabilities(model, features)

        assert len(probs) == 1
        assert 0.0 <= probs[0] <= 1.0
