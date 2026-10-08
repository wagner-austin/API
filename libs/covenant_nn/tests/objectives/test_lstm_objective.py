"""Tests for LSTM objective function.

Tests the LSTM hyperparameter optimization objective using real US bankruptcy data.
"""

from __future__ import annotations

from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams
from platform_ml import RequestedDevice, RequestedPrecision

from covenant_nn.objectives import LSTMObjective, create_lstm_objective

from ..conftest import load_us_bankruptcy_sample


def test_lstm_objective_returns_validation_auc() -> None:
    """LSTMObjective trains LSTM and returns validation AUC."""
    dataset = load_us_bankruptcy_sample()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    objective = create_lstm_objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device=RequestedDevice.CPU,
        precision=RequestedPrecision.FP32,
        feature_preset=FeaturePreset.NONE,
        n_epochs=3,  # Small for fast test
        early_stopping_patience=2,
        sequence_length=4,
    )

    # Verify n_features property
    assert objective.n_features == dataset["n_features"]

    # Sample hyperparameters (LSTM uses num_layers, not n_layers)
    int_params: SampledIntParams = {
        "num_layers": 1,
        "hidden_size": 16,
        "batch_size": 256,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.001,
        "dropout": 0.1,
    }

    # LSTM has no string params
    string_params: SampledStringParams = {}

    # Run objective
    val_auc = objective(
        x_features=x,  # Ignored - uses pre-stored
        y_labels=y,  # Ignored - uses pre-stored
        feature_names=names,  # Ignored - uses pre-stored
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    # AUC should be in valid range and above random baseline
    assert 0.0 <= val_auc <= 1.0
    assert val_auc > 0.5, f"AUC {val_auc} should beat random baseline"


def test_lstm_objective_with_feature_engineering() -> None:
    """LSTMObjective, built directly and bidirectional, applies a non-'none' preset."""
    dataset = load_us_bankruptcy_sample()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    objective = LSTMObjective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device=RequestedDevice.CPU,
        precision=RequestedPrecision.FP32,
        feature_preset=FeaturePreset.LOG_ONLY,  # Apply log transforms
        n_epochs=3,
        early_stopping_patience=2,
        sequence_length=4,
        bidirectional=True,
    )

    # Feature count should be increased by log transforms
    assert objective.n_features > dataset["n_features"]

    # Sample hyperparameters
    int_params: SampledIntParams = {
        "num_layers": 1,
        "hidden_size": 8,
        "batch_size": 256,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.001,
        "dropout": 0.0,
    }

    # LSTM has no string params
    string_params: SampledStringParams = {}

    # Run objective
    val_auc = objective(
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

    # AUC should be valid
    assert 0.0 <= val_auc <= 1.0
