"""Tests for LSTM objective function.

Tests the LSTM hyperparameter optimization objective using real US bankruptcy data.
"""

from __future__ import annotations

from covenant_ml.optimizer.objectives import LSTMObjective, create_lstm_objective
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams

from ...conftest import load_us_bankruptcy_data


def test_lstm_objective_returns_validation_auc() -> None:
    """LSTMObjective trains LSTM and returns validation AUC."""
    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    objective = create_lstm_objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device="cpu",
        precision="fp32",
        feature_preset="none",
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
    """LSTMObjective applies feature engineering when preset is not 'none'."""
    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    # Create objective with feature engineering
    objective = create_lstm_objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device="cpu",
        precision="fp32",
        feature_preset="log_only",  # Apply log transforms
        n_epochs=3,
        early_stopping_patience=2,
        sequence_length=4,
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


def test_lstm_objective_class_direct_instantiation() -> None:
    """LSTMObjective can be instantiated directly."""
    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    # Direct instantiation
    objective = LSTMObjective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device="cpu",
        precision="fp32",
        feature_preset="none",
        n_epochs=2,
        early_stopping_patience=1,
        sequence_length=3,
        bidirectional=False,
    )

    assert objective.n_features == dataset["n_features"]

    # Run with minimal config
    int_params: SampledIntParams = {
        "num_layers": 1,
        "hidden_size": 8,
        "batch_size": 512,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.01,
        "dropout": 0.0,
    }

    # LSTM has no string params
    string_params: SampledStringParams = {}

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
        random_state=123,
    )

    assert 0.0 <= val_auc <= 1.0


def test_lstm_objective_with_bidirectional() -> None:
    """LSTMObjective supports bidirectional LSTM."""
    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    # Create objective with bidirectional=True
    objective = create_lstm_objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device="cpu",
        precision="fp32",
        feature_preset="none",
        n_epochs=2,
        early_stopping_patience=1,
        sequence_length=3,
        bidirectional=True,  # Enable bidirectional
    )

    # Verify n_features property
    assert objective.n_features == dataset["n_features"]

    # Sample hyperparameters
    int_params: SampledIntParams = {
        "num_layers": 1,
        "hidden_size": 8,
        "batch_size": 512,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.01,
        "dropout": 0.0,
    }

    # LSTM has no string params
    string_params: SampledStringParams = {}

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


def test_lstm_objective_with_multiple_layers() -> None:
    """LSTMObjective supports multi-layer LSTM with dropout."""
    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    objective = create_lstm_objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device="cpu",
        precision="fp32",
        feature_preset="none",
        n_epochs=2,
        early_stopping_patience=1,
        sequence_length=3,
    )

    # Sample hyperparameters with multiple layers
    int_params: SampledIntParams = {
        "num_layers": 2,  # Multiple layers
        "hidden_size": 8,
        "batch_size": 256,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.001,
        "dropout": 0.2,  # Dropout between layers
    }

    # LSTM has no string params
    string_params: SampledStringParams = {}

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


def test_lstm_objective_different_sequence_lengths() -> None:
    """LSTMObjective works with various sequence lengths."""
    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    # Test with different sequence lengths
    for seq_len in [2, 4, 6]:
        objective = create_lstm_objective(
            x_features=x,
            y_labels=y,
            feature_names=names,
            device="cpu",
            precision="fp32",
            feature_preset="none",
            n_epochs=2,
            early_stopping_patience=1,
            sequence_length=seq_len,
        )

        int_params: SampledIntParams = {
            "num_layers": 1,
            "hidden_size": 8,
            "batch_size": 512,
        }
        float_params: SampledFloatParams = {
            "learning_rate": 0.01,
            "dropout": 0.0,
        }
        # LSTM has no string params
        string_params: SampledStringParams = {}

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

        assert 0.0 <= val_auc <= 1.0, f"Failed for sequence_length={seq_len}"
