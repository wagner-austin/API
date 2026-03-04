"""Tests for MLP objective function.

Tests the MLP hyperparameter optimization objective using real US bankruptcy data.
"""

from __future__ import annotations

from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams

from covenant_nn.objectives import MLPObjective, create_mlp_objective

from ..conftest import load_us_bankruptcy_data


def test_mlp_objective_returns_validation_auc() -> None:
    """MLPObjective trains MLP and returns validation AUC."""
    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    objective = create_mlp_objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device="cpu",
        precision="fp32",
        feature_preset="none",
        n_epochs=3,  # Small for fast test
        early_stopping_patience=2,
    )

    # Verify n_features property
    assert objective.n_features == dataset["n_features"]

    # Sample hyperparameters
    int_params: SampledIntParams = {
        "n_layers": 2,
        "hidden_size": 32,
        "batch_size": 256,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.001,
        "dropout": 0.1,
    }

    # MLP has no string params
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


def test_mlp_objective_with_feature_engineering() -> None:
    """MLPObjective applies feature engineering when preset is not 'none'."""
    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    # Create objective with feature engineering
    objective = create_mlp_objective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device="cpu",
        precision="fp32",
        feature_preset="log_only",  # Apply log transforms
        n_epochs=3,
        early_stopping_patience=2,
    )

    # Feature count should be increased by log transforms
    assert objective.n_features > dataset["n_features"]

    # Sample hyperparameters
    int_params: SampledIntParams = {
        "n_layers": 1,
        "hidden_size": 16,
        "batch_size": 256,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.001,
        "dropout": 0.0,
    }

    # MLP has no string params
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


def test_mlp_objective_class_direct_instantiation() -> None:
    """MLPObjective can be instantiated directly."""
    dataset = load_us_bankruptcy_data()
    x = dataset["x"]
    y = dataset["y"]
    names = dataset["feature_names"]

    # Direct instantiation
    objective = MLPObjective(
        x_features=x,
        y_labels=y,
        feature_names=names,
        device="cpu",
        precision="fp32",
        feature_preset="none",
        n_epochs=2,
        early_stopping_patience=1,
        optimizer_name="adam",  # Use adam instead of default adamw
    )

    assert objective.n_features == dataset["n_features"]

    # Run with minimal config
    int_params: SampledIntParams = {
        "n_layers": 1,
        "hidden_size": 8,
        "batch_size": 512,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.01,
        "dropout": 0.0,
    }

    # MLP has no string params
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
