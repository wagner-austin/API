"""Tests for LSTM regressor objective function.

Tests the LSTM hyperparameter optimization objective using real US bankruptcy
features with synthetic continuous targets (regression).
"""

from __future__ import annotations

import math

import numpy as np
from covenant_ml.features import FeaturePreset
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams
from numpy.typing import NDArray
from platform_ml import RequestedDevice, RequestedPrecision

from covenant_nn.objectives import LSTMRegressorObjective, create_lstm_regressor_objective

from ..conftest import load_us_bankruptcy_sample


def _make_regression_targets(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Create continuous regression targets from features.

    Uses a deterministic linear combination of the first two features.

    Args:
        x: Feature matrix.

    Returns:
        Continuous target array.
    """
    n_samples = int(x.shape[0])
    y: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        y[i] = float(x.item((i, 0))) * 3.0 + float(x.item((i, 1))) * 1.5 + 2.0
    return y


def test_lstm_regressor_objective_returns_negative_rmse() -> None:
    """LSTMRegressorObjective trains LSTM and returns negative validation RMSE."""
    dataset = load_us_bankruptcy_sample()
    x = dataset["x"]
    names = dataset["feature_names"]
    y = _make_regression_targets(x)

    objective = create_lstm_regressor_objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        device=RequestedDevice.CPU,
        precision=RequestedPrecision.FP32,
        feature_preset=FeaturePreset.NONE,
        n_epochs=3,
        early_stopping_patience=2,
        sequence_length=4,
    )

    # Verify n_features property
    assert objective.n_features == dataset["n_features"]

    int_params: SampledIntParams = {
        "num_layers": 1,
        "hidden_size": 16,
        "batch_size": 256,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.001,
        "dropout": 0.1,
    }
    string_params: SampledStringParams = {}

    result = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert result < 0.0
    assert math.isfinite(result)


def test_lstm_regressor_objective_with_feature_engineering() -> None:
    """LSTMRegressorObjective, built directly and bidirectional, applies a non-'none' preset."""
    dataset = load_us_bankruptcy_sample()
    x = dataset["x"]
    names = dataset["feature_names"]
    y = _make_regression_targets(x)

    objective = LSTMRegressorObjective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        device=RequestedDevice.CPU,
        precision=RequestedPrecision.FP32,
        feature_preset=FeaturePreset.LOG_ONLY,
        n_epochs=3,
        early_stopping_patience=2,
        sequence_length=4,
        bidirectional=True,
    )

    # Feature count should be increased by log transforms
    assert objective.n_features > dataset["n_features"]

    int_params: SampledIntParams = {
        "num_layers": 1,
        "hidden_size": 8,
        "batch_size": 256,
    }
    float_params: SampledFloatParams = {
        "learning_rate": 0.001,
        "dropout": 0.0,
    }
    string_params: SampledStringParams = {}

    result = objective(
        x_features=x,
        y_targets=y,
        feature_names=names,
        int_params=int_params,
        float_params=float_params,
        string_params=string_params,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert result < 0.0
