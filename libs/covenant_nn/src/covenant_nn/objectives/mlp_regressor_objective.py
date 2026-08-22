"""MLP regressor objective function for hyperparameter optimization.

Parallel to mlp_objective.py (classification). Key differences:
- y_targets: NDArray[np.float64] (continuous, not int64 binary)
- Uses create_mlp_regressor_backend() instead of create_mlp_backend()
- Returns negative RMSE (Optuna maximizes; lower RMSE = better)
- Reads best_val_rmse instead of best_val_auc from outcome

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from covenant_ml.backends.regressor_protocol import RegressorProgressCallback
from covenant_ml.features import (
    FeaturePreset,
    engineer_features,
    get_feature_config_for_preset,
)
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams
from covenant_ml.types import (
    MLPConfig,
    MLPOptimizer,
    MLPPrecision,
)
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_nn.backends.mlp import create_mlp_regressor_backend

_log = get_logger(__name__)


class MLPRegressorObjective:
    """MLP regressor objective that trains models and returns negative RMSE.

    Pre-applies feature engineering and stores configuration for
    consistent trial evaluation.
    """

    def __init__(
        self,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        precision: MLPPrecision,
        feature_preset: FeaturePreset,
        n_epochs: int,
        early_stopping_patience: int,
        optimizer_name: MLPOptimizer = "adamw",
        epoch_callback: RegressorProgressCallback | None = None,
    ) -> None:
        """Initialize with data and fixed training configuration.

        Args:
            x_features: Feature matrix.
            y_targets: Continuous target values.
            feature_names: Original feature names.
            device: Device to use for training.
            precision: Precision mode for training.
            feature_preset: Feature engineering preset to apply.
            n_epochs: Maximum training epochs per trial.
            early_stopping_patience: Early stopping patience.
            optimizer_name: Optimizer to use (adamw, adam, sgd).
            epoch_callback: Optional callback for epoch-level progress updates.
        """
        # Apply feature engineering BEFORE storing
        if feature_preset != "none":
            config = get_feature_config_for_preset(feature_preset)
            engineered = engineer_features(x_features, feature_names, config)
            x_engineered = engineered["x"]
            n_original = engineered["n_original"]
            n_ratios = engineered["n_ratios"]
            n_products = engineered["n_products"]
            n_log = engineered["n_log"]
            _log.info(
                "Applied feature engineering for MLP regressor",
                extra={
                    "preset": feature_preset,
                    "n_original": n_original,
                    "n_ratios": n_ratios,
                    "n_products": n_products,
                    "n_log": n_log,
                    "total_features": int(x_engineered.shape[1]),
                },
            )
        else:
            x_engineered = x_features

        # Store data and configuration
        self._x_features = x_engineered
        self._y_targets = y_targets
        self._feature_names = [f"f{i}" for i in range(x_engineered.shape[1])]
        self._n_features = int(x_engineered.shape[1])
        self._device = device
        self._precision = precision
        self._n_epochs = n_epochs
        self._early_stopping_patience = early_stopping_patience
        self._optimizer_name = optimizer_name
        self._epoch_callback = epoch_callback

    @property
    def n_features(self) -> int:
        """Return the actual feature count (after engineering)."""
        return self._n_features

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str],
        int_params: SampledIntParams,
        float_params: SampledFloatParams,
        string_params: SampledStringParams,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        """Train MLP regressor with given hyperparameters and return negative RMSE.

        Args:
            x_features: Ignored (uses pre-stored engineered features).
            y_targets: Ignored (uses pre-stored targets).
            feature_names: Ignored (uses pre-stored names).
            int_params: Sampled integer hyperparameters.
            float_params: Sampled float hyperparameters.
            string_params: String hyperparameters (unused for MLP).
            train_ratio: Train split ratio.
            val_ratio: Validation split ratio.
            test_ratio: Test split ratio.
            random_state: Random seed for reproducibility.

        Returns:
            Negative validation RMSE (higher = better for Optuna).
        """
        # Ignore passed data - use pre-stored engineered features
        _ = x_features, y_targets, feature_names, string_params

        # Extract hyperparameters from typed dicts
        n_layers = int_params["n_layers"]
        hidden_size = int_params["hidden_size"]
        batch_size = int_params["batch_size"]
        learning_rate = float_params["learning_rate"]
        dropout = float_params["dropout"]

        # Build hidden_sizes tuple from n_layers and hidden_size
        hidden_sizes = tuple(hidden_size for _ in range(n_layers))

        # Create MLPConfig
        config: MLPConfig = {
            "device": self._device,
            "precision": self._precision,
            "optimizer": self._optimizer_name,
            "hidden_sizes": hidden_sizes,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": self._n_epochs,
            "dropout": dropout,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_state": random_state,
            "early_stopping_patience": self._early_stopping_patience,
        }

        # Train using MLP regressor backend with temporary output directory
        backend = create_mlp_regressor_backend()
        with TemporaryDirectory() as tmpdir:
            outcome = backend.train(
                x_features=self._x_features,
                y_targets=self._y_targets,
                feature_names=self._feature_names,
                config=config,
                output_dir=Path(tmpdir),
                progress=self._epoch_callback,
            )

        # Extract result before cleanup
        best_val_rmse = outcome["best_val_rmse"]

        # Force aggressive cleanup of PyTorch memory between trials
        del outcome
        del backend
        gc.collect()

        return -best_val_rmse


def create_mlp_regressor_objective(
    x_features: NDArray[np.float64],
    y_targets: NDArray[np.float64],
    feature_names: list[str],
    device: Literal["cpu", "cuda", "auto"],
    precision: MLPPrecision,
    feature_preset: FeaturePreset,
    n_epochs: int,
    early_stopping_patience: int,
    optimizer_name: MLPOptimizer = "adamw",
    epoch_callback: RegressorProgressCallback | None = None,
) -> MLPRegressorObjective:
    """Create an objective function for MLP regressor optimization.

    Applies feature engineering based on preset and stores configuration
    for consistent trial evaluation. The returned objective tracks the
    engineered feature count via its n_features property.

    Args:
        x_features: Feature matrix.
        y_targets: Continuous target values.
        feature_names: Original feature names.
        device: Device to use for training.
        precision: Precision mode for training.
        feature_preset: Feature engineering preset to apply.
        n_epochs: Maximum training epochs per trial.
        early_stopping_patience: Early stopping patience.
        optimizer_name: Optimizer to use (adamw, adam, sgd).
        epoch_callback: Optional callback for epoch-level progress updates.

    Returns:
        Objective callable with n_features property for engineered feature count.
    """
    return MLPRegressorObjective(
        x_features,
        y_targets,
        feature_names,
        device,
        precision,
        feature_preset,
        n_epochs,
        early_stopping_patience,
        optimizer_name,
        epoch_callback,
    )


__all__ = [
    "MLPRegressorObjective",
    "create_mlp_regressor_objective",
]
