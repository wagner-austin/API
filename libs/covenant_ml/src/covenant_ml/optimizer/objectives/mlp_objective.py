"""MLP objective function for hyperparameter optimization.

Provides the objective function that Optuna uses to evaluate MLP
hyperparameter configurations. Pre-splits data once and trains MLP
models for efficient trial evaluation.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.backends.mlp import create_mlp_backend
from covenant_ml.backends.protocol import ProgressCallback
from covenant_ml.features import (
    FeaturePreset,
    engineer_features,
    get_feature_config_for_preset,
)
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams
from covenant_ml.types import MLPConfig, MLPOptimizer, MLPPrecision

_log = get_logger(__name__)


class MLPObjective:
    """MLP objective that trains models and returns validation AUC.

    Pre-applies feature engineering and stores configuration for
    consistent trial evaluation.
    """

    def __init__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        precision: MLPPrecision,
        feature_preset: FeaturePreset,
        n_epochs: int,
        early_stopping_patience: int,
        optimizer_name: MLPOptimizer = "adamw",
        epoch_callback: ProgressCallback | None = None,
    ) -> None:
        """Initialize with data and fixed training configuration.

        Args:
            x_features: Feature matrix.
            y_labels: Binary labels.
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
                "Applied feature engineering for MLP",
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
        self._y_labels = y_labels
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
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        int_params: SampledIntParams,
        float_params: SampledFloatParams,
        string_params: SampledStringParams,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        """Train MLP with given hyperparameters and return validation AUC.

        Args:
            x_features: Ignored (uses pre-stored engineered features).
            y_labels: Ignored (uses pre-stored labels).
            feature_names: Ignored (uses pre-stored names).
            int_params: Sampled integer hyperparameters.
            float_params: Sampled float hyperparameters.
            string_params: String hyperparameters (unused for MLP).
            train_ratio: Train split ratio.
            val_ratio: Validation split ratio.
            test_ratio: Test split ratio.
            random_state: Random seed for reproducibility.

        Returns:
            Validation AUC score.
        """
        # Ignore passed data - use pre-stored engineered features
        _ = x_features, y_labels, feature_names, string_params

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

        # Train using MLP backend with temporary output directory
        backend = create_mlp_backend()
        with TemporaryDirectory() as tmpdir:
            outcome = backend.train(
                x_features=self._x_features,
                y_labels=self._y_labels,
                feature_names=self._feature_names,
                config=config,
                output_dir=Path(tmpdir),
                progress=self._epoch_callback,
            )

        # Extract result before cleanup
        best_val_auc = outcome["best_val_auc"]

        # Force aggressive cleanup of PyTorch memory between trials
        # Delete outcome dict which may hold references to config/metrics
        del outcome
        del backend

        # Run garbage collection to free PyTorch model memory
        import gc

        gc.collect()

        return best_val_auc


def create_mlp_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    device: Literal["cpu", "cuda", "auto"],
    precision: MLPPrecision,
    feature_preset: FeaturePreset,
    n_epochs: int,
    early_stopping_patience: int,
    optimizer_name: MLPOptimizer = "adamw",
    epoch_callback: ProgressCallback | None = None,
) -> MLPObjective:
    """Create an objective function for MLP optimization.

    Applies feature engineering based on preset and stores configuration
    for consistent trial evaluation. The returned objective tracks the
    engineered feature count via its n_features property.

    Args:
        x_features: Feature matrix.
        y_labels: Binary labels.
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
    return MLPObjective(
        x_features,
        y_labels,
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
    "MLPObjective",
    "create_mlp_objective",
]
