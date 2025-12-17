"""LightGBM objective function for hyperparameter optimization.

Provides the objective function that Optuna uses to evaluate LightGBM
hyperparameter configurations. Pre-splits data once and pre-computes
class weights for efficient trial evaluation.

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.features import (
    FeaturePreset,
    engineer_features,
    get_feature_config_for_preset,
)
from covenant_ml.metrics import compute_auc
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams
from covenant_ml.trainer import stratified_split

_log = get_logger(__name__)


# =============================================================================
# LightGBM Protocol Types
# =============================================================================


class _LGBDatasetProtocol(Protocol):
    """Protocol for LightGBM Dataset."""

    def __init__(
        self,
        data: NDArray[np.float64],
        label: NDArray[np.int64] | None = ...,
        *,
        free_raw_data: bool = ...,
    ) -> None: ...


class _LGBBoosterProtocol(Protocol):
    """Protocol for LightGBM Booster."""

    def predict(
        self,
        data: NDArray[np.float64],
        *,
        num_iteration: int | None = ...,
    ) -> NDArray[np.float64]: ...


class _LGBTrainFunc(Protocol):
    """Protocol for lgb.train function."""

    def __call__(
        self,
        params: dict[str, str | int | float],
        train_set: _LGBDatasetProtocol,
        num_boost_round: int = ...,
        *,
        valid_sets: list[_LGBDatasetProtocol] | None = ...,
        valid_names: list[str] | None = ...,
        callbacks: list[_EarlyStoppingCallback] | None = ...,
    ) -> _LGBBoosterProtocol: ...


class _EarlyStoppingCallback(Protocol):
    """Protocol for early stopping callback."""

    stopping_rounds: int


class _EarlyStoppingFactory(Protocol):
    """Protocol for early_stopping callback factory."""

    def __call__(self, stopping_rounds: int, verbose: bool = ...) -> _EarlyStoppingCallback: ...


# =============================================================================
# LightGBM Helpers
# =============================================================================


def _get_lgb_dataset_and_train() -> tuple[
    type[_LGBDatasetProtocol], _LGBTrainFunc, _EarlyStoppingFactory
]:
    """Get Dataset class, train function, and early stopping via dynamic import."""
    lgb_module = __import__("lightgbm", fromlist=["Dataset", "train", "early_stopping"])
    dataset_cls: type[_LGBDatasetProtocol] = lgb_module.Dataset
    train_fn: _LGBTrainFunc = lgb_module.train
    early_stopping: _EarlyStoppingFactory = lgb_module.early_stopping
    return dataset_cls, train_fn, early_stopping


# =============================================================================
# LightGBM Objective
# =============================================================================


class LightGBMObjective:
    """LightGBM objective that trains on pre-split data and returns validation AUC.

    Uses LightGBM's native Dataset format with early stopping for efficient
    hyperparameter optimization.
    """

    def __init__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        feature_preset: FeaturePreset,
        early_stopping_rounds: int = 10,
    ) -> None:
        """Initialize with data and configuration.

        Args:
            x_features: Feature matrix
            y_labels: Binary labels
            feature_names: Original feature names
            device: Device to use for training (cpu/cuda/auto)
            feature_preset: Feature engineering preset to apply
            early_stopping_rounds: Stop if no improvement for this many rounds
        """
        # Apply feature engineering BEFORE splitting
        if feature_preset != "none":
            config = get_feature_config_for_preset(feature_preset)
            engineered = engineer_features(x_features, feature_names, config)
            x_engineered = engineered["x"]
            n_original = engineered["n_original"]
            n_ratios = engineered["n_ratios"]
            n_products = engineered["n_products"]
            n_log = engineered["n_log"]
            _log.info(
                "Applied feature engineering for LightGBM",
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

        # Store actual feature count (after engineering)
        self._n_features = int(x_engineered.shape[1])
        self._early_stopping_rounds = early_stopping_rounds

        # Pre-split data once (stratified)
        self._splits = stratified_split(
            x_engineered,
            y_labels,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        # Resolve device once (LightGBM uses "cpu" or "gpu")
        self._device = "cpu" if device == "auto" else device

        # Calculate scale_pos_weight from training data (once)
        n_pos = int(np.sum(self._splits.y_train))
        n_neg = len(self._splits.y_train) - n_pos
        self._scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0

        # Pre-create Dataset objects
        dataset_cls, train_fn, early_stopping = _get_lgb_dataset_and_train()
        self._train_dataset = dataset_cls(
            self._splits.x_train,
            label=self._splits.y_train,
            free_raw_data=False,
        )
        self._val_dataset = dataset_cls(
            self._splits.x_val,
            label=self._splits.y_val,
            free_raw_data=False,
        )
        self._x_val = self._splits.x_val
        self._y_val = self._splits.y_val
        self._lgb_train = train_fn
        self._early_stopping = early_stopping

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
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        """Train LightGBM and return validation AUC.

        Args:
            x_features: Ignored (uses pre-split data)
            y_labels: Ignored (uses pre-split data)
            feature_names: Ignored (uses pre-split data)
            int_params: Integer hyperparameters from sampler
            float_params: Float hyperparameters from sampler
            train_ratio: Ignored (uses pre-split data)
            val_ratio: Ignored (uses pre-split data)
            test_ratio: Ignored (uses pre-split data)
            random_state: Random seed for reproducibility

        Returns:
            Validation AUC score
        """
        # Ignore passed data - use pre-computed datasets
        _ = x_features, y_labels, feature_names
        _ = train_ratio, val_ratio, test_ratio

        # Extract hyperparameters from typed dicts
        max_depth = int_params["max_depth"]
        n_estimators = int_params["n_estimators"]
        num_leaves = int_params.get("num_leaves", 31)
        min_child_samples = int_params.get("min_child_samples", 20)
        learning_rate = float_params["learning_rate"]
        reg_alpha = float_params["reg_alpha"]
        reg_lambda = float_params["reg_lambda"]
        subsample = float_params["subsample"]
        colsample_bytree = float_params["colsample_bytree"]

        # LightGBM parameters
        params: dict[str, str | int | float] = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_samples": min_child_samples,
            "scale_pos_weight": self._scale_pos_weight,
            "device": self._device,
            "seed": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }

        # Early stopping callback
        early_stop_cb = self._early_stopping(
            stopping_rounds=self._early_stopping_rounds,
            verbose=False,
        )

        # Train with early stopping on validation AUC
        booster = self._lgb_train(
            params,
            self._train_dataset,
            num_boost_round=n_estimators,
            valid_sets=[self._val_dataset],
            valid_names=["valid"],
            callbacks=[early_stop_cb],
        )

        # Predict on validation set
        y_pred_proba: NDArray[np.float64] = np.asarray(
            booster.predict(self._x_val, num_iteration=None),
            dtype=np.float64,
        )

        # Use our typed compute_auc instead of sklearn
        return compute_auc(self._y_val, y_pred_proba)


def create_lightgbm_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    device: Literal["cpu", "cuda", "auto"],
    feature_preset: FeaturePreset,
    early_stopping_rounds: int = 10,
) -> LightGBMObjective:
    """Create an objective function for LightGBM optimization.

    Applies feature engineering based on preset and pre-splits data for efficient
    trial evaluation. The returned objective tracks the engineered feature count
    via its n_features property.

    Args:
        x_features: Feature matrix
        y_labels: Binary labels
        feature_names: Original feature names
        device: Device to use for training
        feature_preset: Feature engineering preset to apply
        early_stopping_rounds: Stop if no improvement for this many rounds

    Returns:
        Objective callable with n_features property for engineered feature count
    """
    return LightGBMObjective(
        x_features,
        y_labels,
        feature_names,
        device,
        feature_preset,
        early_stopping_rounds,
    )


__all__ = [
    "LightGBMObjective",
    "create_lightgbm_objective",
]
