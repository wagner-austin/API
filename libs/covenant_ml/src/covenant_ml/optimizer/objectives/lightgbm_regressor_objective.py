"""LightGBM regressor objective function for hyperparameter optimization.

Parallel to lightgbm_objective.py (classification). Key differences:
- objective="regression" instead of "binary"
- metric="rmse" instead of "auc"
- y_targets: NDArray[np.float64] (continuous, not int64 binary)
- No scale_pos_weight (regression has no class imbalance)
- Returns negative RMSE (Optuna maximizes; lower RMSE = better)
- Uses regression_split instead of stratified_split
- Preprocesses with AutoPreprocessor (y unused in fit)

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

import gc
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.features import (
    FeaturePreset,
    engineer_features,
    get_feature_config_for_preset,
)
from covenant_ml.metrics import compute_rmse
from covenant_ml.optimizer.types import (
    DeviceRequest,
    LightGBMDevice,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.preprocessing import AutoPreprocessor, PreprocessingState
from covenant_ml.trainer import regression_split

_log = get_logger(__name__)


# =============================================================================
# LightGBM Protocol Types
# =============================================================================


class _LGBDatasetProtocol(Protocol):
    """Protocol for LightGBM Dataset."""

    def __init__(
        self,
        data: NDArray[np.float64],
        label: NDArray[np.float64] | None = ...,
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


def _resolve_lightgbm_device(
    device: DeviceRequest,
    *,
    platform: str | None = None,
) -> LightGBMDevice:
    """Resolve user device request to LightGBM-compatible device parameter.

    Args:
        device: User-requested device ("cpu", "cuda", or "auto").
        platform: Override for sys.platform (for testing). If None, uses sys.platform.

    Returns:
        LightGBM device parameter ("cpu", "gpu", or "cuda").
    """
    import sys

    actual_platform = platform if platform is not None else sys.platform

    if device == "auto":
        return "cpu"
    if device == "cuda" and actual_platform == "win32":
        _log.info(
            "LightGBM CUDA not supported on Windows, using OpenCL GPU instead",
            extra={"requested_device": device, "resolved_device": "gpu"},
        )
        return "gpu"
    if device == "cuda":
        return "cuda"
    return "cpu"


# =============================================================================
# LightGBM Regressor Objective
# =============================================================================


class LightGBMRegressorObjective:
    """LightGBM regressor objective that returns negative validation RMSE.

    Uses LightGBM's native Dataset format with early stopping for efficient
    hyperparameter optimization.
    """

    def __init__(
        self,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str],
        device: DeviceRequest,
        feature_preset: FeaturePreset,
        early_stopping_rounds: int = 10,
        n_jobs: int = -1,
    ) -> None:
        """Initialize with data and configuration.

        Args:
            x_features: Feature matrix.
            y_targets: Continuous target values.
            feature_names: Original feature names.
            device: Device to use for training (cpu/cuda/auto).
            feature_preset: Feature engineering preset to apply.
            early_stopping_rounds: Stop if no improvement for this many rounds.
            n_jobs: Number of parallel threads for LightGBM (-1 for all cores).
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
                "Applied feature engineering for LightGBM regressor",
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

        self._n_features = int(x_engineered.shape[1])
        self._early_stopping_rounds = early_stopping_rounds
        self._n_jobs = n_jobs

        # Pre-split data using random regression split
        raw_splits = regression_split(
            x_engineered,
            y_targets,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        # Preprocess features
        preprocessor = AutoPreprocessor()
        dummy_y: NDArray[np.int64] = np.zeros(raw_splits.n_train, dtype=np.int64)
        state: PreprocessingState = preprocessor.fit(raw_splits.x_train, dummy_y)

        x_train_processed = preprocessor.transform(raw_splits.x_train, state)
        x_val_processed = preprocessor.transform(raw_splits.x_val, state)

        # Resolve device
        self._device: LightGBMDevice = _resolve_lightgbm_device(device)

        # Pre-create Dataset objects
        dataset_cls, train_fn, early_stopping = _get_lgb_dataset_and_train()
        self._train_dataset = dataset_cls(
            x_train_processed,
            label=raw_splits.y_train,
            free_raw_data=False,
        )
        self._val_dataset = dataset_cls(
            x_val_processed,
            label=raw_splits.y_val,
            free_raw_data=False,
        )
        self._x_val = x_val_processed
        self._y_val = raw_splits.y_val
        self._lgb_train = train_fn
        self._early_stopping = early_stopping

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
        """Train LightGBM regressor and return negative validation RMSE.

        Args:
            x_features: Ignored (uses pre-split data).
            y_targets: Ignored (uses pre-split data).
            feature_names: Ignored (uses pre-split data).
            int_params: Integer hyperparameters from sampler.
            float_params: Float hyperparameters from sampler.
            string_params: String hyperparameters (boosting_type for DART).
            train_ratio: Ignored (uses pre-split data).
            val_ratio: Ignored (uses pre-split data).
            test_ratio: Ignored (uses pre-split data).
            random_state: Random seed for reproducibility.

        Returns:
            Negative validation RMSE (higher = better for Optuna).
        """
        _ = x_features, y_targets, feature_names
        _ = train_ratio, val_ratio, test_ratio

        # Extract hyperparameters
        n_estimators = int_params["n_estimators"]
        num_leaves = int_params.get("num_leaves", 31)
        min_child_samples = int_params.get("min_child_samples", 20)
        learning_rate = float_params["learning_rate"]
        reg_alpha = float_params["reg_alpha"]
        reg_lambda = float_params["reg_lambda"]
        subsample = float_params["subsample"]
        colsample_bytree = float_params["colsample_bytree"]

        boosting_type = string_params.get("boosting_type", "gbdt")

        # LightGBM regression parameters
        params: dict[str, str | int | float] = {
            "boosting_type": boosting_type,
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": num_leaves,
            "max_depth": -1,
            "learning_rate": learning_rate,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_samples": min_child_samples,
            "device": self._device,
            "seed": random_state,
            "verbose": -1,
            "n_jobs": self._n_jobs,
        }

        # DART-specific params
        if boosting_type == "dart":
            if "drop_rate" in float_params:
                params["drop_rate"] = float_params["drop_rate"]
            if "skip_drop" in float_params:
                params["skip_drop"] = float_params["skip_drop"]
            if "feature_fraction" in float_params:
                params["feature_fraction"] = float_params["feature_fraction"]

        # Train with or without early stopping
        if boosting_type == "dart":
            booster = self._lgb_train(
                params,
                self._train_dataset,
                num_boost_round=n_estimators,
                valid_sets=[self._val_dataset],
                valid_names=["valid"],
            )
        else:
            early_stop_cb = self._early_stopping(
                stopping_rounds=self._early_stopping_rounds,
                verbose=False,
            )
            booster = self._lgb_train(
                params,
                self._train_dataset,
                num_boost_round=n_estimators,
                valid_sets=[self._val_dataset],
                valid_names=["valid"],
                callbacks=[early_stop_cb],
            )

        # Predict on validation set
        y_pred: NDArray[np.float64] = np.asarray(
            booster.predict(self._x_val, num_iteration=None),
            dtype=np.float64,
        )

        # Compute RMSE and return negative (Optuna maximizes)
        rmse = compute_rmse(self._y_val, y_pred)

        del booster
        del y_pred
        gc.collect()

        return -rmse


def create_lightgbm_regressor_objective(
    x_features: NDArray[np.float64],
    y_targets: NDArray[np.float64],
    feature_names: list[str],
    device: DeviceRequest,
    feature_preset: FeaturePreset,
    early_stopping_rounds: int = 10,
    n_jobs: int = -1,
) -> LightGBMRegressorObjective:
    """Create an objective function for LightGBM regression optimization.

    Args:
        x_features: Feature matrix.
        y_targets: Continuous target values.
        feature_names: Original feature names.
        device: Device to use for training (cpu/cuda/auto).
        feature_preset: Feature engineering preset to apply.
        early_stopping_rounds: Stop if no improvement for this many rounds.
        n_jobs: Number of parallel threads for LightGBM (-1 for all cores).

    Returns:
        Objective callable with n_features property.
    """
    return LightGBMRegressorObjective(
        x_features,
        y_targets,
        feature_names,
        device,
        feature_preset,
        early_stopping_rounds,
        n_jobs,
    )


__all__ = [
    "LightGBMRegressorObjective",
    "create_lightgbm_regressor_objective",
]
