"""XGBoost objective function for hyperparameter optimization.

Provides the objective function that Optuna uses to evaluate XGBoost
hyperparameter configurations. Pre-splits data once and creates DMatrix
objects for efficient trial evaluation.

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
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams
from covenant_ml.trainer import preprocess_data_splits, stratified_split

_log = get_logger(__name__)


# =============================================================================
# XGBoost Protocol Types
# =============================================================================


class DMatrixProtocol(Protocol):
    """Protocol for XGBoost DMatrix."""

    def __init__(
        self,
        data: NDArray[np.float64],
        label: NDArray[np.int64] | None = ...,
    ) -> None: ...


class BoosterProtocol(Protocol):
    """Protocol for XGBoost Booster."""

    def predict(self, data: DMatrixProtocol) -> NDArray[np.float64]: ...


class XGBTrainFunc(Protocol):
    """Protocol for xgb.train function."""

    def __call__(
        self,
        params: dict[str, str | int | float],
        dtrain: DMatrixProtocol,
        num_boost_round: int = ...,
        *,
        verbose_eval: bool = ...,
    ) -> BoosterProtocol: ...


class XGBBuildInfoProtocol(Protocol):
    """Protocol for xgboost module's build_info function."""

    def __call__(self) -> dict[str, str]: ...


# =============================================================================
# XGBoost Helpers
# =============================================================================


def _get_xgb_dmatrix_and_train() -> tuple[type[DMatrixProtocol], XGBTrainFunc]:
    """Get DMatrix class and train function via dynamic import."""
    xgb_module = __import__("xgboost")
    dmatrix_cls: type[DMatrixProtocol] = xgb_module.DMatrix
    train_fn: XGBTrainFunc = xgb_module.train
    return dmatrix_cls, train_fn


def _cuda_available() -> bool:
    """Check if CUDA is available for XGBoost."""
    xgb_module = __import__("xgboost")
    build_info_fn: XGBBuildInfoProtocol = xgb_module.build_info
    build_info: dict[str, str] = build_info_fn()
    use_cuda_value = build_info.get("USE_CUDA", "OFF")
    return use_cuda_value == "ON"


# =============================================================================
# XGBoost Objective
# =============================================================================


class XGBoostObjective:
    """XGBoost objective that trains on pre-split data and returns validation AUC.

    Uses DMatrix directly for full GPU pipeline - no scikit-learn wrapper overhead.
    """

    def __init__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        feature_preset: FeaturePreset,
    ) -> None:
        """Initialize with pre-split data and pre-created DMatrix objects.

        Args:
            x_features: Feature matrix
            y_labels: Binary labels
            feature_names: Original feature names
            device: Device to use for training
            feature_preset: Feature engineering preset to apply
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
                "Applied feature engineering",
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

        # Pre-split and preprocess data once
        raw_splits = stratified_split(
            x_engineered,
            y_labels,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )
        self._splits = preprocess_data_splits(raw_splits)

        # Resolve device once
        self._device = ("cuda" if _cuda_available() else "cpu") if device == "auto" else device

        # Calculate scale_pos_weight from training data (once)
        n_pos = int(np.sum(self._splits.y_train))
        n_neg = len(self._splits.y_train) - n_pos
        self._scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Pre-create DMatrix objects with preprocessed data
        dmatrix_cls, train_fn = _get_xgb_dmatrix_and_train()
        self._train_dmatrix = dmatrix_cls(
            self._splits.x_train,
            label=self._splits.y_train,
        )
        self._val_dmatrix = dmatrix_cls(
            self._splits.x_val,
            label=self._splits.y_val,
        )
        self._y_val = self._splits.y_val  # Keep for AUC computation
        self._xgb_train = train_fn

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
        """Train XGBoost using DMatrix directly and return validation AUC.

        Args:
            x_features: Ignored (uses pre-split data)
            y_labels: Ignored (uses pre-split data)
            feature_names: Ignored (uses pre-split data)
            int_params: Integer hyperparameters from sampler
            float_params: Float hyperparameters from sampler
            string_params: String hyperparameters (booster for DART)
            train_ratio: Ignored (uses pre-split data)
            val_ratio: Ignored (uses pre-split data)
            test_ratio: Ignored (uses pre-split data)
            random_state: Random seed for reproducibility

        Returns:
            Validation AUC score
        """
        # Ignore passed data - use pre-computed DMatrix
        _ = x_features, y_labels, feature_names
        _ = train_ratio, val_ratio, test_ratio

        # Extract hyperparameters from typed dicts
        max_depth = int_params["max_depth"]
        n_estimators = int_params["n_estimators"]
        learning_rate = float_params["learning_rate"]
        reg_alpha = float_params["reg_alpha"]
        reg_lambda = float_params["reg_lambda"]
        subsample = float_params["subsample"]
        colsample_bytree = float_params["colsample_bytree"]

        # Get booster type from string_params (defaults to "gbtree" if not present)
        booster_type = string_params.get("booster", "gbtree")

        # XGBoost parameters for direct training
        params: dict[str, str | int | float] = {
            "booster": booster_type,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": self._scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "device": self._device,
            "seed": random_state,
        }

        # Add DART-specific params when using DART booster
        if booster_type == "dart":
            if "rate_drop" in float_params:
                params["rate_drop"] = float_params["rate_drop"]
            if "skip_drop" in float_params:
                params["skip_drop"] = float_params["skip_drop"]

        # Train using xgb.train directly (full GPU pipeline)
        trained_model = self._xgb_train(
            params,
            self._train_dmatrix,
            num_boost_round=n_estimators,
            verbose_eval=False,
        )

        # Predict on validation set (already on GPU)
        y_pred_proba: NDArray[np.float64] = trained_model.predict(self._val_dmatrix)
        # Use our typed compute_auc instead of sklearn
        return compute_auc(self._y_val, y_pred_proba)


def create_xgboost_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    device: Literal["cpu", "cuda", "auto"],
    feature_preset: FeaturePreset,
) -> XGBoostObjective:
    """Create an objective function for XGBoost optimization.

    Applies feature engineering based on preset and pre-splits data for efficient
    trial evaluation. The returned objective tracks the engineered feature count
    via its n_features property.

    Args:
        x_features: Feature matrix
        y_labels: Binary labels
        feature_names: Original feature names
        device: Device to use for training
        feature_preset: Feature engineering preset to apply

    Returns:
        Objective callable with n_features property for engineered feature count
    """
    return XGBoostObjective(x_features, y_labels, feature_names, device, feature_preset)


__all__ = [
    "XGBoostObjective",
    "create_xgboost_objective",
]
