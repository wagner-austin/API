"""XGBoost regressor objective function for hyperparameter optimization.

Parallel to xgboost_objective.py (classification). Key differences:
- objective="reg:squarederror" instead of "binary:logistic"
- eval_metric="rmse" instead of "auc"
- __init__ takes y_targets: NDArray[np.float64] (continuous targets)
- __call__ takes y_labels: NDArray[np.int64] (ObjectiveProtocol compat; ignored)
- No scale_pos_weight (regression has no class imbalance)
- Returns negative RMSE (Optuna maximizes; lower RMSE = better)
- Uses regression_split instead of stratified_split
- Preprocesses with AutoPreprocessor (y unused in fit)

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

import gc
import math
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.features import (
    FeaturePreset,
    engineer_features,
    get_feature_config_for_preset,
)
from covenant_ml.metrics import compute_rmse
from covenant_ml.optimizer.types import SampledFloatParams, SampledIntParams, SampledStringParams
from covenant_ml.preprocessing import AutoPreprocessor, PreprocessingState
from covenant_ml.trainer import regression_split

_log = get_logger(__name__)


# =============================================================================
# XGBoost Protocol Types (same as classification)
# =============================================================================


class _DMatrixProtocol(Protocol):
    """Protocol for XGBoost DMatrix."""

    def __init__(
        self,
        data: NDArray[np.float64],
        label: NDArray[np.float64] | None = ...,
    ) -> None: ...


class _BoosterProtocol(Protocol):
    """Protocol for XGBoost Booster."""

    def predict(self, data: _DMatrixProtocol) -> NDArray[np.float64]: ...


class _XGBTrainFunc(Protocol):
    """Protocol for xgb.train function."""

    def __call__(
        self,
        params: dict[str, str | int | float],
        dtrain: _DMatrixProtocol,
        num_boost_round: int = ...,
        *,
        verbose_eval: bool = ...,
    ) -> _BoosterProtocol: ...


class _XGBBuildInfoProtocol(Protocol):
    """Protocol for xgboost module's build_info function."""

    def __call__(self) -> dict[str, str]: ...


# =============================================================================
# XGBoost Helpers
# =============================================================================


def _get_xgb_dmatrix_and_train() -> tuple[type[_DMatrixProtocol], _XGBTrainFunc]:
    """Get DMatrix class and train function via dynamic import."""
    xgb_module = __import__("xgboost")
    dmatrix_cls: type[_DMatrixProtocol] = xgb_module.DMatrix
    train_fn: _XGBTrainFunc = xgb_module.train
    return dmatrix_cls, train_fn


def _cuda_available() -> bool:
    """Check if CUDA is available for XGBoost."""
    xgb_module = __import__("xgboost")
    build_info_fn: _XGBBuildInfoProtocol = xgb_module.build_info
    build_info: dict[str, str] = build_info_fn()
    use_cuda_value = build_info.get("USE_CUDA", "OFF")
    return use_cuda_value == "ON"


# =============================================================================
# XGBoost Regressor Objective
# =============================================================================


class XGBoostRegressorObjective:
    """XGBoost regressor objective that returns negative validation RMSE.

    Uses DMatrix directly for full GPU pipeline. Pre-splits data once
    and pre-creates DMatrix objects for efficient trial evaluation.
    """

    def __init__(
        self,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str],
        device: Literal["cpu", "cuda", "auto"],
        feature_preset: FeaturePreset,
    ) -> None:
        """Initialize with pre-split and preprocessed data.

        Args:
            x_features: Feature matrix.
            y_targets: Continuous target values.
            feature_names: Original feature names.
            device: Device to use for training.
            feature_preset: Feature engineering preset to apply.
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
                "Applied feature engineering for XGBoost regressor",
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

        # Pre-split data using random (non-stratified) regression split
        raw_splits = regression_split(
            x_engineered,
            y_targets,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42,
        )

        # Preprocess features (AutoPreprocessor.fit uses dummy int64 y)
        preprocessor = AutoPreprocessor()
        dummy_y: NDArray[np.int64] = np.zeros(raw_splits.n_train, dtype=np.int64)
        state: PreprocessingState = preprocessor.fit(raw_splits.x_train, dummy_y)

        x_train_processed = preprocessor.transform(raw_splits.x_train, state)
        x_val_processed = preprocessor.transform(raw_splits.x_val, state)

        # Resolve device once
        self._device = ("cuda" if _cuda_available() else "cpu") if device == "auto" else device

        # Pre-create DMatrix objects with preprocessed data
        dmatrix_cls, train_fn = _get_xgb_dmatrix_and_train()
        self._train_dmatrix = dmatrix_cls(
            x_train_processed,
            label=raw_splits.y_train,
        )
        self._val_dmatrix = dmatrix_cls(
            x_val_processed,
            label=raw_splits.y_val,
        )
        self._y_val = raw_splits.y_val
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
        """Train XGBoost regressor and return negative validation RMSE.

        Conforms to ObjectiveProtocol so existing optimizer strategies can
        call this objective unchanged. The x/y/feature_names parameters are
        ignored — real data is pre-split in __init__. The int64 y_labels
        type satisfies the protocol; the actual regression targets (float64)
        are stored internally from __init__.

        Args:
            x_features: Ignored (uses pre-split data).
            y_labels: Ignored (uses pre-split data; int64 for protocol compat).
            feature_names: Ignored (uses pre-split data).
            int_params: Integer hyperparameters from sampler.
            float_params: Float hyperparameters from sampler.
            string_params: String hyperparameters (booster for DART).
            train_ratio: Ignored (uses pre-split data).
            val_ratio: Ignored (uses pre-split data).
            test_ratio: Ignored (uses pre-split data).
            random_state: Random seed for reproducibility.

        Returns:
            Negative validation RMSE (higher = better for Optuna).
        """
        _ = x_features, y_labels, feature_names
        _ = train_ratio, val_ratio, test_ratio

        # Extract hyperparameters
        max_depth = int_params["max_depth"]
        n_estimators = int_params["n_estimators"]
        learning_rate = float_params["learning_rate"]
        reg_alpha = float_params["reg_alpha"]
        reg_lambda = float_params["reg_lambda"]
        subsample = float_params["subsample"]
        colsample_bytree = float_params["colsample_bytree"]

        booster_type = string_params.get("booster", "gbtree")

        # XGBoost regression parameters
        params: dict[str, str | int | float] = {
            "booster": booster_type,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "device": self._device,
            "seed": random_state,
            "nthread": 1,
        }

        # DART-specific params
        if booster_type == "dart":
            if "rate_drop" in float_params:
                params["rate_drop"] = float_params["rate_drop"]
            if "skip_drop" in float_params:
                params["skip_drop"] = float_params["skip_drop"]

        # Train
        trained_model = self._xgb_train(
            params,
            self._train_dmatrix,
            num_boost_round=n_estimators,
            verbose_eval=False,
        )

        # Predict on validation set
        y_pred: NDArray[np.float64] = trained_model.predict(self._val_dmatrix)

        # Compute RMSE and return negative (Optuna maximizes)
        rmse = compute_rmse(self._y_val, y_pred)

        del trained_model
        del y_pred
        gc.collect()

        return -rmse

    @staticmethod
    def _compute_neg_rmse(
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ) -> float:
        """Compute negative RMSE for optimization.

        Args:
            y_true: True target values.
            y_pred: Predicted values.

        Returns:
            Negative RMSE (higher = better).
        """
        n = len(y_true)
        sse = 0.0
        for i in range(n):
            diff = float(y_true.item(i)) - float(y_pred.item(i))
            sse += diff * diff
        return -math.sqrt(sse / n)


def create_xgboost_regressor_objective(
    x_features: NDArray[np.float64],
    y_targets: NDArray[np.float64],
    feature_names: list[str],
    device: Literal["cpu", "cuda", "auto"],
    feature_preset: FeaturePreset,
) -> XGBoostRegressorObjective:
    """Create an objective function for XGBoost regression optimization.

    Args:
        x_features: Feature matrix.
        y_targets: Continuous target values.
        feature_names: Original feature names.
        device: Device to use for training.
        feature_preset: Feature engineering preset to apply.

    Returns:
        Objective callable with n_features property.
    """
    return XGBoostRegressorObjective(x_features, y_targets, feature_names, device, feature_preset)


__all__ = [
    "XGBoostRegressorObjective",
    "create_xgboost_regressor_objective",
]
