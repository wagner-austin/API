"""Random Forest objective function for hyperparameter optimization.

Provides the objective function that Optuna uses to evaluate Random Forest
hyperparameter configurations. Pre-splits data once for efficient
trial evaluation.

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
from covenant_ml.metrics import compute_auc
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.trainer import preprocess_data_splits, stratified_split

_log = get_logger(__name__)


# =============================================================================
# sklearn Protocol Types
# =============================================================================


class _RandomForestModelProtocol(Protocol):
    """Protocol for sklearn RandomForestClassifier."""

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
    ) -> _RandomForestModelProtocol:
        """Fit the model to training data."""
        ...

    def predict_proba(self, x_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        ...


class _RandomForestCtorProtocol(Protocol):
    """Protocol for RandomForestClassifier constructor."""

    def __call__(
        self,
        *,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
        max_features: str,
        class_weight: str | None,
        random_state: int,
        n_jobs: int,
    ) -> _RandomForestModelProtocol:
        """Construct RandomForestClassifier with given parameters."""
        ...


# =============================================================================
# sklearn Helpers
# =============================================================================


def _get_random_forest_ctor() -> _RandomForestCtorProtocol:
    """Get RandomForestClassifier class via dynamic import.

    Returns:
        RandomForestClassifier constructor satisfying _RandomForestCtorProtocol.
    """
    sklearn_module = __import__(
        "sklearn.ensemble",
        fromlist=["RandomForestClassifier"],
    )
    ctor: _RandomForestCtorProtocol = sklearn_module.RandomForestClassifier
    return ctor


# =============================================================================
# Random Forest Objective
# =============================================================================


class RandomForestObjective:
    """Random Forest objective that trains on pre-split data and returns validation AUC.

    Uses sklearn RandomForestClassifier with class weight balancing for
    efficient hyperparameter optimization.
    """

    def __init__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        feature_preset: FeaturePreset,
    ) -> None:
        """Initialize with data and configuration.

        Args:
            x_features: Feature matrix.
            y_labels: Binary labels.
            feature_names: Original feature names.
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
                "Applied feature engineering for RandomForest",
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

        # Store numpy arrays directly
        self._x_train: NDArray[np.float64] = self._splits.x_train
        self._y_train: NDArray[np.int64] = self._splits.y_train
        self._x_val: NDArray[np.float64] = self._splits.x_val
        self._y_val: NDArray[np.int64] = self._splits.y_val

        # Get RandomForest constructor
        self._rf_ctor = _get_random_forest_ctor()

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
        """Train Random Forest and return validation AUC.

        Args:
            x_features: Ignored (uses pre-split data).
            y_labels: Ignored (uses pre-split data).
            feature_names: Ignored (uses pre-split data).
            int_params: Integer hyperparameters from sampler.
            float_params: Float hyperparameters from sampler.
            string_params: String hyperparameters from sampler.
            train_ratio: Ignored (uses pre-split data).
            val_ratio: Ignored (uses pre-split data).
            test_ratio: Ignored (uses pre-split data).
            random_state: Random seed for reproducibility.

        Returns:
            Validation AUC score.
        """
        # Ignore passed data - use pre-computed datasets
        _ = x_features, y_labels, feature_names
        _ = train_ratio, val_ratio, test_ratio
        _ = float_params  # RandomForest has no float params in search space

        # Extract hyperparameters from typed dicts
        n_estimators = int_params["n_estimators"]
        max_depth = int_params["max_depth"]
        min_samples_split = int_params.get("min_samples_split", 5)
        min_samples_leaf = int_params.get("min_samples_leaf", 2)

        max_features = string_params.get("max_features", "sqrt")

        # Train Random Forest model
        model = self._rf_ctor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(self._x_train, self._y_train)

        # Predict on validation set
        y_pred_proba_raw: NDArray[np.float64] = model.predict_proba(self._x_val)
        y_pred_proba: NDArray[np.float64] = np.asarray(y_pred_proba_raw[:, 1], dtype=np.float64)

        # Use our typed compute_auc instead of sklearn
        auc = compute_auc(self._y_val, y_pred_proba)

        # Force cleanup between trials
        del model
        del y_pred_proba_raw
        del y_pred_proba
        gc.collect()

        return auc


def create_random_forest_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    feature_preset: FeaturePreset,
) -> RandomForestObjective:
    """Create an objective function for Random Forest optimization.

    Applies feature engineering based on preset and pre-splits data for efficient
    trial evaluation. The returned objective tracks the engineered feature count
    via its n_features property.

    Args:
        x_features: Feature matrix.
        y_labels: Binary labels.
        feature_names: Original feature names.
        feature_preset: Feature engineering preset to apply.

    Returns:
        Objective callable with n_features property for engineered feature count.
    """
    return RandomForestObjective(x_features, y_labels, feature_names, feature_preset)


__all__ = [
    "RandomForestObjective",
    "create_random_forest_objective",
]
