"""ClearGBM objective function for hyperparameter optimization.

Provides the objective function that Optuna uses to evaluate ClearGBM
hyperparameter configurations. Pre-splits data once and pre-computes
class weights for efficient trial evaluation.

ClearGBM is a numpy-based gradient boosting implementation with built-in
interpretability features (rule extraction, feature contributions).

Strict typing only: no Any, no casts, no stubs.
"""

from __future__ import annotations

import gc

import numpy as np
from cleargbm.ensemble import predict_proba as cgbm_predict_proba
from cleargbm.ensemble import train_gradient_boosting
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.backends.cleargbm.config_resolution import _compute_class_weight
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


def _extract_positive_class_proba(
    proba: tuple[tuple[float, float], ...],
) -> NDArray[np.float64]:
    """Extract positive class probabilities from predict_proba output.

    ClearGBM's predict_proba returns (prob_class_0, prob_class_1) tuples.
    We extract the second element (positive class probability) for AUC computation.

    Args:
        proba: Tuple of (prob_class_0, prob_class_1) tuples.

    Returns:
        1D numpy array of positive class probabilities.
    """
    positive_probs: list[float] = []
    for pair in proba:
        positive_probs.append(pair[1])
    return np.array(positive_probs, dtype=np.float64)


def _build_trial_config(
    *,
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
    y_train: NDArray[np.int64],
    random_state: int,
    early_stopping_rounds: int,
) -> GradientBoostingConfig:
    """Build the training config one optimization trial runs under.

    Every trial trains under the same auto-computed class weight the
    backend applies to the final model. Trials trained unweighted until
    2026-08-22, so the sweep tuned an objective the production model never
    trained — the exact mismatch that made the pre-fix "optimal" configs
    stale the moment weighting became real.

    Args:
        int_params: Integer hyperparameters from the sampler.
        float_params: Float hyperparameters from the sampler.
        y_train: Training labels the class weight derives from.
        random_state: Random seed for the trial's training.
        early_stopping_rounds: Early stopping patience.

    Returns:
        The trial's training configuration.
    """
    return GradientBoostingConfig(
        n_estimators=int_params["n_estimators"],
        max_depth=int_params["max_depth"],
        learning_rate=float_params["learning_rate"],
        min_samples_split=int_params.get("min_samples_split", 10),
        min_samples_leaf=int_params.get("min_samples_leaf", 5),
        max_features=None,
        max_bins=int_params.get("max_bins", 64),
        subsample=float_params.get("subsample", 1.0),
        random_state=random_state,
        monotonic_constraints=None,
        reg_alpha=0.0,
        reg_lambda=0.0,
        n_jobs=1,  # Sequential for stability
        early_stopping_rounds=early_stopping_rounds,
        growth_strategy="depth_wise",
        num_leaves=None,
        scale_pos_weight=_compute_class_weight(y_train),
    )


class ClearGBMObjective:
    """ClearGBM objective that trains on pre-split data and returns validation AUC.

    Uses numpy-based ClearGBM library with histogram-based split finding
    for efficient hyperparameter optimization.
    """

    def __init__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        feature_preset: FeaturePreset,
        early_stopping_rounds: int = 10,
    ) -> None:
        """Initialize with data and configuration.

        Args:
            x_features: Feature matrix.
            y_labels: Binary labels.
            feature_names: Original feature names.
            feature_preset: Feature engineering preset to apply.
            early_stopping_rounds: Stop if no improvement for this many rounds.
        """
        # Apply feature engineering BEFORE splitting
        if feature_preset != "none":
            config = get_feature_config_for_preset(feature_preset)
            engineered = engineer_features(x_features, feature_names, config)
            x_engineered = engineered["x"]
            engineered_names = engineered["feature_names"]
            n_original = engineered["n_original"]
            n_ratios = engineered["n_ratios"]
            n_products = engineered["n_products"]
            n_log = engineered["n_log"]
            _log.info(
                "Applied feature engineering for ClearGBM",
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
            engineered_names = feature_names

        # Store actual feature count (after engineering)
        self._n_features = int(x_engineered.shape[1])
        self._early_stopping_rounds = early_stopping_rounds
        self._feature_names = tuple(engineered_names)

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

        # Store numpy arrays directly for ClearGBM
        self._x_train: NDArray[np.float64] = self._splits.x_train
        self._y_train: NDArray[np.int64] = self._splits.y_train
        self._x_val: NDArray[np.float64] = self._splits.x_val
        self._y_val: NDArray[np.int64] = self._splits.y_val

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
        """Train ClearGBM and return validation AUC.

        Args:
            x_features: Ignored (uses pre-split data).
            y_labels: Ignored (uses pre-split data).
            feature_names: Ignored (uses pre-split data).
            int_params: Integer hyperparameters from sampler.
            float_params: Float hyperparameters from sampler.
            string_params: String hyperparameters (unused for ClearGBM).
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
        _ = string_params  # ClearGBM has no string params

        config = _build_trial_config(
            int_params=int_params,
            float_params=float_params,
            y_train=self._y_train,
            random_state=random_state,
            early_stopping_rounds=self._early_stopping_rounds,
        )

        model = train_gradient_boosting(
            x_train=self._x_train,
            y_train=self._y_train,
            x_val=self._x_val,
            y_val=self._y_val,
            config=config,
            feature_names=self._feature_names,
        )

        # Predict on validation set
        proba_tuple = cgbm_predict_proba(model, self._x_val)
        y_pred_proba = _extract_positive_class_proba(proba_tuple)

        # Use our typed compute_auc instead of sklearn
        auc = compute_auc(self._y_val, y_pred_proba)

        # Force aggressive cleanup between trials to prevent memory accumulation
        del model
        del proba_tuple
        del y_pred_proba
        gc.collect()

        return auc


def create_cleargbm_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    feature_preset: FeaturePreset,
    early_stopping_rounds: int = 10,
) -> ClearGBMObjective:
    """Create an objective function for ClearGBM optimization.

    Applies feature engineering based on preset and pre-splits data for efficient
    trial evaluation. The returned objective tracks the engineered feature count
    via its n_features property.

    Args:
        x_features: Feature matrix.
        y_labels: Binary labels.
        feature_names: Original feature names.
        feature_preset: Feature engineering preset to apply.
        early_stopping_rounds: Stop if no improvement for this many rounds.

    Returns:
        Objective callable with n_features property for engineered feature count.
    """
    return ClearGBMObjective(
        x_features,
        y_labels,
        feature_names,
        feature_preset,
        early_stopping_rounds,
    )


__all__ = [
    "ClearGBMObjective",
    "create_cleargbm_objective",
]
