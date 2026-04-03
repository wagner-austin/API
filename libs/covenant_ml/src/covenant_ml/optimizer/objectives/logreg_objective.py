"""Logistic Regression objective function for hyperparameter optimization.

Provides the objective function that Optuna uses to evaluate LogReg
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


class _LogRegModelProtocol(Protocol):
    """Protocol for sklearn LogisticRegression classifier."""

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
    ) -> _LogRegModelProtocol:
        """Fit the model to training data."""
        ...

    def predict_proba(self, x_data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        ...


# =============================================================================
# sklearn Helpers
# =============================================================================


def _build_logreg_model(
    *,
    penalty: str | None,
    c_value: float,
    solver: str,
    max_iter: int,
    tol: float,
    random_state: int,
    class_weight: str | None,
    l1_ratio: float | None,
    n_jobs: int,
) -> _LogRegModelProtocol:
    """Build a LogisticRegression model via dynamic import.

    Uses lowercase ``c_value`` to satisfy naming conventions, then passes
    it as sklearn's ``C`` keyword argument.

    Args:
        penalty: Regularization type (l1, l2, elasticnet, or None).
        c_value: Inverse regularization strength (sklearn's C parameter).
        solver: Optimization algorithm.
        max_iter: Maximum iterations for solver convergence.
        tol: Tolerance for stopping criteria.
        random_state: Random seed for reproducibility.
        class_weight: Class weight balancing strategy.
        l1_ratio: ElasticNet mixing parameter (only for elasticnet).
        n_jobs: Number of parallel jobs.

    Returns:
        Fitted-ready LogisticRegression model.
    """
    sklearn_module = __import__(
        "sklearn.linear_model",
        fromlist=["LogisticRegression"],
    )
    model: _LogRegModelProtocol = sklearn_module.LogisticRegression(
        penalty=penalty,
        C=c_value,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        class_weight=class_weight,
        l1_ratio=l1_ratio,
        n_jobs=n_jobs,
    )
    return model


# =============================================================================
# LogReg Objective
# =============================================================================


class LogRegObjective:
    """LogReg objective that trains on pre-split data and returns validation AUC.

    Uses sklearn LogisticRegression with class weight balancing for
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
                "Applied feature engineering for LogReg",
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
        """Train LogReg and return validation AUC.

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

        # Extract hyperparameters from typed dicts
        c_value = float_params["C"]
        tol_value = float_params.get("tol", 1e-4)
        max_iter_value = int_params.get("max_iter", 300)
        l1_ratio_value = float_params.get("l1_ratio", None)

        penalty_value = string_params.get("penalty", "l2")
        solver_value = string_params.get("solver", "saga")

        # Map penalty to sklearn format
        penalty_arg: str | None = None if penalty_value == "none" else penalty_value
        l1_ratio_arg: float | None = l1_ratio_value if penalty_value == "elasticnet" else None

        # Train LogReg model
        model = _build_logreg_model(
            penalty=penalty_arg,
            c_value=c_value,
            solver=solver_value,
            max_iter=max_iter_value,
            tol=tol_value,
            random_state=random_state,
            class_weight="balanced",
            l1_ratio=l1_ratio_arg,
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


def create_logreg_objective(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    feature_names: list[str],
    feature_preset: FeaturePreset,
) -> LogRegObjective:
    """Create an objective function for LogReg optimization.

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
    return LogRegObjective(x_features, y_labels, feature_names, feature_preset)


__all__ = [
    "LogRegObjective",
    "create_logreg_objective",
]
