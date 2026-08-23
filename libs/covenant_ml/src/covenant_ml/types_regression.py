"""Regression training types: metrics, configs, outcomes, XGB regressor protocols."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

from covenant_ml.types import (
    ClearGBMConfig,
    FeatureImportance,
    LightGBMConfig,
    LSTMConfig,
    MLPConfig,
    TrainConfig,
    XGBBoosterProtocol,
)


class XGBRegressorModelProtocol(Protocol):
    """Protocol for XGBoost regressor interface.

    Parallel to XGBModelProtocol for classifiers. Key differences:
    - fit() takes float64 y_targets instead of int64 y_labels.
    - predict() returns 1D float64 instead of predict_proba() returning 2D.
    - No get_xgb_params() (not needed for regression training).
    """

    @property
    def feature_importances_(self) -> NDArray[np.float32]:
        """Feature importance scores (gain-based by default)."""
        ...

    def fit(
        self,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        *,
        verbose: bool = False,
    ) -> XGBRegressorModelProtocol:
        """Fit the regressor on training data.

        Args:
            x_features: Feature matrix, shape (n_samples, n_features).
            y_targets: Continuous target values, shape (n_samples,).
            verbose: Whether to print training progress.

        Returns:
            The fitted regressor (self).
        """
        ...

    def predict(
        self,
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).
        """
        ...

    def get_booster(self) -> XGBBoosterProtocol:
        """Return the underlying Booster object."""
        ...

    def save_model(self, fname: str) -> None:
        """Save the regressor model to file."""
        ...

    def load_model(self, fname: str) -> None:
        """Load a regressor model from file."""
        ...


class XGBRegressorFactory(Protocol):
    """Protocol for XGBRegressor constructor.

    Parallel to XGBClassifierFactory. No scale_pos_weight parameter
    (regression has no class imbalance concept).
    """

    def __call__(
        self,
        *,
        learning_rate: float,
        max_depth: int,
        n_estimators: int,
        subsample: float,
        colsample_bytree: float,
        random_state: int,
        objective: str,
        eval_metric: str,
        n_jobs: int,
        tree_method: str,
        device: str,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
    ) -> XGBRegressorModelProtocol:
        """Construct an XGBRegressor with the given hyperparameters.

        Args:
            learning_rate: Boosting learning rate.
            max_depth: Maximum tree depth.
            n_estimators: Number of boosting rounds.
            subsample: Row subsampling ratio.
            colsample_bytree: Column subsampling ratio.
            random_state: Random seed.
            objective: XGBoost objective function.
            eval_metric: Evaluation metric.
            n_jobs: Number of parallel workers.
            tree_method: Tree construction algorithm.
            device: Device for training ('cpu' or 'cuda').
            reg_alpha: L1 regularization.
            reg_lambda: L2 regularization.

        Returns:
            Configured XGBRegressor instance.
        """
        ...


RegressorBackendName = Literal["xgboost_reg", "lightgbm_reg", "cleargbm_reg", "mlp_reg", "lstm_reg"]


class RegressionMetrics(TypedDict, total=True):
    """Evaluation metrics for regression models.

    All metrics are computed from continuous predictions vs continuous targets.
    No thresholding or class-based logic.

    Args:
        mse: Mean squared error (lower is better).
        rmse: Root mean squared error (lower is better).
        mae: Mean absolute error (lower is better).
        r_squared: Coefficient of determination (higher is better, max 1.0).
        mape: Mean absolute percentage error (lower is better).
    """

    mse: float
    rmse: float
    mae: float
    r_squared: float
    mape: float


class RegressionTrainProgress(TypedDict, total=True):
    """Progress update during regression training.

    Args:
        round: Current training round (1-indexed).
        total_rounds: Total rounds configured.
        train_rmse: Training RMSE at this round.
        val_rmse: Validation RMSE at this round, None if no validation set.
    """

    round: int
    total_rounds: int
    train_rmse: float
    val_rmse: float | None


RegressorTrainConfig = TrainConfig | MLPConfig | LSTMConfig | LightGBMConfig | ClearGBMConfig


class RegressionTrainOutcome(TypedDict, total=True):
    """Complete training outcome for a regression model.

    Parallel to TrainOutcome for classification. Uses RegressionMetrics
    instead of EvalMetrics, best_val_rmse instead of best_val_auc,
    and no scale_pos_weight_computed.

    Args:
        model_path: Path to saved model file.
        model_id: Unique identifier for this trained model.
        samples_total: Total number of samples in dataset.
        samples_train: Number of training samples.
        samples_val: Number of validation samples.
        samples_test: Number of test samples.
        train_metrics: Regression metrics on training set.
        val_metrics: Regression metrics on validation set.
        test_metrics: Regression metrics on test set.
        best_val_rmse: Best validation RMSE achieved (lower is better).
        best_round: Round that achieved best validation RMSE.
        total_rounds: Total training rounds executed.
        early_stopped: Whether training stopped early.
        config: The regressor configuration used for training.
        feature_importances: Feature importances sorted descending.
    """

    model_path: str
    model_id: str
    samples_total: int
    samples_train: int
    samples_val: int
    samples_test: int
    train_metrics: RegressionMetrics
    val_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    best_val_rmse: float
    best_round: int
    total_rounds: int
    early_stopped: bool
    config: RegressorTrainConfig
    feature_importances: list[FeatureImportance]


__all__ = [
    "RegressionMetrics",
    "RegressionTrainOutcome",
    "RegressionTrainProgress",
    "RegressorBackendName",
    "RegressorTrainConfig",
    "XGBRegressorFactory",
    "XGBRegressorModelProtocol",
]
