"""XGBoost regressor backend wrapping the regression trainer API.

Provides a RegressorBackend implementation that defers to
train_regression_model_with_validation. Parallel to backend.py
for classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeGuard

import numpy as np
from numpy.typing import NDArray

from ...metrics import compute_all_regression_metrics
from ...trainer import train_regression_model_with_validation
from ...types import (
    FeatureImportance,
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressorBackendName,
    RegressorTrainConfig,
    TrainConfig,
    XGBRegressorFactory,
    XGBRegressorModelProtocol,
)
from ..protocol import BackendCapabilities
from ..regressor_protocol import (
    PreparedRegressor,
    RegressorBackend,
    RegressorProgressCallback,
)

XGBOOST_REGRESSOR_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": True,
    "model_format": "ubj",
}


class _XGBRegressorPrepared:
    """Loaded XGBoost regressor model for inference.

    Wraps a real XGBRegressorModelProtocol loaded from file.
    """

    def __init__(self, model: XGBRegressorModelProtocol) -> None:
        self._model = model

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).
        """
        return np.asarray(self._model.predict(x), dtype=np.float64)


class XGBoostRegressorBackend:
    """Backend that wraps covenant_ml regression trainer.

    Implements the RegressorBackend protocol for XGBoost regression.
    Delegates training to train_regression_model_with_validation().
    """

    def backend_name(self) -> RegressorBackendName:
        """Return the backend identifier.

        Returns:
            The backend name literal 'xgboost_reg'.
        """
        return "xgboost_reg"

    def capabilities(self) -> BackendCapabilities:
        """Return capability flags for this backend.

        Returns:
            BackendCapabilities describing XGBoost regressor support.
        """
        return XGBOOST_REGRESSOR_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        feature_names: list[str] | None,
    ) -> PreparedRegressor:
        """Prepare is not supported for XGBoost regressor.

        XGBoost uses on-demand training via train(). Use train() to create
        a fitted model, then load() to get a PreparedRegressor for inference.

        Args:
            n_features: Number of input features (unused).
            feature_names: Optional feature names (unused).

        Raises:
            RuntimeError: Always, as prepare is not supported.
        """
        raise RuntimeError(
            "XGBoostRegressorBackend.prepare not supported; use train() then load() for inference."
        )

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str] | None,
        config: RegressorTrainConfig,
        output_dir: Path,
        progress: RegressorProgressCallback | None,
    ) -> RegressionTrainOutcome:
        """Train an XGBoost regressor via the trainer module.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_targets: Continuous target values (n_samples,).
            feature_names: Optional feature names for importances.
            config: Regressor training configuration.
            output_dir: Directory to save model artifacts.
            progress: Optional callback for training progress.

        Returns:
            RegressionTrainOutcome with complete training results.

        Raises:
            RuntimeError: If config is not a TrainConfig (XGBoost).
        """

        def _is_train_config(
            cfg: RegressorTrainConfig,
        ) -> TypeGuard[TrainConfig]:
            return isinstance(cfg, dict) and "n_estimators" in cfg

        if not _is_train_config(config):
            raise RuntimeError("XGBoostRegressorBackend requires TrainConfig")
        cfg = config
        if feature_names is None:
            count = int(x_features.shape[1])
            names = [f"f{i}" for i in range(count)]
        else:
            names = feature_names
        return train_regression_model_with_validation(
            x_features=x_features,
            y_targets=y_targets,
            config=cfg,
            output_dir=output_dir,
            feature_names=names,
            progress_callback=progress,
        )

    def evaluate(
        self,
        *,
        model: PreparedRegressor,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RegressionMetrics:
        """Evaluate a trained regressor on data.

        Args:
            model: A trained regressor.
            x: Feature matrix (n_samples, n_features).
            y: True continuous target values (n_samples,).

        Returns:
            RegressionMetrics with mse, rmse, mae, r_squared, mape.
        """
        preds = model.predict(x)
        return compute_all_regression_metrics(y, preds)

    def save(self, *, model: PreparedRegressor, path: str) -> None:
        """Save is not supported via this method.

        Saving is handled in train() via model.save_model().
        Consumers use RegressionTrainOutcome.model_path.

        Args:
            model: The regressor to save (unused).
            path: File path to save to (unused).

        Raises:
            RuntimeError: Always, as save is handled by train().
        """
        raise RuntimeError(
            "XGBoostRegressorBackend.save not supported; use TrainOutcome.model_path."
        )

    def load(self, *, path: str) -> PreparedRegressor:
        """Load a trained XGBoost regressor from file.

        Args:
            path: Path to the saved model file (.ubj format).

        Returns:
            PreparedRegressor wrapping the loaded XGBRegressor.
        """
        xgb = __import__("xgboost")
        regressor_factory: XGBRegressorFactory = xgb.XGBRegressor
        model = regressor_factory(
            learning_rate=0.1,
            max_depth=3,
            n_estimators=1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=0,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=1,
            tree_method="hist",
            device="cpu",
        )
        model.load_model(path)
        return _XGBRegressorPrepared(model)

    def get_feature_importances(
        self,
        *,
        model: PreparedRegressor,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Feature importances are provided via RegressionTrainOutcome.

        Args:
            model: A trained regressor (unused).
            feature_names: Optional feature names (unused).

        Returns:
            None (importances provided in RegressionTrainOutcome).
        """
        _ = model, feature_names
        return None


def create_xgboost_regressor_backend() -> RegressorBackend:
    """Create an XGBoost regressor backend instance.

    Returns:
        A new XGBoostRegressorBackend.
    """
    return XGBoostRegressorBackend()


__all__ = [
    "XGBOOST_REGRESSOR_CAPABILITIES",
    "XGBoostRegressorBackend",
    "create_xgboost_regressor_backend",
]
