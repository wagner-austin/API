"""ClearGBM backend for tabular regression.

Implements RegressorBackend protocol on cleargbm's native squared-error
objective (single-call Rust training loop). Parallel to
:class:`covenant_ml.backends.cleargbm.backend.ClearGBMBackend` for
classification, with the key regression differences:

- objective="squared_error" and no class weight (regression has no classes)
- predictions come from ``predict_raw`` — under squared error the raw score
  IS the prediction
- random splits (regression — no stratification), early stopping on
  validation MSE inside the Rust core

Strict typing only: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from cleargbm.ensemble import (
    predict_raw as cgbm_predict_raw,
)
from cleargbm.ensemble import (
    train_gradient_boosting_regression,
)
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.metrics_regression import compute_all_regression_metrics
from covenant_ml.types import FeatureImportance
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorBackendName,
    RegressorTrainConfig,
)

from ...optimizer.search_spaces import make_cleargbm_default_space, make_cleargbm_focused_space
from ...optimizer.types import SampledFloatParams, SampledIntParams, SearchSpace
from ...trainer import regression_split
from ..protocol import BackendCapabilities
from ..regressor_protocol import (
    PreparedRegressor,
    RegressorBackend,
    RegressorProgressCallback,
)
from .backend import (
    _py_gbm_model_feature_importances,
    _py_gbm_model_from_json,
    _py_gbm_model_n_trees,
    _py_gbm_model_to_json,
    _PyGbmModelProto,
)
from .config_resolution import (
    _is_cleargbm_regressor_config,
    _resolve_max_features,
    _resolve_monotonic_constraints,
)

_log = get_logger(__name__)


CLEARGBM_REGRESSOR_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": False,
    "supports_early_stopping": True,
    "supports_feature_importance": True,
    "model_format": "json",
}


class _ClearGBMRegressorPrepared:
    """Prepared ClearGBM native model for regression inference.

    Wraps the opaque ``cleargbm_rs.PyGbmModel`` handle produced by the native
    regression training loop. Predictions are the raw scores — the identity
    transform is the squared-error objective's prediction function.

    Args:
        model: Trained native model handle.
    """

    def __init__(self, model: _PyGbmModelProto) -> None:
        """Initialize with a trained native model handle.

        Args:
            model: Opaque ``PyGbmModel`` from
                ``train_gradient_boosting_regression``.
        """
        self._model = model

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).
        """
        return np.asarray(cgbm_predict_raw(self._model, x), dtype=np.float64)

    @property
    def model(self) -> _PyGbmModelProto:
        """Get the underlying native model handle."""
        return self._model


class ClearGBMRegressorBackend:
    """ClearGBM backend for tabular regression.

    Implements RegressorBackend protocol on the native squared-error
    objective. The saved artifact is the same self-describing model JSON the
    classifier backend writes — its embedded config carries
    ``objective: "squared_error"``.
    """

    def backend_name(self) -> RegressorBackendName:
        """Return the backend identifier.

        Returns:
            The backend name literal 'cleargbm_reg'.
        """
        return "cleargbm_reg"

    def capabilities(self) -> BackendCapabilities:
        """Return capability flags for this backend.

        Returns:
            BackendCapabilities describing ClearGBM regressor support.
        """
        return CLEARGBM_REGRESSOR_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        feature_names: list[str] | None,
    ) -> PreparedRegressor:
        """Prepare is not supported for the ClearGBM regressor.

        ClearGBM uses on-demand training via train(). Use train() to create
        a fitted model, then load() to get a PreparedRegressor for inference.

        Args:
            n_features: Number of input features (unused).
            feature_names: Optional feature names (unused).

        Raises:
            RuntimeError: Always, as prepare is not supported.
        """
        raise RuntimeError(
            "ClearGBMRegressorBackend.prepare not supported; use train() then load()."
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
        """Train a ClearGBM regressor on tabular data.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_targets: Continuous target values (n_samples,).
            feature_names: Optional feature names for importances.
            config: Regressor training configuration (must be ClearGBMConfig).
            output_dir: Directory to save model artifacts.
            progress: Optional callback for training progress.

        Returns:
            RegressionTrainOutcome with complete training results.

        Raises:
            RuntimeError: If config is not a ClearGBMConfig.
        """
        if not _is_cleargbm_regressor_config(config):
            raise RuntimeError("ClearGBMRegressorBackend requires ClearGBMConfig")

        cfg = config

        # Random split (regression — no stratification)
        splits = regression_split(
            x_features,
            y_targets,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            random_state=cfg["random_state"],
        )

        # Resolve feature names
        n_feats = int(x_features.shape[1])
        if feature_names is None:
            resolved_names = tuple(f"f{i}" for i in range(n_feats))
        else:
            resolved_names = tuple(feature_names)

        # Build the cleargbm config under the squared-error pairing:
        # objective states the loss, and scale_pos_weight is None — the
        # boundary rejects a weight here, regression has no positive class.
        gbm_config: GradientBoostingConfig = GradientBoostingConfig(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=_resolve_max_features(cfg["max_features"], n_feats),
            colsample_bytree=cfg["colsample_bytree"],
            max_bins=cfg["max_bins"],
            subsample=cfg["subsample"],
            random_state=cfg["random_state"],
            monotonic_constraints=_resolve_monotonic_constraints(
                cfg["monotonic_constraints"], resolved_names
            ),
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            n_jobs=cfg["n_jobs"],
            early_stopping_rounds=cfg["early_stopping_rounds"],
            growth_strategy=cfg["growth_strategy"],
            num_leaves=cfg["num_leaves"],
            objective="squared_error",
            scale_pos_weight=None,
        )

        _log.info(
            "Starting ClearGBM regression training (native Rust loop)",
            extra={
                "n_estimators": cfg["n_estimators"],
                "max_depth": cfg["max_depth"],
                "learning_rate": cfg["learning_rate"],
                "n_train": splits.n_train,
                "n_val": splits.n_val,
            },
        )

        # The native loop does not accept a Python progress callback; early
        # stopping runs inside the Rust core on validation MSE, and the
        # returned model is already trimmed to the best-round ensemble.
        if progress is not None:
            _log.info(
                "ClearGBM native path does not emit per-round progress; "
                "callback receives only the final summary"
            )
        model = train_gradient_boosting_regression(
            x_train=splits.x_train,
            y_train=splits.y_train,
            x_val=splits.x_val,
            y_val=splits.y_val,
            config=gbm_config,
            feature_names=resolved_names,
        )

        prepared = _ClearGBMRegressorPrepared(model)

        # Compute regression metrics on all splits
        train_metrics = compute_all_regression_metrics(
            splits.y_train, prepared.predict(splits.x_train)
        )
        val_metrics = compute_all_regression_metrics(splits.y_val, prepared.predict(splits.x_val))
        test_metrics = compute_all_regression_metrics(
            splits.y_test, prepared.predict(splits.x_test)
        )

        surviving_trees = _py_gbm_model_n_trees(model)
        early_stopped = surviving_trees < cfg["n_estimators"]

        _log.info(
            "ClearGBM regression training complete",
            extra={
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_metrics["rmse"],
                "test_rmse": test_metrics["rmse"],
                "surviving_trees": surviving_trees,
                "early_stopped": early_stopped,
            },
        )

        if progress is not None:
            prog: RegressionTrainProgress = {
                "round": surviving_trees,
                "total_rounds": cfg["n_estimators"],
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_metrics["rmse"],
            }
            progress(prog)

        # Feature importances from the native model
        native_importances = _py_gbm_model_feature_importances(model)
        feature_importance_list: list[FeatureImportance] = []
        for rank, (imp_name, imp_value) in enumerate(native_importances, start=1):
            feature_importance_list.append(
                FeatureImportance(
                    name=imp_name,
                    importance=imp_value,
                    rank=rank,
                )
            )

        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_id = str(uuid.uuid4())
        model_path = output_dir / f"cleargbm_regressor_{model_id}.json"
        self.save(model=prepared, path=str(model_path))

        return RegressionTrainOutcome(
            model_path=str(model_path),
            model_id=model_id,
            samples_total=splits.n_total,
            samples_train=splits.n_train,
            samples_val=splits.n_val,
            samples_test=splits.n_test,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_val_rmse=val_metrics["rmse"],
            best_round=surviving_trees,
            total_rounds=cfg["n_estimators"],
            early_stopped=early_stopped,
            config=cfg,
            feature_importances=feature_importance_list,
        )

    def evaluate(
        self,
        *,
        model: PreparedRegressor,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RegressionMetrics:
        """Evaluate a ClearGBM regressor model.

        Args:
            model: A trained regressor (PreparedRegressor).
            x: Feature matrix (n_samples, n_features).
            y: True continuous target values (n_samples,).

        Returns:
            RegressionMetrics with mse, rmse, mae, r_squared, mape.
        """
        preds = model.predict(x)
        return compute_all_regression_metrics(y, preds)

    def save(self, *, model: PreparedRegressor, path: str) -> None:
        """Save a ClearGBM native regression model to a JSON file.

        Args:
            model: The regressor to save (must be _ClearGBMRegressorPrepared).
            path: Output path.

        Raises:
            RuntimeError: If model is not _ClearGBMRegressorPrepared.
        """
        if not isinstance(model, _ClearGBMRegressorPrepared):
            raise RuntimeError("Model must be _ClearGBMRegressorPrepared")
        json_str = _py_gbm_model_to_json(model.model)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)

    def load(self, *, path: str) -> PreparedRegressor:
        """Load a ClearGBM native regression model from a JSON file.

        Args:
            path: Path to the saved model file.

        Returns:
            PreparedRegressor wrapping the deserialized native model.
        """
        with open(path, encoding="utf-8") as f:
            raw_str = f.read()
        native_model = _py_gbm_model_from_json(raw_str)
        return _ClearGBMRegressorPrepared(native_model)

    def get_feature_importances(
        self,
        *,
        model: PreparedRegressor,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Get feature importances from the native ClearGBM model.

        Args:
            model: A trained regressor.
            feature_names: Feature names (unused; extracted from the native
                model).

        Returns:
            List of feature importances in feature-index order, or None if
            the prepared regressor is not a ClearGBM instance.
        """
        if not isinstance(model, _ClearGBMRegressorPrepared):
            return None

        native_importances = _py_gbm_model_feature_importances(model.model)
        result: list[FeatureImportance] = []
        for rank, (imp_name, imp_value) in enumerate(native_importances, start=1):
            result.append(
                FeatureImportance(
                    name=imp_name,
                    importance=imp_value,
                    rank=rank,
                )
            )
        return result

    def get_default_search_space(self) -> SearchSpace:
        """Return default ClearGBM search space.

        Returns:
            ClearGBMSearchSpace with sensible default ranges.
        """
        return make_cleargbm_default_space()

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Return focused ClearGBM search space around prior best params.

        Args:
            best_int_params: Best integer params (reads max_depth).
            best_float_params: Best float params (reads learning_rate).

        Returns:
            ClearGBMSearchSpace with narrowed ranges.
        """
        return make_cleargbm_focused_space(
            best_max_depth=best_int_params["max_depth"],
            best_learning_rate=best_float_params["learning_rate"],
        )


def create_cleargbm_regressor_backend() -> RegressorBackend:
    """Create a ClearGBM regressor backend instance.

    Returns:
        A new ClearGBMRegressorBackend.
    """
    return ClearGBMRegressorBackend()


__all__ = [
    "CLEARGBM_REGRESSOR_CAPABILITIES",
    "ClearGBMRegressorBackend",
    "create_cleargbm_regressor_backend",
]
