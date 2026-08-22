"""LightGBM backend for tabular regression.

Implements RegressorBackend protocol using LightGBM with:
- Random train/val/test splits (regression — no stratification)
- Early stopping on validation RMSE (lower is better)
- GPU support when available
- Feature importance extraction
- Strict typing (no Any, no casts, no type: ignore)
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeGuard

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from covenant_ml.metrics_regression import compute_all_regression_metrics
from covenant_ml.types import (
    FeatureImportance,
    LightGBMConfig,
)
from covenant_ml.types_regression import (
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressionTrainProgress,
    RegressorBackendName,
    RegressorTrainConfig,
)

from ...optimizer.search_spaces import make_lightgbm_default_space, make_lightgbm_focused_space
from ...optimizer.types import SampledFloatParams, SampledIntParams, SearchSpace
from ...trainer import regression_split
from ..protocol import BackendCapabilities
from ..regressor_protocol import (
    PreparedRegressor,
    RegressorBackend,
    RegressorProgressCallback,
)
from .backend import _resolve_device

_log = get_logger(__name__)


def _is_lightgbm_config(cfg: RegressorTrainConfig) -> TypeGuard[LightGBMConfig]:
    """Check if config is LightGBMConfig by looking for LightGBM-specific keys.

    Args:
        cfg: Regressor training configuration to check.

    Returns:
        True if the config is a LightGBMConfig.
    """
    return (
        isinstance(cfg, dict)
        and "num_leaves" in cfg
        and "min_child_samples" in cfg
        and "n_estimators" in cfg
    )


# ---------------------------------------------------------------------------
# Protocols for LightGBM regressor types
# ---------------------------------------------------------------------------


class _LGBRegressorModelProtocol(Protocol):
    """Protocol for LightGBM regressor model."""

    @property
    def feature_importances_(self) -> NDArray[np.float64]: ...

    @property
    def booster_(self) -> _BoosterProtocol: ...

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        *,
        eval_set: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
        callbacks: list[_EarlyStoppingCallback] | None = None,
        feature_name: list[str] | None = None,
    ) -> _LGBRegressorModelProtocol: ...

    def predict(self, x_data: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _BoosterProtocol(Protocol):
    """Protocol for LightGBM Booster."""

    @property
    def best_iteration(self) -> int: ...

    def save_model(self, filename: str) -> None: ...

    def predict(self, data: NDArray[np.float64]) -> NDArray[np.float64]: ...


class _EarlyStoppingCallback(Protocol):
    """Protocol for early stopping callback."""

    stopping_round: int


class _EarlyStoppingCallbackFactory(Protocol):
    """Protocol for early_stopping callback constructor."""

    def __call__(self, stopping_rounds: int, verbose: bool = ...) -> _EarlyStoppingCallback: ...


class _BoosterFactory(Protocol):
    """Protocol for Booster constructor (loads from file)."""

    def __call__(self, *, model_file: str) -> _BoosterProtocol: ...


class _LGBMRegressorCtor(Protocol):
    """Protocol for LGBMRegressor constructor."""

    def __call__(
        self,
        *,
        boosting_type: str,
        num_leaves: int,
        max_depth: int,
        learning_rate: float,
        n_estimators: int,
        subsample: float,
        colsample_bytree: float,
        reg_alpha: float,
        reg_lambda: float,
        min_child_samples: int,
        random_state: int,
        n_jobs: int,
        device: str,
        objective: str,
        metric: str,
        verbose: int,
    ) -> _LGBRegressorModelProtocol: ...


# ---------------------------------------------------------------------------
# Prepared model wrappers
# ---------------------------------------------------------------------------


class _LGBMRegressorPrepared:
    """Prepared LightGBM model for regression inference (loaded from Booster file).

    LightGBM Booster.predict returns raw predictions for regression models.
    Exposes raw_model for SHAP TreeExplainer compatibility.
    """

    def __init__(self, booster: _BoosterProtocol) -> None:
        self._booster = booster

    @property
    def raw_model(self) -> _BoosterProtocol:
        """Return the underlying LightGBM Booster.

        Needed by SHAP TreeExplainer which requires raw tree models,
        not wrapper objects.

        Returns:
            The raw LightGBM Booster.
        """
        return self._booster

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict continuous values.

        Args:
            x: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).
        """
        return np.asarray(self._booster.predict(x), dtype=np.float64)


LIGHTGBM_REGRESSOR_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": True,
    "model_format": "txt",
}


def _get_lightgbm_regressor_imports() -> tuple[_LGBMRegressorCtor, _EarlyStoppingCallbackFactory]:
    """Get LightGBM regressor constructor and callbacks via dynamic import.

    Returns:
        Tuple of (LGBMRegressor constructor, early_stopping factory).
    """
    lgb_module = __import__("lightgbm", fromlist=["LGBMRegressor", "early_stopping"])
    regressor_ctor: _LGBMRegressorCtor = lgb_module.LGBMRegressor
    early_stopping: _EarlyStoppingCallbackFactory = lgb_module.early_stopping
    return regressor_ctor, early_stopping


class LightGBMRegressorBackend:
    """LightGBM backend for tabular regression.

    Implements RegressorBackend protocol. Parallel to LightGBMBackend
    for classification with key differences:
    - objective="regression", metric="rmse" (not "binary"/"auc")
    - No class_weight (regression has no class imbalance)
    - predict() returns 1D continuous values (not probabilities)
    - Early stopping on validation RMSE (lower is better)
    - Uses regression_split (random, not stratified)
    """

    def backend_name(self) -> RegressorBackendName:
        """Return the backend identifier.

        Returns:
            The backend name literal 'lightgbm_reg'.
        """
        return "lightgbm_reg"

    def capabilities(self) -> BackendCapabilities:
        """Return capability flags for this backend.

        Returns:
            BackendCapabilities describing LightGBM regressor support.
        """
        return LIGHTGBM_REGRESSOR_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        feature_names: list[str] | None,
    ) -> PreparedRegressor:
        """Prepare is not supported for LightGBM regressor.

        LightGBM uses on-demand training via train(). Use train() to create
        a fitted model, then load() to get a PreparedRegressor for inference.

        Args:
            n_features: Number of input features (unused).
            feature_names: Optional feature names (unused).

        Raises:
            RuntimeError: Always, as prepare is not supported.
        """
        raise RuntimeError(
            "LightGBMRegressorBackend.prepare not supported; use train() then load() for inference."
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
        """Train LightGBM regressor on tabular data.

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
            RuntimeError: If config is not a LightGBMConfig.
        """
        if not _is_lightgbm_config(config):
            raise RuntimeError("LightGBMRegressorBackend requires LightGBMConfig")

        cfg = config
        device = _resolve_device(cfg["device"])

        # Random split (regression — no stratification)
        splits = regression_split(
            x_features,
            y_targets,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            random_state=cfg["random_state"],
        )

        # Get LightGBM imports
        lgbm_ctor, early_stopping = _get_lightgbm_regressor_imports()

        # Build regressor model — no class_weight for regression
        model = lgbm_ctor(
            boosting_type="gbdt",
            num_leaves=cfg["num_leaves"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            n_estimators=cfg["n_estimators"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            min_child_samples=cfg["min_child_samples"],
            random_state=cfg["random_state"],
            n_jobs=-1,
            device=device,
            objective="regression",
            metric="rmse",
            verbose=-1,
        )

        # Early stopping callback
        early_stop_cb = early_stopping(
            stopping_rounds=cfg["early_stopping_rounds"],
            verbose=False,
        )

        # Resolve feature names
        n_feats = int(x_features.shape[1])
        if feature_names is None:
            resolved_names = [f"f{i}" for i in range(n_feats)]
        else:
            resolved_names = feature_names

        # Train with validation — eval_set uses float64 targets
        eval_set_data: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = [
            (splits.x_val, splits.y_val)
        ]

        model.fit(
            splits.x_train,
            splits.y_train,
            eval_set=eval_set_data,
            callbacks=[early_stop_cb],
            feature_name=resolved_names,
        )

        # Get best iteration from booster
        booster = model.booster_
        best_iter: int = booster.best_iteration
        best_round: int = best_iter if best_iter > 0 else cfg["n_estimators"]

        # Compute predictions using Booster directly (avoids sklearn warnings)
        train_preds: NDArray[np.float64] = np.asarray(
            booster.predict(splits.x_train), dtype=np.float64
        )
        val_preds: NDArray[np.float64] = np.asarray(booster.predict(splits.x_val), dtype=np.float64)
        test_preds: NDArray[np.float64] = np.asarray(
            booster.predict(splits.x_test), dtype=np.float64
        )

        # Compute regression metrics
        train_metrics = compute_all_regression_metrics(splits.y_train, train_preds)
        val_metrics = compute_all_regression_metrics(splits.y_val, val_preds)
        test_metrics = compute_all_regression_metrics(splits.y_test, test_preds)

        # Report progress if callback provided
        if progress is not None:
            prog: RegressionTrainProgress = {
                "round": best_round,
                "total_rounds": cfg["n_estimators"],
                "train_rmse": train_metrics["rmse"],
                "val_rmse": val_metrics["rmse"],
            }
            progress(prog)

        # Feature importances
        importances_arr: NDArray[np.float64] = np.asarray(
            model.feature_importances_, dtype=np.float64
        )

        # Sort by importance — use explicit indexing to avoid Any from iteration
        sorted_indices: NDArray[np.int64] = np.argsort(importances_arr)[::-1].astype(np.int64)
        n_sorted: int = int(sorted_indices.shape[0])
        feature_importances: list[FeatureImportance] = []
        for rank_idx in range(n_sorted):
            feat_idx_flat = np.asarray(sorted_indices[rank_idx : rank_idx + 1], dtype=np.int64).flat
            feat_idx: int = int(feat_idx_flat[0])
            imp_flat = np.asarray(importances_arr[feat_idx : feat_idx + 1], dtype=np.float64).flat
            importance: float = float(imp_flat[0])
            feature_importances.append(
                {
                    "name": resolved_names[feat_idx],
                    "importance": importance,
                    "rank": rank_idx + 1,
                }
            )

        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "lightgbm_regressor_model.txt"
        booster.save_model(str(model_path))

        return RegressionTrainOutcome(
            model_path=str(model_path),
            model_id="lightgbm_reg",
            samples_total=splits.n_total,
            samples_train=splits.n_train,
            samples_val=splits.n_val,
            samples_test=splits.n_test,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_val_rmse=val_metrics["rmse"],
            best_round=best_round,
            total_rounds=cfg["n_estimators"],
            early_stopped=best_round < cfg["n_estimators"],
            config=cfg,
            feature_importances=feature_importances,
        )

    def evaluate(
        self,
        *,
        model: PreparedRegressor,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RegressionMetrics:
        """Evaluate LightGBM regressor model.

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
        """Save is not supported via this method.

        Saving is handled in train() via booster.save_model().
        Consumers use RegressionTrainOutcome.model_path.

        Args:
            model: The regressor to save (unused).
            path: File path to save to (unused).

        Raises:
            RuntimeError: Always, as save is handled by train().
        """
        raise RuntimeError(
            "LightGBMRegressorBackend.save not supported; use TrainOutcome.model_path."
        )

    def load(self, *, path: str) -> PreparedRegressor:
        """Load a trained LightGBM regressor from file.

        Args:
            path: Path to the saved model file (.txt format).

        Returns:
            PreparedRegressor wrapping the loaded Booster.
        """
        lgb_module = __import__("lightgbm", fromlist=["Booster"])
        booster_ctor: _BoosterFactory = lgb_module.Booster
        booster = booster_ctor(model_file=path)
        return _LGBMRegressorPrepared(booster)

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

    def get_default_search_space(self) -> SearchSpace:
        """Return default LightGBM search space for regression.

        Returns:
            LightGBMSearchSpace with sensible default ranges.
        """
        return make_lightgbm_default_space()

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Return focused LightGBM search space around prior best params.

        Args:
            best_int_params: Best integer params (reads num_leaves).
            best_float_params: Best float params (reads learning_rate).

        Returns:
            LightGBMSearchSpace with narrowed ranges.
        """
        return make_lightgbm_focused_space(
            best_num_leaves=best_int_params["num_leaves"],
            best_learning_rate=best_float_params["learning_rate"],
        )


def create_lightgbm_regressor_backend() -> RegressorBackend:
    """Create LightGBM regressor backend instance.

    Returns:
        A new LightGBMRegressorBackend.
    """
    return LightGBMRegressorBackend()


__all__ = [
    "LIGHTGBM_REGRESSOR_CAPABILITIES",
    "LightGBMRegressorBackend",
    "create_lightgbm_regressor_backend",
]
