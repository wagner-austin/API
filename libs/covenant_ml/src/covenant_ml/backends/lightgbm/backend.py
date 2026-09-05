"""LightGBM backend for tabular binary classification.

Implements ClassifierBackend protocol using LightGBM with:
- Stratified train/val/test splits
- Early stopping on validation AUC
- GPU support when available
- Feature importance extraction
- Strict typing (no Any, no casts, no type: ignore)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeGuard

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ...metrics import compute_all_metrics
from ...optimizer.search_spaces import make_lightgbm_default_space, make_lightgbm_focused_space
from ...optimizer.types import SampledFloatParams, SampledIntParams, SearchSpace
from ...trainer import stratified_split
from ...types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    LightGBMConfig,
    TrainOutcome,
    TrainProgress,
)
from ..protocol import BackendCapabilities, ClassifierBackend, PreparedClassifier

_log = get_logger(__name__)


def _is_lightgbm_config(cfg: ClassifierTrainConfig) -> TypeGuard[LightGBMConfig]:
    """Check if config is LightGBMConfig by looking for LightGBM-specific keys."""
    return (
        isinstance(cfg, dict)
        and "num_leaves" in cfg
        and "min_child_samples" in cfg
        and "n_estimators" in cfg
    )


# Protocols for LightGBM types
class _LGBModelProtocol(Protocol):
    """Protocol for LightGBM classifier."""

    @property
    def feature_importances_(self) -> NDArray[np.float64]: ...

    @property
    def booster_(self) -> _BoosterProtocol: ...

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        *,
        eval_set: list[tuple[NDArray[np.float64], NDArray[np.int64]]],
        callbacks: list[_EarlyStoppingCallback] | None = None,
        feature_name: list[str] | None = None,
    ) -> _LGBModelProtocol: ...

    def predict_proba(self, x_data: NDArray[np.float64]) -> NDArray[np.float64]: ...


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


class _LGBMClassifierCtor(Protocol):
    """Protocol for LGBMClassifier constructor."""

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
        class_weight: str | dict[int, float] | None,
        verbose: int,
    ) -> _LGBModelProtocol: ...


class _LGBMBoosterPrepared:
    """Prepared LightGBM model for inference (loaded from Booster file).

    LightGBM Booster returns probabilities directly for binary classification
    when the model was trained with binary objective.
    """

    def __init__(self, booster: _BoosterProtocol) -> None:
        self._booster = booster

    @property
    def raw_model(self) -> _BoosterProtocol:
        """Return the underlying LightGBM Booster.

        Needed by SHAP TreeExplainer, which reads the native tree structure
        and rejects wrapper objects. Mirrors _LGBMRegressorPrepared.

        Returns:
            The raw LightGBM Booster.
        """
        return self._booster

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)].
        """
        # Booster.predict returns P(class=1) directly for binary classification
        pos_proba: NDArray[np.float64] = np.asarray(self._booster.predict(x), dtype=np.float64)
        neg_proba: NDArray[np.float64] = 1.0 - pos_proba
        result: NDArray[np.float64] = np.column_stack([neg_proba, pos_proba])
        return result


LIGHTGBM_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": True,
    "supports_early_stopping": True,
    "supports_feature_importance": True,
    "model_format": "txt",
}


def _get_lightgbm_imports() -> tuple[_LGBMClassifierCtor, _EarlyStoppingCallbackFactory]:
    """Get LightGBM constructor and callbacks via dynamic import."""
    lgb_module = __import__("lightgbm", fromlist=["LGBMClassifier", "early_stopping"])
    classifier_ctor: _LGBMClassifierCtor = lgb_module.LGBMClassifier
    early_stopping: _EarlyStoppingCallbackFactory = lgb_module.early_stopping
    return classifier_ctor, early_stopping


def _resolve_device(requested: str, *, platform: str | None = None) -> str:
    """Resolve device preference for LightGBM.

    LightGBM has three device modes:
    - "cpu": CPU-only training
    - "gpu": OpenCL-based GPU training (works on Windows, Linux, macOS)
    - "cuda": CUDA Tree Learner (Linux-only, requires CUDA build)

    Since CUDA is not supported on Windows, this function maps "cuda" to "gpu"
    (OpenCL) on Windows to provide transparent GPU acceleration.

    Args:
        requested: Device preference ("cpu", "cuda", or "auto").
        platform: Override for sys.platform (for testing). If None, uses sys.platform.

    Returns:
        Resolved device string for LightGBM ("cpu", "gpu", or "cuda").
    """
    import sys

    actual_platform = platform if platform is not None else sys.platform

    if requested == "auto":
        return "cpu"
    if requested == "cuda" and actual_platform == "win32":
        _log.info(
            "LightGBM CUDA not supported on Windows, using OpenCL GPU instead",
            extra={"requested_device": requested, "resolved_device": "gpu"},
        )
        return "gpu"
    if requested == "cuda":
        return "cuda"
    return "cpu"


def _compute_class_weight(y_train: NDArray[np.int64]) -> float:
    """Compute scale_pos_weight from training labels."""
    pos_mask: NDArray[np.bool_] = y_train == 1
    neg_mask: NDArray[np.bool_] = y_train == 0
    n_positive = int(np.count_nonzero(pos_mask))
    n_negative = int(np.count_nonzero(neg_mask))
    if n_positive == 0:
        raise ValueError("Training set has no positive samples")
    computed = float(n_negative) / float(n_positive)
    _log.info(
        "Auto-calculated scale_pos_weight for LightGBM",
        extra={
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": computed,
        },
    )
    return computed


class LightGBMBackend(ClassifierBackend):
    """LightGBM backend for tabular binary classification."""

    def backend_name(self) -> BackendName:
        return "lightgbm"

    def capabilities(self) -> BackendCapabilities:
        return LIGHTGBM_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        """Prepare is not supported for LightGBM.

        LightGBM uses on-demand training via train(). Use train() to create
        a fitted model, then load() to get a PreparedClassifier for inference.

        Raises:
            RuntimeError: Always, as prepare is not supported.
        """
        raise RuntimeError(
            "LightGBMBackend.prepare not supported; use train() then load() for inference."
        )

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: Callable[[TrainProgress], None] | None,
        groups: NDArray[np.int64] | None = None,
    ) -> TrainOutcome:
        """Train LightGBM model on tabular data."""
        if not _is_lightgbm_config(config):
            raise RuntimeError("LightGBMBackend requires LightGBMConfig")

        cfg = config
        device = _resolve_device(cfg["device"])

        # Stratified split
        splits = stratified_split(
            x_features,
            y_labels,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            random_state=cfg["random_state"],
            groups=groups,
        )

        # Compute class weights for imbalanced data
        scale_pos_weight = _compute_class_weight(splits.y_train)

        # Get LightGBM imports
        lgbm_ctor, early_stopping = _get_lightgbm_imports()

        # Build model
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
            objective="binary",
            metric="auc",
            class_weight={0: 1.0, 1: scale_pos_weight},
            verbose=-1,
        )

        # Early stopping callback
        early_stop_cb = early_stopping(
            stopping_rounds=cfg["early_stopping_rounds"],
            verbose=False,
        )

        # Resolve feature names for training (used for both fit and importance reporting)
        n_feats = int(x_features.shape[1])
        if feature_names is None:
            resolved_names = [f"f{i}" for i in range(n_feats)]
        else:
            resolved_names = feature_names

        # Train with validation
        eval_set_data: list[tuple[NDArray[np.float64], NDArray[np.int64]]] = [
            (splits.x_val, splits.y_val)
        ]

        model.fit(
            splits.x_train,
            splits.y_train,
            eval_set=eval_set_data,
            callbacks=[early_stop_cb],
            feature_name=resolved_names,
        )

        # Get best iteration from model (set by early_stopping callback)
        # LightGBM stores best_iteration on the booster after fitting
        # best_iteration is 0 if no early stopping, otherwise the best round
        booster = model.booster_
        best_iter: int = booster.best_iteration
        best_round: int = best_iter if best_iter > 0 else cfg["n_estimators"]

        # Compute predictions and metrics using Booster directly
        # (avoids sklearn feature name validation warnings)
        # Booster.predict returns P(class=1) for binary classification
        train_proba: NDArray[np.float64] = np.asarray(
            booster.predict(splits.x_train), dtype=np.float64
        )
        val_proba: NDArray[np.float64] = np.asarray(booster.predict(splits.x_val), dtype=np.float64)
        test_proba: NDArray[np.float64] = np.asarray(
            booster.predict(splits.x_test), dtype=np.float64
        )

        train_metrics = compute_all_metrics(splits.y_train, train_proba)
        val_metrics = compute_all_metrics(splits.y_val, val_proba)
        test_metrics = compute_all_metrics(splits.y_test, test_proba)

        # Report progress if callback provided
        if progress is not None:
            prog: TrainProgress = {
                "round": best_round,
                "total_rounds": cfg["n_estimators"],
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["auc"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
            }
            progress(prog)

        # Feature importances (use resolved_names from above)
        importances_arr: NDArray[np.float64] = np.asarray(
            model.feature_importances_, dtype=np.float64
        )

        # Sort by importance - use explicit indexing to avoid Any from iteration
        sorted_indices: NDArray[np.int64] = np.argsort(importances_arr)[::-1].astype(np.int64)
        n_sorted: int = int(sorted_indices.shape[0])
        feature_importances: list[FeatureImportance] = []
        for rank_idx in range(n_sorted):
            # Use slice indexing + flat to get typed int (avoids Any from scalar indexing)
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
        model_path = output_dir / "lightgbm_model.txt"
        booster = model.booster_
        booster.save_model(str(model_path))

        return TrainOutcome(
            model_path=str(model_path),
            model_id="lightgbm",
            samples_total=splits.n_total,
            samples_train=splits.n_train,
            samples_val=splits.n_val,
            samples_test=splits.n_test,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_val_auc=val_metrics["auc"],
            best_round=best_round,
            total_rounds=cfg["n_estimators"],
            early_stopped=best_round < cfg["n_estimators"],
            config=cfg,
            feature_importances=feature_importances,
            scale_pos_weight_computed=scale_pos_weight,
        )

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Evaluate LightGBM model."""
        proba = model.predict_proba(x)
        return compute_all_metrics(y, proba[:, 1])

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        raise RuntimeError("LightGBMBackend.save not supported; use TrainOutcome.model_path.")

    def load(self, *, path: str) -> PreparedClassifier:
        """Load LightGBM model from file.

        Args:
            path: Path to the saved model file (.txt format).

        Returns:
            PreparedClassifier wrapping the loaded Booster.
        """
        lgb_module = __import__("lightgbm", fromlist=["Booster"])
        booster_ctor: _BoosterFactory = lgb_module.Booster
        booster = booster_ctor(model_file=path)
        return _LGBMBoosterPrepared(booster)

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        # Importances are provided via TrainOutcome
        return None

    def get_default_search_space(self) -> SearchSpace:
        """Return default LightGBM search space with DART support.

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


def create_lightgbm_backend() -> LightGBMBackend:
    """Create LightGBM backend instance."""
    return LightGBMBackend()


__all__ = [
    "LIGHTGBM_CAPABILITIES",
    "LightGBMBackend",
    "create_lightgbm_backend",
]
