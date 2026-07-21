"""ClearGBM backend for tabular binary classification.

Implements ClassifierBackend protocol using the ClearGBM library on the native
Rust training path (single-call training loop, no per-tree FFI overhead).

Strict typing only: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import types
import uuid
from pathlib import Path
from typing import Protocol, TypeGuard

import numpy as np
from cleargbm.ensemble import (
    predict_proba as cgbm_predict_proba,
    train_gradient_boosting,
)
from cleargbm.types import GradientBoostingConfig
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ...metrics import compute_all_metrics
from ...optimizer.search_spaces import make_cleargbm_default_space, make_cleargbm_focused_space
from ...optimizer.types import SampledFloatParams, SampledIntParams, SearchSpace
from ...trainer import stratified_split
from ...types import (
    BackendName,
    ClassifierTrainConfig,
    ClearGBMConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
)
from ..protocol import BackendCapabilities, ClassifierBackend, PreparedClassifier, ProgressCallback

_log = get_logger(__name__)


# =============================================================================
# Native model access — Protocol-typed getattr onto the Rust extension
# =============================================================================
#
# The cleargbm_rs Python package re-exports Rust-backed functions as stubs that
# raise ImportError at the top-level; the real implementations live in the
# .pyd submodule at ``cleargbm_rs.cleargbm_rs``. We import that submodule
# directly and pin each function to a Protocol type so mypy sees precise
# signatures instead of Any leaking from the dynamic __import__.


class _PyGbmModelProto(Protocol):
    """Opaque model handle produced by the native training loop."""

    ...


class _ToJsonProto(Protocol):
    """Signature of ``cleargbm_rs.py_gbm_model_to_json_rs``."""

    def __call__(self, model: _PyGbmModelProto) -> str:
        """Serialize a native model to JSON.

        Args:
            model: Trained native model handle.

        Returns:
            JSON representation.
        """
        ...


class _FromJsonProto(Protocol):
    """Signature of ``cleargbm_rs.py_gbm_model_from_json_rs``."""

    def __call__(self, json_str: str) -> _PyGbmModelProto:
        """Deserialize a native model from JSON.

        Args:
            json_str: JSON previously produced by the paired to-JSON function.

        Returns:
            Native model handle.
        """
        ...


class _FeatureImportancesProto(Protocol):
    """Signature of ``cleargbm_rs.py_gbm_model_feature_importances_rs``."""

    def __call__(self, model: _PyGbmModelProto) -> list[tuple[str, float]]:
        """Return split-count feature importances.

        Args:
            model: Trained native model handle.

        Returns:
            List of ``(feature_name, importance)`` pairs in feature-index order,
            normalized to sum to 1.0 when at least one internal split exists.
        """
        ...


class _NTreesProto(Protocol):
    """Signature of ``cleargbm_rs.py_gbm_model_n_trees_rs``."""

    def __call__(self, model: _PyGbmModelProto) -> int:
        """Return the trained tree count.

        Args:
            model: Trained native model handle.

        Returns:
            Number of trees kept in the ensemble.
        """
        ...


_native_mod: types.ModuleType = __import__("cleargbm_rs.cleargbm_rs", fromlist=["cleargbm_rs"])

_py_gbm_model_to_json: _ToJsonProto = _native_mod.py_gbm_model_to_json_rs
_py_gbm_model_from_json: _FromJsonProto = _native_mod.py_gbm_model_from_json_rs
_py_gbm_model_feature_importances: _FeatureImportancesProto = (
    _native_mod.py_gbm_model_feature_importances_rs
)
_py_gbm_model_n_trees: _NTreesProto = _native_mod.py_gbm_model_n_trees_rs

# cleargbm auto-imports the Rust extension at its own module load, so simply
# importing ``cleargbm.ensemble`` above already guarantees the native path is
# live. No hook-activation call is required.


def _is_cleargbm_config(cfg: ClassifierTrainConfig) -> TypeGuard[ClearGBMConfig]:
    """Check if config is ClearGBMConfig by looking for ClearGBM-specific keys.

    Args:
        cfg: Configuration to check.

    Returns:
        True if config is ClearGBMConfig.
    """
    return (
        isinstance(cfg, dict)
        and "min_samples_split" in cfg
        and "min_samples_leaf" in cfg
        and "num_leaves" not in cfg  # Distinguish from LightGBM
    )


def _compute_class_weight(y_train: NDArray[np.int64]) -> float:
    """Compute scale_pos_weight from training labels.

    Args:
        y_train: Training labels.

    Returns:
        Weight for positive class.
    """
    pos_mask: NDArray[np.bool_] = y_train == 1
    neg_mask: NDArray[np.bool_] = y_train == 0
    n_positive = int(np.count_nonzero(pos_mask))
    n_negative = int(np.count_nonzero(neg_mask))
    if n_positive == 0:
        raise ValueError("Training set has no positive samples")
    computed = float(n_negative) / float(n_positive)
    _log.info(
        "Auto-calculated scale_pos_weight for ClearGBM",
        extra={
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": computed,
        },
    )
    return computed


CLEARGBM_CAPABILITIES: BackendCapabilities = {
    "supports_train": True,
    "supports_gpu": False,
    "supports_early_stopping": True,
    "supports_feature_importance": True,
    "model_format": "json",
}


class _ClearGBMPrepared:
    """Wrapper for a trained ClearGBM native model implementing PreparedClassifier.

    Wraps the opaque ``cleargbm_rs.PyGbmModel`` handle produced by the native
    training loop. All inference goes through ``predict_proba``; save,
    load, and feature-importance extraction go through the native
    ``py_gbm_model_*_rs`` module functions imported at module scope.

    Args:
        model: Trained native model handle.
    """

    def __init__(self, model: _PyGbmModelProto) -> None:
        """Initialize with a trained native model handle.

        Args:
            model: Opaque ``PyGbmModel`` from ``train_gradient_boosting``.
        """
        self._model = model

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities.

        Args:
            x: Feature matrix (n_samples, n_features).

        Returns:
            Probability matrix (n_samples, 2).
        """
        proba_tuple = cgbm_predict_proba(self._model, x)
        return np.array(proba_tuple, dtype=np.float64)

    @property
    def model(self) -> _PyGbmModelProto:
        """Get the underlying native model handle."""
        return self._model


def try_extract_cleargbm_model(
    prepared: PreparedClassifier,
) -> _PyGbmModelProto | None:
    """Extract the native model handle from a prepared classifier if ClearGBM.

    Args:
        prepared: Prepared classifier from any backend.

    Returns:
        Native ``PyGbmModel`` handle if prepared is a ClearGBM classifier,
        None otherwise.
    """
    if not isinstance(prepared, _ClearGBMPrepared):
        return None
    return prepared.model


class ClearGBMBackend(ClassifierBackend):
    """ClearGBM backend for tabular binary classification.

    Provides numpy-based gradient boosting with built-in interpretability.
    """

    def backend_name(self) -> BackendName:
        """Return backend identifier.

        Returns:
            Backend name literal.
        """
        return "cleargbm"

    def capabilities(self) -> BackendCapabilities:
        """Return backend capabilities.

        Returns:
            Capabilities dictionary.
        """
        return CLEARGBM_CAPABILITIES

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        """Prepare is not supported for ClearGBM.

        ClearGBM uses on-demand training via train(). Use train() to create
        a fitted model, then load() to get a PreparedClassifier for inference.

        Raises:
            RuntimeError: Always, as prepare is not supported.
        """
        raise RuntimeError(
            "ClearGBMBackend.prepare not supported; use train() then load() for inference."
        )

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
    ) -> TrainOutcome:
        """Train ClearGBM model on tabular data.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_labels: Binary labels (0 or 1).
            feature_names: Optional feature names.
            config: Training configuration (must be ClearGBMConfig).
            output_dir: Directory for model output.
            progress: Optional progress callback.

        Returns:
            Training outcome with metrics and model path.

        Raises:
            RuntimeError: If config is not ClearGBMConfig.
        """
        if not _is_cleargbm_config(config):
            raise RuntimeError("ClearGBMBackend requires ClearGBMConfig")

        cfg = config

        # Stratified split
        splits = stratified_split(
            x_features,
            y_labels,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
            random_state=cfg["random_state"],
        )

        # Compute class weights for imbalanced data
        scale_pos_weight = _compute_class_weight(splits.y_train)

        # Resolve feature names
        n_feats = int(x_features.shape[1])
        if feature_names is None:
            resolved_names = tuple(f"f{i}" for i in range(n_feats))
        else:
            resolved_names = tuple(feature_names)

        # Build ClearGBM config
        gbm_config: GradientBoostingConfig = GradientBoostingConfig(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=None,
            max_bins=cfg["max_bins"],
            subsample=cfg["subsample"],
            random_state=cfg["random_state"],
            track_contributions=True,
            monotonic_constraints=None,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=1,
            early_stopping_rounds=cfg["early_stopping_rounds"],
        )

        # Train model
        _log.info(
            "Starting ClearGBM training (native Rust loop)",
            extra={
                "n_estimators": cfg["n_estimators"],
                "max_depth": cfg["max_depth"],
                "learning_rate": cfg["learning_rate"],
                "n_train": splits.n_train,
                "n_val": splits.n_val,
            },
        )

        # The native Rust training loop does not accept a Python progress
        # callback. Early stopping is handled internally by the Rust core via
        # ``early_stopping_rounds`` in ``gbm_config``; the returned model is
        # already trimmed to the best-round ensemble. The wrapper's earlier
        # ``_EarlyStoppingTracker`` was redundant Python-side tracking of the
        # same signal — removed with the switch to the native path.
        if progress is not None:
            _log.info(
                "ClearGBM native path does not emit per-round progress; "
                "callback will not be invoked"
            )
        model = train_gradient_boosting(
            x_train=splits.x_train,
            y_train=splits.y_train,
            x_val=splits.x_val,
            y_val=splits.y_val,
            config=gbm_config,
            feature_names=resolved_names,
        )

        # Wrap model for evaluation
        prepared = _ClearGBMPrepared(model)

        # Compute metrics on all splits
        train_proba = prepared.predict_proba(splits.x_train)[:, 1]
        val_proba = prepared.predict_proba(splits.x_val)[:, 1]
        test_proba = prepared.predict_proba(splits.x_test)[:, 1]

        train_metrics = compute_all_metrics(splits.y_train, train_proba)
        val_metrics = compute_all_metrics(splits.y_val, val_proba)
        test_metrics = compute_all_metrics(splits.y_test, test_proba)

        # The Rust core returns a model already trimmed to the best-round
        # ensemble; ``best_round`` is the surviving tree count and
        # ``early_stopped`` is inferred from whether that count is below the
        # configured n_estimators. No independent tracker needed.
        surviving_trees = _py_gbm_model_n_trees(model)
        early_stopped = surviving_trees < cfg["n_estimators"]

        _log.info(
            "ClearGBM training complete",
            extra={
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "test_auc": test_metrics["auc"],
                "surviving_trees": surviving_trees,
                "early_stopped": early_stopped,
            },
        )

        # Get feature importances from the native model
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
        model_path = output_dir / f"cleargbm_{model_id}.json"
        self.save(model=prepared, path=str(model_path))

        return TrainOutcome(
            model_path=str(model_path),
            model_id=model_id,
            samples_total=splits.n_total,
            samples_train=splits.n_train,
            samples_val=splits.n_val,
            samples_test=splits.n_test,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_val_auc=val_metrics["auc"],
            best_round=surviving_trees,
            total_rounds=cfg["n_estimators"],
            early_stopped=early_stopped,
            config=cfg,
            feature_importances=feature_importance_list,
            scale_pos_weight_computed=scale_pos_weight,
        )

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Evaluate ClearGBM model.

        Args:
            model: Trained model (PreparedClassifier).
            x: Feature matrix.
            y: True labels.

        Returns:
            Evaluation metrics.
        """
        proba = model.predict_proba(x)
        return compute_all_metrics(y, proba[:, 1])

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        """Save ClearGBM native model to a JSON file.

        Args:
            model: Trained model (must be _ClearGBMPrepared).
            path: Output path.

        Raises:
            RuntimeError: If model is not _ClearGBMPrepared.
        """
        if not isinstance(model, _ClearGBMPrepared):
            raise RuntimeError("Model must be _ClearGBMPrepared")
        json_str = _py_gbm_model_to_json(model.model)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)

    def load(self, *, path: str) -> PreparedClassifier:
        """Load ClearGBM native model from a JSON file.

        Args:
            path: Path to the saved model file.

        Returns:
            PreparedClassifier wrapping the deserialized native model.
        """
        with open(path, encoding="utf-8") as f:
            raw_str = f.read()
        native_model = _py_gbm_model_from_json(raw_str)
        return _ClearGBMPrepared(native_model)

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Get feature importances from the native ClearGBM model.

        Args:
            model: Trained model.
            feature_names: Feature names (not used, extracted from native model).

        Returns:
            List of feature importances in feature-index order, or None if the
            prepared classifier is not a ClearGBM instance.
        """
        if not isinstance(model, _ClearGBMPrepared):
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


def create_cleargbm_backend() -> ClearGBMBackend:
    """Create a ClearGBM backend instance bound to the native Rust training loop.

    The Rust backend is activated at module import time via
    ``cleargbm._rust`` module load; this factory is a plain constructor with no
    additional setup.

    Returns:
        ClearGBM backend.
    """
    return ClearGBMBackend()


__all__ = [
    "CLEARGBM_CAPABILITIES",
    "ClearGBMBackend",
    "create_cleargbm_backend",
]
