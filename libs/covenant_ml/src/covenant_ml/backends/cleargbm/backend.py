"""ClearGBM backend for tabular binary classification.

Implements ClassifierBackend protocol using the numpy-based ClearGBM library.
Provides built-in interpretability (rule extraction, feature contributions).

Strict typing only: no Any, no casts, no type: ignore.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TypeGuard

import numpy as np
from cleargbm.ensemble import predict_proba as cgbm_predict_proba
from cleargbm.ensemble import train_gradient_boosting
from cleargbm.explain import get_feature_importances
from cleargbm.types import (
    GradientBoostingConfig,
    GradientBoostingModel,
    TrainingProgress,
    decode_gradient_boosting_model,
    encode_gradient_boosting_model,
)
from numpy.typing import NDArray
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from ...metrics import compute_all_metrics
from ...trainer import stratified_split
from ...types import (
    BackendName,
    ClassifierTrainConfig,
    ClearGBMConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
)
from ...types import (
    TrainProgress as CovenantTrainProgress,
)
from ..protocol import BackendCapabilities, ClassifierBackend, PreparedClassifier, ProgressCallback

_log = get_logger(__name__)


class _EarlyStoppingTracker:
    """Tracks early stopping state during ClearGBM training.

    Monitors validation loss and determines when to stop training
    if no improvement is seen for a specified number of rounds.

    Attributes:
        best_val_loss: Best validation loss seen so far.
        best_round: Round number where best loss was achieved.
        rounds_without_improvement: Consecutive rounds without improvement.
        early_stopped: Whether early stopping was triggered.
    """

    def __init__(self, early_stopping_rounds: int) -> None:
        """Initialize early stopping tracker.

        Args:
            early_stopping_rounds: Stop after this many rounds without improvement.
        """
        self._early_stopping_rounds = early_stopping_rounds
        self.best_val_loss = float("inf")
        self.best_round = 0
        self.rounds_without_improvement = 0
        self.early_stopped = False

    def update(self, val_loss: float | None, tree_index: int) -> None:
        """Update tracking state with new validation loss.

        Args:
            val_loss: Validation loss for current round, or None if not available.
            tree_index: Zero-based tree/round index.
        """
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_round = tree_index + 1
            self.rounds_without_improvement = 0
        elif val_loss is not None:
            self.rounds_without_improvement += 1

        if self.rounds_without_improvement >= self._early_stopping_rounds:
            self.early_stopped = True


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
    """Wrapper for a trained ClearGBM model implementing PreparedClassifier.

    Args:
        model: Trained ClearGBM model.
    """

    def __init__(self, model: GradientBoostingModel) -> None:
        """Initialize with trained model."""
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
    def model(self) -> GradientBoostingModel:
        """Get the underlying ClearGBM model."""
        return self._model


def try_extract_cleargbm_model(
    prepared: PreparedClassifier,
) -> GradientBoostingModel | None:
    """Extract GradientBoostingModel from a prepared classifier if ClearGBM.

    Args:
        prepared: Prepared classifier from any backend.

    Returns:
        GradientBoostingModel if prepared is a ClearGBM classifier, None otherwise.
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

        # Track best model for early stopping
        tracker = _EarlyStoppingTracker(cfg["early_stopping_rounds"])

        # Progress callback wrapper
        def progress_wrapper(prog: TrainingProgress) -> None:
            tracker.update(prog["val_loss"], prog["tree_index"])

            if progress is not None:
                covenant_progress = CovenantTrainProgress(
                    round=prog["tree_index"] + 1,
                    total_rounds=prog["total_trees"],
                    train_loss=prog["train_loss"],
                    train_auc=0.0,  # Computed after training
                    val_loss=prog["val_loss"],
                    val_auc=None,
                )
                progress(covenant_progress)

        # Train model
        _log.info(
            "Starting ClearGBM training",
            extra={
                "n_estimators": cfg["n_estimators"],
                "max_depth": cfg["max_depth"],
                "learning_rate": cfg["learning_rate"],
                "n_train": splits.n_train,
                "n_val": splits.n_val,
            },
        )

        model = train_gradient_boosting(
            x_train=splits.x_train,
            y_train=splits.y_train,
            x_val=splits.x_val,
            y_val=splits.y_val,
            config=gbm_config,
            feature_names=resolved_names,
            progress_callback=progress_wrapper,
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

        _log.info(
            "ClearGBM training complete",
            extra={
                "train_auc": train_metrics["auc"],
                "val_auc": val_metrics["auc"],
                "test_auc": test_metrics["auc"],
                "best_round": tracker.best_round,
                "early_stopped": tracker.early_stopped,
            },
        )

        # Get feature importances
        importances = get_feature_importances(model)
        feature_importance_list: list[FeatureImportance] = []
        for rank, imp in enumerate(importances, start=1):
            feature_importance_list.append(
                FeatureImportance(
                    name=imp["feature_name"],
                    importance=imp["total_contribution"],
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
            best_round=tracker.best_round if tracker.best_round > 0 else cfg["n_estimators"],
            total_rounds=cfg["n_estimators"],
            early_stopped=tracker.early_stopped,
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
        """Save ClearGBM model to JSON file.

        Args:
            model: Trained model (must be _ClearGBMPrepared).
            path: Output path.

        Raises:
            RuntimeError: If model is not _ClearGBMPrepared.
        """
        if not isinstance(model, _ClearGBMPrepared):
            raise RuntimeError("Model must be _ClearGBMPrepared")
        encoded = encode_gradient_boosting_model(model.model)
        json_str = dump_json_str(encoded, indent=2)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)

    def load(self, *, path: str) -> PreparedClassifier:
        """Load ClearGBM model from JSON file.

        Args:
            path: Path to the saved model file.

        Returns:
            PreparedClassifier wrapping the loaded model.
        """
        with open(path, encoding="utf-8") as f:
            raw_str = f.read()
        raw = narrow_json_to_dict(load_json_str(raw_str))
        model = decode_gradient_boosting_model(raw)
        return _ClearGBMPrepared(model)

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Get feature importances from ClearGBM model.

        Args:
            model: Trained model.
            feature_names: Feature names (not used, extracted from model).

        Returns:
            List of feature importances sorted by importance.
        """
        if not isinstance(model, _ClearGBMPrepared):
            return None

        importances = get_feature_importances(model.model)
        result: list[FeatureImportance] = []
        for rank, imp in enumerate(importances, start=1):
            result.append(
                FeatureImportance(
                    name=imp["feature_name"],
                    importance=imp["total_contribution"],
                    rank=rank,
                )
            )
        return result


def create_cleargbm_backend() -> ClearGBMBackend:
    """Create ClearGBM backend instance.

    Returns:
        ClearGBM backend.
    """
    return ClearGBMBackend()


__all__ = [
    "CLEARGBM_CAPABILITIES",
    "ClearGBMBackend",
    "create_cleargbm_backend",
]
