"""XGBoost model training for covenant breach prediction.

Enhanced trainer with:
- Train/validation/test splits
- Early stopping based on validation AUC
- Comprehensive metrics (loss, AUC, accuracy, precision, recall, F1)
- Progress callbacks for monitoring
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from collections.abc import Callable as TypingCallable
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from .metrics import compute_all_metrics, format_metrics_str
from .types import (
    DMatrixFactory,
    DMatrixProtocol,
    FeatureImportance,
    RequestedDevice,
    ResolvedDevice,
    TrainConfig,
    TrainOutcome,
    TrainProgress,
    XGBClassifierFactory,
    XGBModelProtocol,
)

_log = get_logger(__name__)

# Type for progress callback
ProgressCallback = Callable[[TrainProgress], None]


class SplitData(Protocol):
    """Protocol for split dataset."""

    x: NDArray[np.float64]
    y: NDArray[np.int64]


class DataSplits:
    """Container for train/val/test data splits."""

    def __init__(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        x_val: NDArray[np.float64],
        y_val: NDArray[np.int64],
        x_test: NDArray[np.float64],
        y_test: NDArray[np.int64],
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def n_val(self) -> int:
        return len(self.y_val)

    @property
    def n_test(self) -> int:
        return len(self.y_test)

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test


class _XGBCoreProto(Protocol):
    def build_info(self) -> dict[str, bool]: ...


class _XGBModuleProto(Protocol):
    core: _XGBCoreProto
    XGBClassifier: XGBClassifierFactory
    DMatrix: DMatrixFactory


_cuda_available_hook: TypingCallable[[], bool] | None = None


def set_cuda_available_hook(hook: TypingCallable[[], bool] | None) -> None:
    """Set test hook for CUDA availability detection."""
    global _cuda_available_hook
    _cuda_available_hook = hook


def _detect_cuda_available(xgb_mod: _XGBModuleProto) -> bool:
    """Detect whether xgboost was built with CUDA support."""
    info = xgb_mod.core.build_info()
    return bool(info.get("USE_CUDA", False))


def _cuda_is_available(xgb_mod: _XGBModuleProto) -> bool:
    """Check CUDA availability with optional test hook."""
    if _cuda_available_hook is not None:
        return bool(_cuda_available_hook()) and _detect_cuda_available(xgb_mod)
    return _detect_cuda_available(xgb_mod)


def _resolve_device(requested: RequestedDevice, xgb_mod: _XGBModuleProto) -> ResolvedDevice:
    """Resolve requested device to a concrete device."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not _cuda_is_available(xgb_mod):
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    # requested == "auto"
    return "cuda" if _cuda_is_available(xgb_mod) else "cpu"


def stratified_split(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> DataSplits:
    """Split data into train/val/test with stratification.

    Maintains class proportions across all splits.

    Args:
        x_features: Feature matrix (n_samples, n_features)
        y_labels: Binary labels (n_samples,)
        train_ratio: Fraction for training (e.g., 0.7)
        val_ratio: Fraction for validation (e.g., 0.15)
        test_ratio: Fraction for test holdout (e.g., 0.15)
        random_state: Random seed for reproducibility

    Returns:
        DataSplits container with train/val/test arrays

    Raises:
        ValueError: If ratios don't sum to 1.0 (within tolerance)
    """
    # Validate ratios sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.3f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    rng = np.random.default_rng(random_state)

    # Get indices for each class (np.where returns tuple, take first element)
    pos_mask: NDArray[np.bool_] = y_labels == 1
    neg_mask: NDArray[np.bool_] = y_labels == 0
    pos_indices: NDArray[np.intp] = np.flatnonzero(pos_mask)
    neg_indices: NDArray[np.intp] = np.flatnonzero(neg_mask)

    # Shuffle indices
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    # Calculate split points for each class
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    pos_train_end = int(n_pos * train_ratio)
    pos_val_end = int(n_pos * (train_ratio + val_ratio))

    neg_train_end = int(n_neg * train_ratio)
    neg_val_end = int(n_neg * (train_ratio + val_ratio))

    # Split indices (test is remainder after train+val)
    train_idx: NDArray[np.intp] = np.concatenate(
        [
            pos_indices[:pos_train_end],
            neg_indices[:neg_train_end],
        ]
    )
    val_idx: NDArray[np.intp] = np.concatenate(
        [
            pos_indices[pos_train_end:pos_val_end],
            neg_indices[neg_train_end:neg_val_end],
        ]
    )
    test_idx: NDArray[np.intp] = np.concatenate(
        [
            pos_indices[pos_val_end:],
            neg_indices[neg_val_end:],
        ]
    )

    # Shuffle final indices
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    _log.info(
        "Data split complete",
        extra={
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
        },
    )

    return DataSplits(
        x_train=x_features[train_idx],
        y_train=y_labels[train_idx],
        x_val=x_features[val_idx],
        y_val=y_labels[val_idx],
        x_test=x_features[test_idx],
        y_test=y_labels[test_idx],
    )


def _get_probabilities(
    model: XGBModelProtocol,
    x_features: NDArray[np.float64],
    xgb_module: _XGBModuleProto,
) -> NDArray[np.float64]:
    """Get probability predictions for class 1 (breach).

    Uses DMatrix and booster.predict() - works on both CPU and GPU.
    """
    dmatrix: DMatrixProtocol = xgb_module.DMatrix(x_features)
    booster = model.get_booster()
    raw_preds: NDArray[np.float32] = booster.predict(dmatrix)
    return np.asarray(raw_preds, dtype=np.float64)


def _compute_scale_pos_weight(
    y_labels: NDArray[np.int64],
    config_value: float | None,
) -> float:
    """Compute scale_pos_weight from labels or return provided value.

    Args:
        y_labels: Binary labels array
        config_value: Optional provided scale_pos_weight value

    Returns:
        scale_pos_weight value (provided or auto-calculated)

    Raises:
        ValueError: If no positive samples exist and no value provided
    """
    if config_value is not None:
        _log.info(
            "Using provided scale_pos_weight",
            extra={"scale_pos_weight": config_value},
        )
        return config_value

    pos_mask: NDArray[np.bool_] = y_labels == 1
    neg_mask: NDArray[np.bool_] = y_labels == 0
    n_positive = int(np.count_nonzero(pos_mask))
    n_negative = int(np.count_nonzero(neg_mask))
    if n_positive == 0:
        raise ValueError("Training set has no positive samples (bankruptcies)")
    computed = float(n_negative) / float(n_positive)
    _log.info(
        "Auto-calculated scale_pos_weight",
        extra={
            "n_positive": n_positive,
            "n_negative": n_negative,
            "scale_pos_weight": computed,
        },
    )
    return computed


def extract_feature_importances(
    model: XGBModelProtocol,
    feature_names: list[str],
) -> list[FeatureImportance]:
    """Extract feature importances from trained model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (must match number of model features)

    Returns:
        List of FeatureImportance sorted by importance (descending)

    Raises:
        ValueError: If feature_names length doesn't match model features
    """
    raw_importances = model.feature_importances_
    n_features = len(raw_importances)

    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match model features ({n_features})"
        )

    names = feature_names

    # Create unsorted list with importances
    # Use flat iterator with item() to get typed float values
    unsorted: list[tuple[str, float]] = []
    for i, imp in enumerate(raw_importances.flat):
        imp_float: float = float(imp.item())
        unsorted.append((names[i], imp_float))

    # Sort by importance descending
    def get_importance(pair: tuple[str, float]) -> float:
        return pair[1]

    sorted_by_importance = sorted(unsorted, key=get_importance, reverse=True)

    # Build result with ranks
    result: list[FeatureImportance] = []
    for rank, (name, importance) in enumerate(sorted_by_importance, start=1):
        result.append(
            FeatureImportance(
                name=name,
                importance=importance,
                rank=rank,
            )
        )

    return result


def train_model_with_validation(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    config: TrainConfig,
    output_dir: Path,
    feature_names: list[str],
    progress_callback: ProgressCallback | None = None,
) -> TrainOutcome:
    """Train XGBoost classifier with validation and early stopping.

    Implements proper early stopping based on validation AUC:
    - Trains for up to n_estimators rounds
    - Monitors validation AUC after each round
    - Stops if no improvement for early_stopping_rounds consecutive rounds
    - Restores best model based on validation AUC

    Args:
        x_features: Feature matrix (n_samples, n_features)
        y_labels: Binary labels (n_samples,)
        config: Training configuration with hyperparameters
        output_dir: Directory to save model artifacts
        feature_names: List of feature names for importance reporting
        progress_callback: Optional callback for progress updates

    Returns:
        TrainOutcome with complete training results, metrics, and feature importances
    """
    xgb = __import__("xgboost")
    xgb_module: _XGBModuleProto = xgb
    classifier_factory: XGBClassifierFactory = xgb_module.XGBClassifier
    resolved_device = _resolve_device(config["device"], xgb_module)
    n_jobs = max(1, int(os.cpu_count() or 1))

    # Split data first (needed for auto-calculating scale_pos_weight)
    splits = stratified_split(
        x_features,
        y_labels,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        random_state=config["random_state"],
    )

    # Calculate scale_pos_weight from training set if not provided
    scale_pos_weight_computed = _compute_scale_pos_weight(
        splits.y_train, config.get("scale_pos_weight")
    )

    def _build_classifier(total_estimators: int) -> XGBModelProtocol:
        return classifier_factory(
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            n_estimators=total_estimators,
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            random_state=config["random_state"],
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=n_jobs,
            tree_method="hist",
            device=resolved_device,
            scale_pos_weight=scale_pos_weight_computed,
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
        )

    n_estimators = config["n_estimators"]
    early_stopping_rounds = config["early_stopping_rounds"]

    # Track training history
    train_loss_history: list[float] = []
    train_auc_history: list[float] = []
    val_loss_history: list[float] = []
    val_auc_history: list[float] = []

    # Track best model state
    best_val_auc = 0.0
    best_round = 0
    rounds_no_improve = 0
    early_stopped = False

    # We train incrementally by setting n_estimators=1 and using warm_start
    # This allows us to evaluate after each round and implement early stopping
    model: XGBModelProtocol | None = None
    current_round = 0  # Will be updated by loop; 0 if loop never runs

    for current_round in range(1, n_estimators + 1):
        if model is None:
            # First round: create model with 1 estimator
            model = _build_classifier(1)
            model.fit(splits.x_train, splits.y_train, verbose=False)
        else:
            # Subsequent rounds: create new model with more estimators
            # XGBoost doesn't support true warm_start, so we retrain with more trees
            # This is the standard approach for manual early stopping
            model = _build_classifier(current_round)
            model.fit(splits.x_train, splits.y_train, verbose=False)

        # Evaluate on train and validation sets
        train_proba = _get_probabilities(model, splits.x_train, xgb_module)
        val_proba = _get_probabilities(model, splits.x_val, xgb_module)

        train_metrics = compute_all_metrics(splits.y_train, train_proba)
        val_metrics = compute_all_metrics(splits.y_val, val_proba)

        # Record history
        train_loss_history.append(train_metrics["loss"])
        train_auc_history.append(train_metrics["auc"])
        val_loss_history.append(val_metrics["loss"])
        val_auc_history.append(val_metrics["auc"])

        # Report progress
        if progress_callback is not None:
            progress_callback(
                TrainProgress(
                    round=current_round,
                    total_rounds=n_estimators,
                    train_loss=train_metrics["loss"],
                    train_auc=train_metrics["auc"],
                    val_loss=val_metrics["loss"],
                    val_auc=val_metrics["auc"],
                )
            )

        # Check for improvement (using AUC - higher is better)
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_round = current_round
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1

        # Log progress every 10 rounds or on improvement
        if current_round % 10 == 0 or rounds_no_improve == 0:
            _log.info(
                "Training progress",
                extra={
                    "round": current_round,
                    "total_rounds": n_estimators,
                    "train_auc": train_metrics["auc"],
                    "val_auc": val_metrics["auc"],
                    "best_val_auc": best_val_auc,
                    "best_round": best_round,
                    "rounds_no_improve": rounds_no_improve,
                },
            )

        # Early stopping check
        if rounds_no_improve >= early_stopping_rounds:
            early_stopped = True
            _log.info(
                "Early stopping triggered",
                extra={
                    "stopped_at_round": current_round,
                    "best_round": best_round,
                    "best_val_auc": best_val_auc,
                    "early_stopping_rounds": early_stopping_rounds,
                },
            )
            break

    # If early stopped, retrain with optimal number of estimators
    if early_stopped and best_round < current_round:
        _log.info(
            "Retraining with optimal estimators",
            extra={"best_round": best_round},
        )
        model = _build_classifier(best_round)
        model.fit(splits.x_train, splits.y_train, verbose=False)
        actual_rounds = best_round
    else:
        actual_rounds = current_round

    # Model should never be None at this point (loop always runs at least once)
    if model is None:
        raise RuntimeError("Model not trained - n_estimators must be >= 1")

    # Final evaluation on all splits
    train_proba = _get_probabilities(model, splits.x_train, xgb_module)
    val_proba = _get_probabilities(model, splits.x_val, xgb_module)
    test_proba = _get_probabilities(model, splits.x_test, xgb_module)

    final_train_metrics = compute_all_metrics(splits.y_train, train_proba)
    final_val_metrics = compute_all_metrics(splits.y_val, val_proba)
    final_test_metrics = compute_all_metrics(splits.y_test, test_proba)

    _log.info(
        "Training complete",
        extra={
            "total_rounds_trained": actual_rounds,
            "early_stopped": early_stopped,
            "best_round": best_round,
            "train_metrics": format_metrics_str(final_train_metrics),
            "val_metrics": format_metrics_str(final_val_metrics),
            "test_metrics": format_metrics_str(final_test_metrics),
        },
    )

    # Save model
    model_id = str(uuid.uuid4())
    model_filename = f"covenant_model_{model_id[:8]}.ubj"
    model_path = output_dir / model_filename

    save_model(model, str(model_path))

    _log.info("Model saved", extra={"model_path": str(model_path)})

    # Extract feature importances
    importances = extract_feature_importances(model, feature_names)

    _log.info(
        "Feature importances extracted",
        extra={
            "top_features": [
                {"name": f["name"], "importance": f"{f['importance']:.4f}"} for f in importances[:3]
            ],
        },
    )

    return TrainOutcome(
        model_path=str(model_path),
        model_id=model_id,
        samples_total=splits.n_total,
        samples_train=splits.n_train,
        samples_val=splits.n_val,
        samples_test=splits.n_test,
        train_metrics=final_train_metrics,
        val_metrics=final_val_metrics,
        test_metrics=final_test_metrics,
        best_val_auc=best_val_auc,
        best_round=best_round,
        total_rounds=actual_rounds,
        early_stopped=early_stopped,
        config=config,
        feature_importances=importances,
        scale_pos_weight_computed=scale_pos_weight_computed,
    )


def train_model(
    x_features: NDArray[np.float64],
    y_labels: NDArray[np.int64],
    config: TrainConfig,
) -> XGBModelProtocol:
    """Train XGBoost classifier (simple API without validation).

    Auto-calculates scale_pos_weight if not provided.
    Prefer train_model_with_validation for production use.

    Args:
        x_features: Feature matrix (n_samples, n_features)
        y_labels: Binary labels (n_samples,)
        config: Training configuration

    Returns:
        Trained XGBClassifier model
    """
    xgb = __import__("xgboost")
    xgb_module: _XGBModuleProto = xgb
    classifier_factory: XGBClassifierFactory = xgb_module.XGBClassifier
    resolved_device = _resolve_device(config["device"], xgb_module)
    n_jobs = max(1, int(os.cpu_count() or 1))

    # Auto-calculate scale_pos_weight if not provided
    scale_pos_weight_computed = _compute_scale_pos_weight(y_labels, config.get("scale_pos_weight"))

    model = classifier_factory(
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        n_estimators=config["n_estimators"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        random_state=config["random_state"],
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=n_jobs,
        tree_method="hist",
        device=resolved_device,
        scale_pos_weight=scale_pos_weight_computed,
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
    )

    model.fit(x_features, y_labels)
    return model


def save_model(model: XGBModelProtocol, path: str) -> None:
    """Save trained model to file path.

    Uses get_booster().save_model() for XGBoost 3.x compatibility.
    """
    booster = model.get_booster()
    booster.save_model(path)


__all__ = [
    "DataSplits",
    "ProgressCallback",
    "extract_feature_importances",
    "save_model",
    "stratified_split",
    "train_model",
    "train_model_with_validation",
]
