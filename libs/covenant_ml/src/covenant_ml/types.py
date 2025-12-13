"""Type definitions for covenant ML training and prediction."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

RequestedDevice = Literal["cpu", "cuda", "auto"]
ResolvedDevice = Literal["cpu", "cuda"]


class TrainConfigRequired(TypedDict, total=True):
    """Required configuration fields for XGBoost model training."""

    device: RequestedDevice
    learning_rate: float
    max_depth: int
    n_estimators: int
    subsample: float
    colsample_bytree: float
    random_state: int
    # Train/val/test split ratios (must sum to 1.0)
    train_ratio: float  # e.g., 0.7
    val_ratio: float  # e.g., 0.15
    test_ratio: float  # e.g., 0.15
    early_stopping_rounds: int  # e.g., 10
    # Regularization - prevents overfitting, lets model learn feature importance
    reg_alpha: float  # L1 (sparsity) - pushes weak features to zero. Typical: 0.0-10.0
    reg_lambda: float  # L2 (ridge) - prevents any feature from dominating. Typical: 1.0-10.0


class TrainConfig(TrainConfigRequired, total=False):
    """Configuration for XGBoost model training.

    Inherits required fields from TrainConfigRequired.
    Optional fields below help with class imbalance.
    """

    # Weight for positive class to handle imbalanced data
    # Set to (n_negative / n_positive) for balanced importance
    scale_pos_weight: float


class FeatureImportance(TypedDict, total=True):
    """Feature importance from trained model."""

    name: str  # Feature name
    importance: float  # Importance score (gain-based)
    rank: int  # Rank (1 = most important)


class EvalMetrics(TypedDict, total=True):
    """Evaluation metrics for a dataset split."""

    loss: float  # Log loss (cross-entropy)
    auc: float  # Area under ROC curve
    accuracy: float  # Classification accuracy
    precision: float  # Precision for breach class
    recall: float  # Recall for breach class
    f1_score: float  # F1 score


class TrainOutcome(TypedDict, total=True):
    """Complete training outcome with metrics from all splits."""

    model_path: str
    model_id: str
    samples_total: int
    samples_train: int
    samples_val: int
    samples_test: int
    train_metrics: EvalMetrics
    val_metrics: EvalMetrics
    test_metrics: EvalMetrics
    best_val_auc: float
    best_round: int
    total_rounds: int
    early_stopped: bool
    config: TrainConfig
    feature_importances: list[FeatureImportance]  # Sorted by importance (descending)
    # Class weight used for training (auto-calculated if not provided in config)
    scale_pos_weight_computed: float


class TrainProgress(TypedDict, total=True):
    """Progress update during training."""

    round: int
    total_rounds: int
    train_loss: float
    train_auc: float
    val_loss: float | None
    val_auc: float | None


class Proba2DProtocol(Protocol):
    """Protocol for 2D probability array from predict_proba.

    predict_proba returns shape (n_samples, n_classes).
    For binary classification: (n_samples, 2).
    """

    @property
    def shape(self) -> tuple[int, int]: ...

    def __getitem__(self, idx: tuple[int, int]) -> float: ...


class DMatrixProtocol(Protocol):
    """Protocol for XGBoost DMatrix interface."""

    def set_info(self, *, feature_names: list[str] | None) -> None: ...


class XGBBoosterProtocol(Protocol):
    """Protocol for XGBoost Booster (core model) interface."""

    def save_model(self, fname: str) -> None: ...

    def predict(self, data: DMatrixProtocol) -> NDArray[np.float32]: ...


class XGBParams(TypedDict, total=False):
    """Subset of XGBoost parameters we rely on."""

    n_jobs: int
    tree_method: str
    device: str
    reg_alpha: float
    reg_lambda: float


class XGBModelProtocol(Protocol):
    """Protocol for XGBoost classifier interface."""

    @property
    def feature_importances_(self) -> NDArray[np.float32]:
        """Feature importance scores (gain-based by default)."""
        ...

    def fit(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        *,
        verbose: bool = False,
    ) -> XGBModelProtocol: ...

    def predict_proba(
        self,
        x_features: NDArray[np.float64],
    ) -> Proba2DProtocol: ...

    def get_xgb_params(self) -> XGBParams: ...

    def save_model(self, fname: str) -> None: ...

    def load_model(self, fname: str) -> None: ...

    def get_booster(self) -> XGBBoosterProtocol: ...


class XGBClassifierFactory(Protocol):
    """Protocol for XGBClassifier constructor."""

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
        scale_pos_weight: float | None = None,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
    ) -> XGBModelProtocol: ...


class XGBClassifierLoader(Protocol):
    """Protocol for XGBClassifier loader (no-arg constructor)."""

    def __call__(self) -> XGBModelProtocol: ...


class DMatrixFactory(Protocol):
    """Protocol for XGBoost DMatrix constructor."""

    def __call__(self, data: NDArray[np.float64]) -> DMatrixProtocol: ...


__all__ = [
    "DMatrixFactory",
    "DMatrixProtocol",
    "EvalMetrics",
    "FeatureImportance",
    "Proba2DProtocol",
    "TrainConfig",
    "TrainConfigRequired",
    "TrainOutcome",
    "TrainProgress",
    "XGBBoosterProtocol",
    "XGBClassifierFactory",
    "XGBClassifierLoader",
    "XGBModelProtocol",
]
