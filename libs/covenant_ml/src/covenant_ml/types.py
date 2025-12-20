"""Type definitions for covenant ML training and prediction.

Strict typing only. No Any, casts, or stubs.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

RequestedDevice = Literal["cpu", "cuda", "auto"]
ResolvedDevice = Literal["cpu", "cuda"]

# Pluggable backend naming - all supported classifier backends
BackendName = Literal["xgboost", "mlp", "lstm", "lightgbm"]


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
    ppl: float  # Perplexity (exp(loss))
    auc: float  # Area under ROC curve
    accuracy: float  # Classification accuracy
    precision: float  # Precision for breach class
    recall: float  # Recall for breach class
    f1_score: float  # F1 score


class AMEXMetricResult(TypedDict, total=True):
    """Result of AMEX competition metric calculation.

    The AMEX metric is: 0.5 * (normalized_gini + default_rate_at_4_percent).
    Accounts for 5% negative class subsampling with 20x weight on negatives.
    """

    score: float  # Final competition metric (0 to 1, higher is better)
    normalized_gini: float  # Gini coefficient normalized by perfect prediction
    default_rate_at_4_percent: float  # Recall at top 4% weighted predictions


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
    # Union inlined to avoid forward ref
    config: TrainConfig | MLPConfig | LSTMConfig | LightGBMConfig
    feature_importances: list[FeatureImportance]  # Sorted by importance (descending)
    # Class weight used for training (auto-calculated if not provided in config)
    scale_pos_weight_computed: float


# MLP backend configuration (tabular classifier)
# Precision/device types use platform_ml conventions (no runtime import here).
MLPPrecision = Literal["fp32", "fp16", "bf16", "auto"]
MLPOptimizer = Literal["adamw", "adam", "sgd"]


class MLPConfig(TypedDict, total=True):
    """Strict configuration for MLP backend training.

    Includes explicit train/val/test splits, deterministic seed, and early stopping.
    Optimizer is pluggable and narrowed to a small, supported set.
    """

    device: RequestedDevice
    precision: MLPPrecision
    optimizer: MLPOptimizer
    hidden_sizes: tuple[int, ...]
    learning_rate: float
    batch_size: int
    n_epochs: int
    dropout: float
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int
    early_stopping_patience: int


# LSTM backend configuration (temporal sequence classifier)
LSTMPrecision = Literal["fp32", "fp16", "bf16", "auto"]


class LSTMConfig(TypedDict, total=True):
    """Strict configuration for LSTM backend training.

    LSTM processes temporal sequences of financial data for bankruptcy prediction.
    Each sequence contains multiple years of data for a single company.

    The backend accepts either:
    - Pre-sequenced data: (n_sequences, seq_len, n_features) with sequence_length set
    - Flat data: (n_samples, n_features) reshaped to pseudo-sequences internally

    For proper temporal modeling, use SequenceBuilder to prepare data with
    entity_ids and years before training.
    """

    device: RequestedDevice
    precision: LSTMPrecision
    hidden_size: int  # LSTM hidden state dimension
    num_layers: int  # Number of stacked LSTM layers
    dropout: float  # Dropout between LSTM layers (only if num_layers > 1)
    bidirectional: bool  # Process sequences in both directions
    sequence_length: int  # Number of time periods in each sequence
    learning_rate: float
    batch_size: int
    n_epochs: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int
    early_stopping_patience: int


# LightGBM backend configuration
class LightGBMConfig(TypedDict, total=True):
    """Strict configuration for LightGBM backend training.

    LightGBM is a gradient boosting framework that uses tree-based learning.
    It's faster than XGBoost and handles large datasets efficiently.

    Key differences from XGBoost:
    - num_leaves: Controls tree complexity (instead of just max_depth)
    - min_child_samples: Minimum data in a leaf
    - Uses leaf-wise tree growth (vs level-wise in XGBoost)
    """

    device: RequestedDevice
    learning_rate: float
    max_depth: int
    n_estimators: int
    num_leaves: int  # LightGBM-specific: controls complexity
    min_child_samples: int  # LightGBM-specific: minimum data in leaf
    subsample: float
    colsample_bytree: float
    reg_alpha: float  # L1 regularization
    reg_lambda: float  # L2 regularization
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_state: int
    early_stopping_rounds: int


# Union of backend-specific train configs
ClassifierTrainConfig = TrainConfig | MLPConfig | LSTMConfig | LightGBMConfig


# =============================================================================
# Model Metadata for Inference Loading
# =============================================================================
# These TypedDicts store the minimal architecture info needed to reconstruct
# models from saved state dicts (MLP/LSTM) or verify model compatibility.


class MLPModelMeta(TypedDict, total=True):
    """Metadata required to reconstruct an MLP model for inference.

    Stored as JSON alongside the .pt state dict file. Contains only the
    architecture parameters needed to call _build_model() before loading
    the state dict.

    Args:
        backend: Literal discriminator for union type narrowing.
        n_features: Number of input features the model was trained on.
        hidden_sizes: List of hidden layer sizes (JSON doesn't support tuples).
        dropout: Dropout rate used in the model architecture.
    """

    backend: Literal["mlp"]
    n_features: int
    hidden_sizes: list[int]
    dropout: float


class LSTMModelMeta(TypedDict, total=True):
    """Metadata required to reconstruct an LSTM model for inference.

    Stored as JSON alongside the .pt state dict file. Contains the full
    architecture specification needed to rebuild the LSTM network.

    Args:
        backend: Literal discriminator for union type narrowing.
        n_features: Number of input features per time step.
        sequence_length: Number of time steps in each sequence.
        hidden_size: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.
        bidirectional: Whether LSTM processes sequences in both directions.
        dropout: Dropout rate between LSTM layers.
    """

    backend: Literal["lstm"]
    n_features: int
    sequence_length: int
    hidden_size: int
    num_layers: int
    bidirectional: bool
    dropout: float


class LightGBMModelMeta(TypedDict, total=True):
    """Metadata for LightGBM model.

    LightGBM's .txt format is self-describing, so minimal metadata is needed.
    The backend field enables consistent discriminated union handling.

    Args:
        backend: Literal discriminator for union type narrowing.
    """

    backend: Literal["lightgbm"]


# Union of model metadata types for type-safe dispatch
ModelMeta = MLPModelMeta | LSTMModelMeta | LightGBMModelMeta


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


class PredictorProtocol(Protocol):
    """Minimal protocol for models that can predict probabilities.

    Both XGBoost and MLP models implement this interface.
    Used by predict_probabilities for inference.
    """

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]: ...


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
        x: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

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
    "AMEXMetricResult",
    "BackendName",
    "ClassifierTrainConfig",
    "DMatrixFactory",
    "DMatrixProtocol",
    "EvalMetrics",
    "FeatureImportance",
    "LSTMConfig",
    "LSTMModelMeta",
    "LSTMPrecision",
    "LightGBMConfig",
    "LightGBMModelMeta",
    "MLPConfig",
    "MLPModelMeta",
    "MLPPrecision",
    "ModelMeta",
    "PredictorProtocol",
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
