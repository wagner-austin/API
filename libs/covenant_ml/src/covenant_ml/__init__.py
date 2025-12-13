"""XGBoost wrapper for covenant breach risk prediction."""

from __future__ import annotations

from .manifest import (
    ClassifierManifest,
    ManifestDataset,
    ManifestMetrics,
    ManifestSystem,
    ManifestTraining,
    ManifestVersions,
)
from .metrics import (
    compute_accuracy,
    compute_all_metrics,
    compute_auc,
    compute_f1_score,
    compute_log_loss,
    compute_precision,
    compute_recall,
    format_metrics_str,
)
from .predictor import load_model, predict_probabilities
from .trainer import (
    DataSplits,
    ProgressCallback,
    save_model,
    stratified_split,
    train_model,
    train_model_with_validation,
)
from .types import (
    EvalMetrics,
    FeatureImportance,
    Proba2DProtocol,
    TrainConfig,
    TrainOutcome,
    TrainProgress,
    XGBBoosterProtocol,
    XGBClassifierFactory,
    XGBClassifierLoader,
    XGBModelProtocol,
)

__all__ = [
    "ClassifierManifest",
    "DataSplits",
    "EvalMetrics",
    "FeatureImportance",
    "ManifestDataset",
    "ManifestMetrics",
    "ManifestSystem",
    "ManifestTraining",
    "ManifestVersions",
    "Proba2DProtocol",
    "ProgressCallback",
    "TrainConfig",
    "TrainOutcome",
    "TrainProgress",
    "XGBBoosterProtocol",
    "XGBClassifierFactory",
    "XGBClassifierLoader",
    "XGBModelProtocol",
    "compute_accuracy",
    "compute_all_metrics",
    "compute_auc",
    "compute_f1_score",
    "compute_log_loss",
    "compute_precision",
    "compute_recall",
    "format_metrics_str",
    "load_model",
    "predict_probabilities",
    "save_model",
    "stratified_split",
    "train_model",
    "train_model_with_validation",
]
