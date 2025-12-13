"""Training manifest types for model metadata and reproducibility.

Strict TypedDict definitions for capturing training context, versions,
and system information alongside trained models.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from .types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
)


class ManifestVersions(TypedDict, total=True):
    """Version information for reproducibility.

    Captures library versions used during training to ensure
    compatibility when loading models.
    """

    covenant_ml: str  # covenant-ml package version
    python: str  # Python version (e.g., "3.11.9")
    xgboost: str | None  # XGBoost version if xgboost backend used
    torch: str | None  # PyTorch version if mlp backend used
    numpy: str  # NumPy version
    scikit_learn: str  # scikit-learn version


class ManifestSystem(TypedDict, total=True):
    """System information at training time.

    Captures hardware and environment details for debugging
    and performance analysis.
    """

    platform: str  # e.g., "linux", "win32", "darwin"
    device_used: Literal["cpu", "cuda"]  # Resolved device
    cuda_version: str | None  # CUDA version if GPU used
    gpu_name: str | None  # GPU model name if GPU used


class ManifestDataset(TypedDict, total=True):
    """Dataset statistics captured at training time."""

    name: str  # Dataset identifier (e.g., "taiwan", "us", "polish")
    samples_total: int
    samples_train: int
    samples_val: int
    samples_test: int
    n_features: int
    n_positive: int  # Number of positive class samples
    n_negative: int  # Number of negative class samples
    class_ratio: float  # n_negative / n_positive


class ManifestTraining(TypedDict, total=True):
    """Training execution details."""

    backend: BackendName
    config: ClassifierTrainConfig
    best_round: int
    total_rounds: int
    early_stopped: bool
    scale_pos_weight_computed: float
    training_duration_seconds: float


class ManifestMetrics(TypedDict, total=True):
    """Evaluation metrics from all splits."""

    train: EvalMetrics
    val: EvalMetrics
    test: EvalMetrics
    best_val_auc: float


class ClassifierManifest(TypedDict, total=True):
    """Complete training manifest for a classifier model.

    Contains all information needed to understand, reproduce, and
    deploy a trained model. Written alongside model artifacts.
    """

    # Manifest metadata
    manifest_version: Literal["1.0"]  # Schema version for forward compatibility
    model_id: str  # Unique identifier for this trained model
    model_path: str  # Relative path to model file
    model_format: Literal["ubj", "pt"]  # XGBoost UBJ or PyTorch .pt

    # Training context
    created_at: str  # ISO 8601 timestamp
    versions: ManifestVersions
    system: ManifestSystem

    # Data and training
    dataset: ManifestDataset
    training: ManifestTraining
    metrics: ManifestMetrics

    # Model outputs
    feature_importances: list[FeatureImportance]  # Empty list for MLP


__all__ = [
    "ClassifierManifest",
    "ManifestDataset",
    "ManifestMetrics",
    "ManifestSystem",
    "ManifestTraining",
    "ManifestVersions",
]
