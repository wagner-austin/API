"""Tests for manifest TypedDict types."""

from __future__ import annotations

from covenant_ml.manifest import (
    ClassifierManifest,
    ManifestDataset,
    ManifestMetrics,
    ManifestSystem,
    ManifestTraining,
    ManifestVersions,
)
from covenant_ml.types import EvalMetrics, FeatureImportance, TrainConfig


class TestManifestVersions:
    """Tests for ManifestVersions TypedDict."""

    def test_manifest_versions_required_fields(self) -> None:
        """ManifestVersions accepts all required fields."""
        versions: ManifestVersions = {
            "covenant_ml": "0.1.0",
            "python": "3.11.9",
            "xgboost": "3.1.0",
            "torch": None,
            "numpy": "2.3.0",
            "scikit_learn": "1.7.0",
        }
        assert versions["covenant_ml"] == "0.1.0"
        assert versions["python"] == "3.11.9"
        assert versions["xgboost"] == "3.1.0"
        assert versions["torch"] is None
        assert versions["numpy"] == "2.3.0"
        assert versions["scikit_learn"] == "1.7.0"

    def test_manifest_versions_mlp_backend(self) -> None:
        """ManifestVersions for MLP backend has torch but no xgboost."""
        versions: ManifestVersions = {
            "covenant_ml": "0.1.0",
            "python": "3.11.9",
            "xgboost": None,
            "torch": "2.5.0",
            "numpy": "2.3.0",
            "scikit_learn": "1.7.0",
        }
        assert versions["xgboost"] is None
        assert versions["torch"] == "2.5.0"


class TestManifestSystem:
    """Tests for ManifestSystem TypedDict."""

    def test_manifest_system_cpu(self) -> None:
        """ManifestSystem for CPU training."""
        system: ManifestSystem = {
            "platform": "linux",
            "device_used": "cpu",
            "cuda_version": None,
            "gpu_name": None,
        }
        assert system["platform"] == "linux"
        assert system["device_used"] == "cpu"
        assert system["cuda_version"] is None
        assert system["gpu_name"] is None

    def test_manifest_system_cuda(self) -> None:
        """ManifestSystem for CUDA training."""
        system: ManifestSystem = {
            "platform": "linux",
            "device_used": "cuda",
            "cuda_version": "12.4",
            "gpu_name": "NVIDIA GeForce RTX 4090",
        }
        assert system["device_used"] == "cuda"
        assert system["cuda_version"] == "12.4"
        assert system["gpu_name"] == "NVIDIA GeForce RTX 4090"


class TestManifestDataset:
    """Tests for ManifestDataset TypedDict."""

    def test_manifest_dataset_fields(self) -> None:
        """ManifestDataset accepts all required fields."""
        dataset: ManifestDataset = {
            "name": "us",
            "samples_total": 10000,
            "samples_train": 7000,
            "samples_val": 1500,
            "samples_test": 1500,
            "n_features": 18,
            "n_positive": 500,
            "n_negative": 9500,
            "class_ratio": 19.0,
        }
        assert dataset["name"] == "us"
        assert dataset["samples_total"] == 10000
        assert dataset["samples_train"] == 7000
        assert dataset["samples_val"] == 1500
        assert dataset["samples_test"] == 1500
        assert dataset["n_features"] == 18
        assert dataset["n_positive"] == 500
        assert dataset["n_negative"] == 9500
        assert dataset["class_ratio"] == 19.0


class TestManifestTraining:
    """Tests for ManifestTraining TypedDict."""

    def test_manifest_training_xgboost(self) -> None:
        """ManifestTraining for XGBoost backend."""
        config: TrainConfig = {
            "device": "cpu",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "early_stopping_rounds": 10,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        training: ManifestTraining = {
            "backend": "xgboost",
            "config": config,
            "best_round": 85,
            "total_rounds": 100,
            "early_stopped": True,
            "scale_pos_weight_computed": 19.0,
            "training_duration_seconds": 12.5,
        }
        assert training["backend"] == "xgboost"
        assert training["best_round"] == 85
        assert training["early_stopped"]
        assert training["training_duration_seconds"] == 12.5


class TestManifestMetrics:
    """Tests for ManifestMetrics TypedDict."""

    def test_manifest_metrics_fields(self) -> None:
        """ManifestMetrics accepts all required fields."""
        train_metrics: EvalMetrics = {
            "loss": 0.15,
            "ppl": 1.16,
            "auc": 0.95,
            "accuracy": 0.90,
            "precision": 0.85,
            "recall": 0.80,
            "f1_score": 0.82,
        }
        val_metrics: EvalMetrics = {
            "loss": 0.20,
            "ppl": 1.22,
            "auc": 0.90,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
        }
        test_metrics: EvalMetrics = {
            "loss": 0.22,
            "ppl": 1.25,
            "auc": 0.88,
            "accuracy": 0.83,
            "precision": 0.78,
            "recall": 0.73,
            "f1_score": 0.75,
        }
        metrics: ManifestMetrics = {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "best_val_auc": 0.90,
        }
        assert metrics["train"]["auc"] == 0.95
        assert metrics["val"]["auc"] == 0.90
        assert metrics["test"]["auc"] == 0.88
        assert metrics["best_val_auc"] == 0.90


class TestClassifierManifest:
    """Tests for ClassifierManifest TypedDict."""

    def test_classifier_manifest_xgboost_complete(self) -> None:
        """ClassifierManifest accepts all required fields for XGBoost."""
        versions: ManifestVersions = {
            "covenant_ml": "0.1.0",
            "python": "3.11.9",
            "xgboost": "3.1.0",
            "torch": None,
            "numpy": "2.3.0",
            "scikit_learn": "1.7.0",
        }
        system: ManifestSystem = {
            "platform": "linux",
            "device_used": "cpu",
            "cuda_version": None,
            "gpu_name": None,
        }
        dataset: ManifestDataset = {
            "name": "us",
            "samples_total": 1000,
            "samples_train": 700,
            "samples_val": 150,
            "samples_test": 150,
            "n_features": 18,
            "n_positive": 50,
            "n_negative": 950,
            "class_ratio": 19.0,
        }
        config: TrainConfig = {
            "device": "cpu",
            "learning_rate": 0.1,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "early_stopping_rounds": 10,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }
        training: ManifestTraining = {
            "backend": "xgboost",
            "config": config,
            "best_round": 85,
            "total_rounds": 100,
            "early_stopped": True,
            "scale_pos_weight_computed": 19.0,
            "training_duration_seconds": 5.2,
        }
        eval_metrics: EvalMetrics = {
            "loss": 0.20,
            "ppl": 1.22,
            "auc": 0.90,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
        }
        metrics: ManifestMetrics = {
            "train": eval_metrics,
            "val": eval_metrics,
            "test": eval_metrics,
            "best_val_auc": 0.90,
        }
        importance: FeatureImportance = {
            "name": "feature_1",
            "importance": 0.25,
            "rank": 1,
        }

        manifest: ClassifierManifest = {
            "manifest_version": "1.0",
            "model_id": "model_20240101_120000",
            "model_path": "models/model_20240101_120000.ubj",
            "model_format": "ubj",
            "created_at": "2024-01-01T12:00:00Z",
            "versions": versions,
            "system": system,
            "dataset": dataset,
            "training": training,
            "metrics": metrics,
            "feature_importances": [importance],
        }

        assert manifest["manifest_version"] == "1.0"
        assert manifest["model_format"] == "ubj"
        assert manifest["training"]["backend"] == "xgboost"
        assert len(manifest["feature_importances"]) == 1

    def test_classifier_manifest_mlp_complete(self) -> None:
        """ClassifierManifest accepts all required fields for MLP."""
        from covenant_ml.types import MLPConfig

        versions: ManifestVersions = {
            "covenant_ml": "0.1.0",
            "python": "3.11.9",
            "xgboost": None,
            "torch": "2.5.0",
            "numpy": "2.3.0",
            "scikit_learn": "1.7.0",
        }
        system: ManifestSystem = {
            "platform": "linux",
            "device_used": "cuda",
            "cuda_version": "12.4",
            "gpu_name": "NVIDIA GeForce RTX 4090",
        }
        dataset: ManifestDataset = {
            "name": "taiwan",
            "samples_total": 500,
            "samples_train": 350,
            "samples_val": 75,
            "samples_test": 75,
            "n_features": 23,
            "n_positive": 100,
            "n_negative": 400,
            "class_ratio": 4.0,
        }
        mlp_config: MLPConfig = {
            "device": "cuda",
            "precision": "fp16",
            "optimizer": "adamw",
            "hidden_sizes": (64, 32),
            "learning_rate": 0.001,
            "batch_size": 32,
            "n_epochs": 100,
            "dropout": 0.2,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_state": 42,
            "early_stopping_patience": 10,
        }
        training: ManifestTraining = {
            "backend": "mlp",
            "config": mlp_config,
            "best_round": 45,
            "total_rounds": 100,
            "early_stopped": True,
            "scale_pos_weight_computed": 4.0,
            "training_duration_seconds": 8.7,
        }
        eval_metrics: EvalMetrics = {
            "loss": 0.25,
            "ppl": 1.28,
            "auc": 0.85,
            "accuracy": 0.80,
            "precision": 0.75,
            "recall": 0.70,
            "f1_score": 0.72,
        }
        metrics: ManifestMetrics = {
            "train": eval_metrics,
            "val": eval_metrics,
            "test": eval_metrics,
            "best_val_auc": 0.85,
        }

        manifest: ClassifierManifest = {
            "manifest_version": "1.0",
            "model_id": "model_20240101_130000",
            "model_path": "models/model_20240101_130000.pt",
            "model_format": "pt",
            "created_at": "2024-01-01T13:00:00Z",
            "versions": versions,
            "system": system,
            "dataset": dataset,
            "training": training,
            "metrics": metrics,
            "feature_importances": [],  # MLP has no feature importances
        }

        assert manifest["manifest_version"] == "1.0"
        assert manifest["model_format"] == "pt"
        assert manifest["training"]["backend"] == "mlp"
        assert manifest["feature_importances"] == []


class TestManifestExports:
    """Tests for manifest module exports."""

    def test_package_exports_manifest_types(self) -> None:
        """covenant_ml package exports all manifest types."""
        import covenant_ml

        # Verify they're in __all__ (strong assertion vs hasattr)
        assert "ClassifierManifest" in covenant_ml.__all__
        assert "ManifestDataset" in covenant_ml.__all__
        assert "ManifestMetrics" in covenant_ml.__all__
        assert "ManifestSystem" in covenant_ml.__all__
        assert "ManifestTraining" in covenant_ml.__all__
        assert "ManifestVersions" in covenant_ml.__all__
        # Verify import works by accessing TypedDict __annotations__
        assert "manifest_version" in covenant_ml.ClassifierManifest.__annotations__
        assert "covenant_ml" in covenant_ml.ManifestVersions.__annotations__
