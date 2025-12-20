"""Tests for model saving after optimization.

Comprehensive tests for scripts/optimize/model_saver.py covering:
- Path generation functions
- AUC comparison logic
- Metadata loading and saving
- Full save_best_model workflow with hooks

Strict typing only: no Any, no casts, no type: ignore, no stubs.
No mocks - uses test hooks for dependency injection.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
import scripts._test_hooks as _hooks
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    ClassifierBackend,
    PreparedClassifier,
    ProgressCallback,
)
from covenant_ml.backends.registry import (
    BackendRegistration,
    ClassifierRegistry,
)
from covenant_ml.datasets import DatasetConfig, DatasetMeta, DatasetRegistry, LoadedDataset
from covenant_ml.testing import make_train_config
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainConfig,
    TrainOutcome,
)
from numpy.typing import NDArray
from platform_core.json_utils import dump_json_str
from platform_core.logging import setup_rich_logging
from scripts._test_hooks import XGBoostOptimizationResult
from scripts.optimize.cli import DatasetName
from scripts.optimize.model_saver import (
    MODEL_EXTENSIONS,
    SaveModelResult,
    load_existing_auc,
    save_best_model,
    should_save_model,
)


@pytest.fixture(autouse=True)
def setup_logging() -> None:
    """Ensure rich logging is setup for all tests in this module."""
    setup_rich_logging(level="WARNING", show_time=False)


# =============================================================================
# Fake Implementations for Testing
# =============================================================================


class FakePreparedClassifier:
    """Fake classifier for testing that returns constant predictions."""

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return constant probability predictions.

        Args:
            x: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, 2).
        """
        n_samples: int = int(x.shape[0])
        col_0: NDArray[np.float64] = np.full(n_samples, 0.3, dtype=np.float64)
        col_1: NDArray[np.float64] = np.full(n_samples, 0.7, dtype=np.float64)
        result: NDArray[np.float64] = np.column_stack([col_0, col_1])
        return result


class FakeClassifierBackend:
    """Fake backend for testing model saving workflow.

    Returns deterministic training outcomes without actual ML training.
    """

    def __init__(
        self,
        backend_name_val: BackendName = "xgboost",
        output_filename_override: str | None = None,
    ) -> None:
        """Initialize with configurable backend name and output filename.

        Args:
            backend_name_val: Backend name to return.
            output_filename_override: Optional override for output model filename.
                If set, uses this exact filename instead of default pattern.
        """
        self._backend_name = backend_name_val
        self._train_call_count = 0
        self._output_filename_override = output_filename_override

    def backend_name(self) -> BackendName:
        """Return the configured backend name.

        Returns:
            Backend name string.
        """
        return self._backend_name

    def capabilities(self) -> BackendCapabilities:
        """Return fake backend capabilities.

        Returns:
            BackendCapabilities with all features enabled.
        """
        return {
            "supports_train": True,
            "supports_gpu": False,
            "supports_early_stopping": True,
            "supports_feature_importance": True,
            "model_format": MODEL_EXTENSIONS[self._backend_name],
        }

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        """Prepare a fake classifier.

        Args:
            n_features: Number of input features.
            n_classes: Number of output classes.
            feature_names: Optional feature names.

        Returns:
            FakePreparedClassifier instance.
        """
        return FakePreparedClassifier()

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
        """Return a fake training outcome without actual training.

        Args:
            x_features: Feature matrix.
            y_labels: Label vector.
            feature_names: Optional feature names.
            config: Training configuration.
            output_dir: Output directory for model.
            progress: Optional progress callback.

        Returns:
            TrainOutcome with deterministic values.
        """
        self._train_call_count += 1
        ext = MODEL_EXTENSIONS[self._backend_name]

        # Use override filename if set, otherwise default pattern
        if self._output_filename_override is not None:
            model_path = output_dir / self._output_filename_override
        else:
            model_path = output_dir / f"model_{self._train_call_count}.{ext}"

        # Write a fake model file
        model_path.write_bytes(b"fake model data")

        n_samples: int = int(x_features.shape[0])
        n_train: int = int(n_samples * 0.7)
        n_val: int = int(n_samples * 0.15)
        n_test: int = n_samples - n_train - n_val

        fake_metrics: EvalMetrics = {
            "loss": 0.35,
            "ppl": 1.42,
            "auc": 0.88,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
        }

        # Build feature importances
        n_features: int = int(x_features.shape[1])
        fake_importances: list[FeatureImportance] = [
            {"name": f"feature_{i}", "importance": 1.0 / n_features, "rank": i + 1}
            for i in range(n_features)
        ]

        train_outcome: TrainOutcome = {
            "model_path": str(model_path),
            "model_id": f"fake-model-{self._train_call_count}",
            "samples_total": n_samples,
            "samples_train": n_train,
            "samples_val": n_val,
            "samples_test": n_test,
            "train_metrics": fake_metrics,
            "val_metrics": fake_metrics,
            "test_metrics": fake_metrics,
            "best_val_auc": 0.88,
            "best_round": 50,
            "total_rounds": 100,
            "early_stopped": True,
            "config": config,
            "feature_importances": fake_importances,
            "scale_pos_weight_computed": 1.0,
        }
        return train_outcome

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Return fake evaluation metrics.

        Args:
            model: Prepared classifier.
            x: Feature matrix.
            y: Label vector.

        Returns:
            EvalMetrics with deterministic values.
        """
        return {
            "loss": 0.35,
            "ppl": 1.42,
            "auc": 0.88,
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.75,
            "f1_score": 0.77,
        }

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        """Save fake model to path.

        Args:
            model: Prepared classifier.
            path: Output path.
        """
        Path(path).write_bytes(b"fake model data")

    def load(self, *, path: str) -> PreparedClassifier:
        """Load fake model from path.

        Args:
            path: Model path.

        Returns:
            FakePreparedClassifier instance.
        """
        return FakePreparedClassifier()

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Return fake feature importances.

        Args:
            model: Prepared classifier.
            feature_names: Optional feature names.

        Returns:
            List of fake feature importances.
        """
        if feature_names is None:
            return None
        n_features = len(feature_names)
        return [
            {"name": name, "importance": 1.0 / n_features, "rank": i + 1}
            for i, name in enumerate(feature_names)
        ]


def _make_fake_dataset_config(name: str = "taiwan") -> DatasetConfig:
    """Create a fake dataset config for testing.

    Args:
        name: Dataset name.

    Returns:
        DatasetConfig with minimal valid values.
    """
    return {
        "name": name,
        "display_name": f"Fake {name.title()} Dataset",
        "folder": f"{name}_data",
        "file_name": "data.csv",
        "file_format": "csv",
        "encoding": "utf-8",
        "target": {
            "column_name": "target",
            "label_type": "binary_int",
            "positive_values": (1,),
            "negative_values": (0,),
        },
        "exclude_columns": (),
        "n_samples_expected": 100,
        "n_features_expected": 10,
        "positive_class_ratio_expected": 0.3,
    }


def _make_fake_loaded_dataset(n_samples: int = 100, n_features: int = 10) -> LoadedDataset:
    """Create a fake loaded dataset for testing.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.

    Returns:
        LoadedDataset with random data.
    """
    rng = np.random.default_rng(42)
    x = rng.random((n_samples, n_features))
    y = rng.integers(0, 2, size=n_samples).astype(np.int64)
    n_positive = int(np.sum(y))

    meta: DatasetMeta = {
        "name": "fake_dataset",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_positive": n_positive,
        "n_negative": n_samples - n_positive,
        "positive_ratio": n_positive / n_samples,
        "feature_names": tuple(f"feature_{i}" for i in range(n_features)),
        "categorical_encodings": (),
    }

    return {
        "meta": meta,
        "x": x,
        "y": y,
    }


def _make_fake_train_config() -> TrainConfig:
    """Create a fake training configuration.

    Returns:
        TrainConfig with default values.
    """
    return make_train_config(
        device="cpu",
        learning_rate=0.1,
        max_depth=3,
        n_estimators=10,
    )


def _make_fake_optimization_result(
    best_val_auc: float = 0.85,
    dataset: str = "taiwan",
) -> XGBoostOptimizationResult:
    """Create a fake XGBoost optimization result for testing.

    Args:
        best_val_auc: Best validation AUC.
        dataset: Dataset name.

    Returns:
        XGBoostOptimizationResult with specified values.
    """
    return {
        "backend": "xgboost",
        "status": "complete",
        "dataset": dataset,
        "n_samples": 1000,
        "n_features": 100,
        "feature_preset": "full",
        "n_trials_complete": 10,
        "n_trials_pruned": 2,
        "n_trials_failed": 0,
        "best_trial_number": 5,
        "best_val_auc": best_val_auc,
        "best_max_depth": 6,
        "best_n_estimators": 100,
        "best_learning_rate": 0.1,
        "best_reg_alpha": 0.01,
        "best_reg_lambda": 0.01,
        "best_subsample": 0.8,
        "best_colsample_bytree": 0.8,
        "duration_seconds": 10.0,
        "recommended_config": _make_fake_train_config(),
    }


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def fake_backend() -> FakeClassifierBackend:
    """Create a fake XGBoost backend for testing.

    Returns:
        FakeClassifierBackend instance.
    """
    return FakeClassifierBackend("xgboost")


@pytest.fixture()
def fake_backend_registry(fake_backend: FakeClassifierBackend) -> ClassifierRegistry:
    """Create a registry with a fake backend.

    Args:
        fake_backend: Fake backend to register.

    Returns:
        ClassifierRegistry with fake backend.
    """
    registry = ClassifierRegistry()

    def factory() -> ClassifierBackend:
        return fake_backend

    registry.register("xgboost", BackendRegistration(factory))
    return registry


@pytest.fixture()
def fake_dataset_registry() -> DatasetRegistry:
    """Create a registry with fake dataset configs.

    Returns:
        DatasetRegistry with taiwan, us, polish configs.
    """
    configs = (
        _make_fake_dataset_config("taiwan"),
        _make_fake_dataset_config("us"),
        _make_fake_dataset_config("polish"),
    )
    return DatasetRegistry(configs)


@pytest.fixture()
def hooks_context(
    fake_backend_registry: ClassifierRegistry,
    fake_dataset_registry: DatasetRegistry,
) -> Generator[None, None, None]:
    """Set up and tear down test hooks for model saving.

    Args:
        fake_backend_registry: Registry with fake backend.
        fake_dataset_registry: Registry with fake datasets.

    Yields:
        None after setting up hooks, restores after test.
    """
    original_backend_factory = _hooks.backend_registry_factory
    original_dataset_registry = _hooks.dataset_registry_factory
    original_dataset_loader = _hooks.dataset_loader

    def fake_backend_factory() -> ClassifierRegistry:
        return fake_backend_registry

    def fake_registry_factory() -> DatasetRegistry:
        return fake_dataset_registry

    def fake_loader(config: DatasetConfig, external_dir: Path) -> LoadedDataset:
        return _make_fake_loaded_dataset()

    _hooks.backend_registry_factory = fake_backend_factory
    _hooks.dataset_registry_factory = fake_registry_factory
    _hooks.dataset_loader = fake_loader

    yield

    _hooks.backend_registry_factory = original_backend_factory
    _hooks.dataset_registry_factory = original_dataset_registry
    _hooks.dataset_loader = original_dataset_loader


# =============================================================================
# Tests for should_save_model
# =============================================================================


class TestShouldSaveModel:
    """Tests for should_save_model decision logic."""

    def test_save_when_no_existing_model(self) -> None:
        """Test returns True when no existing model (None AUC)."""
        result = should_save_model(new_auc=0.85, existing_auc=None)
        assert result is True

    def test_save_when_new_is_better(self) -> None:
        """Test returns True when new AUC is strictly better."""
        result = should_save_model(new_auc=0.90, existing_auc=0.85)
        assert result is True

    def test_no_save_when_new_is_worse(self) -> None:
        """Test returns False when new AUC is worse."""
        result = should_save_model(new_auc=0.80, existing_auc=0.85)
        assert result is False

    def test_no_save_when_equal(self) -> None:
        """Test returns False when AUCs are equal (not strictly better)."""
        result = should_save_model(new_auc=0.85, existing_auc=0.85)
        assert result is False

    def test_save_with_minimal_improvement(self) -> None:
        """Test saves with minimal but real improvement."""
        result = should_save_model(new_auc=0.850001, existing_auc=0.85)
        assert result is True


# =============================================================================
# Tests for load_existing_auc
# =============================================================================


class TestLoadExistingAuc:
    """Tests for load_existing_auc metadata loading."""

    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        """Test returns None when metadata file doesn't exist."""
        result = load_existing_auc(tmp_path, "taiwan", "xgboost")
        assert result is None

    def test_loads_auc_from_valid_metadata(self, tmp_path: Path) -> None:
        """Test correctly loads AUC from valid metadata JSON."""
        meta_path = tmp_path / "taiwan_xgboost_best_meta.json"
        meta_content = {
            "backend": "xgboost",
            "dataset": "taiwan",
            "best_val_auc": 0.8765,
            "saved_at": "2024-01-01T00:00:00+00:00",
        }
        meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        result = load_existing_auc(tmp_path, "taiwan", "xgboost")
        assert result == 0.8765

    def test_loads_for_different_backends(self, tmp_path: Path) -> None:
        """Test loads correct file for each backend type."""
        backends: list[BackendName] = ["xgboost", "mlp", "lightgbm", "lstm"]
        expected_aucs = [0.80, 0.82, 0.84, 0.86]

        for backend, auc in zip(backends, expected_aucs, strict=True):
            meta_path = tmp_path / f"taiwan_{backend}_best_meta.json"
            meta_content = {"best_val_auc": auc}
            meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        for backend, expected_auc in zip(backends, expected_aucs, strict=True):
            result = load_existing_auc(tmp_path, "taiwan", backend)
            assert result == expected_auc

    def test_loads_for_different_datasets(self, tmp_path: Path) -> None:
        """Test loads correct file for each dataset."""
        datasets = ["taiwan", "us", "polish"]
        expected_aucs = [0.80, 0.85, 0.90]

        for dataset, auc in zip(datasets, expected_aucs, strict=True):
            meta_path = tmp_path / f"{dataset}_xgboost_best_meta.json"
            meta_content = {"best_val_auc": auc}
            meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        for dataset, expected_auc in zip(datasets, expected_aucs, strict=True):
            result = load_existing_auc(tmp_path, dataset, "xgboost")
            assert result == expected_auc


# =============================================================================
# Tests for MODEL_EXTENSIONS
# =============================================================================


class TestModelExtensions:
    """Tests for MODEL_EXTENSIONS mapping."""

    def test_xgboost_extension(self) -> None:
        """Test XGBoost uses .ubj extension."""
        assert MODEL_EXTENSIONS["xgboost"] == "ubj"

    def test_mlp_extension(self) -> None:
        """Test MLP uses .pt extension."""
        assert MODEL_EXTENSIONS["mlp"] == "pt"

    def test_lightgbm_extension(self) -> None:
        """Test LightGBM uses .txt extension."""
        assert MODEL_EXTENSIONS["lightgbm"] == "txt"

    def test_lstm_extension(self) -> None:
        """Test LSTM uses .pt extension."""
        assert MODEL_EXTENSIONS["lstm"] == "pt"

    def test_all_backends_covered(self) -> None:
        """Test all backend names have extensions defined."""
        backends: list[BackendName] = ["xgboost", "mlp", "lightgbm", "lstm"]
        for backend in backends:
            assert backend in MODEL_EXTENSIONS


# =============================================================================
# Tests for save_best_model
# =============================================================================


class TestSaveBestModel:
    """Tests for save_best_model main workflow."""

    def test_saves_when_no_existing_model(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test saves model when no existing model exists."""
        result = _make_fake_optimization_result(best_val_auc=0.85)

        save_result: SaveModelResult = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        assert save_result["saved"] is True
        assert save_result["reason"] == "New best model"
        # Verify paths contain expected components
        model_path = save_result["model_path"]
        meta_path = save_result["meta_path"]
        train_outcome = save_result["train_outcome"]
        assert model_path is not None and "taiwan_xgboost_best.ubj" in model_path
        assert meta_path is not None and "taiwan_xgboost_best_meta.json" in meta_path
        assert train_outcome is not None and train_outcome["best_val_auc"] == 0.88

        # Verify files exist
        assert Path(model_path).exists()
        assert Path(meta_path).exists()

    def test_skips_when_existing_is_better(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test skips saving when existing model has better AUC."""
        # Create existing metadata with high AUC
        output_dir = tmp_path / "models" / "xgboost"
        output_dir.mkdir(parents=True)
        meta_path = output_dir / "taiwan_xgboost_best_meta.json"
        meta_content = {"best_val_auc": 0.95}
        meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        result = _make_fake_optimization_result(best_val_auc=0.85)

        save_result: SaveModelResult = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        assert save_result["saved"] is False
        assert "not better than existing" in save_result["reason"]
        assert save_result["model_path"] is None
        assert save_result["meta_path"] is None
        assert save_result["train_outcome"] is None

    def test_saves_when_new_is_better(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test saves model when new AUC is better than existing."""
        # Create existing metadata with low AUC
        output_dir = tmp_path / "models" / "xgboost"
        output_dir.mkdir(parents=True)
        meta_path = output_dir / "taiwan_xgboost_best_meta.json"
        meta_content = {"best_val_auc": 0.70}
        meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        result = _make_fake_optimization_result(best_val_auc=0.85)

        save_result: SaveModelResult = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        assert save_result["saved"] is True
        assert save_result["reason"] == "New best model"

    def test_creates_output_directory(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test creates output directory if it doesn't exist."""
        result = _make_fake_optimization_result()

        save_result = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        assert save_result["saved"] is True
        output_dir = tmp_path / "models" / "xgboost"
        assert output_dir.exists()

    def test_replaces_existing_model_file(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test replaces existing model file when saving better model."""
        # Create existing model and metadata with low AUC
        output_dir = tmp_path / "models" / "xgboost"
        output_dir.mkdir(parents=True)

        existing_model = output_dir / "taiwan_xgboost_best.ubj"
        existing_model.write_bytes(b"old model")

        meta_path = output_dir / "taiwan_xgboost_best_meta.json"
        meta_content = {"best_val_auc": 0.50}
        meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        result = _make_fake_optimization_result(best_val_auc=0.85)

        save_result = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        assert save_result["saved"] is True

        # Verify model was replaced
        new_model_content = existing_model.read_bytes()
        assert new_model_content != b"old model"

    def test_metadata_contains_expected_fields(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test saved metadata contains all expected fields."""
        from platform_core.json_utils import load_json_str, narrow_json_to_dict

        result = _make_fake_optimization_result()

        save_result = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        meta_path_str = save_result["meta_path"]
        assert meta_path_str is not None and Path(meta_path_str).exists()
        meta_content = Path(meta_path_str).read_text(encoding="utf-8")
        meta_json = load_json_str(meta_content)
        meta = narrow_json_to_dict(meta_json)

        assert meta["backend"] == "xgboost"
        assert meta["dataset"] == "taiwan"
        assert meta["feature_preset"] == "full"
        # Verify AUC value is reasonable
        from platform_core.json_utils import require_float, require_int, require_str

        best_val_auc = require_float(meta, "best_val_auc")
        assert 0.0 <= best_val_auc <= 1.0
        # Verify saved_at is an ISO timestamp string
        saved_at = require_str(meta, "saved_at")
        assert "T" in saved_at  # ISO format contains T
        # Verify model_path is a string path
        model_path_val = require_str(meta, "model_path")
        assert "xgboost" in model_path_val
        # Verify n_features and n_samples are positive integers
        n_features = require_int(meta, "n_features")
        n_samples = require_int(meta, "n_samples")
        assert n_features > 0
        assert n_samples > 0
        # Verify config is a dict
        config_val = meta["config"]
        assert config_val is not None and isinstance(config_val, dict)

    def test_works_with_different_datasets(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test saving works for all supported datasets."""
        datasets: list[DatasetName] = ["taiwan", "us", "polish"]

        for dataset in datasets:
            result = _make_fake_optimization_result(dataset=dataset)

            save_result = save_best_model(
                result=result,
                dataset=dataset,
                feature_preset="full",
                project_root=tmp_path,
            )

            assert save_result["saved"] is True
            assert dataset in str(save_result["model_path"])

    def test_skips_rename_when_paths_equal(
        self,
        tmp_path: Path,
        fake_dataset_registry: DatasetRegistry,
    ) -> None:
        """Test no rename when trained model is already at best path.

        This covers the branch where trained_model_path == best_model_path.
        """
        # Create backend that writes directly to the best model path
        best_filename = "taiwan_xgboost_best.ubj"
        custom_backend = FakeClassifierBackend(
            backend_name_val="xgboost",
            output_filename_override=best_filename,
        )

        # Set up hooks with custom backend
        custom_registry = ClassifierRegistry()

        def factory() -> ClassifierBackend:
            return custom_backend

        custom_registry.register("xgboost", BackendRegistration(factory))

        original_backend_factory = _hooks.backend_registry_factory
        original_dataset_registry = _hooks.dataset_registry_factory
        original_dataset_loader = _hooks.dataset_loader

        def fake_backend_factory() -> ClassifierRegistry:
            return custom_registry

        def fake_registry_factory() -> DatasetRegistry:
            return fake_dataset_registry

        def fake_loader(config: DatasetConfig, external_dir: Path) -> LoadedDataset:
            return _make_fake_loaded_dataset()

        _hooks.backend_registry_factory = fake_backend_factory
        _hooks.dataset_registry_factory = fake_registry_factory
        _hooks.dataset_loader = fake_loader

        try:
            result = _make_fake_optimization_result()

            save_result = save_best_model(
                result=result,
                dataset="taiwan",
                feature_preset="full",
                project_root=tmp_path,
            )

            # Verify save succeeded
            assert save_result["saved"] is True
            assert save_result["reason"] == "New best model"

            # Verify model exists at best path
            model_path = save_result["model_path"]
            assert model_path is not None and best_filename in model_path
            assert Path(model_path).exists()
        finally:
            _hooks.backend_registry_factory = original_backend_factory
            _hooks.dataset_registry_factory = original_dataset_registry
            _hooks.dataset_loader = original_dataset_loader

    def test_train_outcome_has_correct_structure(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test returned train_outcome has correct TypedDict structure."""
        result = _make_fake_optimization_result()

        save_result = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        # Extract and verify train_outcome
        train_outcome = save_result["train_outcome"]
        assert train_outcome is not None and train_outcome["best_val_auc"] == 0.88
        outcome: TrainOutcome = train_outcome

        # Verify all required TrainOutcome fields have expected values
        # Note: model_path in train_outcome points to original temp location
        # which was renamed to the best model path, so check saved model exists
        saved_model_path = save_result["model_path"]
        assert saved_model_path is not None and Path(saved_model_path).exists()
        # Verify train_outcome has a valid model_path string (even if renamed)
        assert outcome["model_path"] != "" and "model" in outcome["model_path"]
        assert outcome["model_id"].startswith("fake-model-")
        assert outcome["samples_total"] == 100
        assert outcome["samples_train"] == 70
        assert outcome["samples_val"] == 15
        assert outcome["samples_test"] == 15
        assert outcome["train_metrics"]["auc"] == 0.88
        assert outcome["val_metrics"]["auc"] == 0.88
        assert outcome["test_metrics"]["auc"] == 0.88
        assert outcome["best_val_auc"] == 0.88
        assert outcome["best_round"] == 50
        assert outcome["total_rounds"] == 100
        assert outcome["early_stopped"] is True


# =============================================================================
# Tests for SaveModelResult TypedDict
# =============================================================================


class TestSaveModelResultTypedDict:
    """Tests for SaveModelResult TypedDict structure."""

    def test_successful_result_structure(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test successful save result has all fields set with correct values."""
        result = _make_fake_optimization_result()

        save_result = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        # Verify saved is True for success
        assert save_result["saved"] is True
        # Verify reason indicates success
        assert save_result["reason"] == "New best model"
        # Verify paths are set and contain expected components
        model_path = save_result["model_path"]
        meta_path = save_result["meta_path"]
        assert model_path is not None and "taiwan_xgboost_best.ubj" in model_path
        assert meta_path is not None and "taiwan_xgboost_best_meta.json" in meta_path

    def test_skipped_result_structure(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test skipped save result has correct None values."""
        # Create existing high-AUC model
        output_dir = tmp_path / "models" / "xgboost"
        output_dir.mkdir(parents=True)
        meta_path = output_dir / "taiwan_xgboost_best_meta.json"
        meta_content = {"best_val_auc": 0.99}
        meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        result = _make_fake_optimization_result(best_val_auc=0.50)

        save_result = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        assert save_result["saved"] is False
        assert save_result["model_path"] is None
        assert save_result["meta_path"] is None
        assert save_result["train_outcome"] is None
        # Verify reason contains specific text about AUC comparison
        assert "not better than existing" in save_result["reason"]
        assert "0.50" in save_result["reason"]  # New AUC
        assert "0.99" in save_result["reason"]  # Existing AUC
