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
    PreparedClassifier,
    ProgressCallback,
)
from covenant_ml.backends.registry import (
    BackendRegistration,
    ClassifierRegistry,
)
from covenant_ml.datasets import DatasetConfig, DatasetMeta, DatasetRegistry, LoadedDataset
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    SearchSpace,
    XGBoostSearchSpace,
)
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
)
from numpy.typing import NDArray
from platform_core.json_utils import dump_json_str
from platform_core.logging import setup_rich_logging
from scripts._test_hooks import UnifiedOptimizationResult
from scripts.optimize.cli import DatasetName
from scripts.optimize.model_saver import (
    MODEL_EXTENSIONS,
    SaveModelResult,
    _build_cleargbm_config,
    _build_lightgbm_config,
    _build_logreg_config,
    _build_lstm_config,
    _build_mlp_config,
    _build_random_forest_config,
    _build_xgboost_config,
    _narrow_logreg_penalty,
    _narrow_logreg_solver,
    build_train_config,
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

        if self._output_filename_override is not None:
            model_path = output_dir / self._output_filename_override
        else:
            model_path = output_dir / f"model_{self._train_call_count}.{ext}"

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

        n_features_count: int = int(x_features.shape[1])
        fake_importances: list[FeatureImportance] = [
            {"name": f"feature_{i}", "importance": 1.0 / n_features_count, "rank": i + 1}
            for i in range(n_features_count)
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

    def get_default_search_space(self) -> SearchSpace:
        """Return fake default search space.

        Returns:
            XGBoostSearchSpace with minimal ranges.
        """
        return XGBoostSearchSpace(
            max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
            n_estimators=IntRangeSpec(param_type="int", low=50, high=500, log_scale=False),
            learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
            reg_alpha=FloatRangeSpec(param_type="float", low=1e-8, high=10.0, log_scale=True),
            reg_lambda=FloatRangeSpec(param_type="float", low=1e-8, high=10.0, log_scale=True),
            subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
            colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
        )

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Return fake focused search space.

        Args:
            best_int_params: Best integer parameters from previous run.
            best_float_params: Best float parameters from previous run.

        Returns:
            XGBoostSearchSpace with minimal ranges.
        """
        return self.get_default_search_space()


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


def _make_fake_optimization_result(
    best_value: float = 0.85,
    dataset: str = "taiwan",
) -> UnifiedOptimizationResult:
    """Create a fake unified optimization result for testing.

    Args:
        best_value: Best validation AUC.
        dataset: Dataset name.

    Returns:
        UnifiedOptimizationResult with specified values.
    """
    return UnifiedOptimizationResult(
        backend="xgboost",
        status="complete",
        dataset=dataset,
        n_samples=1000,
        n_features=100,
        feature_preset="full",
        n_trials_complete=10,
        n_trials_pruned=2,
        n_trials_failed=0,
        best_trial_number=5,
        best_value=best_value,
        best_int_params=SampledIntParams(max_depth=6, n_estimators=100),
        best_float_params=SampledFloatParams(
            learning_rate=0.1,
            reg_alpha=0.01,
            reg_lambda=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        best_string_params=SampledStringParams(),
        duration_seconds=10.0,
    )


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

    def factory() -> FakeClassifierBackend:
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

    def test_cleargbm_extension(self) -> None:
        """Test ClearGBM uses .json extension."""
        assert MODEL_EXTENSIONS["cleargbm"] == "json"

    def test_logreg_extension(self) -> None:
        """Test LogReg uses .joblib extension."""
        assert MODEL_EXTENSIONS["logreg"] == "joblib"

    def test_random_forest_extension(self) -> None:
        """Test RandomForest uses .joblib extension."""
        assert MODEL_EXTENSIONS["random_forest"] == "joblib"

    def test_all_backends_covered(self) -> None:
        """Test all backend names have extensions defined."""
        backends: list[BackendName] = [
            "xgboost",
            "mlp",
            "lightgbm",
            "lstm",
            "cleargbm",
            "logreg",
            "random_forest",
        ]
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
        result = _make_fake_optimization_result(best_value=0.85)

        save_result: SaveModelResult = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        assert save_result["saved"] is True
        assert save_result["reason"] == "New best model"
        model_path = save_result["model_path"]
        meta_path = save_result["meta_path"]
        train_outcome = save_result["train_outcome"]
        assert model_path is not None and "taiwan_xgboost_best.ubj" in model_path
        assert meta_path is not None and "taiwan_xgboost_best_meta.json" in meta_path
        assert train_outcome is not None and train_outcome["best_val_auc"] == 0.88

        assert Path(model_path).exists()
        assert Path(meta_path).exists()

    def test_skips_when_existing_is_better(
        self,
        tmp_path: Path,
        hooks_context: None,
    ) -> None:
        """Test skips saving when existing model has better AUC."""
        output_dir = tmp_path / "models" / "xgboost"
        output_dir.mkdir(parents=True)
        meta_path = output_dir / "taiwan_xgboost_best_meta.json"
        meta_content = {"best_val_auc": 0.95}
        meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        result = _make_fake_optimization_result(best_value=0.85)

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
        output_dir = tmp_path / "models" / "xgboost"
        output_dir.mkdir(parents=True)
        meta_path = output_dir / "taiwan_xgboost_best_meta.json"
        meta_content = {"best_val_auc": 0.70}
        meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        result = _make_fake_optimization_result(best_value=0.85)

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
        output_dir = tmp_path / "models" / "xgboost"
        output_dir.mkdir(parents=True)

        existing_model = output_dir / "taiwan_xgboost_best.ubj"
        existing_model.write_bytes(b"old model")

        meta_path = output_dir / "taiwan_xgboost_best_meta.json"
        meta_content = {"best_val_auc": 0.50}
        meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        result = _make_fake_optimization_result(best_value=0.85)

        save_result = save_best_model(
            result=result,
            dataset="taiwan",
            feature_preset="full",
            project_root=tmp_path,
        )

        assert save_result["saved"] is True

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
        from platform_core.json_utils import require_float, require_int, require_str

        best_val_auc = require_float(meta, "best_val_auc")
        assert 0.0 <= best_val_auc <= 1.0
        saved_at = require_str(meta, "saved_at")
        assert "T" in saved_at
        model_path_val = require_str(meta, "model_path")
        assert "xgboost" in model_path_val
        n_features = require_int(meta, "n_features")
        n_samples = require_int(meta, "n_samples")
        assert n_features > 0
        assert n_samples > 0

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
        best_filename = "taiwan_xgboost_best.ubj"
        custom_backend = FakeClassifierBackend(
            backend_name_val="xgboost",
            output_filename_override=best_filename,
        )

        custom_registry = ClassifierRegistry()

        def factory() -> FakeClassifierBackend:
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

            assert save_result["saved"] is True
            assert save_result["reason"] == "New best model"

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

        train_outcome = save_result["train_outcome"]
        assert train_outcome is not None and train_outcome["best_val_auc"] == 0.88
        outcome: TrainOutcome = train_outcome

        saved_model_path = save_result["model_path"]
        assert saved_model_path is not None and Path(saved_model_path).exists()
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

        assert save_result["saved"] is True
        assert save_result["reason"] == "New best model"
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
        output_dir = tmp_path / "models" / "xgboost"
        output_dir.mkdir(parents=True)
        meta_path = output_dir / "taiwan_xgboost_best_meta.json"
        meta_content = {"best_val_auc": 0.99}
        meta_path.write_text(dump_json_str(meta_content), encoding="utf-8")

        result = _make_fake_optimization_result(best_value=0.50)

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
        assert "not better than existing" in save_result["reason"]
        assert "0.50" in save_result["reason"]
        assert "0.99" in save_result["reason"]


# =============================================================================
# Tests for build_train_config per-backend
# =============================================================================


def _make_result_for_backend(
    backend: BackendName,
    int_params: SampledIntParams,
    float_params: SampledFloatParams,
    string_params: SampledStringParams | None = None,
) -> UnifiedOptimizationResult:
    """Create optimization result for build_train_config tests.

    Args:
        backend: Backend name.
        int_params: Integer hyperparameters.
        float_params: Float hyperparameters.
        string_params: Optional string hyperparameters.

    Returns:
        UnifiedOptimizationResult with the given params.
    """
    return UnifiedOptimizationResult(
        backend=backend,
        status="complete",
        dataset="taiwan",
        n_samples=1000,
        n_features=50,
        feature_preset="none",
        n_trials_complete=10,
        n_trials_pruned=0,
        n_trials_failed=0,
        best_trial_number=5,
        best_value=0.85,
        best_int_params=int_params,
        best_float_params=float_params,
        best_string_params=string_params or SampledStringParams(),
        duration_seconds=10.0,
    )


class TestBuildTrainConfigDispatch:
    """Tests for build_train_config dispatch to all 7 backends."""

    def test_dispatches_all_backends(self) -> None:
        """Dispatch works for all 7 backends and returns valid config."""
        backend_params: list[
            tuple[BackendName, SampledIntParams, SampledFloatParams, SampledStringParams]
        ] = [
            (
                "xgboost",
                SampledIntParams(max_depth=6, n_estimators=100),
                SampledFloatParams(
                    learning_rate=0.1,
                    reg_alpha=0.01,
                    reg_lambda=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                ),
                SampledStringParams(),
            ),
            (
                "mlp",
                SampledIntParams(n_layers=3, hidden_size=128, batch_size=64),
                SampledFloatParams(learning_rate=0.001, dropout=0.2),
                SampledStringParams(),
            ),
            (
                "lstm",
                SampledIntParams(hidden_size=64, num_layers=2, batch_size=32),
                SampledFloatParams(learning_rate=0.001, dropout=0.3),
                SampledStringParams(),
            ),
            (
                "lightgbm",
                SampledIntParams(
                    max_depth=-1,
                    n_estimators=100,
                    num_leaves=31,
                    min_child_samples=20,
                ),
                SampledFloatParams(
                    learning_rate=0.1,
                    reg_alpha=0.01,
                    reg_lambda=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                ),
                SampledStringParams(),
            ),
            (
                "cleargbm",
                SampledIntParams(
                    max_depth=5,
                    n_estimators=100,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_bins=64,
                ),
                SampledFloatParams(
                    learning_rate=0.1,
                    subsample=1.0,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                ),
                SampledStringParams(),
            ),
            (
                "logreg",
                SampledIntParams(),
                SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.5),
                SampledStringParams(solver="saga", penalty="elasticnet"),
            ),
            (
                "random_forest",
                SampledIntParams(n_estimators=200, min_samples_split=5, min_samples_leaf=2),
                SampledFloatParams(),
                SampledStringParams(max_features="sqrt"),
            ),
        ]
        for backend, int_p, float_p, str_p in backend_params:
            result = _make_result_for_backend(backend, int_p, float_p, str_p)
            config = build_train_config(backend, result)
            # Common fields present on all config types
            assert config["train_ratio"] == 0.7
            assert config["val_ratio"] == 0.15
            assert config["test_ratio"] == 0.15
            assert config["random_state"] == 42

    def test_unknown_backend_raises(self) -> None:
        """Unknown backend raises ValueError."""
        result = _make_result_for_backend(
            "xgboost",
            SampledIntParams(max_depth=6, n_estimators=100),
            SampledFloatParams(
                learning_rate=0.1,
                reg_alpha=0.01,
                reg_lambda=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
        )
        with pytest.raises(ValueError, match="Unknown backend"):
            build_train_config("not_a_backend", result)


class TestBuildXGBoostConfig:
    """Tests for _build_xgboost_config."""

    def test_xgboost_config(self) -> None:
        """XGBoost config has expected fields."""
        config = _build_xgboost_config(
            SampledIntParams(max_depth=6, n_estimators=100),
            SampledFloatParams(
                learning_rate=0.1,
                reg_alpha=0.01,
                reg_lambda=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
        )
        assert config["learning_rate"] == 0.1
        assert config["max_depth"] == 6
        assert config["n_estimators"] == 100
        assert config["subsample"] == 0.8
        assert config["colsample_bytree"] == 0.8
        assert config["reg_alpha"] == 0.01
        assert config["reg_lambda"] == 0.01
        assert config["device"] == "auto"
        assert config["early_stopping_rounds"] == 10


class TestBuildMLPConfig:
    """Tests for _build_mlp_config."""

    def test_mlp_config(self) -> None:
        """MLP config has hidden_sizes tuple from n_layers and hidden_size."""
        config = _build_mlp_config(
            SampledIntParams(n_layers=3, hidden_size=128, batch_size=64),
            SampledFloatParams(learning_rate=0.001, dropout=0.2),
        )
        assert config["hidden_sizes"] == (128, 128, 128)
        assert config["batch_size"] == 64
        assert config["dropout"] == 0.2
        assert config["learning_rate"] == 0.001
        assert config["precision"] == "fp32"
        assert config["optimizer"] == "adamw"
        assert config["n_epochs"] == 50
        assert config["early_stopping_patience"] == 10


class TestBuildLSTMConfig:
    """Tests for _build_lstm_config."""

    def test_lstm_config(self) -> None:
        """LSTM config has hidden_size, num_layers, dropout."""
        config = _build_lstm_config(
            SampledIntParams(hidden_size=64, num_layers=2, batch_size=32),
            SampledFloatParams(learning_rate=0.001, dropout=0.3),
        )
        assert config["hidden_size"] == 64
        assert config["num_layers"] == 2
        assert config["dropout"] == 0.3
        assert config["learning_rate"] == 0.001
        assert config["bidirectional"] is False
        assert config["sequence_length"] == 5
        assert config["n_epochs"] == 50


class TestBuildLightGBMConfig:
    """Tests for _build_lightgbm_config."""

    def test_lightgbm_config(self) -> None:
        """LightGBM config has num_leaves, min_child_samples."""
        config = _build_lightgbm_config(
            SampledIntParams(max_depth=-1, n_estimators=100, num_leaves=31, min_child_samples=20),
            SampledFloatParams(
                learning_rate=0.1,
                reg_alpha=0.01,
                reg_lambda=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
            ),
        )
        assert config["num_leaves"] == 31
        assert config["min_child_samples"] == 20
        assert config["max_depth"] == -1
        assert config["early_stopping_rounds"] == 10


class TestBuildClearGBMConfig:
    """Tests for _build_cleargbm_config."""

    def test_cleargbm_config(self) -> None:
        """ClearGBM config has min_samples_split, min_samples_leaf, max_bins."""
        config = _build_cleargbm_config(
            SampledIntParams(
                max_depth=5,
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                max_bins=64,
            ),
            SampledFloatParams(
                learning_rate=0.1,
                subsample=1.0,
                reg_alpha=0.0,
                reg_lambda=1.0,
            ),
        )
        assert config["min_samples_split"] == 10
        assert config["min_samples_leaf"] == 5
        assert config["max_bins"] == 64
        assert config["track_contributions"] is False
        assert config["monotonic_constraints"] is None
        assert config["n_jobs"] == -1


class TestBuildLogRegConfig:
    """Tests for _build_logreg_config."""

    def test_logreg_config(self) -> None:
        """LogReg config has solver, penalty, C, tol, l1_ratio."""
        config = _build_logreg_config(
            SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.5),
            SampledStringParams(solver="saga", penalty="elasticnet"),
        )
        assert config["solver"] == "saga"
        assert config["penalty"] == "elasticnet"
        assert config["C"] == 1.0
        assert config["tol"] == 0.0001
        assert config["l1_ratio"] == 0.5
        assert config["max_iter"] == 1000
        assert config["class_weight_balanced"] is True

    def test_l2_lbfgs(self) -> None:
        """LogReg config with l2 penalty and lbfgs solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=0.5, tol=0.001, l1_ratio=0.0),
            SampledStringParams(solver="lbfgs", penalty="l2"),
        )
        assert config["solver"] == "lbfgs"
        assert config["penalty"] == "l2"

    def test_l1_liblinear(self) -> None:
        """LogReg config with l1 penalty and liblinear solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=10.0, tol=0.0001, l1_ratio=0.0),
            SampledStringParams(solver="liblinear", penalty="l1"),
        )
        assert config["solver"] == "liblinear"
        assert config["penalty"] == "l1"

    def test_none_penalty_newton_cg(self) -> None:
        """LogReg config with none penalty and newton-cg solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.0),
            SampledStringParams(solver="newton-cg", penalty="none"),
        )
        assert config["solver"] == "newton-cg"
        assert config["penalty"] == "none"

    def test_newton_cholesky(self) -> None:
        """LogReg config with newton-cholesky solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.0),
            SampledStringParams(solver="newton-cholesky", penalty="l2"),
        )
        assert config["solver"] == "newton-cholesky"

    def test_sag(self) -> None:
        """LogReg config with sag solver."""
        config = _build_logreg_config(
            SampledFloatParams(C=1.0, tol=0.0001, l1_ratio=0.0),
            SampledStringParams(solver="sag", penalty="l2"),
        )
        assert config["solver"] == "sag"


class TestBuildRandomForestConfig:
    """Tests for _build_random_forest_config."""

    def test_sqrt_features(self) -> None:
        """RandomForest config with max_features='sqrt'."""
        config = _build_random_forest_config(
            SampledIntParams(n_estimators=200, min_samples_split=5, min_samples_leaf=2),
            SampledFloatParams(),
            SampledStringParams(max_features="sqrt"),
        )
        assert config["max_features"] == "sqrt"
        assert config["n_estimators"] == 200
        assert config["min_samples_split"] == 5
        assert config["min_samples_leaf"] == 2
        assert config["bootstrap"] is True
        assert config["class_weight_balanced"] is True

    def test_log2_features(self) -> None:
        """RandomForest config with max_features='log2'."""
        config = _build_random_forest_config(
            SampledIntParams(n_estimators=100, min_samples_split=2, min_samples_leaf=1),
            SampledFloatParams(),
            SampledStringParams(max_features="log2"),
        )
        assert config["max_features"] == "log2"

    def test_float_features(self) -> None:
        """RandomForest config with float max_features via max_features_float."""
        config = _build_random_forest_config(
            SampledIntParams(n_estimators=100, min_samples_split=2, min_samples_leaf=1),
            SampledFloatParams(max_features_float=0.7),
            SampledStringParams(),
        )
        assert config["max_features"] == 0.7

    def test_default_features(self) -> None:
        """RandomForest config defaults to 'sqrt' when no max_features specified."""
        config = _build_random_forest_config(
            SampledIntParams(n_estimators=100, min_samples_split=2, min_samples_leaf=1),
            SampledFloatParams(),
            SampledStringParams(),
        )
        assert config["max_features"] == "sqrt"


class TestNarrowLogRegSolver:
    """Tests for _narrow_logreg_solver."""

    def test_all_valid_solvers(self) -> None:
        """All valid solver names are narrowed correctly."""
        solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
        for solver in solvers:
            assert _narrow_logreg_solver(solver) == solver

    def test_invalid_solver_raises(self) -> None:
        """Invalid solver name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid LogReg solver"):
            _narrow_logreg_solver("invalid_solver")


class TestNarrowLogRegPenalty:
    """Tests for _narrow_logreg_penalty."""

    def test_all_valid_penalties(self) -> None:
        """All valid penalty names are narrowed correctly."""
        penalties = ["l1", "l2", "elasticnet", "none"]
        for penalty in penalties:
            assert _narrow_logreg_penalty(penalty) == penalty

    def test_invalid_penalty_raises(self) -> None:
        """Invalid penalty name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid LogReg penalty"):
            _narrow_logreg_penalty("invalid_penalty")
