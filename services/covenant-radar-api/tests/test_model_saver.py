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

import pytest
import scripts._test_hooks as _hooks
from covenant_ml.backends.registry import (
    BackendRegistration,
    ClassifierRegistry,
)
from covenant_ml.datasets import DatasetConfig, DatasetRegistry, LoadedDataset
from covenant_ml.types import BackendName, TrainOutcome
from platform_core.json_utils import dump_json_str
from platform_core.rich_logging import setup_rich_logging
from scripts.optimize.cli import DatasetName
from scripts.optimize.model_saver import (
    MODEL_EXTENSIONS,
    SaveModelResult,
    load_existing_auc,
    save_best_model,
    should_save_model,
)

from tests._model_saver_fixtures import (
    FakeClassifierBackend,
    _make_fake_dataset_config,
    _make_fake_loaded_dataset,
    _make_fake_optimization_result,
)


@pytest.fixture(autouse=True)
def setup_logging() -> None:
    """Ensure rich logging is setup for all tests in this module."""
    setup_rich_logging(level="WARNING", show_time=False)


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
