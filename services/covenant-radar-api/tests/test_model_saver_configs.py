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
from covenant_ml.optimizer.types import (
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)
from covenant_ml.types import (
    BackendName,
)
from platform_core.json_utils import dump_json_str
from platform_core.rich_logging import setup_rich_logging
from scripts.optimize.model_saver import (
    build_train_config,
    save_best_model,
)

from covenant_radar_api.worker.optimize_result_types import (
    UnifiedOptimizationResult,
)
from tests._model_saver_fixtures import (
    FakeClassifierBackend,
    _make_fake_dataset_config,
    _make_fake_loaded_dataset,
    _make_fake_optimization_result,
)


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


@pytest.fixture(autouse=True)
def setup_logging() -> None:
    """Ensure rich logging is setup for all tests in this module."""
    setup_rich_logging(level="WARNING", show_time=False)


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
