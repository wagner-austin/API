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
from scripts.optimize._config_builders import (
    _build_cleargbm_config,
    _build_lightgbm_config,
    _build_logreg_config,
    _build_lstm_config,
    _build_mlp_config,
    _build_random_forest_config,
    _build_xgboost_config,
    _narrow_logreg_penalty,
    _narrow_logreg_solver,
)
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
        assert config["growth_strategy"] == "depth_wise"
        assert config["num_leaves"] is None


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
