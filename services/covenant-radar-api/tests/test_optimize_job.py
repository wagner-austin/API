"""Tests for worker/optimize_job.py unified hyperparameter optimization job.

Tests use dependency injection via worker/_test_hooks to verify actual code paths.
All code paths are tested with strong assertions on actual behavior.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    PreparedClassifier,
    ProgressCallback,
)
from covenant_ml.backends.registry import (
    BackendRegistration,
    ClassifierRegistry,
)
from covenant_ml.datasets import (
    DatasetConfig,
    DatasetMeta,
    DatasetRegistry,
    LoadedDataset,
    TimeSeriesDatasetRegistry,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.optimizer import (
    OptimizerStrategyRegistry,
    SearchSpace,
)
from covenant_ml.optimizer.protocol import ObjectiveProtocol, TrialCallbackProtocol
from covenant_ml.optimizer.registry import OptimizerStrategyRegistration
from covenant_ml.optimizer.strategy_protocol import (
    OptimizerStrategyCapabilities,
    OptimizerStrategyName,
)
from covenant_ml.optimizer.types import (
    FloatRangeSpec,
    IntRangeSpec,
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    TrialResult,
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
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from covenant_radar_api.worker import _test_hooks as worker_hooks
from covenant_radar_api.worker._optimize_common import (
    parse_bidirectional,
    parse_nn_optimizer,
    parse_precision,
)
from covenant_radar_api.worker._test_hooks import ObjectiveWithFeatureCount
from covenant_radar_api.worker.optimize_job import (
    _parse_optimize_config,
    run_optimization,
)
from covenant_radar_api.worker.optimize_types import (
    LoadingProgressInfo,
    PhaseProgressInfo,
    TrialProgressInfo,
    UnifiedOptimizeParseResult,
)

# =============================================================================
# Fake Implementations for Testing
# =============================================================================


def _make_fake_dataset(name: str = "taiwan") -> LoadedDataset:
    """Create fake dataset for testing.

    Args:
        name: Dataset name.

    Returns:
        LoadedDataset with synthetic data.
    """
    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((100, 10)).astype(np.float64)
    y: NDArray[np.int64] = rng.integers(0, 2, size=100).astype(np.int64)
    n_positive = int(np.sum(y))
    meta: DatasetMeta = {
        "name": name,
        "n_samples": 100,
        "n_features": 10,
        "n_positive": n_positive,
        "n_negative": 100 - n_positive,
        "positive_ratio": n_positive / 100,
        "feature_names": tuple(f"feature_{i}" for i in range(10)),
        "categorical_encodings": (),
    }
    return {"meta": meta, "x": x, "y": y, "groups": None}


def _make_fake_standard_config(name: str) -> DatasetConfig:
    """Create fake standard dataset config.

    Args:
        name: Dataset name.

    Returns:
        DatasetConfig for standard dataset.
    """
    return {
        "name": name,
        "display_name": f"Fake {name}",
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


def _make_fake_standard_registry() -> DatasetRegistry:
    """Create fake standard dataset registry.

    Returns:
        DatasetRegistry with taiwan, us, polish datasets.
    """
    configs = (
        _make_fake_standard_config("taiwan"),
        _make_fake_standard_config("us"),
        _make_fake_standard_config("polish"),
    )
    return DatasetRegistry(configs)


def _make_fake_timeseries_registry() -> TimeSeriesDatasetRegistry:
    """Create empty time-series registry (no TS datasets in optimize tests).

    Returns:
        Empty TimeSeriesDatasetRegistry.
    """
    return TimeSeriesDatasetRegistry(())


class _FakeObjective:
    """Fake objective function implementing ObjectiveWithFeatureCount.

    Returns a fixed AUC value and tracks the number of engineered features.
    """

    def __init__(self, n_features: int = 10, return_value: float = 0.85) -> None:
        """Initialize fake objective.

        Args:
            n_features: Number of features to report.
            return_value: Fixed AUC value to return.
        """
        self._n_features = n_features
        self._return_value = return_value

    @property
    def n_features(self) -> int:
        """Return the feature count."""
        return self._n_features

    def __call__(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        int_params: SampledIntParams,
        float_params: SampledFloatParams,
        string_params: SampledStringParams,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_state: int,
    ) -> float:
        """Return fixed value, ignoring all parameters.

        Args:
            x_features: Feature matrix.
            y_labels: Binary labels.
            feature_names: Feature column names.
            int_params: Integer hyperparameters.
            float_params: Float hyperparameters.
            string_params: String hyperparameters.
            train_ratio: Training data fraction.
            val_ratio: Validation data fraction.
            test_ratio: Test data fraction.
            random_state: Random seed.

        Returns:
            Fixed AUC value.
        """
        del x_features, y_labels, feature_names
        del int_params, float_params, string_params
        del train_ratio, val_ratio, test_ratio, random_state
        return self._return_value


class _FakeOptimizer:
    """Fake optimizer that records calls and returns predetermined results."""

    def __init__(self, result: OptimizationSummary | None = None) -> None:
        """Initialize fake optimizer.

        Args:
            result: Predetermined result to return. If None, generates simple result.
        """
        self._result = result
        self._optimize_call_count = 0
        self._last_search_space: SearchSpace | None = None
        self._last_config: OptimizationConfig | None = None

    @property
    def optimize_call_count(self) -> int:
        """Get the number of times optimize was called."""
        return self._optimize_call_count

    @property
    def last_search_space(self) -> SearchSpace | None:
        """Get the search space from the last optimize call."""
        return self._last_search_space

    @property
    def last_config(self) -> OptimizationConfig | None:
        """Get the config from the last optimize call."""
        return self._last_config

    def strategy_name(self) -> OptimizerStrategyName:
        """Return 'optuna_tpe' as the strategy name."""
        return "optuna_tpe"

    def capabilities(self) -> OptimizerStrategyCapabilities:
        """Return default capabilities."""
        return OptimizerStrategyCapabilities(
            supports_pruning=True,
            supports_parallel=True,
            is_deterministic=False,
            requires_bounds=True,
        )

    def optimize(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: SearchSpace,
        config: OptimizationConfig,
        objective: ObjectiveProtocol,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> OptimizationSummary:
        """Record call and return predetermined result.

        Args:
            x_features: Feature matrix.
            y_labels: Binary labels.
            feature_names: Feature column names.
            search_space: Hyperparameter search space.
            config: Optimization configuration.
            objective: Objective function.
            trial_callback: Optional trial callback.

        Returns:
            Predetermined or generated optimization summary.
        """
        del x_features, y_labels, feature_names, objective, trial_callback

        self._optimize_call_count += 1
        self._last_search_space = search_space
        self._last_config = config

        if self._result is not None:
            return self._result

        return OptimizationSummary(
            best_trial_number=0,
            best_value=0.85,
            best_int_params=SampledIntParams(max_depth=5, n_estimators=100),
            best_float_params=SampledFloatParams(learning_rate=0.1),
            best_string_params=SampledStringParams(),
            n_trials_total=config["n_trials"],
            n_trials_complete=config["n_trials"],
            n_trials_pruned=0,
            n_trials_failed=0,
            total_duration_seconds=1.0,
        )


_FAKE_SEARCH_SPACE: XGBoostSearchSpace = XGBoostSearchSpace(
    max_depth=IntRangeSpec(param_type="int", low=3, high=10, log_scale=False),
    n_estimators=IntRangeSpec(param_type="int", low=50, high=500, log_scale=False),
    learning_rate=FloatRangeSpec(param_type="float", low=0.01, high=0.3, log_scale=True),
    reg_alpha=FloatRangeSpec(param_type="float", low=1e-8, high=10.0, log_scale=True),
    reg_lambda=FloatRangeSpec(param_type="float", low=1e-8, high=10.0, log_scale=True),
    subsample=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
    colsample_bytree=FloatRangeSpec(param_type="float", low=0.5, high=1.0, log_scale=False),
)


class _FakeBackend:
    """Fake ClassifierBackend satisfying full protocol for testing.

    Only get_default_search_space is exercised by optimize tests.
    Other protocol methods raise NotImplementedError.
    """

    def __init__(self, name: BackendName = "xgboost") -> None:
        """Initialize fake backend.

        Args:
            name: Backend name to return.
        """
        self._name = name
        self._get_search_space_called = False

    @property
    def get_search_space_called(self) -> bool:
        """Whether get_default_search_space was called."""
        return self._get_search_space_called

    def backend_name(self) -> BackendName:
        """Return the backend name."""
        return self._name

    def capabilities(self) -> BackendCapabilities:
        """Return dummy capabilities."""
        return BackendCapabilities(
            supports_train=True,
            supports_gpu=False,
            supports_early_stopping=True,
            supports_feature_importance=True,
            model_format="ubj",
        )

    def prepare(
        self,
        *,
        n_features: int,
        n_classes: int,
        feature_names: list[str] | None,
    ) -> PreparedClassifier:
        """Not used in optimize tests."""
        raise NotImplementedError

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str] | None,
        config: ClassifierTrainConfig,
        output_dir: Path,
        progress: ProgressCallback | None,
        groups: NDArray[np.int64] | None = None,
    ) -> TrainOutcome:
        """Not used in optimize tests."""
        raise NotImplementedError

    def evaluate(
        self,
        *,
        model: PreparedClassifier,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> EvalMetrics:
        """Not used in optimize tests."""
        raise NotImplementedError

    def save(self, *, model: PreparedClassifier, path: str) -> None:
        """Not used in optimize tests."""
        raise NotImplementedError

    def load(self, *, path: str) -> PreparedClassifier:
        """Not used in optimize tests."""
        raise NotImplementedError

    def get_feature_importances(
        self,
        *,
        model: PreparedClassifier,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Not used in optimize tests."""
        raise NotImplementedError

    def get_default_search_space(self) -> SearchSpace:
        """Return fake XGBoost search space."""
        self._get_search_space_called = True
        return _FAKE_SEARCH_SPACE

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Not used in optimize tests."""
        raise NotImplementedError


def _make_fake_backend_registry(backend: _FakeBackend) -> ClassifierRegistry:
    """Create a fake registry that returns the given backend.

    Args:
        backend: Fake backend to register.

    Returns:
        ClassifierRegistry with the fake backend registered for all 7 names.
    """
    reg = ClassifierRegistry()
    backends: tuple[BackendName, ...] = (
        "xgboost",
        "mlp",
        "lstm",
        "lightgbm",
        "cleargbm",
        "logreg",
        "random_forest",
    )
    for name in backends:

        def _factory(b: _FakeBackend = backend) -> _FakeBackend:
            return b

        reg.register(name, BackendRegistration(_factory))
    return reg


def _make_fake_optimizer_registry(optimizer: _FakeOptimizer) -> OptimizerStrategyRegistry:
    """Create a registry containing the given fake optimizer.

    Args:
        optimizer: Fake optimizer to register.

    Returns:
        OptimizerStrategyRegistry with 'optuna_tpe' mapped to fake optimizer.
    """
    registry = OptimizerStrategyRegistry()

    def _factory(o: _FakeOptimizer = optimizer) -> _FakeOptimizer:
        return o

    registry.register("optuna_tpe", OptimizerStrategyRegistration(_factory))
    return registry


def _make_config_json(
    backend: str = "xgboost",
    dataset: str = "taiwan",
    n_trials: int = 10,
    **overrides: JSONValue,
) -> str:
    """Build a JSON config string for testing.

    Args:
        backend: Backend name.
        dataset: Dataset name.
        n_trials: Number of trials.
        **overrides: Additional or override fields.

    Returns:
        JSON string.
    """
    config: dict[str, JSONValue] = {
        "backend": backend,
        "dataset": dataset,
        "n_trials": n_trials,
    }
    config.update(overrides)
    return dump_json_str(config)


# =============================================================================
# Tests for parse_precision
# =============================================================================


class TestParsePrecision:
    """Tests for parse_precision function."""

    def test_defaults_to_fp32(self) -> None:
        """None input returns 'fp32'."""
        assert parse_precision(None) == "fp32"

    def test_accepts_fp32(self) -> None:
        """'fp32' is accepted."""
        assert parse_precision("fp32") == "fp32"

    def test_accepts_fp16(self) -> None:
        """'fp16' is accepted."""
        assert parse_precision("fp16") == "fp16"

    def test_accepts_bf16(self) -> None:
        """'bf16' is accepted."""
        assert parse_precision("bf16") == "bf16"

    def test_accepts_auto(self) -> None:
        """'auto' is accepted."""
        assert parse_precision("auto") == "auto"

    def test_rejects_invalid_string(self) -> None:
        """Invalid precision raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="precision must be one of"):
            parse_precision("fp64")

    def test_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="precision must be a string"):
            parse_precision(16)


# =============================================================================
# Tests for parse_nn_optimizer
# =============================================================================


class TestParseNnOptimizer:
    """Tests for parse_nn_optimizer function."""

    def test_defaults_to_adamw(self) -> None:
        """None input returns 'adamw'."""
        assert parse_nn_optimizer(None) == "adamw"

    def test_accepts_adamw(self) -> None:
        """'adamw' is accepted."""
        assert parse_nn_optimizer("adamw") == "adamw"

    def test_accepts_adam(self) -> None:
        """'adam' is accepted."""
        assert parse_nn_optimizer("adam") == "adam"

    def test_accepts_sgd(self) -> None:
        """'sgd' is accepted."""
        assert parse_nn_optimizer("sgd") == "sgd"

    def test_rejects_invalid_string(self) -> None:
        """Invalid optimizer raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="optimizer must be one of"):
            parse_nn_optimizer("rmsprop")

    def test_rejects_non_string(self) -> None:
        """Non-string input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="optimizer must be a string"):
            parse_nn_optimizer(42)


# =============================================================================
# Tests for parse_bidirectional
# =============================================================================


class TestParseBidirectional:
    """Tests for parse_bidirectional function."""

    def test_defaults_to_false(self) -> None:
        """None input returns False."""
        assert parse_bidirectional(None) is False

    def test_accepts_true(self) -> None:
        """True is accepted."""
        assert parse_bidirectional(True) is True

    def test_accepts_false(self) -> None:
        """False is accepted."""
        assert parse_bidirectional(False) is False

    def test_rejects_non_bool(self) -> None:
        """Non-boolean input raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="bidirectional must be a boolean"):
            parse_bidirectional("yes")


# =============================================================================
# Tests for _parse_optimize_config
# =============================================================================


class TestParseOptimizeConfig:
    """Tests for _parse_optimize_config function."""

    def setup_method(self) -> None:
        """Install fake dataset registries before each test."""
        self._orig_dataset_registry = worker_hooks.dataset_registry_factory
        self._orig_ts_registry = worker_hooks.timeseries_registry_factory
        worker_hooks.dataset_registry_factory = _make_fake_standard_registry
        worker_hooks.timeseries_registry_factory = _make_fake_timeseries_registry

    def teardown_method(self) -> None:
        """Restore original hooks after each test."""
        worker_hooks.dataset_registry_factory = self._orig_dataset_registry
        worker_hooks.timeseries_registry_factory = self._orig_ts_registry

    def test_minimal_config_returns_defaults(self) -> None:
        """Minimal config uses defaults for all optional fields."""
        config_json = _make_config_json()
        result = _parse_optimize_config(config_json)

        assert result["backend"] == "xgboost"
        assert result["dataset"] == "taiwan"
        assert result["n_trials"] == 10
        assert result["timeout_seconds"] is None
        assert result["device"] == "auto"
        assert result["feature_preset"] == "none"
        assert result["random_state"] == 42
        assert result["early_stopping_rounds"] == 10
        assert result["n_jobs"] == -1
        assert result["precision"] == "fp32"
        assert result["nn_optimizer"] == "adamw"
        assert result["n_epochs"] == 50
        assert result["early_stopping_patience"] == 10
        assert result["sequence_length"] == 5
        assert result["bidirectional"] is False

    def test_full_config_all_fields(self) -> None:
        """Full config with all fields specified."""
        config_json = _make_config_json(
            backend="mlp",
            dataset="us",
            n_trials=100,
            timeout_seconds=3600,
            device="cuda",
            feature_preset="full",
            random_state=123,
            early_stopping_rounds=20,
            n_jobs=4,
            precision="fp16",
            optimizer="adam",
            n_epochs=100,
            early_stopping_patience=15,
            sequence_length=10,
            bidirectional=True,
        )
        result = _parse_optimize_config(config_json)

        assert result["backend"] == "mlp"
        assert result["dataset"] == "us"
        assert result["n_trials"] == 100
        assert result["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["feature_preset"] == "full"
        assert result["random_state"] == 123
        assert result["early_stopping_rounds"] == 20
        assert result["n_jobs"] == 4
        assert result["precision"] == "fp16"
        assert result["nn_optimizer"] == "adam"
        assert result["n_epochs"] == 100
        assert result["early_stopping_patience"] == 15
        assert result["sequence_length"] == 10
        assert result["bidirectional"] is True

    def test_all_seven_backends_accepted(self) -> None:
        """All 7 backends are accepted."""
        backends: tuple[str, ...] = (
            "xgboost",
            "mlp",
            "lstm",
            "lightgbm",
            "cleargbm",
            "logreg",
            "random_forest",
        )
        for backend in backends:
            config_json = _make_config_json(backend=backend)
            result = _parse_optimize_config(config_json)
            assert result["backend"] == backend

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        config_json = _make_config_json(backend="invalid")
        with pytest.raises(ValueError, match="backend must be one of"):
            _parse_optimize_config(config_json)

    def test_missing_dataset_raises(self) -> None:
        """Missing dataset field raises JSONTypeError."""
        config_json = dump_json_str({"backend": "xgboost", "n_trials": 10})
        with pytest.raises(JSONTypeError):
            _parse_optimize_config(config_json)

    def test_invalid_dataset_raises(self) -> None:
        """Invalid dataset name raises ValueError."""
        config_json = _make_config_json(dataset="nonexistent")
        with pytest.raises(ValueError, match="dataset must be one of"):
            _parse_optimize_config(config_json)

    def test_missing_n_trials_raises(self) -> None:
        """Missing n_trials field raises JSONTypeError."""
        config_json = dump_json_str({"backend": "xgboost", "dataset": "taiwan"})
        with pytest.raises(JSONTypeError):
            _parse_optimize_config(config_json)

    def test_non_object_config_raises(self) -> None:
        """Non-object config raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_optimize_config('"just a string"')

    def test_invalid_precision_raises(self) -> None:
        """Invalid precision raises JSONTypeError."""
        config_json = _make_config_json(precision="fp64")
        with pytest.raises(JSONTypeError, match="precision must be one of"):
            _parse_optimize_config(config_json)

    def test_invalid_nn_optimizer_raises(self) -> None:
        """Invalid optimizer raises JSONTypeError."""
        config_json = _make_config_json(optimizer="rmsprop")
        with pytest.raises(JSONTypeError, match="optimizer must be one of"):
            _parse_optimize_config(config_json)

    def test_invalid_bidirectional_raises(self) -> None:
        """Invalid bidirectional raises JSONTypeError."""
        config_json = _make_config_json(bidirectional="yes")
        with pytest.raises(JSONTypeError, match="bidirectional must be a boolean"):
            _parse_optimize_config(config_json)

    def test_invalid_timeout_type_raises(self) -> None:
        """Non-integer timeout_seconds raises JSONTypeError."""
        config_json = _make_config_json(timeout_seconds="an_hour")
        with pytest.raises(JSONTypeError, match="timeout_seconds must be an integer"):
            _parse_optimize_config(config_json)

    def test_null_timeout_is_none(self) -> None:
        """Explicit null timeout_seconds results in None."""
        config_json = _make_config_json(timeout_seconds=None)
        result = _parse_optimize_config(config_json)
        assert result["timeout_seconds"] is None


# =============================================================================
# Tests for run_optimization
# =============================================================================


class TestRunOptimization:
    """Tests for run_optimization using fake worker_hooks."""

    def setup_method(self) -> None:
        """Install all fake hooks before each test."""
        self._orig_registry = worker_hooks.registry_factory
        self._orig_optimizer = worker_hooks.optimizer_registry_factory
        self._orig_objective = worker_hooks.objective_factory
        self._orig_dataset_registry = worker_hooks.dataset_registry_factory
        self._orig_ts_registry = worker_hooks.timeseries_registry_factory
        self._orig_dataset_loader = worker_hooks.dataset_loader

        # Install fake dataset registries
        worker_hooks.dataset_registry_factory = _make_fake_standard_registry
        worker_hooks.timeseries_registry_factory = _make_fake_timeseries_registry

    def teardown_method(self) -> None:
        """Restore all original hooks after each test."""
        worker_hooks.registry_factory = self._orig_registry
        worker_hooks.optimizer_registry_factory = self._orig_optimizer
        worker_hooks.objective_factory = self._orig_objective
        worker_hooks.dataset_registry_factory = self._orig_dataset_registry
        worker_hooks.timeseries_registry_factory = self._orig_ts_registry
        worker_hooks.dataset_loader = self._orig_dataset_loader

    def _install_fakes(
        self,
        backend: _FakeBackend | None = None,
        optimizer: _FakeOptimizer | None = None,
        objective: _FakeObjective | None = None,
        dataset: LoadedDataset | None = None,
    ) -> tuple[_FakeBackend, _FakeOptimizer, _FakeObjective]:
        """Install fake hooks and return the fakes for assertion.

        Args:
            backend: Optional fake backend (defaults to new _FakeBackend).
            optimizer: Optional fake optimizer (defaults to new _FakeOptimizer).
            objective: Optional fake objective (defaults to new _FakeObjective).
            dataset: Optional fake dataset (defaults to _make_fake_dataset).

        Returns:
            Tuple of (fake_backend, fake_optimizer, fake_objective).
        """
        fake_backend = backend or _FakeBackend()
        fake_optimizer = optimizer or _FakeOptimizer()
        fake_objective = objective or _FakeObjective()
        fake_dataset = dataset or _make_fake_dataset()

        fake_registry = _make_fake_backend_registry(fake_backend)
        fake_optimizer_registry = _make_fake_optimizer_registry(fake_optimizer)

        worker_hooks.registry_factory = lambda: fake_registry

        worker_hooks.optimizer_registry_factory = lambda: fake_optimizer_registry

        def _fake_objective_factory(
            backend_name: BackendName,
            x: NDArray[np.float64],
            y: NDArray[np.int64],
            feature_names: list[str],
            config: UnifiedOptimizeParseResult,
        ) -> ObjectiveWithFeatureCount:
            del backend_name, x, y, feature_names, config
            return fake_objective

        worker_hooks.objective_factory = _fake_objective_factory

        def _fake_loader(
            config: DatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            del config, external_dir, progress_callback
            return fake_dataset

        worker_hooks.dataset_loader = _fake_loader

        return fake_backend, fake_optimizer, fake_objective

    def test_run_optimization_returns_result(self, tmp_path: Path) -> None:
        """run_optimization returns UnifiedOptimizationResult."""
        _, _, _ = self._install_fakes()
        config_json = _make_config_json()
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["backend"] == "xgboost"
        assert result["status"] == "complete"
        assert result["dataset"] == "taiwan"
        assert result["n_samples"] == 100
        assert result["n_features"] == 10
        assert result["best_value"] == 0.85
        assert result["n_trials_complete"] == 10

    def test_optimizer_called_once(self, tmp_path: Path) -> None:
        """Optimizer.optimize is called exactly once."""
        _, fake_optimizer, _ = self._install_fakes()
        config_json = _make_config_json()
        run_optimization(config_json, tmp_path, tmp_path / "output")

        assert fake_optimizer.optimize_call_count == 1

    def test_backend_search_space_used(self, tmp_path: Path) -> None:
        """Backend's get_default_search_space is called and passed to optimizer."""
        fake_backend, fake_optimizer, _ = self._install_fakes()
        config_json = _make_config_json()
        run_optimization(config_json, tmp_path, tmp_path / "output")

        assert fake_backend.get_search_space_called
        assert fake_optimizer.last_search_space == _FAKE_SEARCH_SPACE

    def test_optimization_config_populated(self, tmp_path: Path) -> None:
        """OptimizationConfig is correctly built from parsed config."""
        _, fake_optimizer, _ = self._install_fakes()
        config_json = _make_config_json(n_trials=25, timeout_seconds=1800, random_state=99)
        run_optimization(config_json, tmp_path, tmp_path / "output")

        config = fake_optimizer.last_config
        if config is None:
            pytest.fail("last_config must be set after optimize")
        assert config["n_trials"] == 25
        assert config["timeout_seconds"] == 1800
        assert config["random_state"] == 99

    def test_results_saved_to_output_dir(self, tmp_path: Path) -> None:
        """Optimization results are saved as JSON files."""
        _, _, _ = self._install_fakes()
        output_dir = tmp_path / "optuna"
        config_json = _make_config_json()
        run_optimization(config_json, tmp_path, output_dir)

        result_file = output_dir / "taiwan_xgboost_optuna_result.json"
        config_file = output_dir / "taiwan_xgboost_optimal_config.json"
        assert result_file.exists()
        assert config_file.exists()

        raw = load_json_str(result_file.read_text())
        result_data = narrow_json_to_dict(raw)
        assert result_data["backend"] == "xgboost"
        assert result_data["dataset"] == "taiwan"

    def test_phase_callbacks_called(self, tmp_path: Path) -> None:
        """Phase callback receives all 4 phases in order."""
        _, _, _ = self._install_fakes()
        phases: list[str] = []

        def _phase_cb(info: PhaseProgressInfo) -> None:
            phases.append(info["phase"])

        config_json = _make_config_json()
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            phase_callback=_phase_cb,
        )

        assert phases == ["loading_data", "feature_engineering", "optimizing", "saving"]

    def test_phase_callbacks_include_backend_and_dataset(self, tmp_path: Path) -> None:
        """Phase callback info includes correct backend and dataset."""
        _, _, _ = self._install_fakes()
        infos: list[PhaseProgressInfo] = []

        def _phase_cb(info: PhaseProgressInfo) -> None:
            infos.append(info)

        config_json = _make_config_json(backend="lightgbm", dataset="us")
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            phase_callback=_phase_cb,
        )

        for info in infos:
            assert info["backend"] == "lightgbm"
            assert info["dataset"] == "us"

    def test_loading_progress_callback(self, tmp_path: Path) -> None:
        """Loading progress callback is invoked when provided."""
        _, _, _ = self._install_fakes()

        # Override the dataset loader to call the progress callback
        fake_dataset = _make_fake_dataset()

        def _loading_loader(
            config: DatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            del config, external_dir
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "reading",
                        "bytes_read": 500,
                        "bytes_total": 1000,
                        "percent_complete": 50.0,
                        "rows_processed": 50,
                        "rows_total": 100,
                        "message": "Reading CSV",
                    }
                )
            return fake_dataset

        worker_hooks.dataset_loader = _loading_loader

        loading_infos: list[LoadingProgressInfo] = []

        def _loading_cb(info: LoadingProgressInfo) -> None:
            loading_infos.append(info)

        config_json = _make_config_json()
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            loading_progress_callback=_loading_cb,
        )

        assert len(loading_infos) == 1
        assert loading_infos[0]["phase"] == "reading"
        assert loading_infos[0]["percent_complete"] == 50.0
        assert loading_infos[0]["dataset"] == "taiwan"

    def test_progress_callback_on_trial(self, tmp_path: Path) -> None:
        """Trial progress callback receives info when optimizer calls trial_callback."""
        # Create an optimizer that calls the trial callback
        summary = OptimizationSummary(
            best_trial_number=0,
            best_value=0.88,
            best_int_params=SampledIntParams(max_depth=6),
            best_float_params=SampledFloatParams(learning_rate=0.05),
            best_string_params=SampledStringParams(),
            n_trials_total=5,
            n_trials_complete=5,
            n_trials_pruned=0,
            n_trials_failed=0,
            total_duration_seconds=2.0,
        )

        class _CallbackOptimizer(_FakeOptimizer):
            """Optimizer that invokes the trial callback."""

            def optimize(
                self,
                x_features: NDArray[np.float64],
                y_labels: NDArray[np.int64],
                feature_names: list[str],
                search_space: SearchSpace,
                config: OptimizationConfig,
                objective: ObjectiveProtocol,
                trial_callback: TrialCallbackProtocol | None = None,
            ) -> OptimizationSummary:
                """Call trial callback then return summary."""
                del x_features, y_labels, feature_names, search_space, objective
                self._optimize_call_count += 1
                self._last_config = config
                if trial_callback is not None:
                    trial_callback(
                        TrialResult(
                            trial_number=0,
                            int_params=SampledIntParams(max_depth=6),
                            float_params=SampledFloatParams(learning_rate=0.05),
                            string_params=SampledStringParams(),
                            value=0.88,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                return summary

        callback_optimizer = _CallbackOptimizer()
        self._install_fakes(optimizer=callback_optimizer)

        trial_infos: list[TrialProgressInfo] = []

        def _trial_cb(info: TrialProgressInfo) -> None:
            trial_infos.append(info)

        config_json = _make_config_json(n_trials=5)
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            progress_callback=_trial_cb,
        )

        assert len(trial_infos) == 1
        assert trial_infos[0]["trial_number"] == 0
        assert trial_infos[0]["current_value"] == 0.88
        assert trial_infos[0]["is_best"] is True
        assert trial_infos[0]["backend"] == "xgboost"

    def test_result_includes_best_params(self, tmp_path: Path) -> None:
        """Result includes best hyperparameters from optimizer summary."""
        summary = OptimizationSummary(
            best_trial_number=7,
            best_value=0.91,
            best_int_params=SampledIntParams(max_depth=8, n_estimators=300),
            best_float_params=SampledFloatParams(learning_rate=0.03, reg_alpha=0.5),
            best_string_params=SampledStringParams(booster="gbtree"),
            n_trials_total=50,
            n_trials_complete=48,
            n_trials_pruned=1,
            n_trials_failed=1,
            total_duration_seconds=60.0,
        )
        fake_optimizer = _FakeOptimizer(result=summary)
        self._install_fakes(optimizer=fake_optimizer)

        config_json = _make_config_json(n_trials=50)
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["best_trial_number"] == 7
        assert result["best_value"] == 0.91
        assert result["best_int_params"]["max_depth"] == 8
        assert result["best_int_params"]["n_estimators"] == 300
        assert result["best_float_params"]["learning_rate"] == 0.03
        assert result["best_string_params"]["booster"] == "gbtree"
        assert result["n_trials_complete"] == 48
        assert result["n_trials_pruned"] == 1
        assert result["n_trials_failed"] == 1
        assert result["duration_seconds"] == 60.0

    def test_feature_preset_passed_through(self, tmp_path: Path) -> None:
        """Feature preset from config appears in result."""
        _, _, _ = self._install_fakes()
        config_json = _make_config_json(feature_preset="full")
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["feature_preset"] == "full"

    def test_no_callbacks_ok(self, tmp_path: Path) -> None:
        """run_optimization works without any callbacks."""
        _, _, _ = self._install_fakes()
        config_json = _make_config_json()
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["status"] == "complete"

    def test_n_features_from_objective(self, tmp_path: Path) -> None:
        """n_features in result comes from objective.n_features."""
        fake_objective = _FakeObjective(n_features=42)
        self._install_fakes(objective=fake_objective)
        config_json = _make_config_json()
        result = run_optimization(config_json, tmp_path, tmp_path / "output")

        assert result["n_features"] == 42

    def test_trial_callback_non_best_trial(self, tmp_path: Path) -> None:
        """Trial callback correctly reports is_best=False for non-best trials."""
        summary = OptimizationSummary(
            best_trial_number=1,
            best_value=0.90,
            best_int_params=SampledIntParams(max_depth=6),
            best_float_params=SampledFloatParams(learning_rate=0.05),
            best_string_params=SampledStringParams(),
            n_trials_total=3,
            n_trials_complete=3,
            n_trials_pruned=0,
            n_trials_failed=0,
            total_duration_seconds=2.0,
        )

        class _MultiTrialOptimizer(_FakeOptimizer):
            """Optimizer that calls trial callback with best and non-best trials."""

            def optimize(
                self,
                x_features: NDArray[np.float64],
                y_labels: NDArray[np.int64],
                feature_names: list[str],
                search_space: SearchSpace,
                config: OptimizationConfig,
                objective: ObjectiveProtocol,
                trial_callback: TrialCallbackProtocol | None = None,
            ) -> OptimizationSummary:
                """Call trial callback with multiple trials."""
                del x_features, y_labels, feature_names, search_space, objective
                self._optimize_call_count += 1
                self._last_config = config
                if trial_callback is not None:
                    # First trial: best
                    trial_callback(
                        TrialResult(
                            trial_number=0,
                            int_params=SampledIntParams(max_depth=5),
                            float_params=SampledFloatParams(learning_rate=0.1),
                            string_params=SampledStringParams(),
                            value=0.85,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                    # Second trial: new best
                    trial_callback(
                        TrialResult(
                            trial_number=1,
                            int_params=SampledIntParams(max_depth=6),
                            float_params=SampledFloatParams(learning_rate=0.05),
                            string_params=SampledStringParams(),
                            value=0.90,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                    # Third trial: not best
                    trial_callback(
                        TrialResult(
                            trial_number=2,
                            int_params=SampledIntParams(max_depth=4),
                            float_params=SampledFloatParams(learning_rate=0.2),
                            string_params=SampledStringParams(),
                            value=0.80,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                return summary

        multi_optimizer = _MultiTrialOptimizer()
        self._install_fakes(optimizer=multi_optimizer)

        trial_infos: list[TrialProgressInfo] = []

        def _trial_cb(info: TrialProgressInfo) -> None:
            trial_infos.append(info)

        config_json = _make_config_json(n_trials=3)
        run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
            progress_callback=_trial_cb,
        )

        assert len(trial_infos) == 3
        # First trial is best (first one always is)
        assert trial_infos[0]["is_best"] is True
        assert trial_infos[0]["best_value"] == 0.85
        # Second trial is new best
        assert trial_infos[1]["is_best"] is True
        assert trial_infos[1]["best_value"] == 0.90
        # Third trial is NOT best
        assert trial_infos[2]["is_best"] is False
        assert trial_infos[2]["best_value"] == 0.90
        assert trial_infos[2]["current_value"] == 0.80

    def test_trial_callback_without_progress_callback(self, tmp_path: Path) -> None:
        """Trial callback works correctly without external progress callback."""
        summary = OptimizationSummary(
            best_trial_number=0,
            best_value=0.85,
            best_int_params=SampledIntParams(max_depth=5),
            best_float_params=SampledFloatParams(learning_rate=0.1),
            best_string_params=SampledStringParams(),
            n_trials_total=2,
            n_trials_complete=2,
            n_trials_pruned=0,
            n_trials_failed=0,
            total_duration_seconds=1.0,
        )

        class _NoProgressOptimizer(_FakeOptimizer):
            """Optimizer that calls trial callback without progress callback."""

            def optimize(
                self,
                x_features: NDArray[np.float64],
                y_labels: NDArray[np.int64],
                feature_names: list[str],
                search_space: SearchSpace,
                config: OptimizationConfig,
                objective: ObjectiveProtocol,
                trial_callback: TrialCallbackProtocol | None = None,
            ) -> OptimizationSummary:
                """Call trial callback to exercise internal state tracking."""
                del x_features, y_labels, feature_names, search_space, objective
                self._optimize_call_count += 1
                self._last_config = config
                if trial_callback is not None:
                    trial_callback(
                        TrialResult(
                            trial_number=0,
                            int_params=SampledIntParams(max_depth=5),
                            float_params=SampledFloatParams(learning_rate=0.1),
                            string_params=SampledStringParams(),
                            value=0.85,
                            state="complete",
                            duration_seconds=0.5,
                        )
                    )
                return summary

        no_progress_optimizer = _NoProgressOptimizer()
        self._install_fakes(optimizer=no_progress_optimizer)

        config_json = _make_config_json(n_trials=2)
        # Run WITHOUT progress_callback (tests the None branch)
        result = run_optimization(
            config_json,
            tmp_path,
            tmp_path / "output",
        )

        assert result["status"] == "complete"
        assert result["best_value"] == 0.85


# =============================================================================
# Tests for process_optimize_job
# =============================================================================


class TestProcessOptimizeJob:
    """Tests for process_optimize_job RQ entry point."""

    def setup_method(self) -> None:
        """Install all fake hooks before each test."""
        self._orig_registry = worker_hooks.registry_factory
        self._orig_optimizer = worker_hooks.optimizer_registry_factory
        self._orig_objective = worker_hooks.objective_factory
        self._orig_dataset_registry = worker_hooks.dataset_registry_factory
        self._orig_ts_registry = worker_hooks.timeseries_registry_factory
        self._orig_dataset_loader = worker_hooks.dataset_loader

        worker_hooks.dataset_registry_factory = _make_fake_standard_registry
        worker_hooks.timeseries_registry_factory = _make_fake_timeseries_registry

    def teardown_method(self) -> None:
        """Restore all original hooks after each test."""
        worker_hooks.registry_factory = self._orig_registry
        worker_hooks.optimizer_registry_factory = self._orig_optimizer
        worker_hooks.objective_factory = self._orig_objective
        worker_hooks.dataset_registry_factory = self._orig_dataset_registry
        worker_hooks.timeseries_registry_factory = self._orig_ts_registry
        worker_hooks.dataset_loader = self._orig_dataset_loader

    def test_process_optimize_job_returns_encoded_result(
        self,
        tmp_path: Path,
    ) -> None:
        """process_optimize_job returns JSON-serializable dict."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv

        from covenant_radar_api.worker.optimize_job import process_optimize_job

        fake_backend = _FakeBackend()
        fake_optimizer = _FakeOptimizer()
        fake_objective = _FakeObjective()
        fake_dataset = _make_fake_dataset()

        fake_registry = _make_fake_backend_registry(fake_backend)
        fake_optimizer_registry = _make_fake_optimizer_registry(fake_optimizer)

        worker_hooks.registry_factory = lambda: fake_registry
        worker_hooks.optimizer_registry_factory = lambda: fake_optimizer_registry

        def _fake_objective_factory(
            backend_name: BackendName,
            x: NDArray[np.float64],
            y: NDArray[np.int64],
            feature_names: list[str],
            config: UnifiedOptimizeParseResult,
        ) -> ObjectiveWithFeatureCount:
            del backend_name, x, y, feature_names, config
            return fake_objective

        worker_hooks.objective_factory = _fake_objective_factory

        def _fake_loader(
            config: DatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> LoadedDataset:
            del config, external_dir, progress_callback
            return fake_dataset

        worker_hooks.dataset_loader = _fake_loader

        fake_env = FakeEnv(
            {
                "APP__DATA_ROOT": str(tmp_path),
                "APP__MODELS_ROOT": str(tmp_path / "models"),
                "DATABASE_URL": "postgresql://test@localhost/test",
                "REDIS_URL": "redis://localhost:6379/0",
            }
        )

        orig_get_env = config_hooks.get_env
        config_hooks.get_env = fake_env
        try:
            config_json = _make_config_json()
            result = process_optimize_job(config_json)

            assert result["backend"] == "xgboost"
            assert result["status"] == "complete"
            assert result["dataset"] == "taiwan"
        finally:
            config_hooks.get_env = orig_get_env
