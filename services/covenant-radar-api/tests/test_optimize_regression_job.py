"""Tests for worker/optimize_regression_job.py regression hyperparameter optimization.

Tests use dependency injection via worker/_regression_hooks and worker/_test_hooks
to verify actual code paths with fake backends and optimizers.

Strict typing only: no Any, no casts, no type: ignore, no stubs, no mocks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from covenant_ml.backends.protocol import BackendCapabilities
from covenant_ml.backends.regressor_protocol import (
    PreparedRegressor,
    RegressorBackend,
    RegressorProgressCallback,
)
from covenant_ml.backends.regressor_registry import (
    RegressorBackendRegistration,
    RegressorRegistry,
)
from covenant_ml.datasets import (
    DatasetRegistry,
    RegressionDatasetConfig,
    RegressionDatasetRegistry,
    RegressionLoadedDataset,
    TimeSeriesDatasetRegistry,
)
from covenant_ml.datasets.protocol import ProgressCallbackProtocol
from covenant_ml.datasets.types import RegressionDatasetMeta, RegressionTargetSpec
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
    TrialState,
    XGBoostSearchSpace,
)
from covenant_ml.types import (
    FeatureImportance,
    RegressionMetrics,
    RegressionTrainOutcome,
    RegressorBackendName,
    RegressorTrainConfig,
)
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    dump_json_str,
)

from covenant_radar_api.worker import _regression_hooks as regression_hooks
from covenant_radar_api.worker import _test_hooks as hooks
from covenant_radar_api.worker._test_hooks import ObjectiveWithFeatureCount
from covenant_radar_api.worker.optimize_regression_job import (
    _make_regression_trial_callback,
    _parse_regression_optimize_config,
    _report_regression_phase,
    run_regression_optimization,
)
from covenant_radar_api.worker.optimize_regression_types import (
    RegressionLoadingProgressInfo,
    RegressionOptimizePhase,
    RegressionPhaseProgressInfo,
    RegressionTrialProgressInfo,
    UnifiedRegressionOptimizeParseResult,
)

# =============================================================================
# Fake Implementations for Testing
# =============================================================================


def _make_fake_regression_dataset(name: str = "financial_distress") -> RegressionLoadedDataset:
    """Create fake regression dataset for testing.

    Args:
        name: Dataset name.

    Returns:
        RegressionLoadedDataset with synthetic data.
    """
    rng = np.random.default_rng(42)
    x: NDArray[np.float64] = rng.random((100, 10)).astype(np.float64)
    y: NDArray[np.float64] = rng.random(100).astype(np.float64)
    meta: RegressionDatasetMeta = {
        "name": name,
        "n_samples": 100,
        "n_features": 10,
        "feature_names": tuple(f"feature_{i}" for i in range(10)),
        "target_mean": 0.5,
        "target_std": 0.3,
        "target_min": 0.0,
        "target_max": 1.0,
        "categorical_encodings": (),
    }
    return {"meta": meta, "x": x, "y": y}


def _make_fake_regression_config(name: str) -> RegressionDatasetConfig:
    """Create fake regression dataset config.

    Args:
        name: Dataset name.

    Returns:
        RegressionDatasetConfig for testing.
    """
    return RegressionDatasetConfig(
        name=name,
        display_name=f"Fake {name}",
        folder=f"{name}_data",
        file_name="data.csv",
        file_format="csv",
        encoding="utf-8",
        target=RegressionTargetSpec(column_name="target"),
        exclude_columns=(),
        n_samples_expected=100,
        n_features_expected=10,
        target_mean_expected=0.5,
    )


def _make_fake_regression_registry() -> RegressionDatasetRegistry:
    """Create fake regression dataset registry.

    Returns:
        RegressionDatasetRegistry with financial_distress.
    """
    configs = (_make_fake_regression_config("financial_distress"),)
    return RegressionDatasetRegistry(configs)


def _make_fake_regression_loader(
    config: RegressionDatasetConfig,
    external_dir: Path,
    progress_callback: ProgressCallbackProtocol | None = None,
) -> RegressionLoadedDataset:
    """Fake regression dataset loader.

    Args:
        config: Dataset config (used for name).
        external_dir: Ignored.
        progress_callback: Ignored.

    Returns:
        Fake RegressionLoadedDataset.
    """
    return _make_fake_regression_dataset(config["name"])


class _FakeRegressorObjective:
    """Fake objective for regression optimization.

    Returns a fixed negative RMSE value.
    """

    def __init__(self, n_features: int = 10, return_value: float = -0.25) -> None:
        """Initialize fake objective.

        Args:
            n_features: Number of features to report.
            return_value: Fixed negative RMSE value.
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
        """Return fixed value.

        Args:
            x_features: Feature matrix.
            y_labels: Labels (ignored for regression).
            feature_names: Feature column names.
            int_params: Integer hyperparameters.
            float_params: Float hyperparameters.
            string_params: String hyperparameters.
            train_ratio: Training data fraction.
            val_ratio: Validation data fraction.
            test_ratio: Test data fraction.
            random_state: Random seed.

        Returns:
            Fixed negative RMSE value.
        """
        del x_features, y_labels, feature_names
        del int_params, float_params, string_params
        del train_ratio, val_ratio, test_ratio, random_state
        return self._return_value


def _make_trial_result(
    trial_number: int = 0,
    value: float = -0.25,
    duration_seconds: float = 0.1,
) -> TrialResult:
    """Create a TrialResult with all required fields for testing.

    Args:
        trial_number: Trial number.
        value: Objective value.
        duration_seconds: Trial duration.

    Returns:
        Complete TrialResult.
    """
    state: TrialState = "complete"
    return TrialResult(
        trial_number=trial_number,
        int_params=SampledIntParams(max_depth=5),
        float_params=SampledFloatParams(learning_rate=0.1),
        string_params=SampledStringParams(),
        value=value,
        state=state,
        duration_seconds=duration_seconds,
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


class _FakeRegressorBackend:
    """Fake RegressorBackend for testing optimize_regression_job.

    Only get_default_search_space is exercised.
    """

    def __init__(self, name: RegressorBackendName = "xgboost_reg") -> None:
        """Initialize fake regressor backend.

        Args:
            name: Backend name.
        """
        self._name = name

    def backend_name(self) -> RegressorBackendName:
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
        feature_names: list[str] | None,
    ) -> PreparedRegressor:
        """Not used in optimize tests."""
        raise NotImplementedError

    def train(
        self,
        *,
        x_features: NDArray[np.float64],
        y_targets: NDArray[np.float64],
        feature_names: list[str] | None,
        config: RegressorTrainConfig,
        output_dir: Path,
        progress: RegressorProgressCallback | None,
    ) -> RegressionTrainOutcome:
        """Not used in optimize tests."""
        raise NotImplementedError

    def evaluate(
        self,
        *,
        model: PreparedRegressor,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> RegressionMetrics:
        """Not used in optimize tests."""
        raise NotImplementedError

    def save(self, *, model: PreparedRegressor, path: str) -> None:
        """Not used in optimize tests."""
        raise NotImplementedError

    def load(self, *, path: str) -> PreparedRegressor:
        """Not used in optimize tests."""
        raise NotImplementedError

    def get_feature_importances(
        self,
        *,
        model: PreparedRegressor,
        feature_names: list[str] | None,
    ) -> list[FeatureImportance] | None:
        """Not used in optimize tests."""
        raise NotImplementedError

    def get_default_search_space(self) -> SearchSpace:
        """Return fake search space."""
        return _FAKE_SEARCH_SPACE

    def get_focused_search_space(
        self,
        *,
        best_int_params: SampledIntParams,
        best_float_params: SampledFloatParams,
    ) -> SearchSpace:
        """Not used in optimize tests."""
        raise NotImplementedError


class _FakeOptimizer:
    """Fake optimizer that records calls and returns predetermined results."""

    def __init__(self, result: OptimizationSummary | None = None) -> None:
        """Initialize fake optimizer.

        Args:
            result: Predetermined result. If None, generates one from config.
        """
        self._result = result
        self._optimize_call_count = 0

    @property
    def optimize_call_count(self) -> int:
        """Get the number of times optimize was called."""
        return self._optimize_call_count

    def strategy_name(self) -> OptimizerStrategyName:
        """Return 'optuna_tpe'."""
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
            y_labels: Labels (int64 zeros for regression).
            feature_names: Feature column names.
            search_space: Hyperparameter search space.
            config: Optimization configuration.
            objective: Objective function.
            trial_callback: Optional trial callback.

        Returns:
            Predetermined or generated optimization summary.
        """
        del x_features, y_labels, feature_names, objective
        del search_space, trial_callback

        self._optimize_call_count += 1

        if self._result is not None:
            return self._result

        return OptimizationSummary(
            best_trial_number=0,
            best_value=-0.25,
            best_int_params=SampledIntParams(max_depth=5, n_estimators=100),
            best_float_params=SampledFloatParams(learning_rate=0.1),
            best_string_params=SampledStringParams(),
            n_trials_total=config["n_trials"],
            n_trials_complete=config["n_trials"],
            n_trials_pruned=0,
            n_trials_failed=0,
            total_duration_seconds=1.0,
        )


def _make_fake_regressor_registry(backend: _FakeRegressorBackend) -> RegressorRegistry:
    """Create a fake regressor registry.

    Args:
        backend: Fake backend to register.

    Returns:
        RegressorRegistry with the fake backend for xgboost_reg and lightgbm_reg.
    """
    reg = RegressorRegistry()
    names: tuple[RegressorBackendName, ...] = ("xgboost_reg", "lightgbm_reg", "mlp_reg", "lstm_reg")
    for name in names:
        _b = backend

        def _factory(_backend: _FakeRegressorBackend = _b) -> RegressorBackend:
            return _backend

        reg.register(name, RegressorBackendRegistration(_factory))
    return reg


def _make_fake_optimizer_registry(optimizer: _FakeOptimizer) -> OptimizerStrategyRegistry:
    """Create a fake optimizer strategy registry.

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


def _make_fake_objective_factory(
    backend_name: RegressorBackendName,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    feature_names: list[str],
    config: UnifiedRegressionOptimizeParseResult,
) -> ObjectiveWithFeatureCount:
    """Fake objective factory returning _FakeRegressorObjective.

    Args:
        backend_name: Backend name (ignored).
        x: Feature matrix (used for n_features).
        y: Target values (ignored).
        feature_names: Feature names (ignored).
        config: Parsed config (ignored).

    Returns:
        Fake objective with correct feature count.
    """
    return _FakeRegressorObjective(n_features=x.shape[1])


def _make_config_json(
    backend: str = "xgboost_reg",
    dataset: str = "financial_distress",
    n_trials: int = 5,
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


def _make_fake_standard_registry() -> DatasetRegistry:
    """Create empty standard dataset registry (unused for regression tests).

    Returns:
        Empty DatasetRegistry.
    """
    return DatasetRegistry(())


def _make_fake_timeseries_registry() -> TimeSeriesDatasetRegistry:
    """Create empty time-series dataset registry (unused for regression tests).

    Returns:
        Empty TimeSeriesDatasetRegistry.
    """
    return TimeSeriesDatasetRegistry(())


# =============================================================================
# Tests: _parse_regression_optimize_config
# =============================================================================


class TestParseRegressionOptimizeConfig:
    """Tests for _parse_regression_optimize_config function."""

    def setup_method(self) -> None:
        """Install fake regression registry before each test."""
        self._orig_regression_registry = regression_hooks.regression_registry_factory
        regression_hooks.regression_registry_factory = _make_fake_regression_registry

    def teardown_method(self) -> None:
        """Restore original hooks after each test."""
        regression_hooks.regression_registry_factory = self._orig_regression_registry

    def test_minimal_config_returns_defaults(self) -> None:
        """Minimal config uses defaults for all optional fields."""
        config_json = _make_config_json()
        result = _parse_regression_optimize_config(config_json)

        assert result["backend"] == "xgboost_reg"
        assert result["dataset"] == "financial_distress"
        assert result["n_trials"] == 5
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
            backend="lightgbm_reg",
            dataset="financial_distress",
            n_trials=50,
            timeout_seconds=3600,
            device="cuda",
            feature_preset="full",
            random_state=123,
            early_stopping_rounds=20,
            n_jobs=4,
            precision="fp16",
            nn_optimizer="adam",
            n_epochs=100,
            early_stopping_patience=20,
            sequence_length=10,
            bidirectional=True,
        )
        result = _parse_regression_optimize_config(config_json)

        assert result["backend"] == "lightgbm_reg"
        assert result["dataset"] == "financial_distress"
        assert result["n_trials"] == 50
        assert result["timeout_seconds"] == 3600
        assert result["device"] == "cuda"
        assert result["feature_preset"] == "full"
        assert result["random_state"] == 123
        assert result["early_stopping_rounds"] == 20
        assert result["n_jobs"] == 4
        assert result["precision"] == "fp16"
        assert result["nn_optimizer"] == "adam"
        assert result["n_epochs"] == 100
        assert result["early_stopping_patience"] == 20
        assert result["sequence_length"] == 10
        assert result["bidirectional"] is True

    def test_all_four_backends_accepted(self) -> None:
        """All 4 regressor backends are accepted."""
        backends: tuple[str, ...] = ("xgboost_reg", "lightgbm_reg", "mlp_reg", "lstm_reg")
        for backend in backends:
            config_json = _make_config_json(backend=backend)
            result = _parse_regression_optimize_config(config_json)
            assert result["backend"] == backend

    def test_invalid_backend_raises(self) -> None:
        """Invalid backend raises ValueError."""
        config_json = _make_config_json(backend="invalid")
        with pytest.raises(ValueError, match="backend must be one of"):
            _parse_regression_optimize_config(config_json)

    def test_missing_dataset_raises(self) -> None:
        """Missing dataset field raises JSONTypeError."""
        config_json = dump_json_str({"backend": "xgboost_reg", "n_trials": 5})
        with pytest.raises(JSONTypeError, match="Missing required field 'dataset'"):
            _parse_regression_optimize_config(config_json)

    def test_invalid_dataset_raises(self) -> None:
        """Invalid dataset name raises ValueError."""
        config_json = _make_config_json(dataset="nonexistent")
        with pytest.raises(ValueError, match="dataset must be one of"):
            _parse_regression_optimize_config(config_json)

    def test_missing_n_trials_raises(self) -> None:
        """Missing n_trials raises JSONTypeError."""
        config_json = dump_json_str({"backend": "xgboost_reg", "dataset": "financial_distress"})
        with pytest.raises(JSONTypeError, match="Missing required field 'n_trials'"):
            _parse_regression_optimize_config(config_json)

    def test_non_dict_config_raises(self) -> None:
        """Non-object JSON raises JSONTypeError."""
        with pytest.raises(JSONTypeError, match="config must be a JSON object"):
            _parse_regression_optimize_config('"just a string"')

    def test_invalid_timeout_type_raises(self) -> None:
        """Non-integer timeout raises JSONTypeError."""
        config_json = _make_config_json(timeout_seconds="fast")
        with pytest.raises(JSONTypeError, match="timeout_seconds must be an integer"):
            _parse_regression_optimize_config(config_json)

    def test_null_timeout_allowed(self) -> None:
        """Null timeout results in None."""
        config_json = _make_config_json(timeout_seconds=None)
        result = _parse_regression_optimize_config(config_json)
        assert result["timeout_seconds"] is None

    def test_device_cpu(self) -> None:
        """CPU device is accepted."""
        config_json = _make_config_json(device="cpu")
        result = _parse_regression_optimize_config(config_json)
        assert result["device"] == "cpu"

    def test_feature_preset_log_only(self) -> None:
        """log_only feature preset is accepted."""
        config_json = _make_config_json(feature_preset="log_only")
        result = _parse_regression_optimize_config(config_json)
        assert result["feature_preset"] == "log_only"


# =============================================================================
# Tests: _report_regression_phase
# =============================================================================


class TestReportRegressionPhase:
    """Tests for _report_regression_phase function."""

    def test_calls_callback_with_correct_info(self) -> None:
        """Callback receives correctly populated RegressionPhaseProgressInfo."""
        received: list[RegressionPhaseProgressInfo] = []

        def _callback(info: RegressionPhaseProgressInfo) -> None:
            received.append(info)

        _report_regression_phase(
            _callback,
            "loading_data",
            "xgboost_reg",
            "financial_distress",
            100,
            10,
        )

        assert len(received) == 1
        info = received[0]
        assert info["phase"] == "loading_data"
        assert info["backend"] == "xgboost_reg"
        assert info["dataset"] == "financial_distress"
        assert info["n_samples"] == 100
        assert info["n_features"] == 10

    def test_none_callback_is_safe(self) -> None:
        """None callback does not raise."""
        _report_regression_phase(None, "optimizing", "xgboost_reg", "test", 50, 5)

    def test_all_phases(self) -> None:
        """All four phases can be reported."""
        received: list[RegressionPhaseProgressInfo] = []

        def _callback(info: RegressionPhaseProgressInfo) -> None:
            received.append(info)

        phases: tuple[RegressionOptimizePhase, ...] = (
            "loading_data",
            "feature_engineering",
            "optimizing",
            "saving",
        )
        for phase in phases:
            _report_regression_phase(_callback, phase, "lightgbm_reg", "test", 0, 0)

        assert len(received) == 4
        assert [r["phase"] for r in received] == list(phases)


# =============================================================================
# Tests: _make_regression_trial_callback
# =============================================================================


class TestMakeRegressionTrialCallback:
    """Tests for _make_regression_trial_callback function."""

    def test_tracks_best_value(self) -> None:
        """Trial callback tracks best value correctly."""
        received: list[RegressionTrialProgressInfo] = []

        def _progress(info: RegressionTrialProgressInfo) -> None:
            received.append(info)

        callback = _make_regression_trial_callback("xgboost_reg", 5, _progress)

        # First trial is always best
        callback(_make_trial_result(trial_number=0, value=-0.5))
        assert received[0]["is_best"] is True
        assert received[0]["best_value"] == -0.5
        assert received[0]["best_trial"] == 0

        # Worse trial (more negative = worse for neg RMSE)
        callback(_make_trial_result(trial_number=1, value=-0.8))
        assert received[1]["is_best"] is False
        assert received[1]["best_value"] == -0.5
        assert received[1]["best_trial"] == 0

        # Better trial (less negative = better)
        callback(_make_trial_result(trial_number=2, value=-0.3))
        assert received[2]["is_best"] is True
        assert received[2]["best_value"] == -0.3
        assert received[2]["best_trial"] == 2

    def test_none_callback_is_safe(self) -> None:
        """None progress callback does not raise."""
        callback = _make_regression_trial_callback("xgboost_reg", 5, None)
        callback(_make_trial_result(trial_number=0, value=-0.5))

    def test_backend_name_propagated(self) -> None:
        """Backend name is included in progress info."""
        received: list[RegressionTrialProgressInfo] = []

        def _progress(info: RegressionTrialProgressInfo) -> None:
            received.append(info)

        callback = _make_regression_trial_callback("lightgbm_reg", 10, _progress)
        callback(_make_trial_result(trial_number=0, value=-0.5))

        assert received[0]["backend"] == "lightgbm_reg"
        assert received[0]["n_trials_total"] == 10

    def test_current_value_always_reported(self) -> None:
        """Current trial value is always reported regardless of best status."""
        received: list[RegressionTrialProgressInfo] = []

        def _progress(info: RegressionTrialProgressInfo) -> None:
            received.append(info)

        callback = _make_regression_trial_callback("xgboost_reg", 3, _progress)
        callback(_make_trial_result(trial_number=0, value=-0.5))
        callback(_make_trial_result(trial_number=1, value=-0.9))

        assert received[0]["current_value"] == -0.5
        assert received[1]["current_value"] == -0.9


# =============================================================================
# Tests: run_regression_optimization
# =============================================================================


class TestRunRegressionOptimization:
    """Tests for run_regression_optimization function."""

    def setup_method(self) -> None:
        """Install all fake hooks before each test."""
        # Regression hooks
        self._orig_regression_registry = regression_hooks.regression_registry_factory
        self._orig_regression_loader = regression_hooks.regression_dataset_loader
        self._orig_regressor_registry = regression_hooks.regressor_registry_factory
        self._orig_regressor_objective = regression_hooks.regressor_objective_factory

        # Classifier hooks (for optimizer registry)
        self._orig_optimizer_registry = hooks.optimizer_registry_factory
        self._orig_dataset_registry = hooks.dataset_registry_factory
        self._orig_ts_registry = hooks.timeseries_registry_factory

        # Install fakes
        self._fake_backend = _FakeRegressorBackend()
        self._fake_optimizer = _FakeOptimizer()

        regression_hooks.regression_registry_factory = _make_fake_regression_registry
        regression_hooks.regression_dataset_loader = _make_fake_regression_loader
        regression_hooks.regressor_registry_factory = (
            lambda b=self._fake_backend: _make_fake_regressor_registry(b)
        )
        regression_hooks.regressor_objective_factory = _make_fake_objective_factory

        hooks.optimizer_registry_factory = (
            lambda o=self._fake_optimizer: _make_fake_optimizer_registry(o)
        )
        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

    def teardown_method(self) -> None:
        """Restore original hooks after each test."""
        regression_hooks.regression_registry_factory = self._orig_regression_registry
        regression_hooks.regression_dataset_loader = self._orig_regression_loader
        regression_hooks.regressor_registry_factory = self._orig_regressor_registry
        regression_hooks.regressor_objective_factory = self._orig_regressor_objective

        hooks.optimizer_registry_factory = self._orig_optimizer_registry
        hooks.dataset_registry_factory = self._orig_dataset_registry
        hooks.timeseries_registry_factory = self._orig_ts_registry

    def test_basic_optimization(self, tmp_path: Path) -> None:
        """Basic regression optimization runs end to end."""
        config_json = _make_config_json()
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["backend"] == "xgboost_reg"
        assert result["status"] == "complete"
        assert result["dataset"] == "financial_distress"
        assert result["n_samples"] == 100
        assert result["n_features"] == 10
        assert result["best_value"] == -0.25
        assert result["n_trials_complete"] == 5
        assert result["duration_seconds"] == 1.0
        assert self._fake_optimizer.optimize_call_count == 1

    def test_lightgbm_backend(self, tmp_path: Path) -> None:
        """LightGBM regressor backend works end to end."""
        config_json = _make_config_json(backend="lightgbm_reg")
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["backend"] == "lightgbm_reg"
        assert result["status"] == "complete"

    def test_with_timeout(self, tmp_path: Path) -> None:
        """Timeout parameter is forwarded to optimizer."""
        config_json = _make_config_json(timeout_seconds=3600)
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["status"] == "complete"

    def test_custom_feature_preset(self, tmp_path: Path) -> None:
        """Feature preset is captured in result."""
        config_json = _make_config_json(feature_preset="full")
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["feature_preset"] == "full"

    def test_phase_callbacks(self, tmp_path: Path) -> None:
        """Phase callbacks are invoked for all four phases."""
        phases: list[str] = []

        def _phase_callback(info: RegressionPhaseProgressInfo) -> None:
            phases.append(info["phase"])

        config_json = _make_config_json()
        run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
            phase_callback=_phase_callback,
        )

        assert phases == ["loading_data", "feature_engineering", "optimizing", "saving"]

    def test_trial_progress_callback(self, tmp_path: Path) -> None:
        """Trial progress callback is invoked during optimization."""

        # Use an optimizer that invokes the trial callback
        def _optimizer_with_callback() -> _FakeOptimizer:
            class _InvokingOptimizer(_FakeOptimizer):
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
                    self._optimize_call_count += 1
                    if trial_callback is not None:
                        trial_callback(_make_trial_result(trial_number=0, value=-0.25))
                    return OptimizationSummary(
                        best_trial_number=0,
                        best_value=-0.25,
                        best_int_params=SampledIntParams(max_depth=5),
                        best_float_params=SampledFloatParams(learning_rate=0.1),
                        best_string_params=SampledStringParams(),
                        n_trials_total=config["n_trials"],
                        n_trials_complete=config["n_trials"],
                        n_trials_pruned=0,
                        n_trials_failed=0,
                        total_duration_seconds=0.5,
                    )

            return _InvokingOptimizer()

        invoking_optimizer = _optimizer_with_callback()
        hooks.optimizer_registry_factory = lambda: _make_fake_optimizer_registry(invoking_optimizer)

        trial_infos: list[RegressionTrialProgressInfo] = []

        def _trial_callback(info: RegressionTrialProgressInfo) -> None:
            trial_infos.append(info)

        config_json = _make_config_json()
        run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
            progress_callback=_trial_callback,
        )

        assert len(trial_infos) == 1
        assert trial_infos[0]["backend"] == "xgboost_reg"
        assert trial_infos[0]["current_value"] == -0.25
        assert trial_infos[0]["is_best"] is True

    def test_loading_progress_callback(self, tmp_path: Path) -> None:
        """Loading progress callback is invoked during dataset loading."""
        from covenant_ml.datasets.types import LoadProgress

        loading_infos: list[RegressionLoadingProgressInfo] = []

        def _loading_callback(info: RegressionLoadingProgressInfo) -> None:
            loading_infos.append(info)

        # Use a loader that invokes the progress callback
        def _loader_with_progress(
            config: RegressionDatasetConfig,
            external_dir: Path,
            progress_callback: ProgressCallbackProtocol | None = None,
        ) -> RegressionLoadedDataset:
            if progress_callback is not None:
                progress_callback(
                    LoadProgress(
                        phase="reading",
                        bytes_read=1000,
                        bytes_total=1000,
                        percent_complete=100.0,
                        rows_processed=100,
                        rows_total=100,
                        message="Done",
                    )
                )
            return _make_fake_regression_dataset(config["name"])

        regression_hooks.regression_dataset_loader = _loader_with_progress

        config_json = _make_config_json()
        run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
            loading_progress_callback=_loading_callback,
        )

        assert len(loading_infos) == 1
        assert loading_infos[0]["dataset"] == "financial_distress"
        assert loading_infos[0]["phase"] == "reading"
        assert loading_infos[0]["percent_complete"] == 100.0

    def test_result_has_best_params(self, tmp_path: Path) -> None:
        """Result includes best hyperparameters from optimizer."""
        config_json = _make_config_json()
        result = run_regression_optimization(
            config_json,
            tmp_path / "external",
            tmp_path / "output",
        )

        assert result["best_int_params"]["max_depth"] == 5
        assert result["best_int_params"]["n_estimators"] == 100
        assert result["best_float_params"]["learning_rate"] == 0.1
        assert result["best_trial_number"] == 0

    def test_saves_results_to_output_dir(self, tmp_path: Path) -> None:
        """Results are saved to the output directory."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        config_json = _make_config_json()
        run_regression_optimization(
            config_json,
            tmp_path / "external",
            output_dir,
        )

        # save_optimization_results creates a dataset subdirectory with result + config files
        saved_files = sorted(output_dir.rglob("*.json"))
        assert saved_files[0].name.endswith(".json")
        assert len(saved_files) == 2


# =============================================================================
# Tests: process_regression_optimize_job
# =============================================================================


class TestProcessRegressionOptimizeJob:
    """Tests for process_regression_optimize_job RQ entry point."""

    def setup_method(self) -> None:
        """Install all fake hooks before each test."""
        self._orig_regression_registry = regression_hooks.regression_registry_factory
        self._orig_regression_loader = regression_hooks.regression_dataset_loader
        self._orig_regressor_registry = regression_hooks.regressor_registry_factory
        self._orig_regressor_objective = regression_hooks.regressor_objective_factory
        self._orig_optimizer_registry = hooks.optimizer_registry_factory
        self._orig_dataset_registry = hooks.dataset_registry_factory
        self._orig_ts_registry = hooks.timeseries_registry_factory

        regression_hooks.regression_registry_factory = _make_fake_regression_registry
        regression_hooks.regression_dataset_loader = _make_fake_regression_loader
        hooks.dataset_registry_factory = _make_fake_standard_registry
        hooks.timeseries_registry_factory = _make_fake_timeseries_registry

    def teardown_method(self) -> None:
        """Restore all original hooks after each test."""
        regression_hooks.regression_registry_factory = self._orig_regression_registry
        regression_hooks.regression_dataset_loader = self._orig_regression_loader
        regression_hooks.regressor_registry_factory = self._orig_regressor_registry
        regression_hooks.regressor_objective_factory = self._orig_regressor_objective
        hooks.optimizer_registry_factory = self._orig_optimizer_registry
        hooks.dataset_registry_factory = self._orig_dataset_registry
        hooks.timeseries_registry_factory = self._orig_ts_registry

    def test_process_regression_optimize_job_returns_encoded_result(
        self,
        tmp_path: Path,
    ) -> None:
        """process_regression_optimize_job returns JSON-serializable dict."""
        from platform_core.config import _test_hooks as config_hooks
        from platform_core.testing import FakeEnv

        from covenant_radar_api.worker.optimize_regression_job import (
            process_regression_optimize_job,
        )

        fake_backend = _FakeRegressorBackend()
        fake_optimizer = _FakeOptimizer()

        regression_hooks.regressor_registry_factory = (
            lambda b=fake_backend: _make_fake_regressor_registry(b)
        )
        regression_hooks.regressor_objective_factory = _make_fake_objective_factory
        hooks.optimizer_registry_factory = lambda o=fake_optimizer: _make_fake_optimizer_registry(o)

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
            result = process_regression_optimize_job(config_json)

            assert result["backend"] == "xgboost_reg"
            assert result["status"] == "complete"
            assert result["dataset"] == "financial_distress"
        finally:
            config_hooks.get_env = orig_get_env
