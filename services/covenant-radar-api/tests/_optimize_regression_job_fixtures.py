"""Shared fixtures and helpers for test_optimize_regression_job splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    JSONValue,
    dump_json_str,
)

from covenant_radar_api.worker._hook_protocols import (
    ObjectiveWithFeatureCount,
)
from covenant_radar_api.worker.optimize_regression_types import (
    UnifiedRegressionOptimizeParseResult,
)


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
