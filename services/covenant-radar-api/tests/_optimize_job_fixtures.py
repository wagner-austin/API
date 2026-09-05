"""Shared fixtures and helpers for test_optimize_job splits."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from covenant_ml.backends.protocol import (
    BackendCapabilities,
    PreparedClassifier,
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
    XGBoostSearchSpace,
)
from covenant_ml.types import (
    BackendName,
    ClassifierTrainConfig,
    EvalMetrics,
    FeatureImportance,
    TrainOutcome,
    TrainProgress,
)
from numpy.typing import NDArray
from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
)


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
        progress: Callable[[TrainProgress], None] | None,
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
