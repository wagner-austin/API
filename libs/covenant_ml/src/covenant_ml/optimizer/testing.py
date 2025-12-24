"""Testing utilities for optimizer module.

Provides factory functions and fake implementations for optimizer tests.
This module is exported for consumers to use in their test suites.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .protocol import ObjectiveProtocol, TrialCallbackProtocol
from .registry import OptimizerStrategyRegistration, OptimizerStrategyRegistry
from .strategy_protocol import (
    HyperparameterOptimizerProtocol,
    OptimizerStrategyCapabilities,
    OptimizerStrategyName,
)
from .types import (
    OptimizationConfig,
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    SearchSpace,
    TrialResult,
)


class FakeHyperparameterOptimizer:
    """Fake optimizer for testing.

    Returns predetermined results for predictable test behavior.
    Useful for testing code that depends on optimization without
    actually running the expensive optimization process.
    """

    def __init__(
        self,
        name: OptimizerStrategyName = "optuna_tpe",
        capabilities: OptimizerStrategyCapabilities | None = None,
        result: OptimizationSummary | None = None,
    ) -> None:
        """Initialize fake optimizer.

        Args:
            name: Strategy name to return.
            capabilities: Capabilities to return. If None, uses defaults.
            result: Predetermined result to return. If None, generates simple result.
        """
        self._name = name
        self._capabilities = capabilities or OptimizerStrategyCapabilities(
            supports_pruning=True,
            supports_parallel=True,
            is_deterministic=False,
            requires_bounds=True,
        )
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
        """Return the configured strategy name."""
        return self._name

    def capabilities(self) -> OptimizerStrategyCapabilities:
        """Return the configured capabilities."""
        return self._capabilities

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
        """Return predetermined or generated optimization result."""
        del x_features, y_labels, feature_names, objective, trial_callback

        self._optimize_call_count += 1
        self._last_search_space = search_space
        self._last_config = config

        if self._result is not None:
            return self._result

        # Generate simple result
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


class FakeObjective:
    """Fake objective function for testing."""

    def __init__(self, return_value: float = 0.85) -> None:
        """Initialize with a fixed return value.

        Args:
            return_value: The AUC value to return from all calls.
        """
        self._return_value = return_value
        self._call_count = 0
        self._calls: list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]] = []

    @property
    def call_count(self) -> int:
        """Get the number of times the objective was called."""
        return self._call_count

    @property
    def calls(
        self,
    ) -> list[tuple[SampledIntParams, SampledFloatParams, SampledStringParams]]:
        """Get the list of parameter tuples from all calls."""
        return self._calls

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
        """Record call and return fixed value."""
        del x_features, y_labels, feature_names
        del train_ratio, val_ratio, test_ratio, random_state

        self._call_count += 1
        self._calls.append((int_params, float_params, string_params))
        return self._return_value


class FakeTrialCallback:
    """Fake trial callback for testing."""

    def __init__(self) -> None:
        """Initialize with empty results list."""
        self._results: list[TrialResult] = []

    @property
    def results(self) -> list[TrialResult]:
        """Get the list of trial results."""
        return self._results

    def __call__(self, result: TrialResult) -> None:
        """Record trial result."""
        self._results.append(result)


def make_fake_optimizer(
    name: OptimizerStrategyName = "optuna_tpe",
    best_value: float = 0.85,
) -> FakeHyperparameterOptimizer:
    """Create a FakeHyperparameterOptimizer with specified settings.

    Args:
        name: Strategy name to use.
        best_value: Best value to return in the summary.

    Returns:
        A configured FakeHyperparameterOptimizer instance.
    """
    result = OptimizationSummary(
        best_trial_number=0,
        best_value=best_value,
        best_int_params=SampledIntParams(max_depth=5, n_estimators=100),
        best_float_params=SampledFloatParams(learning_rate=0.1),
        best_string_params=SampledStringParams(),
        n_trials_total=10,
        n_trials_complete=10,
        n_trials_pruned=0,
        n_trials_failed=0,
        total_duration_seconds=1.0,
    )
    return FakeHyperparameterOptimizer(name=name, result=result)


def make_test_optimization_config(
    n_trials: int = 10,
    random_state: int = 42,
) -> OptimizationConfig:
    """Create an OptimizationConfig for testing.

    Args:
        n_trials: Number of trials.
        random_state: Random seed.

    Returns:
        A minimal OptimizationConfig for tests.
    """
    return OptimizationConfig(
        n_trials=n_trials,
        timeout_seconds=None,
        n_startup_trials=5,
        random_state=random_state,
        direction="maximize",
        pruning_enabled=False,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )


def make_test_optimizer_registry() -> OptimizerStrategyRegistry:
    """Create a test optimizer registry with fake optimizers.

    Returns:
        OptimizerStrategyRegistry populated with FakeHyperparameterOptimizer instances.
    """
    registry = OptimizerStrategyRegistry()

    def create_fake_optuna() -> HyperparameterOptimizerProtocol:
        return make_fake_optimizer("optuna_tpe")

    registry.register(
        "optuna_tpe",
        OptimizerStrategyRegistration(create_fake_optuna),
    )

    def create_fake_random() -> HyperparameterOptimizerProtocol:
        caps = OptimizerStrategyCapabilities(
            supports_pruning=False,
            supports_parallel=True,
            is_deterministic=True,
            requires_bounds=True,
        )
        return FakeHyperparameterOptimizer(name="random_search", capabilities=caps)

    registry.register(
        "random_search",
        OptimizerStrategyRegistration(create_fake_random),
    )

    def create_fake_grid() -> HyperparameterOptimizerProtocol:
        caps = OptimizerStrategyCapabilities(
            supports_pruning=False,
            supports_parallel=True,
            is_deterministic=True,
            requires_bounds=True,
        )
        return FakeHyperparameterOptimizer(name="grid_search", capabilities=caps)

    registry.register(
        "grid_search",
        OptimizerStrategyRegistration(create_fake_grid),
    )

    return registry


__all__ = [
    "FakeHyperparameterOptimizer",
    "FakeObjective",
    "FakeTrialCallback",
    "make_fake_optimizer",
    "make_test_optimization_config",
    "make_test_optimizer_registry",
]
