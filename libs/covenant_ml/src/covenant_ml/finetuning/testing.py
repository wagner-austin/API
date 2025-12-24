"""Testing utilities for fine-tuning module.

Provides factory functions and fake implementations for fine-tuning tests.
This module is exported for consumers to use in their test suites.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..optimizer.protocol import ObjectiveProtocol, TrialCallbackProtocol
from ..optimizer.types import (
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    SearchSpace,
)
from .protocol import (
    FineTuningCapabilities,
    FineTuningStrategyName,
    FineTuningStrategyProtocol,
)
from .registry import FineTuningRegistration, FineTuningRegistry
from .types import (
    FineTuningConfig,
    FineTuningResult,
    StageResult,
    WarmStartConfig,
)


class FakeFineTuningStrategy:
    """Fake fine-tuning strategy for testing.

    Returns predetermined results for predictable test behavior.
    """

    def __init__(
        self,
        name: FineTuningStrategyName = "staged",
        capabilities: FineTuningCapabilities | None = None,
        result: FineTuningResult | None = None,
    ) -> None:
        """Initialize fake strategy.

        Args:
            name: Strategy name to return.
            capabilities: Capabilities to return. If None, uses defaults.
            result: Predetermined result to return. If None, generates simple result.
        """
        self._name = name
        self._capabilities = capabilities or FineTuningCapabilities(
            supports_warm_start=True,
            supports_staged=True,
            supports_early_stop=True,
            preserves_prior_params=True,
        )
        self._result = result
        self._fine_tune_call_count = 0

    @property
    def fine_tune_call_count(self) -> int:
        """Get the number of times fine_tune was called."""
        return self._fine_tune_call_count

    def strategy_name(self) -> FineTuningStrategyName:
        """Return the configured strategy name."""
        return self._name

    def capabilities(self) -> FineTuningCapabilities:
        """Return the configured capabilities."""
        return self._capabilities

    def fine_tune(
        self,
        x_features: NDArray[np.float64],
        y_labels: NDArray[np.int64],
        feature_names: list[str],
        search_space: SearchSpace,
        config: FineTuningConfig,
        objective: ObjectiveProtocol,
        warm_start: WarmStartConfig | None = None,
        trial_callback: TrialCallbackProtocol | None = None,
    ) -> FineTuningResult:
        """Return predetermined or generated result."""
        del x_features, y_labels, feature_names, search_space
        del objective, warm_start, trial_callback

        self._fine_tune_call_count += 1

        if self._result is not None:
            return self._result

        # Generate simple result
        stage_result = StageResult(
            stage_name="exploration",
            optimization_summary=OptimizationSummary(
                best_trial_number=0,
                best_value=0.85,
                best_int_params=SampledIntParams(max_depth=5, n_estimators=100),
                best_float_params=SampledFloatParams(learning_rate=0.1),
                best_string_params=SampledStringParams(),
                n_trials_total=config["stages"][0]["n_trials"] if config["stages"] else 10,
                n_trials_complete=config["stages"][0]["n_trials"] if config["stages"] else 10,
                n_trials_pruned=0,
                n_trials_failed=0,
                total_duration_seconds=1.0,
            ),
            improvement_over_previous=0.0,
            cumulative_trials=config["stages"][0]["n_trials"] if config["stages"] else 10,
        )

        return FineTuningResult(
            stage_results=(stage_result,),
            final_best_value=0.85,
            final_int_params=SampledIntParams(max_depth=5, n_estimators=100),
            final_float_params=SampledFloatParams(learning_rate=0.1),
            final_string_params=SampledStringParams(),
            total_trials=config["stages"][0]["n_trials"] if config["stages"] else 10,
            total_duration_seconds=1.0,
            stages_completed=1,
            early_stopped=False,
        )


def make_fake_finetuning_strategy(
    name: FineTuningStrategyName = "staged",
    best_value: float = 0.85,
) -> FakeFineTuningStrategy:
    """Create a FakeFineTuningStrategy with specified settings.

    Args:
        name: Strategy name to use.
        best_value: Best value to return in the result.

    Returns:
        A configured FakeFineTuningStrategy instance.
    """
    stage_result = StageResult(
        stage_name="exploration",
        optimization_summary=OptimizationSummary(
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
        ),
        improvement_over_previous=0.0,
        cumulative_trials=10,
    )

    result = FineTuningResult(
        stage_results=(stage_result,),
        final_best_value=best_value,
        final_int_params=SampledIntParams(max_depth=5, n_estimators=100),
        final_float_params=SampledFloatParams(learning_rate=0.1),
        final_string_params=SampledStringParams(),
        total_trials=10,
        total_duration_seconds=1.0,
        stages_completed=1,
        early_stopped=False,
    )

    return FakeFineTuningStrategy(name=name, result=result)


def make_test_finetuning_registry() -> FineTuningRegistry:
    """Create a test fine-tuning registry with fake strategies.

    Returns:
        FineTuningRegistry populated with FakeFineTuningStrategy instances.
    """
    registry = FineTuningRegistry()

    def create_fake_staged() -> FineTuningStrategyProtocol:
        return make_fake_finetuning_strategy("staged")

    registry.register(
        "staged",
        FineTuningRegistration(create_fake_staged),
    )

    def create_fake_warm_start() -> FineTuningStrategyProtocol:
        caps = FineTuningCapabilities(
            supports_warm_start=True,
            supports_staged=False,
            supports_early_stop=False,
            preserves_prior_params=True,
        )
        return FakeFineTuningStrategy(name="warm_start", capabilities=caps)

    registry.register(
        "warm_start",
        FineTuningRegistration(create_fake_warm_start),
    )

    def create_fake_iterative() -> FineTuningStrategyProtocol:
        return make_fake_finetuning_strategy("iterative_refinement")

    registry.register(
        "iterative_refinement",
        FineTuningRegistration(create_fake_iterative),
    )

    return registry


__all__ = [
    "FakeFineTuningStrategy",
    "make_fake_finetuning_strategy",
    "make_test_finetuning_registry",
]
