"""Iterative refinement fine-tuning strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Implements repeated refinement until convergence or max iterations.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ...optimizer.protocol import ObjectiveProtocol, TrialCallbackProtocol
from ...optimizer.strategies.random_search import RandomSearchOptimizer
from ...optimizer.types import (
    OptimizationConfig,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
    SearchSpace,
)
from ..protocol import FineTuningCapabilities, FineTuningStrategyName
from ..space_narrowing import narrow_search_space
from ..types import (
    FineTuningConfig,
    FineTuningResult,
    FineTuningStage,
    StageResult,
    WarmStartConfig,
)

_log = get_logger(__name__)


class IterativeRefinementFineTuning:
    """Iterative refinement fine-tuning with automatic convergence detection.

    Repeatedly runs optimization with progressively narrowing search spaces
    until improvement falls below a threshold or max iterations reached.
    Each iteration uses the same number of trials but a smaller search radius.

    Unlike staged fine-tuning which uses predefined stages, iterative
    refinement adapts the number of iterations based on convergence.
    """

    def __init__(
        self,
        trials_per_iteration: int = 20,
        max_iterations: int = 10,
        radius_decay: float = 0.7,
    ) -> None:
        """Initialize iterative refinement strategy.

        Args:
            trials_per_iteration: Trials to run in each iteration.
            max_iterations: Maximum iterations before stopping.
            radius_decay: Factor to multiply search radius each iteration.
        """
        self._trials_per_iteration = trials_per_iteration
        self._max_iterations = max_iterations
        self._radius_decay = radius_decay

    @property
    def trials_per_iteration(self) -> int:
        """Get trials per iteration."""
        return self._trials_per_iteration

    @property
    def max_iterations(self) -> int:
        """Get maximum iterations."""
        return self._max_iterations

    @property
    def radius_decay(self) -> float:
        """Get radius decay factor."""
        return self._radius_decay

    def strategy_name(self) -> FineTuningStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'iterative_refinement'.
        """
        return "iterative_refinement"

    def capabilities(self) -> FineTuningCapabilities:
        """Return the capabilities of this strategy.

        Returns:
            Capabilities indicating iterative supports all features.
        """
        return FineTuningCapabilities(
            supports_warm_start=True,
            supports_staged=False,
            supports_early_stop=True,
            preserves_prior_params=True,
        )

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
        """Execute iterative refinement fine-tuning.

        Runs iterations until convergence or max iterations reached.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_labels: Binary labels (n_samples,).
            feature_names: Names for each feature column.
            search_space: Base parameter ranges to search.
            config: Fine-tuning configuration (uses early_stop_threshold).
            objective: Function to evaluate hyperparameters.
            warm_start: Optional warm-start from prior optimization.
            trial_callback: Optional callback after each trial.

        Returns:
            Complete fine-tuning result with best parameters.
        """
        start_time = time.perf_counter()

        _log.info(
            "Starting iterative refinement fine-tuning",
            extra={
                "trials_per_iteration": self._trials_per_iteration,
                "max_iterations": self._max_iterations,
                "radius_decay": self._radius_decay,
                "early_stop_threshold": config["early_stop_threshold"],
            },
        )

        # Initialize best params
        best_value: float = float("-inf")
        best_int_params: SampledIntParams = {}
        best_float_params: SampledFloatParams = {}
        best_string_params: SampledStringParams = {}
        current_radius: float = 1.0

        # Override with warm start if provided
        if warm_start is not None:
            best_value = warm_start["prior_summary"]["best_value"]
            best_int_params = warm_start["prior_summary"]["best_int_params"]
            best_float_params = warm_start["prior_summary"]["best_float_params"]
            best_string_params = warm_start["prior_summary"]["best_string_params"]
            current_radius = warm_start["narrow_factor"]

        stage_results: list[StageResult] = []
        cumulative_trials = 0
        early_stopped = False

        optimizer = RandomSearchOptimizer()

        for iteration in range(self._max_iterations):
            # Check if we've exceeded max trials
            if cumulative_trials >= config["max_total_trials"]:
                _log.info(
                    "Stopping: max total trials reached",
                    extra={"cumulative_trials": cumulative_trials},
                )
                break

            # Calculate trials for this iteration
            remaining_trials = config["max_total_trials"] - cumulative_trials
            iter_trials = min(self._trials_per_iteration, remaining_trials)

            _log.info(
                "Starting refinement iteration",
                extra={
                    "iteration": iteration,
                    "n_trials": iter_trials,
                    "current_radius": current_radius,
                },
            )

            # Narrow search space if we have previous best
            if best_value > float("-inf"):
                current_space = narrow_search_space(
                    search_space,
                    best_int_params,
                    best_float_params,
                    best_string_params,
                    current_radius,
                )
            else:
                current_space = search_space

            # Create optimization config for this iteration
            opt_config = OptimizationConfig(
                n_trials=iter_trials,
                timeout_seconds=None,
                n_startup_trials=min(5, iter_trials // 2),
                random_state=config["random_state"] + iteration,
                direction="maximize",
                pruning_enabled=False,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            # Run optimization
            summary = optimizer.optimize(
                x_features=x_features,
                y_labels=y_labels,
                feature_names=feature_names,
                search_space=current_space,
                config=opt_config,
                objective=objective,
                trial_callback=trial_callback,
            )

            cumulative_trials += summary["n_trials_complete"]

            # Calculate improvement
            improvement = 0.0
            if summary["best_value"] > best_value:
                if best_value > float("-inf"):
                    improvement = (summary["best_value"] - best_value) / abs(best_value)
                best_value = summary["best_value"]
                best_int_params = summary["best_int_params"]
                best_float_params = summary["best_float_params"]
                best_string_params = summary["best_string_params"]

            # Map iteration to stage name
            stage_name: FineTuningStage
            if iteration == 0:
                stage_name = "exploration"
            elif iteration == self._max_iterations - 1:
                stage_name = "final"
            else:
                stage_name = "refinement"

            stage_result = StageResult(
                stage_name=stage_name,
                optimization_summary=summary,
                improvement_over_previous=improvement,
                cumulative_trials=cumulative_trials,
            )
            stage_results.append(stage_result)

            _log.info(
                "Refinement iteration complete",
                extra={
                    "iteration": iteration,
                    "best_value": summary["best_value"],
                    "improvement": improvement,
                    "cumulative_trials": cumulative_trials,
                },
            )

            # Check for early stopping (after first iteration)
            if (
                iteration > 0
                and config["early_stop_threshold"] > 0
                and improvement < config["early_stop_threshold"]
            ):
                _log.info(
                    "Early stopping: insufficient improvement",
                    extra={
                        "improvement": improvement,
                        "threshold": config["early_stop_threshold"],
                    },
                )
                early_stopped = True
                break

            # Decay the search radius for next iteration
            current_radius *= self._radius_decay

        total_duration = time.perf_counter() - start_time

        result = FineTuningResult(
            stage_results=tuple(stage_results),
            final_best_value=best_value,
            final_int_params=best_int_params,
            final_float_params=best_float_params,
            final_string_params=best_string_params,
            total_trials=cumulative_trials,
            total_duration_seconds=total_duration,
            stages_completed=len(stage_results),
            early_stopped=early_stopped,
        )

        _log.info(
            "Iterative refinement complete",
            extra={
                "final_best_value": result["final_best_value"],
                "total_trials": result["total_trials"],
                "iterations": result["stages_completed"],
                "early_stopped": result["early_stopped"],
                "total_duration_sec": result["total_duration_seconds"],
            },
        )

        return result


def create_iterative_refinement_finetuning() -> IterativeRefinementFineTuning:
    """Factory function to create an IterativeRefinementFineTuning instance.

    Returns:
        A new IterativeRefinementFineTuning instance with defaults.
    """
    return IterativeRefinementFineTuning(
        trials_per_iteration=20,
        max_iterations=10,
        radius_decay=0.7,
    )


__all__ = [
    "IterativeRefinementFineTuning",
    "create_iterative_refinement_finetuning",
]
