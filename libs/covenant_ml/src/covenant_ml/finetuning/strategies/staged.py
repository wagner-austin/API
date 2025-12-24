"""Staged fine-tuning strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Implements multi-stage optimization with narrowing search spaces.
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
    StageResult,
    WarmStartConfig,
)

_log = get_logger(__name__)


class StagedFineTuning:
    """Multi-stage fine-tuning with progressively narrowing search spaces.

    Executes optimization in multiple stages, where each stage narrows
    the search space around the best parameters from the previous stage.
    This allows broad exploration initially, followed by focused refinement.

    Typical workflow:
    1. Exploration stage: Wide search space, more trials
    2. Refinement stage: Narrowed around best, fewer trials
    3. Final stage: Tight search for precise optimization
    """

    def __init__(self) -> None:
        """Initialize staged fine-tuning strategy."""
        pass

    def strategy_name(self) -> FineTuningStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'staged'.
        """
        return "staged"

    def capabilities(self) -> FineTuningCapabilities:
        """Return the capabilities of this strategy.

        Returns:
            Capabilities indicating staged supports all features.
        """
        return FineTuningCapabilities(
            supports_warm_start=True,
            supports_staged=True,
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
        """Execute multi-stage fine-tuning.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_labels: Binary labels (n_samples,).
            feature_names: Names for each feature column.
            search_space: Base parameter ranges to search.
            config: Fine-tuning configuration with stages.
            objective: Function to evaluate hyperparameters.
            warm_start: Optional warm-start from prior optimization.
            trial_callback: Optional callback after each trial.

        Returns:
            Complete fine-tuning result with best parameters.
        """
        start_time = time.perf_counter()

        _log.info(
            "Starting staged fine-tuning",
            extra={
                "n_stages": len(config["stages"]),
                "max_total_trials": config["max_total_trials"],
                "early_stop_threshold": config["early_stop_threshold"],
            },
        )

        # Initialize from warm start if provided
        # Initialize best params
        best_value: float = float("-inf")
        best_int_params: SampledIntParams = {}
        best_float_params: SampledFloatParams = {}
        best_string_params: SampledStringParams = {}
        current_space: SearchSpace = search_space

        if warm_start is not None:
            current_space = narrow_search_space(
                search_space,
                warm_start["prior_summary"]["best_int_params"],
                warm_start["prior_summary"]["best_float_params"],
                warm_start["prior_summary"]["best_string_params"],
                warm_start["narrow_factor"],
            )
            best_value = warm_start["prior_summary"]["best_value"]
            best_int_params = warm_start["prior_summary"]["best_int_params"]
            best_float_params = warm_start["prior_summary"]["best_float_params"]
            best_string_params = warm_start["prior_summary"]["best_string_params"]

        stage_results: list[StageResult] = []
        cumulative_trials = 0
        early_stopped = False

        # Use RandomSearchOptimizer as the underlying optimizer for simplicity
        # Could be made configurable to use any optimizer strategy
        optimizer = RandomSearchOptimizer()

        for stage_config in config["stages"]:
            # Check if we've exceeded max trials
            if cumulative_trials >= config["max_total_trials"]:
                _log.info(
                    "Stopping: max total trials reached",
                    extra={"cumulative_trials": cumulative_trials},
                )
                break

            # Adjust n_trials if we're near the limit
            remaining_trials = config["max_total_trials"] - cumulative_trials
            stage_trials = min(stage_config["n_trials"], remaining_trials)

            _log.info(
                "Starting fine-tuning stage",
                extra={
                    "stage": stage_config["stage_name"],
                    "n_trials": stage_trials,
                    "search_radius": stage_config["search_radius"],
                    "use_previous_best": stage_config["use_previous_best"],
                },
            )

            # Narrow search space if using previous best
            if stage_config["use_previous_best"] and best_value > float("-inf"):
                current_space = narrow_search_space(
                    search_space,
                    best_int_params,
                    best_float_params,
                    best_string_params,
                    stage_config["search_radius"],
                )

            # Create optimization config for this stage
            opt_config = OptimizationConfig(
                n_trials=stage_trials,
                timeout_seconds=None,
                n_startup_trials=min(5, stage_trials // 2),
                random_state=config["random_state"] + len(stage_results),
                direction="maximize",
                pruning_enabled=False,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            # Run optimization for this stage
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

            stage_result = StageResult(
                stage_name=stage_config["stage_name"],
                optimization_summary=summary,
                improvement_over_previous=improvement,
                cumulative_trials=cumulative_trials,
            )
            stage_results.append(stage_result)

            _log.info(
                "Fine-tuning stage complete",
                extra={
                    "stage": stage_config["stage_name"],
                    "best_value": summary["best_value"],
                    "improvement": improvement,
                    "cumulative_trials": cumulative_trials,
                },
            )

            # Check for early stopping
            if (
                len(stage_results) > 1
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
            "Staged fine-tuning complete",
            extra={
                "final_best_value": result["final_best_value"],
                "total_trials": result["total_trials"],
                "stages_completed": result["stages_completed"],
                "early_stopped": result["early_stopped"],
                "total_duration_sec": result["total_duration_seconds"],
            },
        )

        return result


def create_staged_finetuning() -> StagedFineTuning:
    """Factory function to create a StagedFineTuning instance.

    Returns:
        A new StagedFineTuning instance.
    """
    return StagedFineTuning()


__all__ = [
    "StagedFineTuning",
    "create_staged_finetuning",
]
