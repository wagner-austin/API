"""Warm-start fine-tuning strategy.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Implements single-stage optimization from prior results.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from platform_core.logging import get_logger

from ...optimizer.protocol import ObjectiveProtocol, TrialCallbackProtocol
from ...optimizer.strategies.random_search import RandomSearchOptimizer
from ...optimizer.types import OptimizationConfig, SearchSpace
from ..protocol import FineTuningCapabilities, FineTuningStrategyName
from ..space_narrowing import narrow_search_space
from ..types import (
    FineTuningConfig,
    FineTuningResult,
    StageResult,
    WarmStartConfig,
)

_log = get_logger(__name__)


class WarmStartFineTuning:
    """Single-stage fine-tuning initialized from prior optimization results.

    Takes a previous optimization result and runs a single optimization
    stage with a narrowed search space centered on the best parameters.
    Simpler than staged fine-tuning when you just want to refine existing
    results without the multi-stage complexity.
    """

    def __init__(self) -> None:
        """Initialize warm-start fine-tuning strategy."""
        pass

    def strategy_name(self) -> FineTuningStrategyName:
        """Return the strategy name.

        Returns:
            The literal string 'warm_start'.
        """
        return "warm_start"

    def capabilities(self) -> FineTuningCapabilities:
        """Return the capabilities of this strategy.

        Returns:
            Capabilities indicating warm-start supports prior params.
        """
        return FineTuningCapabilities(
            supports_warm_start=True,
            supports_staged=False,
            supports_early_stop=False,
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
        """Execute single-stage warm-start fine-tuning.

        If warm_start is provided, narrows the search space around prior best.
        Otherwise, runs a single stage with the first stage config.

        Args:
            x_features: Feature matrix (n_samples, n_features).
            y_labels: Binary labels (n_samples,).
            feature_names: Names for each feature column.
            search_space: Base parameter ranges to search.
            config: Fine-tuning configuration (uses first stage).
            objective: Function to evaluate hyperparameters.
            warm_start: Required for warm-start - prior optimization results.
            trial_callback: Optional callback after each trial.

        Returns:
            Fine-tuning result with best parameters.
        """
        start_time = time.perf_counter()

        # Use first stage config
        if len(config["stages"]) == 0:
            raise ValueError("FineTuningConfig must have at least one stage")
        stage_config = config["stages"][0]
        n_trials = min(stage_config["n_trials"], config["max_total_trials"])

        _log.info(
            "Starting warm-start fine-tuning",
            extra={
                "n_trials": n_trials,
                "has_warm_start": warm_start is not None,
            },
        )

        # Apply warm start if provided
        if warm_start is not None:
            current_space = narrow_search_space(
                search_space,
                warm_start["prior_summary"]["best_int_params"],
                warm_start["prior_summary"]["best_float_params"],
                warm_start["prior_summary"]["best_string_params"],
                warm_start["narrow_factor"],
            )
            prior_best = warm_start["prior_summary"]["best_value"]
        else:
            current_space = search_space
            prior_best = float("-inf")

        # Create optimization config
        opt_config = OptimizationConfig(
            n_trials=n_trials,
            timeout_seconds=None,
            n_startup_trials=min(5, n_trials // 2),
            random_state=config["random_state"],
            direction="maximize",
            pruning_enabled=False,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        # Run optimization
        optimizer = RandomSearchOptimizer()
        summary = optimizer.optimize(
            x_features=x_features,
            y_labels=y_labels,
            feature_names=feature_names,
            search_space=current_space,
            config=opt_config,
            objective=objective,
            trial_callback=trial_callback,
        )

        # Calculate improvement
        improvement = 0.0
        if prior_best > float("-inf") and summary["best_value"] > prior_best:
            improvement = (summary["best_value"] - prior_best) / abs(prior_best)

        stage_result = StageResult(
            stage_name="refinement",
            optimization_summary=summary,
            improvement_over_previous=improvement,
            cumulative_trials=summary["n_trials_complete"],
        )

        total_duration = time.perf_counter() - start_time

        result = FineTuningResult(
            stage_results=(stage_result,),
            final_best_value=summary["best_value"],
            final_int_params=summary["best_int_params"],
            final_float_params=summary["best_float_params"],
            final_string_params=summary["best_string_params"],
            total_trials=summary["n_trials_complete"],
            total_duration_seconds=total_duration,
            stages_completed=1,
            early_stopped=False,
        )

        _log.info(
            "Warm-start fine-tuning complete",
            extra={
                "final_best_value": result["final_best_value"],
                "improvement": improvement,
                "total_trials": result["total_trials"],
                "total_duration_sec": result["total_duration_seconds"],
            },
        )

        return result


def create_warm_start_finetuning() -> WarmStartFineTuning:
    """Factory function to create a WarmStartFineTuning instance.

    Returns:
        A new WarmStartFineTuning instance.
    """
    return WarmStartFineTuning()


__all__ = [
    "WarmStartFineTuning",
    "create_warm_start_finetuning",
]
