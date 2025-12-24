"""Type definitions for fine-tuning.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
Defines configuration types for fine-tuning workflows.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from ..optimizer.types import (
    OptimizationSummary,
    SampledFloatParams,
    SampledIntParams,
    SampledStringParams,
)

# =============================================================================
# Fine-Tuning Stage Types
# =============================================================================

FineTuningStage = Literal["exploration", "refinement", "final"]


class StageConfig(TypedDict, total=True):
    """Configuration for a single fine-tuning stage.

    Attributes:
        stage_name: Identifier for this stage.
        n_trials: Number of trials in this stage.
        search_radius: How much to narrow search space (0.5 = half width).
        use_previous_best: Whether to center search on previous best params.
    """

    stage_name: FineTuningStage
    n_trials: int
    search_radius: float
    use_previous_best: bool


class FineTuningConfig(TypedDict, total=True):
    """Configuration for multi-stage fine-tuning.

    Attributes:
        stages: Tuple of stage configurations to execute in order.
        random_state: Base random seed for reproducibility.
        early_stop_threshold: Stop if improvement less than this (0 to disable).
        max_total_trials: Maximum trials across all stages.
    """

    stages: tuple[StageConfig, ...]
    random_state: int
    early_stop_threshold: float
    max_total_trials: int


# =============================================================================
# Fine-Tuning Result Types
# =============================================================================


class StageResult(TypedDict, total=True):
    """Result from a single fine-tuning stage.

    Attributes:
        stage_name: Which stage produced this result.
        optimization_summary: Full optimization summary from this stage.
        improvement_over_previous: Relative improvement from previous stage.
        cumulative_trials: Total trials used including this stage.
    """

    stage_name: FineTuningStage
    optimization_summary: OptimizationSummary
    improvement_over_previous: float
    cumulative_trials: int


class FineTuningResult(TypedDict, total=True):
    """Complete result from multi-stage fine-tuning.

    Attributes:
        stage_results: Results from each stage executed.
        final_best_value: Best objective value achieved.
        final_int_params: Best integer parameters found.
        final_float_params: Best float parameters found.
        final_string_params: Best string parameters found.
        total_trials: Total number of trials executed.
        total_duration_seconds: Total time for all stages.
        stages_completed: Number of stages that ran (may be < total if early stopped).
        early_stopped: Whether fine-tuning stopped early due to insufficient improvement.
    """

    stage_results: tuple[StageResult, ...]
    final_best_value: float
    final_int_params: SampledIntParams
    final_float_params: SampledFloatParams
    final_string_params: SampledStringParams
    total_trials: int
    total_duration_seconds: float
    stages_completed: int
    early_stopped: bool


# =============================================================================
# Warm Start Configuration
# =============================================================================


class WarmStartConfig(TypedDict, total=True):
    """Configuration for warm-starting optimization from previous results.

    Attributes:
        prior_summary: Previous optimization summary to warm-start from.
        narrow_factor: Factor to narrow search space (0.5 = half the range).
        inherit_string_params: Whether to fix string params from prior.
    """

    prior_summary: OptimizationSummary
    narrow_factor: float
    inherit_string_params: bool


# =============================================================================
# Factory Functions for Default Configs
# =============================================================================


def make_default_stage_config(
    stage_name: FineTuningStage,
    n_trials: int = 50,
    search_radius: float = 0.5,
) -> StageConfig:
    """Create a default stage configuration.

    Args:
        stage_name: Name of the stage.
        n_trials: Number of trials for this stage.
        search_radius: How much to narrow search (0.5 = half width).

    Returns:
        A configured StageConfig.
    """
    return StageConfig(
        stage_name=stage_name,
        n_trials=n_trials,
        search_radius=search_radius,
        use_previous_best=stage_name != "exploration",
    )


def make_default_finetuning_config(
    random_state: int = 42,
    exploration_trials: int = 50,
    refinement_trials: int = 30,
    final_trials: int = 20,
) -> FineTuningConfig:
    """Create a default three-stage fine-tuning configuration.

    The default stages are:
    1. Exploration: Broad search with full parameter ranges
    2. Refinement: Narrowed search centered on best from exploration
    3. Final: Tight search for precise optimization

    Args:
        random_state: Random seed for reproducibility.
        exploration_trials: Trials for exploration stage.
        refinement_trials: Trials for refinement stage.
        final_trials: Trials for final stage.

    Returns:
        A configured FineTuningConfig.
    """
    return FineTuningConfig(
        stages=(
            StageConfig(
                stage_name="exploration",
                n_trials=exploration_trials,
                search_radius=1.0,
                use_previous_best=False,
            ),
            StageConfig(
                stage_name="refinement",
                n_trials=refinement_trials,
                search_radius=0.5,
                use_previous_best=True,
            ),
            StageConfig(
                stage_name="final",
                n_trials=final_trials,
                search_radius=0.25,
                use_previous_best=True,
            ),
        ),
        random_state=random_state,
        early_stop_threshold=0.001,
        max_total_trials=exploration_trials + refinement_trials + final_trials,
    )


def make_warm_start_config(
    prior_summary: OptimizationSummary,
    narrow_factor: float = 0.5,
) -> WarmStartConfig:
    """Create a warm-start configuration from a previous optimization.

    Args:
        prior_summary: The optimization summary to warm-start from.
        narrow_factor: How much to narrow the search space.

    Returns:
        A configured WarmStartConfig.
    """
    return WarmStartConfig(
        prior_summary=prior_summary,
        narrow_factor=narrow_factor,
        inherit_string_params=True,
    )


__all__ = [
    "FineTuningConfig",
    "FineTuningResult",
    "FineTuningStage",
    "StageConfig",
    "StageResult",
    "WarmStartConfig",
    "make_default_finetuning_config",
    "make_default_stage_config",
    "make_warm_start_config",
]
