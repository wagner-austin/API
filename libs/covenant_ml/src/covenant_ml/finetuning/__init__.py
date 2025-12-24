"""Fine-tuning module for staged hyperparameter optimization.

Provides pluggable fine-tuning strategies for multi-stage optimization,
warm-starting from prior results, and iterative refinement until convergence.

Key components:
- FineTuningStrategyProtocol: Protocol for fine-tuning strategies
- FineTuningRegistry: Registry for fine-tuning strategy implementations
- StagedFineTuning: Multi-stage optimization with narrowing search spaces
- WarmStartFineTuning: Single-stage optimization from prior results
- IterativeRefinementFineTuning: Repeated refinement until convergence

Usage:
    from covenant_ml.finetuning import (
        default_finetuning_registry,
        make_default_finetuning_config,
        make_warm_start_config,
    )

    # Get fine-tuning strategy from registry
    registry = default_finetuning_registry()
    strategy = registry.get("staged")

    # Create configuration
    config = make_default_finetuning_config(
        exploration_trials=50,
        refinement_trials=30,
        final_trials=20,
    )

    # Run fine-tuning
    result = strategy.fine_tune(
        x_features=X,
        y_labels=y,
        feature_names=names,
        search_space=space,
        config=config,
        objective=my_objective_function,
    )

    # Or warm-start from previous optimization
    warm_start = make_warm_start_config(prior_summary)
    result = strategy.fine_tune(..., warm_start=warm_start)
"""

from .protocol import (
    FineTuningCapabilities,
    FineTuningStrategyFactory,
    FineTuningStrategyName,
    FineTuningStrategyProtocol,
)
from .registry import (
    FineTuningRegistration,
    FineTuningRegistry,
    default_finetuning_registry,
)
from .space_narrowing import (
    narrow_lightgbm_space,
    narrow_lstm_space,
    narrow_mlp_space,
    narrow_search_space,
    narrow_xgboost_space,
)
from .strategies import (
    IterativeRefinementFineTuning,
    StagedFineTuning,
    WarmStartFineTuning,
    create_iterative_refinement_finetuning,
    create_staged_finetuning,
    create_warm_start_finetuning,
)
from .types import (
    FineTuningConfig,
    FineTuningResult,
    FineTuningStage,
    StageConfig,
    StageResult,
    WarmStartConfig,
    make_default_finetuning_config,
    make_default_stage_config,
    make_warm_start_config,
)

__all__ = [
    "FineTuningCapabilities",
    "FineTuningConfig",
    "FineTuningRegistration",
    "FineTuningRegistry",
    "FineTuningResult",
    "FineTuningStage",
    "FineTuningStrategyFactory",
    "FineTuningStrategyName",
    "FineTuningStrategyProtocol",
    "IterativeRefinementFineTuning",
    "StageConfig",
    "StageResult",
    "StagedFineTuning",
    "WarmStartConfig",
    "WarmStartFineTuning",
    "create_iterative_refinement_finetuning",
    "create_staged_finetuning",
    "create_warm_start_finetuning",
    "default_finetuning_registry",
    "make_default_finetuning_config",
    "make_default_stage_config",
    "make_warm_start_config",
    "narrow_lightgbm_space",
    "narrow_lstm_space",
    "narrow_mlp_space",
    "narrow_search_space",
    "narrow_xgboost_space",
]
