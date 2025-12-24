"""Fine-tuning strategy implementations.

Provides pluggable fine-tuning implementations that satisfy FineTuningStrategyProtocol.
Each strategy can be registered in FineTuningRegistry and used interchangeably.

Strategies:
- StagedFineTuning: Multi-stage optimization with narrowing search spaces
- WarmStartFineTuning: Single-stage optimization from prior results
- IterativeRefinementFineTuning: Repeated refinement until convergence
"""

from .iterative import (
    IterativeRefinementFineTuning,
    create_iterative_refinement_finetuning,
)
from .staged import (
    StagedFineTuning,
    create_staged_finetuning,
)
from .warm_start import (
    WarmStartFineTuning,
    create_warm_start_finetuning,
)

__all__ = [
    "IterativeRefinementFineTuning",
    "StagedFineTuning",
    "WarmStartFineTuning",
    "create_iterative_refinement_finetuning",
    "create_staged_finetuning",
    "create_warm_start_finetuning",
]
