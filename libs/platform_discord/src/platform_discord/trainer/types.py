"""Trainer types for Discord embed generation.

Re-exports platform_core trainer metrics event types for DRY.
"""

from __future__ import annotations

from platform_core.trainer_metrics_events import (
    TrainerCompletedMetricsV1 as FinalMetrics,
)
from platform_core.trainer_metrics_events import (
    TrainerConfigV1 as TrainingConfig,
)
from platform_core.trainer_metrics_events import (
    TrainerProgressMetricsV1 as Progress,
)

__all__ = ["FinalMetrics", "Progress", "TrainingConfig"]
