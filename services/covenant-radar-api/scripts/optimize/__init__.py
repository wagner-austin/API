"""Hyperparameter optimization CLI package.

Modular structure:
- cli: Argument parsing and types
- display: Rich console output formatting
- history: Run history tracking and comparison
- logging_config: Logging setup and suppression
- main: Main entry point function
- modes: Run modes (single, compare, all-datasets)
- runner: Core optimization execution
- state: Lifecycle state management
"""

from __future__ import annotations

from scripts.optimize.cli import (
    PRESET_DESCRIPTIONS,
    DatasetName,
    FeaturePreset,
    OptimizeArgs,
    parse_args,
)
from scripts.optimize.history import (
    HISTORY_FILENAME,
    OptimizationHistory,
    UnifiedHistoryEntry,
    result_to_entry,
)
from scripts.optimize.logging_config import (
    set_verbose_mode,
    suppress_verbose_logging,
)
from scripts.optimize.main import main
from scripts.optimize.modes import (
    compare_presets,
    run_all_datasets,
)
from scripts.optimize.runner import (
    RunResult,
    get_project_root,
    run_backend,
)
from scripts.optimize.state import (
    OptimizationState,
    get_state,
    is_interrupted,
    managed_execution,
)

__all__ = [
    "HISTORY_FILENAME",
    "PRESET_DESCRIPTIONS",
    "DatasetName",
    "FeaturePreset",
    "OptimizationHistory",
    "OptimizationState",
    "OptimizeArgs",
    "RunResult",
    "UnifiedHistoryEntry",
    "compare_presets",
    "get_project_root",
    "get_state",
    "is_interrupted",
    "main",
    "managed_execution",
    "parse_args",
    "result_to_entry",
    "run_all_datasets",
    "run_backend",
    "set_verbose_mode",
    "suppress_verbose_logging",
]
