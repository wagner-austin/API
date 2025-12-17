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
    LightGBMHistoryEntry,
    LSTMHistoryEntry,
    MLPHistoryEntry,
    OptimizationHistory,
    UnifiedHistoryEntry,
    XGBoostHistoryEntry,
    lightgbm_result_to_entry,
    lstm_result_to_entry,
    mlp_result_to_entry,
    xgboost_result_to_entry,
)
from scripts.optimize.logging_config import (
    set_verbose_mode,
    suppress_verbose_logging,
)
from scripts.optimize.main import main
from scripts.optimize.modes import (
    compare_presets,
    run_all_datasets,
    run_single_with_progress,
)
from scripts.optimize.runner import (
    LightGBMOptimizationResult,
    LSTMOptimizationResult,
    MLPOptimizationResult,
    RunResult,
    UnifiedOptimizationResult,
    XGBoostOptimizationResult,
    get_project_root,
    run_lightgbm,
    run_lstm,
    run_mlp,
    run_xgboost,
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
    "LSTMHistoryEntry",
    "LSTMOptimizationResult",
    "LightGBMHistoryEntry",
    "LightGBMOptimizationResult",
    "MLPHistoryEntry",
    "MLPOptimizationResult",
    "OptimizationHistory",
    "OptimizationState",
    "OptimizeArgs",
    "RunResult",
    "UnifiedHistoryEntry",
    "UnifiedOptimizationResult",
    "XGBoostHistoryEntry",
    "XGBoostOptimizationResult",
    "compare_presets",
    "get_project_root",
    "get_state",
    "is_interrupted",
    "lightgbm_result_to_entry",
    "lstm_result_to_entry",
    "main",
    "managed_execution",
    "mlp_result_to_entry",
    "parse_args",
    "run_all_datasets",
    "run_lightgbm",
    "run_lstm",
    "run_mlp",
    "run_single_with_progress",
    "run_xgboost",
    "set_verbose_mode",
    "suppress_verbose_logging",
    "xgboost_result_to_entry",
]
