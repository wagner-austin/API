"""Core optimization runner with unified backend routing.

Routes optimization requests to the unified runner for any backend.
Supports all 7 backends via a single dispatch path.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from typing import TypedDict

from covenant_ml.types import BackendName
from platform_core.json_utils import JSONValue, dump_json_str

import scripts._test_hooks as _hooks
from scripts._test_hooks import (
    LoadingProgressCallbackProtocol,
    PhaseProgressCallbackProtocol,
    TrialProgressCallbackProtocol,
    UnifiedOptimizationResult,
    get_project_root,
)
from scripts.optimize.cli import DatasetName, FeaturePreset
from scripts.optimize.history import UnifiedHistoryEntry


class RunResult(TypedDict):
    """Result of an optimization run with history context.

    Bundles the optimization result with timing and historical comparison
    data for display purposes.
    """

    backend: BackendName
    result: UnifiedOptimizationResult
    elapsed: float
    previous_best: UnifiedHistoryEntry | None
    all_time_best: UnifiedHistoryEntry | None
    is_new_best: bool


def _build_config(
    backend: str,
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
) -> str:
    """Build optimization config JSON for any backend.

    The unified optimize job handles all backend-specific defaults internally.

    Args:
        backend: Backend name (xgboost, mlp, lstm, lightgbm, cleargbm, logreg, random_forest).
        dataset: Dataset to optimize on (taiwan, us, polish).
        n_trials: Number of Optuna trials to run.
        feature_preset: Feature engineering preset.
        device: Device for training (cuda/cpu/auto).
        timeout: Optional timeout in seconds, or None for no limit.

    Returns:
        JSON configuration string for the optimizer.
    """
    config_dict: dict[str, JSONValue] = {
        "backend": backend,
        "dataset": dataset,
        "n_trials": n_trials,
        "feature_preset": feature_preset,
        "device": device,
        "random_state": 42,
    }
    if timeout is not None:
        config_dict["timeout_seconds"] = timeout
    return dump_json_str(config_dict)


def run_backend(
    backend: BackendName,
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    progress_callback: TrialProgressCallbackProtocol | None = None,
    phase_callback: PhaseProgressCallbackProtocol | None = None,
    loading_progress_callback: LoadingProgressCallbackProtocol | None = None,
) -> UnifiedOptimizationResult:
    """Run hyperparameter optimization for any backend.

    Args:
        backend: Backend name.
        dataset: Dataset to optimize on (taiwan, us, polish).
        n_trials: Number of Optuna trials to run.
        feature_preset: Feature engineering preset.
        device: Device for training (cuda/cpu/auto).
        timeout: Optional timeout in seconds, or None for no limit.
        progress_callback: Optional callback for trial progress updates.
        phase_callback: Optional callback for phase transitions.
        loading_progress_callback: Optional callback for loading progress.

    Returns:
        UnifiedOptimizationResult with best hyperparameters.
    """
    project_root = get_project_root()
    external_dir = project_root / "data" / "external"
    output_dir = project_root / "models" / backend
    output_dir.mkdir(parents=True, exist_ok=True)

    config_json = _build_config(backend, dataset, n_trials, feature_preset, device, timeout)
    return _hooks.optimization_runner(
        config_json,
        external_dir,
        output_dir,
        progress_callback,
        phase_callback,
        loading_progress_callback,
    )


__all__ = [
    "RunResult",
    "UnifiedOptimizationResult",
    "get_project_root",
    "run_backend",
]
