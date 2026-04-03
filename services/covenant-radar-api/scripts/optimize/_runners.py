"""Unified backend runner with progress tracking.

Wraps the core optimization runner with progress bar integration
and history tracking. Supports all 7 backends via a single function.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import time
from pathlib import Path

from covenant_ml.types import BackendName
from platform_core.logging import (
    RichProgressProtocol,
    create_rich_progress,
    get_rich_console,
)

from scripts._test_hooks import (
    LoadingProgressInfo,
    PhaseProgressInfo,
    TrialProgressInfo,
    UnifiedOptimizationResult,
    get_project_root,
)
from scripts.optimize._formatters import (
    format_elapsed,
    format_loading_progress,
    format_trial_progress,
)
from scripts.optimize.cli import DatasetName, FeaturePreset
from scripts.optimize.history import (
    OptimizationHistory,
    UnifiedHistoryEntry,
    result_to_entry,
)
from scripts.optimize.model_saver import SaveModelResult, save_best_model
from scripts.optimize.runner import (
    RunResult,
    run_backend,
)


def _run_backend_with_progress(
    backend: BackendName,
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    history: OptimizationHistory,
    progress: RichProgressProtocol | None = None,
) -> tuple[UnifiedOptimizationResult, float, UnifiedHistoryEntry]:
    """Run optimization for any backend with progress bar.

    Args:
        backend: Backend to use for optimization.
        dataset: Dataset to optimize on (taiwan, us, polish).
        n_trials: Number of Optuna trials to run.
        feature_preset: Feature engineering preset.
        device: Device for training (cuda/cpu/auto).
        timeout: Optional timeout in seconds, or None for no limit.
        history: History manager for saving results.
        progress: Optional existing progress bar.

    Returns:
        Tuple of (optimization result, elapsed time in seconds, history entry).
    """
    console = get_rich_console()
    start = time.perf_counter()

    if progress is None:
        with create_rich_progress(console) as new_progress:
            return _run_backend_with_progress(
                backend,
                dataset,
                n_trials,
                feature_preset,
                device,
                timeout,
                history,
                new_progress,
            )

    task_id = progress.add_task(
        f"[dim]0s[/dim] [yellow]Loading {dataset} dataset...[/yellow]",
        total=float(n_trials),
    )

    def phase_callback(info: PhaseProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        elapsed_str = f"[dim]{format_elapsed(elapsed)}[/dim]"
        phase = info["phase"]
        if phase == "loading_data":
            desc = f"{elapsed_str} [yellow]Loading {info['dataset']} dataset...[/yellow]"
        elif phase == "feature_engineering":
            n_samples = info["n_samples"]
            n_features = info["n_features"]
            desc = (
                f"{elapsed_str} [yellow]Loaded {n_samples:,} samples, "
                f"{n_features} features. Applying feature engineering...[/yellow]"
            )
        elif phase == "optimizing":
            n_features = info["n_features"]
            desc = (
                f"{elapsed_str} [green]Ready ({n_features} features). "
                f"Starting optimization...[/green]"
            )
        else:
            desc = f"{elapsed_str} [cyan]Saving results...[/cyan]"
        progress.update(task_id, description=desc)

    def progress_callback(info: TrialProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        progress.update(task_id, description=format_trial_progress(info, elapsed))
        progress.advance(task_id)

    def loading_progress_callback(info: LoadingProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        desc = format_loading_progress(
            info["dataset"],
            info["phase"],
            info["percent_complete"],
            info["rows_processed"],
            info["rows_total"],
            elapsed,
        )
        progress.update(task_id, description=desc)

    result = run_backend(
        backend,
        dataset,
        n_trials,
        feature_preset,
        device,
        timeout,
        progress_callback,
        phase_callback,
        loading_progress_callback,
    )
    elapsed = time.perf_counter() - start

    entry = result_to_entry(result, elapsed)
    history.append(entry)
    return result, elapsed, entry


def run_single_with_progress(
    backend: BackendName,
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    save_model: bool = True,
    project_root: Path | None = None,
    progress: RichProgressProtocol | None = None,
) -> RunResult:
    """Run single optimization with progress bar and history tracking.

    Args:
        backend: Backend to use (any of 7 registered backends).
        dataset: Dataset to optimize on (taiwan, us, polish).
        n_trials: Number of Optuna trials to run.
        feature_preset: Feature engineering preset.
        device: Device for training (cuda/cpu/auto).
        timeout: Optional timeout in seconds, or None for no limit.
        save_model: If True, train and save the best model after optimization.
            Only saves if the new model is better than any existing saved model.
        project_root: Project root directory. If None, uses default.
        progress: Optional existing progress bar.

    Returns:
        RunResult with optimization result, timing, and history context.
    """
    if project_root is None:
        project_root = get_project_root()
    output_dir = project_root / "models"

    # Load history for comparison BEFORE running
    history = OptimizationHistory.for_output_dir(output_dir)
    history.load()
    previous_best = history.get_previous_best(backend, dataset, feature_preset)
    all_time_best = history.get_all_time_best(backend, dataset, feature_preset)

    # Run unified optimization
    result, elapsed, _ = _run_backend_with_progress(
        backend, dataset, n_trials, feature_preset, device, timeout, history, progress
    )

    # Determine if new best
    current_auc = result["best_value"]
    is_new_best = all_time_best is None or current_auc > all_time_best["best_val_auc"]

    # Save best model if requested
    if save_model:
        _save_result: SaveModelResult = save_best_model(
            result=result,
            dataset=dataset,
            feature_preset=feature_preset,
            project_root=project_root,
        )

    return RunResult(
        backend=backend,
        result=result,
        elapsed=elapsed,
        previous_best=previous_best,
        all_time_best=all_time_best,
        is_new_best=is_new_best,
    )


__all__ = [
    "_run_backend_with_progress",
    "run_single_with_progress",
]
