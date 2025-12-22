"""Backend-specific runners with progress tracking.

Contains functions that wrap the core optimization runners with
progress bar integration and history tracking.

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

import scripts._test_hooks as _hooks
from scripts.optimize._formatters import (
    format_elapsed,
    format_lightgbm_progress,
    format_loading_progress,
    format_lstm_progress,
    format_mlp_progress,
    format_xgboost_progress,
)
from scripts.optimize.cli import DatasetName, FeaturePreset
from scripts.optimize.history import (
    OptimizationHistory,
    UnifiedHistoryEntry,
    lightgbm_result_to_entry,
    lstm_result_to_entry,
    mlp_result_to_entry,
    xgboost_result_to_entry,
)
from scripts.optimize.model_saver import SaveModelResult, save_best_model
from scripts.optimize.runner import (
    RunResult,
    UnifiedOptimizationResult,
    get_project_root,
    run_lightgbm,
    run_lstm,
    run_mlp,
    run_xgboost,
)


def _run_xgboost_with_progress(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    history: OptimizationHistory,
    progress: RichProgressProtocol | None = None,
) -> tuple[UnifiedOptimizationResult, float, UnifiedHistoryEntry]:
    """Run XGBoost optimization with progress bar.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        history (OptimizationHistory): History manager for saving results.
        progress (RichProgressProtocol | None): Optional existing progress bar.

    Returns:
        tuple[UnifiedOptimizationResult, float, UnifiedHistoryEntry]: Tuple of
            (optimization result, elapsed time in seconds, history entry).
    """
    console = get_rich_console()
    start = time.perf_counter()

    if progress is None:
        with create_rich_progress(console) as new_progress:
            return _run_xgboost_with_progress(
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

    def phase_callback(info: _hooks.XGBoostPhaseInfo) -> None:
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

    def progress_callback(info: _hooks.XGBoostProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        progress.update(task_id, description=format_xgboost_progress(info, elapsed))
        progress.advance(task_id)

    def loading_progress_callback(info: _hooks.XGBoostLoadingProgressInfo) -> None:
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

    result = run_xgboost(
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

    entry = xgboost_result_to_entry(result, elapsed)
    history.append(entry)
    return result, elapsed, entry


def _run_mlp_with_progress(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    history: OptimizationHistory,
    progress: RichProgressProtocol | None = None,
) -> tuple[UnifiedOptimizationResult, float, UnifiedHistoryEntry]:
    """Run MLP optimization with progress bar.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        history (OptimizationHistory): History manager for saving results.
        progress (RichProgressProtocol | None): Optional existing progress bar.

    Returns:
        tuple[UnifiedOptimizationResult, float, UnifiedHistoryEntry]: Tuple of
            (optimization result, elapsed time in seconds, history entry).
    """
    console = get_rich_console()
    start = time.perf_counter()

    if progress is None:
        with create_rich_progress(console) as new_progress:
            return _run_mlp_with_progress(
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

    def phase_callback(info: _hooks.MLPPhaseInfo) -> None:
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

    def progress_callback(info: _hooks.MLPTrialProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        progress.update(task_id, description=format_mlp_progress(info, elapsed))
        progress.advance(task_id)

    def loading_progress_callback(info: _hooks.MLPLoadingProgressInfo) -> None:
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

    result = run_mlp(
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

    entry = mlp_result_to_entry(result, elapsed)
    history.append(entry)
    return result, elapsed, entry


def _run_lightgbm_with_progress(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    history: OptimizationHistory,
    progress: RichProgressProtocol | None = None,
) -> tuple[UnifiedOptimizationResult, float, UnifiedHistoryEntry]:
    """Run LightGBM optimization with progress bar.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        history (OptimizationHistory): History manager for saving results.
        progress (RichProgressProtocol | None): Optional existing progress bar.

    Returns:
        tuple[UnifiedOptimizationResult, float, UnifiedHistoryEntry]: Tuple of
            (optimization result, elapsed time in seconds, history entry).
    """
    console = get_rich_console()
    start = time.perf_counter()

    if progress is None:
        with create_rich_progress(console) as new_progress:
            return _run_lightgbm_with_progress(
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

    def phase_callback(info: _hooks.LightGBMPhaseInfo) -> None:
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

    def progress_callback(info: _hooks.LightGBMTrialProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        progress.update(task_id, description=format_lightgbm_progress(info, elapsed))
        progress.advance(task_id)

    def loading_progress_callback(info: _hooks.LightGBMLoadingProgressInfo) -> None:
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

    result = run_lightgbm(
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

    entry = lightgbm_result_to_entry(result, elapsed)
    history.append(entry)
    return result, elapsed, entry


def _run_lstm_with_progress(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    history: OptimizationHistory,
    progress: RichProgressProtocol | None = None,
) -> tuple[UnifiedOptimizationResult, float, UnifiedHistoryEntry]:
    """Run LSTM optimization with progress bar.

    Args:
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        history (OptimizationHistory): History manager for saving results.
        progress (RichProgressProtocol | None): Optional existing progress bar.

    Returns:
        tuple[UnifiedOptimizationResult, float, UnifiedHistoryEntry]: Tuple of
            (optimization result, elapsed time in seconds, history entry).
    """
    console = get_rich_console()
    start = time.perf_counter()

    if progress is None:
        with create_rich_progress(console) as new_progress:
            return _run_lstm_with_progress(
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

    def phase_callback(info: _hooks.LSTMPhaseInfo) -> None:
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

    def progress_callback(info: _hooks.LSTMTrialProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        progress.update(task_id, description=format_lstm_progress(info, elapsed))
        progress.advance(task_id)

    def loading_progress_callback(info: _hooks.LSTMLoadingProgressInfo) -> None:
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

    result = run_lstm(
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

    entry = lstm_result_to_entry(result, elapsed)
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
        backend (BackendName): Backend to use (xgboost, mlp, lightgbm, lstm).
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds, or None for no limit.
        save_model (bool): If True, train and save the best model after optimization.
            Only saves if the new model is better than any existing saved model.
        project_root (Path | None): Project root directory. If None, uses default.
        progress (RichProgressProtocol | None): Optional existing progress bar.

    Returns:
        RunResult: Result dict with optimization result, timing, and history context.
    """
    if project_root is None:
        project_root = get_project_root()
    output_dir = project_root / "models"

    # Load history for comparison BEFORE running
    history = OptimizationHistory.for_output_dir(output_dir)
    history.load()
    previous_best = history.get_previous_best(backend, dataset, feature_preset)
    all_time_best = history.get_all_time_best(backend, dataset, feature_preset)

    # Run backend-specific optimization
    if backend == "xgboost":
        result, elapsed, _ = _run_xgboost_with_progress(
            dataset, n_trials, feature_preset, device, timeout, history, progress
        )
    elif backend == "mlp":
        result, elapsed, _ = _run_mlp_with_progress(
            dataset, n_trials, feature_preset, device, timeout, history, progress
        )
    elif backend == "lightgbm":
        result, elapsed, _ = _run_lightgbm_with_progress(
            dataset, n_trials, feature_preset, device, timeout, history, progress
        )
    else:
        # backend must be "lstm" here - mypy validates exhaustiveness
        result, elapsed, _ = _run_lstm_with_progress(
            dataset, n_trials, feature_preset, device, timeout, history, progress
        )

    # Determine if new best
    current_auc = result["best_val_auc"]
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
    "_run_lightgbm_with_progress",
    "_run_lstm_with_progress",
    "_run_mlp_with_progress",
    "_run_xgboost_with_progress",
    "run_single_with_progress",
]
