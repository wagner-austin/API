"""Run modes for optimization: single, preset comparison, and multi-dataset.

Supports all backends (XGBoost, MLP, LightGBM, LSTM) with backend-specific
progress display and history tracking.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import time
from pathlib import Path

from covenant_ml.types import BackendName
from platform_core.logging import (
    RichProgressProtocol,
    create_rich_panel,
    create_rich_progress,
    create_rich_table,
    get_rich_console,
)

import scripts._test_hooks as _hooks
from scripts.optimize.cli import PRESET_DESCRIPTIONS, DatasetName, FeaturePreset
from scripts.optimize.display import print_result
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

# =============================================================================
# Progress Display Formatters
# =============================================================================


def _format_elapsed(seconds: float) -> str:
    """Format elapsed time for display.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted time string (e.g., "1m 23s" or "45s").
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def _format_xgboost_progress(info: _hooks.XGBoostProgressInfo, elapsed: float = 0.0) -> str:
    """Format XGBoost trial progress for display.

    Args:
        info (XGBoostProgressInfo): XGBoost trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        str: Rich-formatted progress string with trial number, AUC, and hyperparameters.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{_format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker} "
        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
        f"Est: [blue]{info['best_n_estimators']}[/blue]"
    )


def _format_mlp_progress(info: _hooks.MLPTrialProgressInfo, elapsed: float = 0.0) -> str:
    """Format MLP trial progress for display.

    Args:
        info (MLPTrialProgressInfo): MLP trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        str: Rich-formatted progress string with trial number, AUC, and hyperparameters.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{_format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker} "
        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
        f"Layers: [magenta]{info['best_n_layers']}[/magenta] "
        f"Hidden: [blue]{info['best_hidden_size']}[/blue]"
    )


def _format_lightgbm_progress(info: _hooks.LightGBMTrialProgressInfo, elapsed: float = 0.0) -> str:
    """Format LightGBM trial progress for display.

    Args:
        info (LightGBMTrialProgressInfo): LightGBM trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        str: Rich-formatted progress string with trial number, AUC, and hyperparameters.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{_format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker} "
        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
        f"Leaves: [magenta]{info['best_num_leaves']}[/magenta] "
        f"Est: [blue]{info['best_n_estimators']}[/blue]"
    )


def _format_lstm_progress(info: _hooks.LSTMTrialProgressInfo, elapsed: float = 0.0) -> str:
    """Format LSTM trial progress for display.

    Args:
        info (LSTMTrialProgressInfo): LSTM trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        str: Rich-formatted progress string with trial number, AUC, and hyperparameters.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{_format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker} "
        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
        f"Layers: [magenta]{info['best_num_layers']}[/magenta] "
        f"Hidden: [blue]{info['best_hidden_size']}[/blue]"
    )


# =============================================================================
# Loading Progress Formatters
# =============================================================================


def _format_loading_progress(
    dataset: str,
    phase: str,
    percent_complete: float,
    rows_processed: int,
    rows_total: int,
    elapsed: float = 0.0,
) -> str:
    """Format loading progress for display.

    Args:
        dataset: Dataset name being loaded.
        phase: Current loading phase (reading, parsing, encoding).
        percent_complete: Percent complete (0-100).
        rows_processed: Number of rows processed so far.
        rows_total: Total rows to process.
        elapsed: Elapsed time in seconds.

    Returns:
        Rich-formatted progress string with loading details.
    """
    elapsed_str = f"[dim]{_format_elapsed(elapsed)}[/dim]"
    phase_color = {"reading": "cyan", "parsing": "yellow", "encoding": "green"}.get(phase, "white")

    if rows_total > 0:
        return (
            f"{elapsed_str} [{phase_color}]Loading {dataset}[/{phase_color}] "
            f"[dim]({phase})[/dim] "
            f"[bold cyan]{rows_processed:,}[/bold cyan]/[dim]{rows_total:,}[/dim] rows "
            f"[magenta]{percent_complete:.1f}%[/magenta]"
        )
    return (
        f"{elapsed_str} [{phase_color}]Loading {dataset}[/{phase_color}] "
        f"[dim]({phase})[/dim] "
        f"[magenta]{percent_complete:.1f}%[/magenta]"
    )


# =============================================================================
# Single Run with Progress
# =============================================================================


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
        elapsed_str = f"[dim]{_format_elapsed(elapsed)}[/dim]"
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
        progress.update(task_id, description=_format_xgboost_progress(info, elapsed))
        progress.advance(task_id)

    def loading_progress_callback(info: _hooks.XGBoostLoadingProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        desc = _format_loading_progress(
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
        elapsed_str = f"[dim]{_format_elapsed(elapsed)}[/dim]"
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
        progress.update(task_id, description=_format_mlp_progress(info, elapsed))
        progress.advance(task_id)

    def loading_progress_callback(info: _hooks.MLPLoadingProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        desc = _format_loading_progress(
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
        elapsed_str = f"[dim]{_format_elapsed(elapsed)}[/dim]"
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
        progress.update(task_id, description=_format_lightgbm_progress(info, elapsed))
        progress.advance(task_id)

    def loading_progress_callback(info: _hooks.LightGBMLoadingProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        desc = _format_loading_progress(
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
        elapsed_str = f"[dim]{_format_elapsed(elapsed)}[/dim]"
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
        progress.update(task_id, description=_format_lstm_progress(info, elapsed))
        progress.advance(task_id)

    def loading_progress_callback(info: _hooks.LSTMLoadingProgressInfo) -> None:
        elapsed = time.perf_counter() - start
        desc = _format_loading_progress(
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


# =============================================================================
# Comparison Modes
# =============================================================================


def compare_presets(
    backend: BackendName,
    dataset: DatasetName,
    n_trials: int,
    device: str,
    timeout: int | None,
    save_model: bool = True,
    project_root: Path | None = None,
) -> None:
    """Run all presets and compare AUC performance.

    Args:
        backend (BackendName): Backend to use (xgboost, mlp, lightgbm, lstm).
        dataset (DatasetName): Dataset to optimize on (taiwan, us, polish).
        n_trials (int): Number of Optuna trials per preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds per preset, or None.
        save_model (bool): If True, save the best model for each preset if it improves.
        project_root (Path | None): Project root directory. If None, uses default.
    """
    if project_root is None:
        project_root = get_project_root()

    console = get_rich_console()
    presets: list[FeaturePreset] = ["none", "log_only", "ratios_only", "full"]
    results: list[tuple[FeaturePreset, float, int, float]] = []

    console.print()
    console.print(
        create_rich_panel(
            f"[bold magenta]Comparing Feature Presets[/bold magenta]\n"
            f"[cyan]Backend:[/cyan] [green]{backend.upper()}[/green] | "
            f"[cyan]Dataset:[/cyan] [yellow]{dataset.upper()}[/yellow] | "
            f"[cyan]Trials:[/cyan] [yellow]{n_trials}[/yellow] | "
            f"[cyan]Device:[/cyan] [yellow]{device.upper()}[/yellow]"
        )
    )
    console.print()
    output_dir = project_root / "models"
    history = OptimizationHistory.for_output_dir(output_dir)
    history.load()

    with create_rich_progress(console) as progress:
        task = progress.add_task(
            "[bold blue]Running presets...[/bold blue]",
            total=float(len(presets)),
        )

        for preset in presets:
            progress.update(task, description=f"Running [bold cyan]{preset}[/bold cyan]...")

            if backend == "xgboost":
                result, elapsed, _ = _run_xgboost_with_progress(
                    dataset, n_trials, preset, device, timeout, history, progress
                )
            elif backend == "mlp":
                result, elapsed, _ = _run_mlp_with_progress(
                    dataset, n_trials, preset, device, timeout, history, progress
                )
            elif backend == "lightgbm":
                result, elapsed, _ = _run_lightgbm_with_progress(
                    dataset, n_trials, preset, device, timeout, history, progress
                )
            else:
                # backend must be "lstm" here - mypy validates exhaustiveness
                result, elapsed, _ = _run_lstm_with_progress(
                    dataset, n_trials, preset, device, timeout, history, progress
                )

            # Save best model for this preset if requested
            if save_model:
                _ = save_best_model(
                    result=result,
                    dataset=dataset,
                    feature_preset=preset,
                    project_root=project_root,
                )

            results.append((preset, result["best_val_auc"], result["n_features"], elapsed))
            progress.advance(task)

    _print_preset_comparison_summary(results)


def _print_preset_comparison_summary(
    results: list[tuple[FeaturePreset, float, int, float]],
) -> None:
    """Print the preset comparison summary table.

    Args:
        results (list[tuple[FeaturePreset, float, int, float]]): List of
            (preset, auc, n_features, elapsed_seconds) tuples.
    """
    console = get_rich_console()

    console.print()
    table = create_rich_table(
        title="[bold magenta]Feature Preset Comparison[/bold magenta] [dim](sorted by AUC)[/dim]"
    )
    table.add_column("Rank", style="bold white", justify="center")
    table.add_column("Preset", style="cyan")
    table.add_column("Features", style="blue", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Time", style="dim", justify="right")

    def _get_auc(item: tuple[FeaturePreset, float, int, float]) -> float:
        return item[1]

    sorted_results = sorted(results, key=_get_auc, reverse=True)
    for i, (preset, auc, n_features, elapsed) in enumerate(sorted_results):
        if i == 0:
            rank = "[bold yellow]1st[/bold yellow]"
            auc_str = f"[bold green on black] {auc:.4f} [/bold green on black]"
        elif i == 1:
            rank = "[white]2nd[/white]"
            auc_str = f"[green]{auc:.4f}[/green]"
        elif i == 2:
            rank = "[dim]3rd[/dim]"
            auc_str = f"[yellow]{auc:.4f}[/yellow]"
        else:
            rank = f"[dim]{i + 1}th[/dim]"
            auc_str = f"{auc:.4f}"

        table.add_row(rank, preset, str(n_features), auc_str, f"{elapsed:.1f}s")

    console.print(table)
    console.print()

    winner = sorted_results[0]
    console.print(
        f"[bold white on blue] Winner: {winner[0]} with AUC {winner[1]:.4f} [/bold white on blue]"
    )
    console.print()


def run_all_datasets(
    backend: BackendName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    save_model: bool = True,
) -> None:
    """Run optimization on all three datasets (taiwan, us, polish).

    Args:
        backend (BackendName): Backend to use (xgboost, mlp, lightgbm, lstm).
        n_trials (int): Number of Optuna trials per dataset.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
        timeout (int | None): Optional timeout in seconds per dataset, or None.
        save_model (bool): If True, save the best model for each dataset if it improves.
    """
    console = get_rich_console()
    datasets: list[DatasetName] = ["taiwan", "us", "polish"]
    all_results: list[tuple[DatasetName, RunResult]] = []

    _print_multi_dataset_config(backend, n_trials, feature_preset, device)

    for i, dataset in enumerate(datasets):
        console.print(
            f"[bold white on blue] Dataset {i + 1}/3: {dataset.upper()} [/bold white on blue]"
        )

        run_result = run_single_with_progress(
            backend,
            dataset,
            n_trials,
            feature_preset,
            device,
            timeout,
            save_model,
        )
        all_results.append((dataset, run_result))

        print_result(
            run_result["backend"],
            run_result["result"],
            run_result["elapsed"],
            run_result["previous_best"],
            run_result["all_time_best"],
        )

    _print_multi_dataset_summary(all_results)


def _print_multi_dataset_config(
    backend: BackendName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
) -> None:
    """Print multi-dataset configuration panel.

    Args:
        backend (BackendName): Backend being used (xgboost, mlp, lightgbm, lstm).
        n_trials (int): Number of Optuna trials per dataset.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
    """
    console = get_rich_console()

    console.print()
    console.print(create_rich_panel("[bold magenta]Multi-Dataset Optimization[/bold magenta]"))
    console.print()

    config_table = create_rich_table(
        title="[bold cyan]Run Configuration[/bold cyan]",
        show_header=False,
    )
    config_table.add_column("Setting", style="bold cyan")
    config_table.add_column("Value", style="bold yellow")
    config_table.add_column("Description", style="dim italic")
    config_table.add_row(
        "[white]Backend[/white]",
        f"[green]{backend.upper()}[/green]",
        "ML model backend",
    )
    config_table.add_row(
        "[white]Datasets[/white]",
        "[green]TAIWAN[/green], [blue]US[/blue], [magenta]POLISH[/magenta]",
        "All external bankruptcy datasets",
    )
    config_table.add_row(
        "[white]Trials/Dataset[/white]",
        f"[yellow]{n_trials}[/yellow]",
        "Optuna TPE Bayesian optimization",
    )
    config_table.add_row(
        "[white]Feature Preset[/white]",
        f"[blue]{feature_preset}[/blue]",
        PRESET_DESCRIPTIONS.get(feature_preset, ""),
    )
    config_table.add_row(
        "[white]Device[/white]",
        f"[yellow]{device.upper()}[/yellow]",
        "GPU (CUDA) or CPU",
    )
    config_table.add_row(
        "[white]Optimizer[/white]",
        "[cyan]Optuna TPE[/cyan]",
        "Tree-structured Parzen Estimator",
    )
    console.print(config_table)
    console.print()


def _print_multi_dataset_summary(
    all_results: list[tuple[DatasetName, RunResult]],
) -> None:
    """Print final multi-dataset summary table.

    Args:
        all_results (list[tuple[DatasetName, RunResult]]): List of
            (dataset_name, run_result) tuples from each dataset run.
    """
    console = get_rich_console()

    console.print()
    console.print(create_rich_panel("[bold green]Final Summary - All Datasets[/bold green]"))
    console.print()

    table = create_rich_table(title="[bold magenta]Cross-Dataset Results[/bold magenta]")
    table.add_column("Dataset", style="bold cyan")
    table.add_column("AUC", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Samples", style="blue", justify="right")
    table.add_column("Features", style="blue", justify="right")
    table.add_column("Time", style="dim", justify="right")

    best_auc = max(r[1]["result"]["best_val_auc"] for r in all_results)

    for dataset, run_result in all_results:
        auc = run_result["result"]["best_val_auc"]
        if auc == best_auc:
            auc_str = f"[bold green on black] {auc:.4f} [/bold green on black]"
        else:
            auc_str = f"[green]{auc:.4f}[/green]"

        if run_result["all_time_best"] is not None:
            delta = auc - run_result["all_time_best"]["best_val_auc"]
            if delta > 0.001:
                delta_str = f"[bold green]+{delta:.4f}[/bold green]"
            elif delta < -0.001:
                delta_str = f"[bold red]{delta:.4f}[/bold red]"
            else:
                delta_str = f"[dim]{delta:+.4f}[/dim]"
        else:
            delta_str = "[green]NEW[/green]"

        table.add_row(
            f"[bold yellow]{dataset.upper()}[/bold yellow]",
            auc_str,
            delta_str,
            f"{run_result['result']['n_samples']:,}",
            f"{run_result['result']['n_features']:,}",
            f"{run_result['elapsed']:.1f}s",
        )

    console.print(table)
    console.print()

    def _get_result_auc(item: tuple[DatasetName, RunResult]) -> float:
        return item[1]["result"]["best_val_auc"]

    best_result = max(all_results, key=_get_result_auc)
    console.print(
        f"[bold white on green] Best: {best_result[0].upper()} "
        f"with AUC {best_result[1]['result']['best_val_auc']:.4f} [/bold white on green]"
    )
    console.print()


__all__ = [
    "_format_elapsed",
    "compare_presets",
    "run_all_datasets",
    "run_single_with_progress",
]
