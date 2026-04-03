"""Run modes for optimization: single, preset comparison, multi-dataset, and multi-backend.

Supports all 7 backends with unified progress display and history tracking.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from pathlib import Path

from covenant_ml.types import BackendName
from platform_core.logging import (
    create_rich_panel,
    create_rich_progress,
    create_rich_table,
    get_rich_console,
)

from scripts.optimize._runners import (
    _run_backend_with_progress,
    run_single_with_progress,
)
from scripts.optimize.cli import PRESET_DESCRIPTIONS, DatasetName, FeaturePreset
from scripts.optimize.display import print_result
from scripts.optimize.history import OptimizationHistory
from scripts.optimize.model_saver import save_best_model
from scripts.optimize.runner import RunResult, get_project_root

# =============================================================================
# Preset Comparison Mode
# =============================================================================


def compare_presets(
    backends: tuple[BackendName, ...],
    dataset: DatasetName,
    n_trials: int,
    device: str,
    timeout: int | None,
    save_model: bool = True,
    project_root: Path | None = None,
) -> None:
    """Run all presets for all backends and compare AUC performance.

    Args:
        backends: Tuple of backends to use.
        dataset: Dataset to optimize on (taiwan, us, polish).
        n_trials: Number of Optuna trials per preset.
        device: Device for training (cuda/cpu/auto).
        timeout: Optional timeout in seconds per preset, or None.
        save_model: If True, save the best model for each preset if it improves.
        project_root: Project root directory. If None, uses default.
    """
    if project_root is None:
        project_root = get_project_root()

    console = get_rich_console()
    presets: list[FeaturePreset] = ["none", "log_only", "ratios_only", "full"]
    # Results: list of (backend, preset, auc, n_features, elapsed)
    results: list[tuple[BackendName, FeaturePreset, float, int, float]] = []

    backends_str = ", ".join(b.upper() for b in backends)
    console.print()
    console.print(
        create_rich_panel(
            f"[bold magenta]Comparing Feature Presets[/bold magenta]\n"
            f"[cyan]Backends:[/cyan] [green]{backends_str}[/green] | "
            f"[cyan]Dataset:[/cyan] [yellow]{dataset.upper()}[/yellow] | "
            f"[cyan]Trials:[/cyan] [yellow]{n_trials}[/yellow] | "
            f"[cyan]Device:[/cyan] [yellow]{device.upper()}[/yellow]"
        )
    )
    console.print()
    output_dir = project_root / "models"
    history = OptimizationHistory.for_output_dir(output_dir)
    history.load()

    total_runs = len(backends) * len(presets)

    with create_rich_progress(console) as progress:
        task = progress.add_task(
            "[bold blue]Running...[/bold blue]",
            total=float(total_runs),
        )

        for backend in backends:
            for preset in presets:
                desc = (
                    f"Running [bold cyan]{backend.upper()}[/bold cyan] "
                    f"/ [yellow]{preset}[/yellow]..."
                )
                progress.update(task, description=desc)

                result, elapsed, _ = _run_backend_with_progress(
                    backend,
                    dataset,
                    n_trials,
                    preset,
                    device,
                    timeout,
                    history,
                    progress,
                )

                # Save best model for this preset if requested
                if save_model:
                    _ = save_best_model(
                        result=result,
                        dataset=dataset,
                        feature_preset=preset,
                        project_root=project_root,
                    )

                results.append(
                    (
                        backend,
                        preset,
                        result["best_value"],
                        result["n_features"],
                        elapsed,
                    )
                )
                progress.advance(task)

    _print_multi_backend_preset_comparison(results, dataset)


def _print_preset_comparison_summary(
    results: list[tuple[FeaturePreset, float, int, float]],
) -> None:
    """Print the preset comparison summary table.

    Args:
        results: List of (preset, auc, n_features, elapsed_seconds) tuples.
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


def _print_multi_backend_preset_comparison(
    results: list[tuple[BackendName, FeaturePreset, float, int, float]],
    dataset: DatasetName,
) -> None:
    """Print comparison table for multiple backends across all presets.

    Args:
        results: List of (backend, preset, auc, n_features, elapsed) tuples.
        dataset: Dataset name used for optimization.
    """
    console = get_rich_console()

    console.print()
    title = (
        f"[bold magenta]Backend x Preset Comparison - {dataset.upper()}[/bold magenta] "
        "[dim](sorted by AUC)[/dim]"
    )
    table = create_rich_table(title=title)
    table.add_column("Rank", style="bold white", justify="center")
    table.add_column("Backend", style="cyan")
    table.add_column("Preset", style="blue")
    table.add_column("Features", style="dim", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Time", style="dim", justify="right")

    def _get_auc(item: tuple[BackendName, FeaturePreset, float, int, float]) -> float:
        return item[2]

    sorted_results = sorted(results, key=_get_auc, reverse=True)

    for i, (backend, preset, auc, n_features, elapsed) in enumerate(sorted_results):
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

        table.add_row(
            rank,
            f"[bold]{backend.upper()}[/bold]",
            preset,
            str(n_features),
            auc_str,
            f"{elapsed:.1f}s",
        )

    console.print(table)
    console.print()

    winner = sorted_results[0]
    console.print(
        f"[bold white on green] Winner: {winner[0].upper()} + {winner[1]} "
        f"with AUC {winner[2]:.4f} [/bold white on green]"
    )
    console.print()


# =============================================================================
# Multi-Dataset Mode
# =============================================================================


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
        backend: Backend to use.
        n_trials: Number of Optuna trials per dataset.
        feature_preset: Feature engineering preset.
        device: Device for training (cuda/cpu/auto).
        timeout: Optional timeout in seconds per dataset, or None.
        save_model: If True, save the best model for each dataset if it improves.
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
        backend: Backend being used.
        n_trials: Number of Optuna trials per dataset.
        feature_preset: Feature engineering preset.
        device: Device for training (cuda/cpu/auto).
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
        all_results: List of (dataset_name, run_result) tuples from each dataset run.
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

    best_auc = max(r[1]["result"]["best_value"] for r in all_results)

    for dataset, run_result in all_results:
        auc = run_result["result"]["best_value"]
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
        return item[1]["result"]["best_value"]

    best_result = max(all_results, key=_get_result_auc)
    console.print(
        f"[bold white on green] Best: {best_result[0].upper()} "
        f"with AUC {best_result[1]['result']['best_value']:.4f} [/bold white on green]"
    )
    console.print()


# =============================================================================
# Multi-Backend Mode
# =============================================================================


def run_multiple_backends(
    backends: tuple[BackendName, ...],
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    save_model: bool = True,
    project_root: Path | None = None,
) -> None:
    """Run optimization on multiple backends sequentially.

    Args:
        backends: Tuple of backends to run.
        dataset: Dataset to optimize on.
        n_trials: Number of Optuna trials per backend.
        feature_preset: Feature engineering preset.
        device: Device for training (cuda/cpu/auto).
        timeout: Optional timeout in seconds per backend.
        save_model: If True, save the best model for each backend.
        project_root: Project root directory. If None, uses default.
    """
    if project_root is None:
        project_root = get_project_root()

    console = get_rich_console()
    all_results: list[tuple[BackendName, RunResult]] = []

    # Print header
    console.print()
    backends_str = ", ".join(b.upper() for b in backends)
    console.print(
        create_rich_panel(
            f"[bold magenta]Multi-Backend Optimization[/bold magenta]\n"
            f"[cyan]Backends:[/cyan] [green]{backends_str}[/green] | "
            f"[cyan]Dataset:[/cyan] [yellow]{dataset.upper()}[/yellow] | "
            f"[cyan]Trials/Backend:[/cyan] [yellow]{n_trials}[/yellow]"
        )
    )
    console.print()

    # Run each backend
    for i, backend in enumerate(backends):
        label = f"Backend {i + 1}/{len(backends)}: {backend.upper()}"
        console.print(f"[bold white on blue] {label} [/bold white on blue]")

        run_result = run_single_with_progress(
            backend,
            dataset,
            n_trials,
            feature_preset,
            device,
            timeout,
            save_model,
            project_root,
        )
        all_results.append((backend, run_result))

        print_result(
            run_result["backend"],
            run_result["result"],
            run_result["elapsed"],
            run_result["previous_best"],
            run_result["all_time_best"],
        )

    # Print summary if multiple backends
    if len(all_results) > 1:
        _print_multi_backend_summary(all_results, dataset)


def _print_multi_backend_summary(
    all_results: list[tuple[BackendName, RunResult]],
    dataset: DatasetName,
) -> None:
    """Print final multi-backend comparison table.

    Args:
        all_results: List of (backend, run_result) tuples.
        dataset: Dataset name that was optimized.
    """
    console = get_rich_console()

    console.print()
    console.print(
        create_rich_panel(f"[bold green]Backend Comparison - {dataset.upper()}[/bold green]")
    )
    console.print()

    table = create_rich_table(title="[bold magenta]Results by Backend[/bold magenta]")
    table.add_column("Rank", style="bold white", justify="center")
    table.add_column("Backend", style="cyan")
    table.add_column("AUC", justify="right")
    table.add_column("Best Trial", style="blue", justify="right")
    table.add_column("Time", style="dim", justify="right")

    def _get_auc(item: tuple[BackendName, RunResult]) -> float:
        return item[1]["result"]["best_value"]

    sorted_results = sorted(all_results, key=_get_auc, reverse=True)

    for i, (backend, run_result) in enumerate(sorted_results):
        auc = run_result["result"]["best_value"]
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

        table.add_row(
            rank,
            f"[bold]{backend.upper()}[/bold]",
            auc_str,
            f"#{run_result['result']['best_trial_number']}",
            f"{run_result['elapsed']:.1f}s",
        )

    console.print(table)
    console.print()

    winner = sorted_results[0]
    console.print(
        f"[bold white on green] Winner: {winner[0].upper()} "
        f"with AUC {winner[1]['result']['best_value']:.4f} [/bold white on green]"
    )
    console.print()


__all__ = [
    "compare_presets",
    "run_all_datasets",
    "run_multiple_backends",
    "run_single_with_progress",
]
