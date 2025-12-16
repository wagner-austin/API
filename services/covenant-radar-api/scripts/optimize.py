"""CLI entry point for running hyperparameter optimization.

Usage:
    python -m scripts.optimize --dataset taiwan --n-trials 50
    python -m scripts.optimize --dataset taiwan --n-trials 100 --feature-preset full
    python -m scripts.optimize --dataset taiwan --n-trials 50 --compare-presets
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from platform_core.json_utils import dump_json_str
from platform_core.logging import (
    RichTableProtocol,
    create_rich_panel,
    create_rich_progress,
    create_rich_table,
    get_logger,
    get_rich_console,
    setup_rich_logging,
    stdlib_logging,
)

import scripts._test_hooks as _hooks
from covenant_radar_api.worker.optimize_job import OptimizationResult

# Module-level verbose flag for optuna logging control
_verbose_mode: bool = False

# Feature preset descriptions
PRESET_DESCRIPTIONS: dict[str, str] = {
    "none": "Original features only",
    "log_only": "Original + log transforms",
    "ratios_only": "Original + pairwise ratios (capped at 500)",
    "full": "Original + log + ratios + products (max ~800 features)",
}

FeaturePreset = Literal["none", "log_only", "ratios_only", "full"]
DatasetName = Literal["taiwan", "us", "polish"]

logger = get_logger(__name__)


def _suppress_verbose_logging() -> None:
    """Suppress verbose logging unless verbose mode is enabled.

    Suppresses Optuna and optimize_job loggers to WARNING level
    to avoid spammy output. Call this right before optimization starts.
    """
    if _verbose_mode:
        return

    # Suppress optuna and its subloggers
    optuna_logger = stdlib_logging.getLogger("optuna")
    optuna_logger.setLevel(stdlib_logging.WARNING)

    for name in ("optuna.trial", "optuna.study", "optuna._optimize"):
        stdlib_logging.getLogger(name).setLevel(stdlib_logging.WARNING)

    # Suppress optimize_job logging when using progress display
    stdlib_logging.getLogger("covenant_radar_api.worker.optimize_job").setLevel(
        stdlib_logging.WARNING
    )

    # Suppress covenant_ml optuna backend logging (Trial complete messages)
    stdlib_logging.getLogger("covenant_ml.optimizer.optuna_backend").setLevel(
        stdlib_logging.WARNING
    )

    # Also suppress real_data loading messages
    stdlib_logging.getLogger("covenant_radar_api.seeding.real_data").setLevel(
        stdlib_logging.WARNING
    )


class OptimizeArgs:
    """Parsed command line arguments."""

    dataset: DatasetName
    n_trials: int
    feature_preset: FeaturePreset
    device: str
    timeout: int | None
    compare_presets: bool
    all_datasets: bool
    verbose: bool

    def __init__(self) -> None:
        """Initialize with defaults."""
        self.dataset = "taiwan"
        self.n_trials = 300
        self.feature_preset = "full"
        self.device = "cuda"
        self.timeout = None
        self.compare_presets = False
        self.all_datasets = False
        self.verbose = False


def _print_help() -> None:
    """Print help message."""
    console = get_rich_console()
    help_text = """
[bold]Usage:[/bold] python -m scripts.optimize [OPTIONS]

[bold]Options:[/bold]
  -d, --dataset         Dataset: taiwan, us, polish (default: taiwan)
  -n, --n-trials        Number of trials (default: 300)
  -f, --feature-preset  Preset: none, log_only, ratios_only, full (default: full)
  --device              Device: auto, cpu, cuda (default: cuda)
  -t, --timeout         Timeout in seconds (optional)
  -c, --compare-presets Run all presets on one dataset and compare
  -a, --all-datasets    Run on all three datasets
  -v, --verbose         Show Optuna trial logs (default: quiet)
  -h, --help            Show this help
"""
    console.print(help_text)


def _get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


def _parse_dataset(val: str) -> DatasetName:
    """Parse dataset value."""
    console = get_rich_console()
    if val == "taiwan":
        return "taiwan"
    if val == "us":
        return "us"
    if val == "polish":
        return "polish"
    console.print(f"[red]Invalid dataset: {val}. Must be taiwan, us, or polish.[/red]")
    raise SystemExit(1)


def _parse_preset(val: str) -> FeaturePreset:
    """Parse feature preset value."""
    console = get_rich_console()
    if val == "none":
        return "none"
    if val == "log_only":
        return "log_only"
    if val == "ratios_only":
        return "ratios_only"
    if val == "full":
        return "full"
    console.print(f"[red]Invalid preset: {val}. Must be none, log_only, ratios_only, full.[/red]")
    raise SystemExit(1)


def _handle_flag(result: OptimizeArgs, arg: str) -> bool:
    """Handle boolean flags. Returns True if flag was handled."""
    if arg in ("--compare-presets", "-c"):
        result.compare_presets = True
        return True
    if arg in ("--all-datasets", "-a"):
        result.all_datasets = True
        return True
    if arg in ("--verbose", "-v"):
        result.verbose = True
        return True
    if arg in ("--help", "-h"):
        _print_help()
        raise SystemExit(0)
    return False


def _parse_args(argv: Sequence[str]) -> OptimizeArgs:
    """Parse command line arguments."""
    args = list(argv)
    result = OptimizeArgs()

    i = 0
    while i < len(args):
        arg = args[i]
        if _handle_flag(result, arg):
            i += 1
        elif arg in ("--dataset", "-d") and i + 1 < len(args):
            result.dataset = _parse_dataset(args[i + 1])
            i += 2
        elif arg in ("--n-trials", "-n") and i + 1 < len(args):
            result.n_trials = int(args[i + 1])
            i += 2
        elif arg in ("--feature-preset", "-f") and i + 1 < len(args):
            result.feature_preset = _parse_preset(args[i + 1])
            i += 2
        elif arg in ("--device",) and i + 1 < len(args):
            result.device = args[i + 1]
            i += 2
        elif arg in ("--timeout", "-t") and i + 1 < len(args):
            result.timeout = int(args[i + 1])
            i += 2
        else:
            i += 1

    return result


def _run_single(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
    progress_callback: _hooks.TrialProgressCallbackProtocol | None = None,
) -> OptimizationResult:
    """Run a single optimization."""
    project_root = _get_project_root()
    external_dir = project_root / "data" / "external"
    output_dir = project_root / "models"
    output_dir.mkdir(exist_ok=True)

    config_dict: dict[str, str | int | None] = {
        "dataset": dataset,
        "n_trials": n_trials,
        "feature_preset": feature_preset,
        "device": device,
        "random_state": 42,
    }
    if timeout is not None:
        config_dict["timeout_seconds"] = timeout

    config_json = dump_json_str(config_dict)

    return _hooks.optimization_runner(config_json, external_dir, output_dir, progress_callback)


def _create_result_table(result: OptimizationResult, elapsed: float) -> RichTableProtocol:
    """Create a rich table for optimization results."""
    table = create_rich_table(
        title="[bold magenta]Optimization Results[/bold magenta]",
        show_header=False,
    )
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    # Dataset and preset with distinct colors
    table.add_row("[cyan]Dataset[/cyan]", f"[bold yellow]{result['dataset'].upper()}[/bold yellow]")
    table.add_row("[cyan]Feature Preset[/cyan]", f"[magenta]{result['feature_preset']}[/magenta]")

    # Data dimensions
    table.add_row("[cyan]Samples[/cyan]", f"[blue]{result['n_samples']:,}[/blue]")
    table.add_row("[cyan]Features[/cyan]", f"[blue]{result['n_features']:,}[/blue]")

    # Key metric - highlighted prominently
    table.add_row(
        "[cyan]Best AUC[/cyan]",
        f"[bold green on black] {result['best_val_auc']:.4f} [/bold green on black]",
    )
    table.add_row("[cyan]Best Trial[/cyan]", f"[yellow]#{result['best_trial_number']}[/yellow]")

    # Trial counts with color-coded status
    complete_str = f"[green]{result['n_trials_complete']}[/green] complete"
    pruned_str = f"[yellow]{result['n_trials_pruned']}[/yellow] pruned"
    failed_str = f"[red]{result['n_trials_failed']}[/red] failed"
    table.add_row("[cyan]Trials[/cyan]", f"{complete_str}, {pruned_str}, {failed_str}")

    # Timing
    table.add_row("[cyan]Time[/cyan]", f"[dim]{elapsed:.1f}s[/dim]")

    return table


def _create_hyperparams_table(result: OptimizationResult) -> RichTableProtocol:
    """Create a rich table for best hyperparameters."""
    table = create_rich_table(
        title="[bold blue]Best Hyperparameters[/bold blue]",
        show_header=True,
    )
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="bold yellow", justify="right")

    # Tree structure params
    table.add_row("[green]max_depth[/green]", f"[bold]{result['best_max_depth']}[/bold]")
    table.add_row("[green]n_estimators[/green]", f"[bold]{result['best_n_estimators']}[/bold]")

    # Learning params
    table.add_row("[magenta]learning_rate[/magenta]", f"{result['best_learning_rate']:.6f}")

    # Regularization
    table.add_row("[yellow]reg_alpha[/yellow]", f"{result['best_reg_alpha']:.6f}")
    table.add_row("[yellow]reg_lambda[/yellow]", f"{result['best_reg_lambda']:.6f}")

    # Sampling params
    table.add_row("[blue]subsample[/blue]", f"{result['best_subsample']:.4f}")
    table.add_row("[blue]colsample_bytree[/blue]", f"{result['best_colsample_bytree']:.4f}")

    return table


def _print_config(
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
) -> None:
    """Print detailed configuration before running."""
    console = get_rich_console()
    config_table = create_rich_table(
        title="[bold cyan]Run Configuration[/bold cyan]",
        show_header=False,
    )
    config_table.add_column("Setting", style="bold cyan")
    config_table.add_column("Value", style="bold yellow")
    config_table.add_column("Description", style="dim italic")

    config_table.add_row(
        "[white]Dataset[/white]",
        f"[bold green]{dataset.upper()}[/bold green]",
        "External bankruptcy dataset",
    )
    config_table.add_row(
        "[white]Trials[/white]",
        f"[magenta]{n_trials}[/magenta]",
        "Optuna TPE Bayesian optimization trials",
    )
    config_table.add_row(
        "[white]Feature Preset[/white]",
        f"[blue]{feature_preset}[/blue]",
        PRESET_DESCRIPTIONS.get(feature_preset, ""),
    )
    config_table.add_row(
        "[white]Device[/white]",
        f"[yellow]{device.upper()}[/yellow]",
        "GPU (CUDA) or CPU for XGBoost training",
    )
    config_table.add_row(
        "[white]Model[/white]",
        "[green]XGBoost[/green]",
        "Gradient boosted trees (DMatrix API)",
    )
    config_table.add_row(
        "[white]Optimizer[/white]",
        "[cyan]Optuna TPE[/cyan]",
        "Tree-structured Parzen Estimator",
    )
    config_table.add_row(
        "[white]Metric[/white]",
        "[bold magenta]AUC-ROC[/bold magenta]",
        "Area under ROC curve (validation set)",
    )

    console.print()
    console.print(create_rich_panel("[bold magenta]Hyperparameter Optimization[/bold magenta]"))
    console.print()
    console.print(config_table)
    console.print()


def _print_result(result: OptimizationResult, elapsed: float) -> None:
    """Print optimization result with rich formatting."""
    console = get_rich_console()
    console.print()
    console.print(create_rich_panel("[bold green]OPTIMIZATION COMPLETE[/bold green]"))
    console.print()

    result_table = _create_result_table(result, elapsed)
    console.print(result_table)
    console.print()

    params_table = _create_hyperparams_table(result)
    console.print(params_table)
    console.print()

    # Print final AUC highlight
    auc_value = result["best_val_auc"]
    console.print(f"[bold white on green] Best AUC: {auc_value:.4f} [/bold white on green]")
    console.print()


def _compare_presets(
    dataset: DatasetName,
    n_trials: int,
    device: str,
    timeout: int | None,
) -> None:
    """Run all presets and compare."""
    console = get_rich_console()
    presets: list[FeaturePreset] = ["none", "log_only", "ratios_only", "full"]
    results: list[tuple[FeaturePreset, float, int, float]] = []

    console.print()
    console.print(
        create_rich_panel(
            f"[bold magenta]Comparing Feature Presets[/bold magenta]\n"
            f"[cyan]Dataset:[/cyan] [yellow]{dataset.upper()}[/yellow] | "
            f"[cyan]Trials:[/cyan] [yellow]{n_trials}[/yellow] | "
            f"[cyan]Device:[/cyan] [yellow]{device.upper()}[/yellow]"
        )
    )
    console.print()

    with create_rich_progress(console) as progress:
        task = progress.add_task(
            "[bold blue]Running presets...[/bold blue]",
            total=float(len(presets)),
        )

        for preset in presets:
            progress.update(task, description=f"Running [bold cyan]{preset}[/bold cyan]...")

            def progress_callback(
                info: _hooks.TrialProgressInfo, current_preset: FeaturePreset = preset
            ) -> None:
                """Update progress bar description with trial info, including more metrics."""
                best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
                desc = (
                    f"Running [bold cyan]{current_preset}[/bold cyan]... "
                    f"[cyan]Trial {info['trial_number']}/{info['n_trials_total']}[/cyan] "
                    f"Best: [green]{info['best_auc']:.4f}[/green] "
                    f"(#{info['best_trial']}) {best_marker} "
                    f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
                    f"Depth: [magenta]{info['best_max_depth']}[/magenta] "
                    f"Estimators: [blue]{info['best_n_estimators']}[/blue]"
                )
                progress.update(task, description=desc)

            start = time.perf_counter()
            result = _run_single(
                dataset,
                n_trials,
                preset,
                device,
                timeout,
                progress_callback,
            )
            elapsed = time.perf_counter() - start
            results.append((preset, result["best_val_auc"], result["n_features"], elapsed))
            progress.advance(task)

    # Summary table
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
        # Colorful rank indicators
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

    # Winner highlight
    winner = sorted_results[0]
    console.print(
        f"[bold white on blue] Winner: {winner[0]} with AUC {winner[1]:.4f} [/bold white on blue]"
    )
    console.print()


def _run_all_datasets(
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
    timeout: int | None,
) -> None:
    """Run optimization on all three datasets."""
    console = get_rich_console()
    datasets: list[DatasetName] = ["taiwan", "us", "polish"]
    all_results: list[tuple[DatasetName, OptimizationResult, float]] = []

    # Print overall configuration
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
        "GPU (CUDA) or CPU for XGBoost",
    )
    config_table.add_row(
        "[white]Model[/white]",
        "[green]XGBoost[/green]",
        "Gradient boosted trees (DMatrix API)",
    )
    config_table.add_row(
        "[white]Optimizer[/white]",
        "[cyan]Optuna TPE[/cyan]",
        "Tree-structured Parzen Estimator",
    )
    console.print(config_table)
    console.print()

    for i, dataset in enumerate(datasets):
        console.print(
            f"[bold white on blue] Dataset {i + 1}/3: {dataset.upper()} [/bold white on blue]"
        )

        with create_rich_progress(console) as progress:
            task_id = progress.add_task(
                f"[cyan]Running {n_trials} trials...[/cyan]",
                total=float(n_trials),
            )

            def progress_callback(
                info: _hooks.TrialProgressInfo, current_task_id: int = task_id
            ) -> None:
                best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
                desc = (
                    f"[cyan]Trial {info['trial_number']}/{info['n_trials_total']}[/cyan] "
                    f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
                    f"(#{info['best_trial']}) {best_marker} "
                    f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
                    f"Depth: [magenta]{info['best_max_depth']}[/magenta] "
                    f"Estimators: [blue]{info['best_n_estimators']}[/blue]"
                )
                progress.update(current_task_id, description=desc)
                progress.advance(current_task_id)

            start = time.perf_counter()
            result = _run_single(
                dataset,
                n_trials,
                feature_preset,
                device,
                timeout,
                progress_callback,
            )
            elapsed = time.perf_counter() - start

        all_results.append((dataset, result, elapsed))
        _print_result(result, elapsed)

    # Final summary table
    console.print()
    console.print(create_rich_panel("[bold green]Final Summary - All Datasets[/bold green]"))
    console.print()

    table = create_rich_table(title="[bold magenta]Cross-Dataset Results[/bold magenta]")
    table.add_column("Dataset", style="bold cyan")
    table.add_column("AUC", justify="right")
    table.add_column("Samples", style="blue", justify="right")
    table.add_column("Features", style="blue", justify="right")
    table.add_column("Time", style="dim", justify="right")

    # Find best AUC for highlighting
    best_auc = max(r[1]["best_val_auc"] for r in all_results)

    for dataset, result, elapsed in all_results:
        auc = result["best_val_auc"]
        if auc == best_auc:
            auc_str = f"[bold green on black] {auc:.4f} [/bold green on black]"
        else:
            auc_str = f"[green]{auc:.4f}[/green]"

        table.add_row(
            f"[bold yellow]{dataset.upper()}[/bold yellow]",
            auc_str,
            f"{result['n_samples']:,}",
            f"{result['n_features']:,}",
            f"{elapsed:.1f}s",
        )

    console.print(table)
    console.print()

    # Overall winner - find dataset with best AUC
    def _get_result_auc(
        item: tuple[DatasetName, OptimizationResult, float],
    ) -> float:
        return item[1]["best_val_auc"]

    best_result = max(all_results, key=_get_result_auc)
    console.print(
        f"[bold white on green] Best: {best_result[0].upper()} "
        f"with AUC {best_result[1]['best_val_auc']:.4f} [/bold white on green]"
    )
    console.print()


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success).
    """
    global _verbose_mode

    # Setup rich logging early so error messages can use console
    # Default to INFO, will update if --verbose is passed
    setup_rich_logging(level="INFO", show_time=False)

    try:
        raw_args = list(argv) if argv is not None else list(sys.argv[1:])
        args = _parse_args(raw_args)

        # Set verbose mode for optuna logging control
        _verbose_mode = args.verbose

        # Update logging level if verbose mode requested
        if args.verbose:
            setup_rich_logging(level="DEBUG", show_time=False)

        # Suppress verbose logging from optuna and workers BEFORE any optimization runs
        _suppress_verbose_logging()

        if args.all_datasets:
            _run_all_datasets(args.n_trials, args.feature_preset, args.device, args.timeout)
        elif args.compare_presets:
            _compare_presets(args.dataset, args.n_trials, args.device, args.timeout)
        else:
            # Print detailed configuration
            _print_config(args.dataset, args.n_trials, args.feature_preset, args.device)

            console = get_rich_console()
            with create_rich_progress(console) as progress:
                task_id = progress.add_task(
                    f"[cyan]Trial 0/{args.n_trials}[/cyan] Best: [green]0.0000[/green]",
                    total=float(args.n_trials),
                )

                def progress_callback(info: _hooks.TrialProgressInfo) -> None:
                    """Update progress bar with trial info, including more metrics."""
                    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
                    desc = (
                        f"[cyan]Trial {info['trial_number']}/{info['n_trials_total']}[/cyan] "
                        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
                        f"(#{info['best_trial']}) {best_marker} "
                        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
                        f"Depth: [magenta]{info['best_max_depth']}[/magenta] "
                        f"Estimators: [blue]{info['best_n_estimators']}[/blue]"
                    )
                    progress.update(task_id, description=desc)
                    progress.advance(task_id)

                start = time.perf_counter()
                result = _run_single(
                    args.dataset,
                    args.n_trials,
                    args.feature_preset,
                    args.device,
                    args.timeout,
                    progress_callback,
                )
                elapsed = time.perf_counter() - start

            _print_result(result, elapsed)

    except KeyboardInterrupt:
        console = get_rich_console()
        console.print()
        console.print(create_rich_panel("[bold red]Process Interrupted by User[/bold red]"))
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
