"""Rich console display formatting for multi-backend optimization output.

Supports all 7 backends with unified hyperparameter display from
best_int_params/best_float_params/best_string_params.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from covenant_ml.types import BackendName
from platform_core.json_utils import require_float, require_int, require_str
from platform_core.rich_logging import (
    RichTableProtocol,
    create_rich_panel,
    create_rich_table,
    get_rich_console,
)

from covenant_radar_api.worker._optimize_param_codec import (
    encode_sampled_float_params,
    encode_sampled_int_params,
    encode_sampled_string_params,
)
from scripts._test_hooks import UnifiedOptimizationResult
from scripts.optimize.cli import PRESET_DESCRIPTIONS, DatasetName, FeaturePreset
from scripts.optimize.history import UnifiedHistoryEntry

# =============================================================================
# Backend Display Names
# =============================================================================

BACKEND_DISPLAY_NAMES: dict[BackendName, str] = {
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
    "cleargbm": "ClearGBM",
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
}


# =============================================================================
# Result Table (Common fields)
# =============================================================================


def create_result_table(
    backend: BackendName,
    result: UnifiedOptimizationResult,
    elapsed: float,
) -> RichTableProtocol:
    """Create a rich table for optimization results.

    Args:
        backend: Backend used for optimization.
        result: Optimization result.
        elapsed: Elapsed time in seconds.

    Returns:
        Rich table with result summary.
    """
    table = create_rich_table(
        title="[bold magenta]Optimization Results[/bold magenta]",
        show_header=False,
    )
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    # Backend
    backend_display = BACKEND_DISPLAY_NAMES.get(backend, backend.upper())
    table.add_row("[cyan]Backend[/cyan]", f"[bold green]{backend_display}[/bold green]")

    # Dataset and preset with distinct colors
    table.add_row("[cyan]Dataset[/cyan]", f"[bold yellow]{result['dataset'].upper()}[/bold yellow]")
    table.add_row("[cyan]Feature Preset[/cyan]", f"[magenta]{result['feature_preset']}[/magenta]")

    # Data dimensions
    table.add_row("[cyan]Samples[/cyan]", f"[blue]{result['n_samples']:,}[/blue]")
    table.add_row("[cyan]Features[/cyan]", f"[blue]{result['n_features']:,}[/blue]")

    # Key metric - highlighted prominently
    table.add_row(
        "[cyan]Best AUC[/cyan]",
        f"[bold green on black] {result['best_value']:.4f} [/bold green on black]",
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


# =============================================================================
# Unified Hyperparameter Table
# =============================================================================


def create_hyperparams_table(
    backend: BackendName,
    result: UnifiedOptimizationResult,
) -> RichTableProtocol:
    """Create a rich table for best hyperparameters from optimization result.

    Displays all parameters from best_int_params, best_float_params,
    and best_string_params in a single table.

    Args:
        backend: Backend used for optimization.
        result: Optimization result with nested best params.

    Returns:
        Rich table with hyperparameters.
    """
    backend_display = BACKEND_DISPLAY_NAMES.get(backend, backend.upper())
    table = create_rich_table(
        title=f"[bold blue]Best Hyperparameters ({backend_display})[/bold blue]",
        show_header=True,
    )
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="bold yellow", justify="right")

    # Integer params — encode then extract with require_int
    int_encoded = encode_sampled_int_params(result["best_int_params"])
    for name in int_encoded:
        val = require_int(int_encoded, name)
        table.add_row(f"[green]{name}[/green]", f"[bold]{val}[/bold]")

    # Float params — encode then extract with require_float
    float_encoded = encode_sampled_float_params(result["best_float_params"])
    for name in float_encoded:
        fval = require_float(float_encoded, name)
        table.add_row(f"[magenta]{name}[/magenta]", f"{fval:.6f}")

    # String params — encode then extract with require_str
    string_encoded = encode_sampled_string_params(result["best_string_params"])
    for name in string_encoded:
        sval = require_str(string_encoded, name)
        table.add_row(f"[blue]{name}[/blue]", sval)

    return table


# =============================================================================
# History Comparison Table
# =============================================================================


def create_history_comparison_table(
    current_auc: float,
    previous_best: UnifiedHistoryEntry | None,
    all_time_best: UnifiedHistoryEntry | None,
) -> RichTableProtocol:
    """Create a table comparing current run against previous runs.

    Args:
        current_auc: Current run's best AUC score.
        previous_best: Previous best history entry, or None if no previous runs.
        all_time_best: All-time best history entry, or None if first run.

    Returns:
        Rich table with run comparison.
    """
    table = create_rich_table(
        title="[bold cyan]Run Comparison[/bold cyan]",
        show_header=True,
    )
    table.add_column("Metric", style="bold white")
    table.add_column("AUC", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Run", style="dim")

    # Current run
    table.add_row(
        "[bold green]Current Run[/bold green]",
        f"[bold green]{current_auc:.4f}[/bold green]",
        "",
        "[dim]now[/dim]",
    )

    # Previous run comparison
    if previous_best is not None:
        prev_auc = previous_best["best_val_auc"]
        delta = current_auc - prev_auc
        delta_str = _format_delta(delta)
        table.add_row(
            "[yellow]Previous Best[/yellow]",
            f"[yellow]{prev_auc:.4f}[/yellow]",
            delta_str,
            f"[dim]{previous_best['timestamp'][:10]}[/dim]",
        )
    else:
        table.add_row(
            "[yellow]Previous Best[/yellow]",
            "[dim]N/A[/dim]",
            "[dim]first run[/dim]",
            "",
        )

    # All-time best comparison
    if all_time_best is not None:
        all_time_auc = all_time_best["best_val_auc"]
        delta = current_auc - all_time_auc
        delta_str = _format_delta(delta)
        is_new_best = current_auc >= all_time_auc
        marker = " [bold yellow]★[/bold yellow]" if is_new_best else ""
        table.add_row(
            f"[cyan]All-Time Best[/cyan]{marker}",
            f"[cyan]{all_time_auc:.4f}[/cyan]",
            delta_str,
            f"[dim]{all_time_best['timestamp'][:10]}[/dim]",
        )
    else:
        table.add_row(
            "[cyan]All-Time Best[/cyan] [bold yellow]★[/bold yellow]",
            f"[cyan]{current_auc:.4f}[/cyan]",
            "[green]NEW[/green]",
            "[dim]now[/dim]",
        )

    return table


def _format_delta(delta: float) -> str:
    """Format AUC delta with color coding.

    Args:
        delta: AUC delta value (current - baseline).

    Returns:
        Rich-formatted delta string with color (green positive, red negative).
    """
    if delta > 0.001:
        return f"[bold green]+{delta:.4f}[/bold green]"
    if delta < -0.001:
        return f"[bold red]{delta:.4f}[/bold red]"
    return f"[dim]{delta:+.4f}[/dim]"


# =============================================================================
# Configuration Display
# =============================================================================


def print_config(
    backend: BackendName,
    dataset: DatasetName,
    n_trials: int,
    feature_preset: FeaturePreset,
    device: str,
) -> None:
    """Print detailed configuration before running.

    Args:
        backend: Backend to use.
        dataset: Dataset name (taiwan, us, polish).
        n_trials: Number of Optuna trials to run.
        feature_preset: Feature engineering preset.
        device: Device for training (cuda/cpu/auto).
    """
    console = get_rich_console()
    config_table = create_rich_table(
        title="[bold cyan]Run Configuration[/bold cyan]",
        show_header=False,
    )
    config_table.add_column("Setting", style="bold cyan")
    config_table.add_column("Value", style="bold yellow")
    config_table.add_column("Description", style="dim italic")

    backend_display = BACKEND_DISPLAY_NAMES.get(backend, backend.upper())
    config_table.add_row(
        "[white]Backend[/white]",
        f"[bold green]{backend_display}[/bold green]",
        "ML model backend",
    )
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
        "GPU (CUDA) or CPU",
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


# =============================================================================
# Main Result Printer
# =============================================================================


def print_result(
    backend: BackendName,
    result: UnifiedOptimizationResult,
    elapsed: float,
    previous_best: UnifiedHistoryEntry | None = None,
    all_time_best: UnifiedHistoryEntry | None = None,
) -> None:
    """Print optimization result with rich formatting and history comparison.

    Args:
        backend: Backend used for optimization.
        result: Optimization result from any backend.
        elapsed: Elapsed time in seconds.
        previous_best: Previous best history entry, or None.
        all_time_best: All-time best history entry, or None.
    """
    console = get_rich_console()
    console.print()
    console.print(create_rich_panel("[bold green]OPTIMIZATION COMPLETE[/bold green]"))
    console.print()

    result_table = create_result_table(backend, result, elapsed)
    console.print(result_table)
    console.print()

    params_table = create_hyperparams_table(backend, result)
    console.print(params_table)
    console.print()

    # History comparison
    comparison_table = create_history_comparison_table(
        result["best_value"],
        previous_best,
        all_time_best,
    )
    console.print(comparison_table)
    console.print()

    # Print final AUC highlight with improvement indicator
    auc_value = result["best_value"]
    is_new_best = all_time_best is None or auc_value >= all_time_best["best_val_auc"]

    if is_new_best:
        console.print(
            f"[bold white on green] ★ NEW BEST AUC: {auc_value:.4f} ★ [/bold white on green]"
        )
    else:
        console.print(f"[bold white on blue] Best AUC: {auc_value:.4f} [/bold white on blue]")
    console.print()


__all__ = [
    "BACKEND_DISPLAY_NAMES",
    "_format_delta",
    "create_history_comparison_table",
    "create_hyperparams_table",
    "create_result_table",
    "print_config",
    "print_result",
]
