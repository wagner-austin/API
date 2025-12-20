"""Rich console display formatting for multi-backend optimization output.

Supports all backends (XGBoost, MLP, LightGBM, LSTM) with backend-specific
hyperparameter display and result formatting.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from covenant_ml.types import BackendName
from platform_core.logging import (
    RichTableProtocol,
    create_rich_panel,
    create_rich_table,
    get_rich_console,
)

from scripts._test_hooks import (
    LightGBMOptimizationResult,
    LSTMOptimizationResult,
    MLPOptimizationResult,
    XGBoostOptimizationResult,
)
from scripts.optimize.cli import PRESET_DESCRIPTIONS, DatasetName, FeaturePreset
from scripts.optimize.history import UnifiedHistoryEntry
from scripts.optimize.runner import UnifiedOptimizationResult

# =============================================================================
# Backend Display Names
# =============================================================================

BACKEND_DISPLAY_NAMES: dict[BackendName, str] = {
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
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
        backend (BackendName): Backend used for optimization.
        result (UnifiedOptimizationResult): Optimization result from any backend.
        elapsed (float): Elapsed time in seconds.

    Returns:
        RichTableProtocol: Rich table with result summary.
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


# =============================================================================
# Backend-Specific Hyperparameter Tables
# =============================================================================


def _create_xgboost_hyperparams_table(result: XGBoostOptimizationResult) -> RichTableProtocol:
    """Create hyperparameters table for XGBoost results.

    Args:
        result (XGBoostOptimizationResult): XGBoost optimization result.

    Returns:
        RichTableProtocol: Rich table with XGBoost hyperparameters.
    """
    table = create_rich_table(
        title="[bold blue]Best Hyperparameters (XGBoost)[/bold blue]",
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


def _create_mlp_hyperparams_table(result: MLPOptimizationResult) -> RichTableProtocol:
    """Create hyperparameters table for MLP results.

    Args:
        result (MLPOptimizationResult): MLP optimization result.

    Returns:
        RichTableProtocol: Rich table with MLP hyperparameters.
    """
    table = create_rich_table(
        title="[bold blue]Best Hyperparameters (MLP)[/bold blue]",
        show_header=True,
    )
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="bold yellow", justify="right")

    # Architecture params
    table.add_row("[green]n_layers[/green]", f"[bold]{result['best_n_layers']}[/bold]")
    table.add_row("[green]hidden_size[/green]", f"[bold]{result['best_hidden_size']}[/bold]")

    # Learning params
    table.add_row("[magenta]learning_rate[/magenta]", f"{result['best_learning_rate']:.6f}")

    # Regularization
    table.add_row("[yellow]dropout[/yellow]", f"{result['best_dropout']:.4f}")

    # Training params
    table.add_row("[blue]batch_size[/blue]", f"[bold]{result['best_batch_size']}[/bold]")

    return table


def _create_lightgbm_hyperparams_table(
    result: LightGBMOptimizationResult,
) -> RichTableProtocol:
    """Create hyperparameters table for LightGBM results.

    Note: max_depth is fixed at -1 (unlimited) and not displayed. LightGBM uses
    leaf-wise growth where num_leaves is the primary complexity control.

    Args:
        result (LightGBMOptimizationResult): LightGBM optimization result.

    Returns:
        RichTableProtocol: Rich table with LightGBM hyperparameters.
    """
    table = create_rich_table(
        title="[bold blue]Best Hyperparameters (LightGBM)[/bold blue]",
        show_header=True,
    )
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="bold yellow", justify="right")

    # Tree structure params (max_depth fixed at -1, num_leaves controls complexity)
    table.add_row("[green]num_leaves[/green]", f"[bold]{result['best_num_leaves']}[/bold]")
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


def _create_lstm_hyperparams_table(result: LSTMOptimizationResult) -> RichTableProtocol:
    """Create hyperparameters table for LSTM results.

    Args:
        result (LSTMOptimizationResult): LSTM optimization result.

    Returns:
        RichTableProtocol: Rich table with LSTM hyperparameters.
    """
    table = create_rich_table(
        title="[bold blue]Best Hyperparameters (LSTM)[/bold blue]",
        show_header=True,
    )
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="bold yellow", justify="right")

    # Architecture params
    table.add_row("[green]num_layers[/green]", f"[bold]{result['best_num_layers']}[/bold]")
    table.add_row("[green]hidden_size[/green]", f"[bold]{result['best_hidden_size']}[/bold]")

    # Learning params
    table.add_row("[magenta]learning_rate[/magenta]", f"{result['best_learning_rate']:.6f}")

    # Regularization
    table.add_row("[yellow]dropout[/yellow]", f"{result['best_dropout']:.4f}")

    # Training params
    table.add_row("[blue]batch_size[/blue]", f"[bold]{result['best_batch_size']}[/bold]")

    return table


def create_hyperparams_table(
    backend: BackendName,
    result: UnifiedOptimizationResult,
) -> RichTableProtocol:
    """Create a rich table for best hyperparameters based on backend.

    Uses discriminated union pattern - each result type has a `backend`
    field with a Literal type that mypy uses for type narrowing.

    Args:
        backend (BackendName): Backend used for optimization.
        result (UnifiedOptimizationResult): Optimization result with backend
            discriminator field.

    Returns:
        RichTableProtocol: Rich table with backend-specific hyperparameters.
    """
    # Use backend parameter to ensure it's used (avoids unused argument warning)
    _ = backend
    if result["backend"] == "xgboost":
        return _create_xgboost_hyperparams_table(result)
    if result["backend"] == "mlp":
        return _create_mlp_hyperparams_table(result)
    if result["backend"] == "lightgbm":
        return _create_lightgbm_hyperparams_table(result)
    # result["backend"] must be "lstm" here - mypy validates exhaustiveness via return type
    return _create_lstm_hyperparams_table(result)


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
        current_auc (float): Current run's best AUC score.
        previous_best (UnifiedHistoryEntry | None): Previous best history entry,
            or None if no previous runs exist.
        all_time_best (UnifiedHistoryEntry | None): All-time best history entry,
            or None if this is the first run.

    Returns:
        RichTableProtocol: Rich table with run comparison.
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
        delta (float): AUC delta value (current - baseline).

    Returns:
        str: Rich-formatted delta string with color (green positive, red negative).
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
        backend (BackendName): Backend to use (xgboost, mlp, lightgbm, lstm).
        dataset (DatasetName): Dataset name (taiwan, us, polish).
        n_trials (int): Number of Optuna trials to run.
        feature_preset (FeaturePreset): Feature engineering preset.
        device (str): Device for training (cuda/cpu/auto).
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
        backend (BackendName): Backend used for optimization.
        result (UnifiedOptimizationResult): Optimization result from any backend.
        elapsed (float): Elapsed time in seconds.
        previous_best (UnifiedHistoryEntry | None): Previous best history entry,
            or None if no previous runs exist. Defaults to None.
        all_time_best (UnifiedHistoryEntry | None): All-time best history entry,
            or None if this is the first run. Defaults to None.
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
        result["best_val_auc"],
        previous_best,
        all_time_best,
    )
    console.print(comparison_table)
    console.print()

    # Print final AUC highlight with improvement indicator
    auc_value = result["best_val_auc"]
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
