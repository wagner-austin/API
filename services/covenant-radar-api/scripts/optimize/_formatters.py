"""Progress display formatters for optimization output.

Contains formatting utilities for trial progress, loading progress,
and elapsed time display. Used by runners to show real-time feedback.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import scripts._test_hooks as _hooks


def format_elapsed(seconds: float) -> str:
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


def format_xgboost_progress(info: _hooks.XGBoostProgressInfo, elapsed: float = 0.0) -> str:
    """Format XGBoost trial progress for display.

    Args:
        info (XGBoostProgressInfo): XGBoost trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        str: Rich-formatted progress string with trial number, AUC, and hyperparameters.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker} "
        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
        f"Est: [blue]{info['best_n_estimators']}[/blue]"
    )


def format_mlp_progress(info: _hooks.MLPTrialProgressInfo, elapsed: float = 0.0) -> str:
    """Format MLP trial progress for display.

    Args:
        info (MLPTrialProgressInfo): MLP trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        str: Rich-formatted progress string with trial number, AUC, and hyperparameters.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker} "
        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
        f"Layers: [magenta]{info['best_n_layers']}[/magenta] "
        f"Hidden: [blue]{info['best_hidden_size']}[/blue]"
    )


def format_lightgbm_progress(info: _hooks.LightGBMTrialProgressInfo, elapsed: float = 0.0) -> str:
    """Format LightGBM trial progress for display.

    Args:
        info (LightGBMTrialProgressInfo): LightGBM trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        str: Rich-formatted progress string with trial number, AUC, and hyperparameters.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker} "
        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
        f"Leaves: [magenta]{info['best_num_leaves']}[/magenta] "
        f"Est: [blue]{info['best_n_estimators']}[/blue]"
    )


def format_lstm_progress(info: _hooks.LSTMTrialProgressInfo, elapsed: float = 0.0) -> str:
    """Format LSTM trial progress for display.

    Args:
        info (LSTMTrialProgressInfo): LSTM trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        str: Rich-formatted progress string with trial number, AUC, and hyperparameters.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker} "
        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
        f"Layers: [magenta]{info['best_num_layers']}[/magenta] "
        f"Hidden: [blue]{info['best_hidden_size']}[/blue]"
    )


def format_cleargbm_progress(info: _hooks.ClearGBMTrialProgressInfo, elapsed: float = 0.0) -> str:
    """Format ClearGBM trial progress for display.

    Args:
        info (ClearGBMTrialProgressInfo): ClearGBM trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        str: Rich-formatted progress string with trial number, AUC, and hyperparameters.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_auc']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker} "
        f"LR: [yellow]{info['best_learning_rate']:.4f}[/yellow] "
        f"Depth: [magenta]{info['best_max_depth']}[/magenta] "
        f"Est: [blue]{info['best_n_estimators']}[/blue]"
    )


def format_loading_progress(
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
    elapsed_str = f"[dim]{format_elapsed(elapsed)}[/dim]"
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


__all__ = [
    "format_cleargbm_progress",
    "format_elapsed",
    "format_lightgbm_progress",
    "format_loading_progress",
    "format_lstm_progress",
    "format_mlp_progress",
    "format_xgboost_progress",
]
