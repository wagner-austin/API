"""Progress display formatters for optimization output.

Contains formatting utilities for trial progress, loading progress,
and elapsed time display. Used by runners to show real-time feedback.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from scripts._test_hooks import TrialProgressInfo


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


def format_trial_progress(info: TrialProgressInfo, elapsed: float = 0.0) -> str:
    """Format trial progress for display (any backend).

    Args:
        info: Trial progress information dict.
        elapsed: Elapsed time in seconds.

    Returns:
        Rich-formatted progress string with trial number and AUC.
    """
    best_marker = "[yellow]*[/yellow]" if info["is_best"] else ""
    elapsed_str = f"[dim]{format_elapsed(elapsed)}[/dim] "
    return (
        f"{elapsed_str}"
        f"[cyan]Trial {info['trial_number'] + 1}/{info['n_trials_total']}[/cyan] "
        f"Best: [bold green]{info['best_value']:.4f}[/bold green] "
        f"(#{info['best_trial']}) {best_marker}"
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
    "format_elapsed",
    "format_loading_progress",
    "format_trial_progress",
]
