"""Rich console display formatting for explanation output.

Provides formatted output for feature importance results with
ranking tables and summary panels.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from covenant_ml.explainers.types import SupportedExplainer
from covenant_ml.types import BackendName
from platform_core.logging import (
    RichTableProtocol,
    create_rich_panel,
    create_rich_table,
    get_rich_console,
)

from scripts.explain.cli import EXPLAINER_DESCRIPTIONS, DatasetName
from scripts.explain.runner import ExplainRunResult

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
# Configuration Display
# =============================================================================


def print_config(
    backend: BackendName,
    dataset: DatasetName,
    explainer: SupportedExplainer,
    n_samples: int,
    model_path: str | None,
) -> None:
    """Print explanation configuration panel.

    Args:
        backend: ML backend type.
        dataset: Dataset name.
        explainer: Explainer method.
        n_samples: Number of samples.
        model_path: Model path or None for default.
    """
    console = get_rich_console()
    backend_display = BACKEND_DISPLAY_NAMES.get(backend, backend.upper())
    explainer_desc = EXPLAINER_DESCRIPTIONS.get(explainer, explainer)
    model_display = model_path if model_path else "(default best model)"

    config_text = f"""[bold cyan]Backend:[/bold cyan] {backend_display}
[bold cyan]Dataset:[/bold cyan] {dataset.upper()}
[bold cyan]Explainer:[/bold cyan] {explainer} - {explainer_desc}
[bold cyan]Samples:[/bold cyan] {n_samples:,}
[bold cyan]Model:[/bold cyan] {model_display}"""

    panel = create_rich_panel(
        config_text,
        title="[bold magenta]Feature Importance Explanation[/bold magenta]",
    )
    console.print(panel)
    console.print()


# =============================================================================
# Result Summary Table
# =============================================================================


def _create_summary_table(result: ExplainRunResult) -> RichTableProtocol:
    """Create summary table for explanation results.

    Args:
        result: Explanation run result.

    Returns:
        Rich table with summary information.
    """
    table = create_rich_table(
        title="[bold magenta]Explanation Summary[/bold magenta]",
        show_header=False,
    )
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")

    backend_display = BACKEND_DISPLAY_NAMES.get(result["backend"], result["backend"].upper())

    table.add_row("[cyan]Backend[/cyan]", f"[bold green]{backend_display}[/bold green]")
    table.add_row("[cyan]Dataset[/cyan]", f"[bold yellow]{result['dataset'].upper()}[/bold yellow]")
    table.add_row("[cyan]Explainer[/cyan]", f"[magenta]{result['explainer']}[/magenta]")
    table.add_row(
        "[cyan]Samples Used[/cyan]", f"[blue]{result['result']['n_samples_used']:,}[/blue]"
    )
    table.add_row("[cyan]Features[/cyan]", f"[blue]{result['result']['n_features']:,}[/blue]")
    table.add_row(
        "[cyan]Target Class[/cyan]", f"[yellow]{result['result']['target_class']}[/yellow]"
    )
    table.add_row("[cyan]Time[/cyan]", f"[dim]{result['elapsed']:.1f}s[/dim]")
    table.add_row("[cyan]Model[/cyan]", f"[dim]{result['model_path']}[/dim]")

    return table


# =============================================================================
# Feature Importance Table
# =============================================================================


def _create_importance_table(result: ExplainRunResult, top_n: int) -> RichTableProtocol:
    """Create feature importance ranking table.

    Args:
        result: Explanation run result.
        top_n: Number of top features to show.

    Returns:
        Rich table with feature importance rankings.
    """
    table = create_rich_table(
        title=f"[bold magenta]Top {top_n} Feature Importances[/bold magenta]",
        show_header=True,
    )
    table.add_column("Rank", style="bold yellow", justify="right")
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", style="green", justify="right")
    table.add_column("Bar", style="blue")

    importances = result["result"]["feature_importances"]

    # Get max importance for bar scaling
    if not importances:
        return table

    max_importance = max(float(fi["importance"]) for fi in importances[:top_n])
    bar_width = 30

    for i, fi in enumerate(importances[:top_n]):
        rank = i + 1
        feature_name = fi["name"]
        importance = float(fi["importance"])

        # Create visual bar
        bar_len = int((importance / max_importance) * bar_width) if max_importance > 0 else 0
        bar = "█" * bar_len

        # Format importance value
        imp_str = f"{importance:.4f}" if importance >= 0.01 else f"{importance:.2e}"

        table.add_row(str(rank), feature_name, imp_str, bar)

    return table


# =============================================================================
# Main Display Function
# =============================================================================


def print_result(result: ExplainRunResult, top_n: int) -> None:
    """Print explanation results with summary and feature table.

    Args:
        result: Explanation run result.
        top_n: Number of top features to show.
    """
    console = get_rich_console()
    console.print()

    # Print summary table
    summary_table = _create_summary_table(result)
    console.print(summary_table)
    console.print()

    # Print feature importance table
    importance_table = _create_importance_table(result, top_n)
    console.print(importance_table)
    console.print()

    # Print completion message
    n_features = result["result"]["n_features"]
    n_shown = min(top_n, len(result["result"]["feature_importances"]))
    console.print(
        f"[green]✓[/green] Computed importance for {n_features} features, showing top {n_shown}."
    )


__all__ = [
    "BACKEND_DISPLAY_NAMES",
    "print_config",
    "print_result",
]
