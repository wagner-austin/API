"""Main entry point for dataset discovery CLI.

Scans external datasets and generates DatasetConfig entries.
Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

from platform_core.rich_logging import setup_rich_logging

from scripts.discover_datasets import _test_hooks
from scripts.discover_datasets.scanner import scan_external_dir
from scripts.discover_datasets.types import DiscoveredDataset, DiscoverySummary


class ParsedArgs(TypedDict, total=True):
    """Typed container for parsed command line arguments.

    Attributes:
        external_dir: Path to external datasets directory.
        detail: Dataset folder name for detailed view, or empty string.
        generate: Whether to generate DatasetConfig code.
        validate: Whether to validate discovered configs.
        verbose: Whether to enable verbose output.
    """

    external_dir: Path
    detail: str
    generate: bool
    validate: bool
    verbose: bool


class ValidationCounts(TypedDict, total=True):
    """Counts from validation mode.

    Attributes:
        n_pass: Number of datasets that passed validation.
        n_warn: Number of datasets with warnings.
        n_skip: Number of datasets skipped.
    """

    n_pass: int
    n_warn: int
    n_skip: int


def _format_dataset_row(ds: DiscoveredDataset) -> str:
    """Format a single dataset as a table row.

    Args:
        ds: Discovered dataset result.

    Returns:
        Formatted string for table display.
    """
    status_colors = {
        "success": "green",
        "warning": "yellow",
        "error": "red",
    }
    color = status_colors.get(ds["status"], "white")

    return (
        f"[{color}]{ds['folder_name']:<30}[/{color}] "
        f"{ds['file_format']:<6} "
        f"{ds['n_rows']:>8,} rows "
        f"{ds['n_columns']:>4} cols "
        f"[{color}]{ds['status']:<8}[/{color}] "
        f"{ds['message']}"
    )


def _print_summary(summary: DiscoverySummary) -> None:
    """Print discovery summary to console.

    Args:
        summary: Discovery summary to print.
    """
    console = _test_hooks.console_factory()

    console.print("\n[bold cyan]Dataset Discovery Results[/bold cyan]\n")
    console.print(
        f"[bold]Total:[/bold] {summary['n_total']} datasets | "
        f"[green]Success: {summary['n_success']}[/green] | "
        f"[yellow]Warnings: {summary['n_warning']}[/yellow] | "
        f"[red]Errors: {summary['n_error']}[/red]\n"
    )

    console.print("[bold]Datasets:[/bold]")
    console.print("-" * 100)

    for ds in summary["datasets"]:
        console.print(_format_dataset_row(ds))

    console.print("-" * 100)


def _print_dataset_detail(ds: DiscoveredDataset) -> None:
    """Print detailed information about a dataset.

    Args:
        ds: Discovered dataset to print details for.
    """
    console = _test_hooks.console_factory()

    console.print(f"\n[bold cyan]{ds['folder_name']}[/bold cyan]")
    console.print(f"  File: {ds['file_name']} ({ds['file_format']})")
    console.print(f"  Encoding: {ds['encoding']}")
    console.print(f"  Size: {ds['n_rows']:,} rows x {ds['n_columns']} columns")

    if len(ds["target_candidates"]) > 0:
        console.print("  Target candidates:")
        for candidate in ds["target_candidates"]:
            binary_str = "[binary]" if candidate["is_binary"] else ""
            console.print(
                f"    - {candidate['column_name']}: "
                f"{candidate['n_unique']} unique values {binary_str}"
            )
            if len(candidate["unique_values"]) > 0:
                values_str = ", ".join(candidate["unique_values"][:5])
                console.print(f"      Values: {values_str}")

    if ds["recommended_target"]:
        console.print(f"  [green]Recommended target: {ds['recommended_target']}[/green]")

    if len(ds["recommended_exclude"]) > 0:
        console.print(f"  Exclude columns: {', '.join(ds['recommended_exclude'])}")


def _print_validation(ds: DiscoveredDataset) -> None:
    """Print validation details for a dataset config.

    Args:
        ds: Discovered dataset to validate.
    """
    console = _test_hooks.console_factory()

    # Skip errors
    if ds["status"] == "error":
        console.print(f"[red]SKIP[/red] {ds['folder_name']}: {ds['message']}")
        return

    # Skip no target
    if not ds["recommended_target"]:
        console.print(f"[yellow]SKIP[/yellow] {ds['folder_name']}: No target column")
        return

    # Check config completeness
    issues: list[str] = []

    if not ds["target_positive_value"]:
        issues.append("Missing positive_value")
    if not ds["target_negative_value"]:
        issues.append("Missing negative_value")
    if ds["positive_class_ratio"] == 0.0 and ds["target_positive_value"]:
        issues.append("Ratio is 0.0 (may be incorrect)")

    # Print result
    if len(issues) == 0:
        console.print(f"[green]PASS[/green] {ds['folder_name']}")
        console.print(f"       Target: {ds['recommended_target']}")
        console.print(
            f"       Values: pos={ds['target_positive_value']!r}, "
            f"neg={ds['target_negative_value']!r}"
        )
        console.print(
            f"       Type: {ds['target_label_type']}, Ratio: {ds['positive_class_ratio']:.1%}"
        )
        if len(ds["recommended_exclude"]) > 0:
            console.print(f"       Exclude: {', '.join(ds['recommended_exclude'])}")
    else:
        console.print(f"[yellow]WARN[/yellow] {ds['folder_name']}")
        console.print(f"       Target: {ds['recommended_target']}")
        console.print(
            f"       Values: pos={ds['target_positive_value']!r}, "
            f"neg={ds['target_negative_value']!r}"
        )
        for issue in issues:
            console.print(f"       [yellow]! {issue}[/yellow]")

    console.print()


def _is_valid_numeric(value: str) -> bool:
    """Check if a string represents a valid numeric value.

    Args:
        value: String to check.

    Returns:
        True if the string is a valid numeric representation.
    """
    if not value:
        return False
    # Remove one decimal point and leading minus sign, then check if remaining is digits
    cleaned = value.replace(".", "", 1).lstrip("-")
    return cleaned.isdigit()


def _format_value_tuple(
    pos_val: str,
    neg_val: str,
    label_type: Literal["binary_int", "binary_str"],
) -> tuple[str, str]:
    """Format positive and negative values as Python tuple strings.

    Args:
        pos_val: Positive class value.
        neg_val: Negative class value.
        label_type: Type of the label (binary_int or binary_str).

    Returns:
        Tuple of (positive_values_str, negative_values_str) as Python code.
    """
    if label_type == "binary_int":
        # Convert to int for cleaner output if valid numeric
        if _is_valid_numeric(pos_val) and _is_valid_numeric(neg_val):
            pos_int = int(float(pos_val))
            neg_int = int(float(neg_val))
            return f"({pos_int},)", f"({neg_int},)"
        pos_str = f'("{pos_val}",)' if pos_val else "(1,)"
        neg_str = f'("{neg_val}",)' if neg_val else "(0,)"
        return pos_str, neg_str

    pos_str = f'("{pos_val}",)' if pos_val else '("",)'
    neg_str = f'("{neg_val}",)' if neg_val else '("",)'
    return pos_str, neg_str


def _generate_config_code(ds: DiscoveredDataset) -> str:
    """Generate Python code for a DatasetConfig.

    Args:
        ds: Discovered dataset to generate config for.

    Returns:
        Python code string for the config.
    """
    if ds["status"] == "error" or not ds["recommended_target"]:
        return f"# Skipped {ds['folder_name']}: {ds['message']}"

    target_col = ds["recommended_target"]
    label_type = ds["target_label_type"]
    pos_val = ds["target_positive_value"]
    neg_val = ds["target_negative_value"]
    pos_ratio = ds["positive_class_ratio"]

    pos_values_str, neg_values_str = _format_value_tuple(pos_val, neg_val, label_type)

    # Build exclude columns tuple
    if len(ds["recommended_exclude"]) > 0:
        exclude_str = ", ".join(f'"{col}"' for col in ds["recommended_exclude"])
        exclude_tuple = f"({exclude_str},)"
    else:
        exclude_tuple = "()"

    # Calculate n_features (subtract target and exclude columns)
    n_features = ds["n_columns"] - 1 - len(ds["recommended_exclude"])

    return f'''DatasetConfig(
    name="{ds["folder_name"]}",
    display_name="{ds["folder_name"].replace("_", " ").title()}",
    folder="{ds["folder_name"]}",
    file_name="{ds["file_name"]}",
    file_format="{ds["file_format"]}",
    encoding="{ds["encoding"]}",
    target=TargetColumnSpec(
        column_name="{target_col}",
        label_type="{label_type}",
        positive_values={pos_values_str},
        negative_values={neg_values_str},
    ),
    exclude_columns={exclude_tuple},
    n_samples_expected={ds["n_rows"]},
    n_features_expected={n_features},
    positive_class_ratio_expected={pos_ratio},
),'''


def parse_args(argv: Sequence[str]) -> ParsedArgs:
    """Parse command line arguments.

    Args:
        argv: Command line arguments.

    Returns:
        Typed dictionary with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="discover_datasets",
        description="Scan external datasets and generate DatasetConfig entries",
    )

    parser.add_argument(
        "--external-dir",
        type=Path,
        default=Path("data/external"),
        help="Path to external datasets directory (default: data/external)",
    )

    parser.add_argument(
        "--detail",
        type=str,
        default="",
        help="Show detailed info for a specific dataset folder",
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate DatasetConfig Python code",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate discovered configs (check target, values, ratios)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    ns = parser.parse_args(argv)

    # Extract typed values from namespace
    external_dir_val: Path = ns.external_dir
    detail_val: str = ns.detail
    generate_val: bool = ns.generate
    validate_val: bool = ns.validate
    verbose_val: bool = ns.verbose

    return ParsedArgs(
        external_dir=external_dir_val,
        detail=detail_val,
        generate=generate_val,
        validate=validate_val,
        verbose=verbose_val,
    )


def _classify_dataset_for_validation(ds: DiscoveredDataset) -> Literal["pass", "warn", "skip"]:
    """Classify a dataset for validation counts.

    Args:
        ds: Discovered dataset to classify.

    Returns:
        Classification as 'pass', 'warn', or 'skip'.
    """
    if ds["status"] == "error" or not ds["recommended_target"]:
        return "skip"
    has_pos = bool(ds["target_positive_value"])
    has_neg = bool(ds["target_negative_value"])
    has_ratio = ds["positive_class_ratio"] > 0.0
    if has_pos and has_neg and has_ratio:
        return "pass"
    return "warn"


def _run_detail_mode(summary: DiscoverySummary, detail_name: str) -> int:
    """Run in detail mode showing info for a specific dataset.

    Args:
        summary: Discovery summary with all datasets.
        detail_name: Name of the dataset folder to show details for.

    Returns:
        Exit code (0 for success, 1 if dataset not found).
    """
    console = _test_hooks.console_factory()
    for ds in summary["datasets"]:
        if ds["folder_name"] == detail_name:
            _print_dataset_detail(ds)
            return 0
    console.print(f"[red]Dataset not found:[/red] {detail_name}")
    return 1


def _run_validate_mode(summary: DiscoverySummary) -> None:
    """Run in validation mode showing config validation for all datasets.

    Args:
        summary: Discovery summary with all datasets.
    """
    console = _test_hooks.console_factory()
    console.print("\n[bold cyan]Config Validation:[/bold cyan]\n")

    counts = ValidationCounts(n_pass=0, n_warn=0, n_skip=0)
    for ds in summary["datasets"]:
        classification = _classify_dataset_for_validation(ds)
        if classification == "pass":
            counts["n_pass"] += 1
        elif classification == "warn":
            counts["n_warn"] += 1
        else:
            counts["n_skip"] += 1
        _print_validation(ds)

    console.print("-" * 60)
    console.print(
        f"[bold]Summary:[/bold] "
        f"[green]{counts['n_pass']} PASS[/green] | "
        f"[yellow]{counts['n_warn']} WARN[/yellow] | "
        f"[dim]{counts['n_skip']} SKIP[/dim]"
    )


def _run_generate_mode(summary: DiscoverySummary) -> None:
    """Run in generate mode printing DatasetConfig code for all datasets.

    Args:
        summary: Discovery summary with all datasets.
    """
    console = _test_hooks.console_factory()
    console.print("\n[bold cyan]Generated DatasetConfig entries:[/bold cyan]\n")
    for ds in summary["datasets"]:
        code = _generate_config_code(ds)
        console.print(code)
        console.print()


def run(argv: Sequence[str]) -> int:
    """Run discovery with parsed arguments.

    Args:
        argv: Command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args(argv)
    console = _test_hooks.console_factory()

    if args["verbose"]:
        setup_rich_logging(level="DEBUG", show_time=False)

    external_dir = args["external_dir"].resolve()

    if not external_dir.exists():
        console.print(f"[red]Error:[/red] Directory not found: {external_dir}")
        return 1

    console.print(f"[cyan]Scanning:[/cyan] {external_dir}")
    summary = scan_external_dir(external_dir)

    if args["detail"]:
        return _run_detail_mode(summary, args["detail"])

    if args["validate"]:
        _run_validate_mode(summary)
        return 0

    if args["generate"]:
        _run_generate_mode(summary)
        return 0

    _print_summary(summary)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for error).

    Raises:
        FileNotFoundError: If a file cannot be found during processing.
        KeyboardInterrupt: If user interrupts execution.
    """
    setup_rich_logging(level="INFO", show_time=False)

    raw_args = list(argv) if argv is not None else list(sys.argv[1:])

    return run(raw_args)


__all__ = ["main", "run"]
