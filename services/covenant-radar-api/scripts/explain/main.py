"""Main entry point for explain CLI.

Runs feature importance explanation on trained models using the
pluggable explainer registry.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core.logging import get_logger
from platform_core.rich_logging import get_rich_console, setup_rich_logging

from scripts.explain.cli import parse_args
from scripts.explain.display import print_config, print_result
from scripts.explain.runner import run_explanation

logger = get_logger(__name__)


def run(argv: Sequence[str]) -> int:
    """Run explanation with parsed arguments.

    Args:
        argv: Command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args(argv)
    console = get_rich_console()

    if args.verbose:
        setup_rich_logging(level="DEBUG", show_time=False)

    print_config(
        args.backend,
        args.dataset,
        args.explainer,
        args.n_samples,
        args.model_path,
    )

    console.print("[cyan]Running explanation...[/cyan]")

    result = run_explanation(
        backend=args.backend,
        dataset=args.dataset,
        explainer=args.explainer,
        model_path=args.model_path,
        n_samples=args.n_samples,
        target_class=args.target_class,
    )

    print_result(result, args.top_n)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point with error handling.

    Args:
        argv: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, 1 for error, 130 for keyboard interrupt).
    """
    setup_rich_logging(level="INFO", show_time=False)
    console = get_rich_console()

    raw_args = list(argv) if argv is not None else list(sys.argv[1:])

    try:
        return run(raw_args)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error("Model file not found", extra={"error": str(e)})
        return 1
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error("Invalid configuration", extra={"error": str(e)})
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        logger.info("Keyboard interrupt received")
        return 130


__all__ = ["main", "run"]
