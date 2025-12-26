"""Main entry point for optimization CLI.

Supports all backends (XGBoost, MLP, LightGBM, LSTM) with backend-aware
command line argument handling.

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from platform_core.logging import get_logger, setup_rich_logging

from scripts.optimize.cli import parse_args
from scripts.optimize.display import print_config, print_result
from scripts.optimize.logging_config import set_verbose_mode, suppress_verbose_logging
from scripts.optimize.modes import (
    compare_presets,
    run_all_datasets,
    run_multiple_backends,
    run_single_with_progress,
)
from scripts.optimize.state import managed_execution

logger = get_logger(__name__)


def run(argv: Sequence[str]) -> int:
    """Run optimization with parsed arguments.

    Args:
        argv: Command line arguments.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args(argv)

    set_verbose_mode(args.verbose)
    if args.verbose:
        setup_rich_logging(level="DEBUG", show_time=False)

    suppress_verbose_logging()

    # For all_datasets mode, use first backend only
    first_backend = args.backends[0]

    if args.all_datasets:
        run_all_datasets(
            first_backend,
            args.n_trials,
            args.feature_preset,
            args.device,
            args.timeout,
            args.save_model,
        )
    elif args.compare_presets:
        # Compare presets supports multiple backends
        compare_presets(
            args.backends,
            args.dataset,
            args.n_trials,
            args.device,
            args.timeout,
            args.save_model,
        )
    elif len(args.backends) > 1:
        # Multiple backends specified - run each sequentially
        run_multiple_backends(
            args.backends,
            args.dataset,
            args.n_trials,
            args.feature_preset,
            args.device,
            args.timeout,
            args.save_model,
        )
    else:
        # Single backend
        print_config(
            first_backend,
            args.dataset,
            args.n_trials,
            args.feature_preset,
            args.device,
        )

        run_result = run_single_with_progress(
            first_backend,
            args.dataset,
            args.n_trials,
            args.feature_preset,
            args.device,
            args.timeout,
            args.save_model,
        )

        print_result(
            run_result["backend"],
            run_result["result"],
            run_result["elapsed"],
            run_result["previous_best"],
            run_result["all_time_best"],
        )

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point with proper lifecycle management.

    Args:
        argv: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, 130 for keyboard interrupt).
    """
    from scripts.optimize.state import get_state

    setup_rich_logging(level="INFO", show_time=False)

    raw_args = list(argv) if argv is not None else list(sys.argv[1:])

    try:
        with managed_execution():
            return run(raw_args)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        get_state().print_interrupted_message()
        return 130


__all__ = ["main", "run"]
