"""CLI entry point for AMEX competition ensemble pipeline.

Usage:
    python -m scripts.amex [OPTIONS]

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

from covenant_ml.types import BackendName
from platform_core.logging import setup_rich_logging

from scripts.amex._hooks import configure_real_scipy, get_console, get_project_root
from scripts.amex.pipeline import run_pipeline
from scripts.amex.types import AMEXPipelineConfig

# =============================================================================
# Types
# =============================================================================


class ParsedArgs(TypedDict, total=True):
    """Parsed command-line arguments.

    Attributes:
        backends: ML backends to use.
        n_folds: Number of CV folds.
        n_estimators: Number of boosting rounds.
        learning_rate: Learning rate.
        aggregation: Time-series aggregation strategy.
        include_rank_features: Whether to include rank features.
        include_diff_features: Whether to include diff features.
        include_window_features: Whether to include window features.
        window_sizes: Window sizes for window features.
        random_state: Random seed.
        train_dir: Training data directory.
        test_dir: Test data directory.
        output_path: Output submission file path.
    """

    backends: tuple[BackendName, ...]
    n_folds: int
    n_estimators: int
    learning_rate: float
    aggregation: Literal["last", "first", "mean", "statistics"]
    include_rank_features: bool
    include_diff_features: bool
    include_window_features: bool
    window_sizes: tuple[int, ...]
    random_state: int
    train_dir: Path
    test_dir: Path
    output_path: Path


class _ArgState:
    """Mutable argument parsing state."""

    backends: tuple[BackendName, ...]
    n_folds: int
    n_estimators: int
    learning_rate: float
    aggregation: Literal["last", "first", "mean", "statistics"]
    include_rank_features: bool
    include_diff_features: bool
    include_window_features: bool
    window_sizes: tuple[int, ...]
    random_state: int
    train_dir: Path
    test_dir: Path
    output_path: Path

    def __init__(self, project_root: Path) -> None:
        """Initialize with defaults."""
        self.backends = ("lightgbm", "xgboost")
        self.n_folds = 5
        self.n_estimators = 1000
        self.learning_rate = 0.05
        self.aggregation = "statistics"
        self.include_rank_features = True
        self.include_diff_features = True
        self.include_window_features = True
        self.window_sizes = (3, 6)
        self.random_state = 42
        self.train_dir = project_root / "data" / "external" / "amex_train"
        self.test_dir = project_root / "data" / "external" / "amex_test"
        self.output_path = project_root / "data" / "submissions" / "amex_submission.csv"

    def to_parsed_args(self) -> ParsedArgs:
        """Convert to immutable ParsedArgs."""
        return ParsedArgs(
            backends=self.backends,
            n_folds=self.n_folds,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            aggregation=self.aggregation,
            include_rank_features=self.include_rank_features,
            include_diff_features=self.include_diff_features,
            include_window_features=self.include_window_features,
            window_sizes=self.window_sizes,
            random_state=self.random_state,
            train_dir=self.train_dir,
            test_dir=self.test_dir,
            output_path=self.output_path,
        )


# =============================================================================
# Argument Parsing
# =============================================================================


def _parse_backends(val: str) -> tuple[BackendName, ...]:
    """Parse comma-separated backend names.

    Args:
        val: Comma-separated backend names.

    Returns:
        Tuple of validated backend names.

    Raises:
        SystemExit: If any backend name is invalid.
    """
    result: list[BackendName] = []
    for name in val.split(","):
        name = name.strip()
        if name == "lightgbm":
            result.append("lightgbm")
        elif name == "xgboost":
            result.append("xgboost")
        else:
            console = get_console()
            console.write(f"Invalid backend: {name}. Must be lightgbm, xgboost.")
            raise SystemExit(1)
    return tuple(result)


def _parse_aggregation(val: str) -> Literal["last", "first", "mean", "statistics"]:
    """Parse aggregation strategy value.

    Args:
        val: Aggregation string from CLI.

    Returns:
        Validated aggregation literal.

    Raises:
        SystemExit: If aggregation is invalid.
    """
    if val == "last":
        return "last"
    if val == "first":
        return "first"
    if val == "mean":
        return "mean"
    if val == "statistics":
        return "statistics"
    console = get_console()
    console.write(f"Invalid aggregation: {val}. Must be last, first, mean, statistics.")
    raise SystemExit(1)


def _parse_window_sizes(val: str) -> tuple[int, ...]:
    """Parse comma-separated window sizes.

    Args:
        val: Comma-separated integers.

    Returns:
        Tuple of window sizes.

    Raises:
        SystemExit: If parsing fails.
    """
    result: list[int] = []
    for s in val.split(","):
        s = s.strip()
        result.append(int(s))
    return tuple(result)


def print_help() -> None:
    """Print help message with all available options."""
    console = get_console()
    help_text = """
Usage: python -m scripts.amex [OPTIONS]

AMEX Competition Ensemble Pipeline
===================================
Trains multiple models with k-fold CV, optimizes ensemble weights,
and generates submission predictions.

Options:
  -b, --backends           Backends: lightgbm,xgboost (default: lightgbm,xgboost)
  -k, --n-folds            Number of CV folds (default: 5)
  -n, --n-estimators       Number of boosting rounds (default: 1000)
  -l, --learning-rate      Learning rate (default: 0.05)
  -a, --aggregation        Aggregation: last, first, mean, statistics (default: statistics)
  -w, --window-sizes       Window sizes: e.g., 3,6 (default: 3,6)
  -s, --random-state       Random seed (default: 42)
  --no-rank-features       Disable per-entity rank features
  --no-diff-features       Disable row-to-row diff features
  --no-window-features     Disable window features
  --train-dir              Training data directory
  --test-dir               Test data directory
  -o, --output             Output submission file path
  -h, --help               Show this help

Examples:
  python -m scripts.amex
  python -m scripts.amex -b lightgbm,xgboost -k 5 -n 1000
  python -m scripts.amex -a statistics --no-window-features
"""
    console.write(help_text)


def _handle_model_args(state: _ArgState, arg: str, next_val: str) -> bool:
    """Handle model-related arguments.

    Args:
        state: Mutable argument state.
        arg: Current argument.
        next_val: Next argument value.

    Returns:
        True if argument was handled.
    """
    if arg in ("--backends", "-b"):
        state.backends = _parse_backends(next_val)
        return True
    if arg in ("--n-folds", "-k"):
        state.n_folds = int(next_val)
        return True
    if arg in ("--n-estimators", "-n"):
        state.n_estimators = int(next_val)
        return True
    if arg in ("--learning-rate", "-l"):
        state.learning_rate = float(next_val)
        return True
    if arg in ("--random-state", "-s"):
        state.random_state = int(next_val)
        return True
    return False


def _handle_feature_and_path_args(state: _ArgState, arg: str, next_val: str) -> bool:
    """Handle feature and path arguments.

    Args:
        state: Mutable argument state.
        arg: Current argument.
        next_val: Next argument value.

    Returns:
        True if argument was handled.
    """
    if arg in ("--aggregation", "-a"):
        state.aggregation = _parse_aggregation(next_val)
        return True
    if arg in ("--window-sizes", "-w"):
        state.window_sizes = _parse_window_sizes(next_val)
        return True
    if arg == "--train-dir":
        state.train_dir = Path(next_val)
        return True
    if arg == "--test-dir":
        state.test_dir = Path(next_val)
        return True
    if arg in ("--output", "-o"):
        state.output_path = Path(next_val)
        return True
    return False


def _handle_value_arg(state: _ArgState, arg: str, next_val: str) -> bool:
    """Handle value-based arguments.

    Args:
        state: Mutable argument state.
        arg: Current argument.
        next_val: Next argument value.

    Returns:
        True if argument was handled.
    """
    if _handle_model_args(state, arg, next_val):
        return True
    return _handle_feature_and_path_args(state, arg, next_val)


def _handle_flag_arg(state: _ArgState, arg: str) -> bool:
    """Handle flag-based arguments.

    Args:
        state: Mutable argument state.
        arg: Current argument.

    Returns:
        True if argument was handled.

    Raises:
        SystemExit: If --help flag.
    """
    if arg in ("--help", "-h"):
        print_help()
        raise SystemExit(0)
    if arg == "--no-rank-features":
        state.include_rank_features = False
        return True
    if arg == "--no-diff-features":
        state.include_diff_features = False
        return True
    if arg == "--no-window-features":
        state.include_window_features = False
        return True
    return False


def parse_args(argv: Sequence[str]) -> ParsedArgs:
    """Parse command-line arguments.

    Args:
        argv: Command-line argument sequence.

    Returns:
        ParsedArgs with all settings.

    Raises:
        SystemExit: If --help is specified or invalid arguments.
    """
    args = list(argv)
    state = _ArgState(get_project_root())

    i = 0
    while i < len(args):
        arg = args[i]
        has_next = i + 1 < len(args)

        if _handle_flag_arg(state, arg):
            i += 1
        elif has_next and _handle_value_arg(state, arg, args[i + 1]):
            i += 2
        else:
            i += 1

    return state.to_parsed_args()


# =============================================================================
# Main Entry Point
# =============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the AMEX pipeline.

    Args:
        argv: Command-line arguments (excluding script name).
            If None, reads from sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    setup_rich_logging()
    configure_real_scipy()

    raw_argv: Sequence[str] = sys.argv[1:] if argv is None else argv
    args = parse_args(raw_argv)

    config = AMEXPipelineConfig(
        backends=args["backends"],
        n_folds=args["n_folds"],
        n_estimators=args["n_estimators"],
        learning_rate=args["learning_rate"],
        aggregation=args["aggregation"],
        include_rank_features=args["include_rank_features"],
        include_diff_features=args["include_diff_features"],
        include_window_features=args["include_window_features"],
        window_sizes=args["window_sizes"],
        random_state=args["random_state"],
    )

    result = run_pipeline(
        train_dir=args["train_dir"],
        test_dir=args["test_dir"],
        output_path=args["output_path"],
        config=config,
    )

    console = get_console()
    console.write("\nFinal Results:")
    console.write(f"  Training samples: {result['n_samples_train']}")
    console.write(f"  Test samples: {result['n_samples_test']}")
    console.write(f"  Features: {result['n_features']}")
    console.write(f"  Ensemble score: {result['ensemble_result']['optimized_score']:.5f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
