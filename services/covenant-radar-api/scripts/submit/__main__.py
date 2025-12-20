"""CLI entry point for Kaggle AMEX submission pipeline.

Usage:
    python -m scripts.submit [OPTIONS]

Strict typing only: no Any, no casts, no type: ignore, no stubs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

from covenant_ml.types import BackendName
from platform_core.logging import setup_rich_logging

from scripts.submit._hooks import get_console, get_project_root
from scripts.submit.pipeline import SubmitConfig, run_pipeline

# =============================================================================
# Types
# =============================================================================


class ParsedArgs(TypedDict, total=True):
    """Parsed command-line arguments.

    Attributes:
        backend: ML backend to use.
        n_estimators: Number of boosting rounds.
        learning_rate: Learning rate.
        num_leaves: Maximum leaves per tree.
        max_depth: Maximum tree depth.
        aggregation: Time-series aggregation strategy.
        include_rank_features: Whether to include rank features.
        include_diff_features: Whether to include diff features.
        train_dir: Training data directory.
        test_dir: Test data directory.
        output_path: Output submission file path.
    """

    backend: BackendName
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    aggregation: Literal["last", "first", "mean", "statistics"]
    include_rank_features: bool
    include_diff_features: bool
    train_dir: Path
    test_dir: Path
    output_path: Path


class _ArgState:
    """Mutable argument parsing state."""

    backend: BackendName
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    aggregation: Literal["last", "first", "mean", "statistics"]
    include_rank_features: bool
    include_diff_features: bool
    train_dir: Path
    test_dir: Path
    output_path: Path

    def __init__(self, project_root: Path) -> None:
        """Initialize with defaults."""
        self.backend = "lightgbm"
        self.n_estimators = 1000
        self.learning_rate = 0.05
        self.num_leaves = 31
        self.max_depth = -1
        self.aggregation = "statistics"
        self.include_rank_features = True
        self.include_diff_features = True
        self.train_dir = project_root / "data" / "external" / "amex_train"
        self.test_dir = project_root / "data" / "external" / "amex_test"
        self.output_path = project_root / "data" / "submissions" / "submission.csv"

    def to_parsed_args(self) -> ParsedArgs:
        """Convert to immutable ParsedArgs."""
        return ParsedArgs(
            backend=self.backend,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            aggregation=self.aggregation,
            include_rank_features=self.include_rank_features,
            include_diff_features=self.include_diff_features,
            train_dir=self.train_dir,
            test_dir=self.test_dir,
            output_path=self.output_path,
        )


# =============================================================================
# Argument Parsing
# =============================================================================


def _parse_backend(val: str) -> BackendName:
    """Parse backend value.

    Args:
        val: Backend name string from CLI.

    Returns:
        Validated backend name literal.

    Raises:
        SystemExit: If backend name is invalid.
    """
    if val == "lightgbm":
        return "lightgbm"
    if val == "xgboost":
        return "xgboost"
    if val == "mlp":
        return "mlp"
    if val == "lstm":
        return "lstm"
    console = get_console()
    console.write(f"Invalid backend: {val}. Must be lightgbm, xgboost, mlp, lstm.")
    raise SystemExit(1)


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


def print_help() -> None:
    """Print help message with all available options."""
    console = get_console()
    help_text = """
Usage: python -m scripts.submit [OPTIONS]

Options:
  -b, --backend          Backend: lightgbm, xgboost, mlp, lstm (default: lightgbm)
  -n, --n-estimators     Number of boosting rounds (default: 1000)
  -l, --learning-rate    Learning rate (default: 0.05)
  --num-leaves           Maximum leaves per tree (default: 31)
  --max-depth            Maximum tree depth, -1 for unlimited (default: -1)
  -a, --aggregation      Aggregation: last, first, mean, statistics (default: statistics)
  --no-rank-features     Disable per-entity rank features
  --no-diff-features     Disable row-to-row diff features
  --train-dir            Training data directory
  --test-dir             Test data directory
  -o, --output           Output submission file path
  -h, --help             Show this help

Backends:
  lightgbm   Gradient boosted trees (LightGBM) - fast, efficient
  xgboost    Gradient boosted trees (XGBoost DMatrix API)
  mlp        Multi-layer perceptron (PyTorch)
  lstm       Long short-term memory network (PyTorch)

Examples:
  python -m scripts.submit
  python -m scripts.submit -b xgboost -n 500 -l 0.1
  python -m scripts.submit -b lightgbm -a statistics
"""
    console.write(help_text)


def _handle_value_arg(state: _ArgState, arg: str, next_val: str) -> bool:
    """Handle value-based arguments.

    Args:
        state: Mutable argument state.
        arg: Current argument.
        next_val: Next argument value.

    Returns:
        True if argument was handled.
    """
    if arg in ("--backend", "-b"):
        state.backend = _parse_backend(next_val)
        return True
    if arg in ("--n-estimators", "-n"):
        state.n_estimators = int(next_val)
        return True
    if arg in ("--learning-rate", "-l"):
        state.learning_rate = float(next_val)
        return True
    if arg == "--num-leaves":
        state.num_leaves = int(next_val)
        return True
    if arg == "--max-depth":
        state.max_depth = int(next_val)
        return True
    if arg in ("--aggregation", "-a"):
        state.aggregation = _parse_aggregation(next_val)
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


def main(argv: Sequence[str]) -> int:
    """Main entry point for the submission pipeline.

    Args:
        argv: Command-line arguments (excluding script name).

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    setup_rich_logging()

    args = parse_args(argv)

    config = SubmitConfig(
        backend=args["backend"],
        n_estimators=args["n_estimators"],
        learning_rate=args["learning_rate"],
        num_leaves=args["num_leaves"],
        max_depth=args["max_depth"],
        aggregation=args["aggregation"],
        include_rank_features=args["include_rank_features"],
        include_diff_features=args["include_diff_features"],
    )

    result = run_pipeline(
        train_dir=args["train_dir"],
        test_dir=args["test_dir"],
        output_path=args["output_path"],
        config=config,
    )

    console = get_console()
    console.write(f"Generated {result['n_samples']} predictions")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
