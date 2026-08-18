"""Measure what leaf-wise growth buys, using XGBoost as the instrument.

Thin entry point: argument parsing, wiring, and output. All measurement logic
lives in :mod:`covenant_ml.growth_policy`, where it is unit tested.

Five arms over the American-bankruptcy workload: XGBoost depth-wise, XGBoost
leaf-wise at two leaf budgets, and LightGBM and ClearGBM as anchors. Every arm
at one seed sees the identical company-disjoint partition.

Usage:
    poetry run python -m scripts.experiment_growth_policy_xgb_instrument
    poetry run python -m scripts.experiment_growth_policy_xgb_instrument \
        --seeds 42 43 44 --repeats 3 --out runs/growth-policy.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.json_utils import dump_json_str

from covenant_ml.growth_policy import (
    DEFAULT_LEAF_BUDGETS,
    DEFAULT_MAX_DEPTH,
    DEFAULT_REPEATS,
    DEFAULT_SEEDS,
    DEFAULT_WARMUPS,
    describe_dataset,
    encode_growth_policy_report,
    load_bankruptcy,
    make_anchor_trainers,
    make_arm_specs,
    make_experiment_config,
    make_group_split_factory,
    make_metrics,
    make_xgb_trainers,
    render_report,
    run_experiment,
)

#: Default input, relative to this library's root.
DEFAULT_CSV = Path("tests") / "data" / "american_bankruptcy.csv"


def _write(message: str) -> None:
    """Write a message to stdout.

    Args:
        message: Text to emit.
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    default_seeds: list[int] = list(DEFAULT_SEEDS)
    default_budgets: list[int] = list(DEFAULT_LEAF_BUDGETS)
    parser = argparse.ArgumentParser(
        description="Measure growth policy with XGBoost as the instrument."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=default_seeds, help="Seeds to measure."
    )
    parser.add_argument(
        "--leaf-budgets",
        type=int,
        nargs="+",
        default=default_budgets,
        help="Leaf budgets, one leaf-wise arm each.",
    )
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH, help="Depth budget.")
    parser.add_argument("--estimators", type=int, default=200, help="Boosting rounds per arm.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Timed fits per arm.")
    parser.add_argument(
        "--warmups", type=int, default=DEFAULT_WARMUPS, help="Discarded fits per arm."
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write the report as JSON to this path."
    )
    parser.add_argument(
        "--skip-anchors",
        action="store_true",
        help="Measure only the XGBoost arms, omitting LightGBM and ClearGBM.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the growth-policy experiment.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code.
    """
    parsed = build_parser().parse_args(argv)
    csv_path: Path = parsed.csv
    seeds: list[int] = parsed.seeds
    leaf_budgets: list[int] = parsed.leaf_budgets
    max_depth: int = parsed.max_depth
    estimators: int = parsed.estimators
    repeats: int = parsed.repeats
    warmups: int = parsed.warmups
    out_path: Path | None = parsed.out
    skip_anchors: bool = parsed.skip_anchors

    dataset = load_bankruptcy(csv_path)
    info = describe_dataset("american-bankruptcy", dataset.features, dataset.labels)
    config = make_experiment_config(n_estimators=estimators, repeats=repeats, warmups=warmups)
    trainers = make_xgb_trainers(config, make_arm_specs(max_depth, leaf_budgets))
    if not skip_anchors:
        trainers = trainers + make_anchor_trainers(config, max_depth=max_depth)

    report = run_experiment(
        trainers,
        make_group_split_factory(dataset.features, dataset.labels, dataset.groups),
        seeds,
        make_metrics(),
        config,
        info,
    )

    _write(render_report(report))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(dump_json_str(encode_growth_policy_report(report)), encoding="utf-8")
        _write(f"\nwrote {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
