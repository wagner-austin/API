"""Run the growth-policy arms on datasets of a different shape.

Thin entry point: argument parsing, wiring, and output. All measurement logic
lives in :mod:`covenant_ml.growth_policy`, where it is unit tested.

The same three XGBoost arms as the instrument run, applied to two smaller
datasets so the single-dataset caveat is answered rather than carried. Neither
dataset has a grouping key, so both are partitioned by a stratified split.

The external data root is an argument with a repository-relative default. The
run these figures came from hardcoded an absolute path, which made the script
unrunnable on any other machine and on the measurement fleet.

Usage:
    poetry run python -m scripts.experiment_growth_policy_multi_dataset
    poetry run python -m scripts.experiment_growth_policy_multi_dataset \
        --external-root ../../services/covenant-radar-api/data/external
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
    GrowthPolicyReport,
    PlainDataset,
    describe_dataset,
    encode_growth_policy_report,
    load_german_credit,
    load_taiwan_bankruptcy,
    make_arm_specs,
    make_experiment_config,
    make_metrics,
    make_stratified_split_factory,
    make_xgb_trainers,
    render_report,
    run_experiment,
)
from covenant_ml.growth_policy.types import ExperimentConfig

#: External data root, relative to this library's root.
DEFAULT_EXTERNAL_ROOT = Path("..") / ".." / "services" / "covenant-radar-api" / "data" / "external"

#: Path of each dataset beneath the external root.
TAIWAN_RELATIVE = Path("kaggle_taiwan_bankruptcy") / "data.csv"
GERMAN_RELATIVE = Path("german_credit") / "german.data"


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
        description="Run the growth-policy arms across datasets of differing shape."
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=DEFAULT_EXTERNAL_ROOT,
        help="Directory holding the external datasets.",
    )
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
        "--out-dir",
        type=Path,
        default=None,
        help="Write each dataset's report as JSON into this directory.",
    )
    return parser


def run_one(
    name: str,
    dataset: PlainDataset,
    seeds: list[int],
    leaf_budgets: list[int],
    max_depth: int,
    config: ExperimentConfig,
) -> GrowthPolicyReport:
    """Measure every arm on one ungrouped dataset.

    Args:
        name: Human-readable dataset name.
        dataset: The loaded features and labels.
        seeds: Seeds to measure at.
        leaf_budgets: Leaf budgets, one leaf-wise arm each.
        max_depth: Depth budget for the depth-wise arm.
        config: Hyperparameters shared across arms.

    Returns:
        The dataset's report.
    """
    return run_experiment(
        make_xgb_trainers(config, make_arm_specs(max_depth, leaf_budgets)),
        make_stratified_split_factory(dataset.features, dataset.labels),
        seeds,
        make_metrics(),
        config,
        describe_dataset(name, dataset.features, dataset.labels),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the growth-policy arms across both external datasets.

    Args:
        argv: Command-line arguments, or ``None`` to read ``sys.argv``.

    Returns:
        Process exit code.
    """
    parsed = build_parser().parse_args(argv)
    external_root: Path = parsed.external_root
    seeds: list[int] = parsed.seeds
    leaf_budgets: list[int] = parsed.leaf_budgets
    max_depth: int = parsed.max_depth
    estimators: int = parsed.estimators
    repeats: int = parsed.repeats
    warmups: int = parsed.warmups
    out_dir: Path | None = parsed.out_dir

    config = make_experiment_config(n_estimators=estimators, repeats=repeats, warmups=warmups)
    loaded: list[tuple[str, PlainDataset]] = [
        ("taiwan-bankruptcy", load_taiwan_bankruptcy(external_root / TAIWAN_RELATIVE)),
        ("german-credit", load_german_credit(external_root / GERMAN_RELATIVE)),
    ]
    for name, dataset in loaded:
        report = run_one(name, dataset, seeds, leaf_budgets, max_depth, config)
        _write("\n" + render_report(report))
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"growth-policy-{name}.json"
            target.write_text(dump_json_str(encode_growth_policy_report(report)), encoding="utf-8")
            _write(f"wrote {target}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
