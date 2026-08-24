"""Benchmark the four regression arms on a registered dataset.

Thin entry point: argument parsing, wiring, and output. All measurement
logic lives in :mod:`covenant_ml.benchmarking.regression_quality`, where
it is unit tested.

Usage:
    poetry run python -m scripts.benchmark_cleargbm_regression \
        --dataset rw_value \
        --external-dir ../../services/covenant-radar-api/data/external \
        --out docs/BENCHMARK_MANIFEST_rw_value.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.json_utils import dump_json_str

from covenant_ml.benchmarking.regression_quality import (
    RegressionBenchConfig,
    encode_regression_manifest,
    run_regression_benchmark,
)


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
    default_seeds: list[int] = [42, 43, 44, 45, 46]
    parser = argparse.ArgumentParser(
        description="Benchmark ClearGBM, LightGBM and XGBoost regression on a registry dataset."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Registry regression dataset name."
    )
    parser.add_argument(
        "--external-dir",
        type=Path,
        required=True,
        help="Root directory holding the registered datasets.",
    )
    parser.add_argument("--trees", type=int, default=300, help="Boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--num-leaves", type=int, default=31, help="Leaf-wise arm budget.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Shrinkage.")
    parser.add_argument("--max-bins", type=int, default=64, help="Histogram bin count.")
    parser.add_argument("--min-samples-leaf", type=int, default=20, help="Minimum rows per leaf.")
    parser.add_argument(
        "--early-stopping", type=int, default=30, help="Early-stopping patience on validation."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=default_seeds, help="Split seeds to measure."
    )
    parser.add_argument("--out", type=Path, default=None, help="Manifest JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and report.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parsed = build_parser().parse_args(argv)
    dataset: str = parsed.dataset
    external_dir: Path = parsed.external_dir
    seeds: list[int] = parsed.seeds
    out_path: Path | None = parsed.out
    n_estimators: int = parsed.trees
    max_depth: int = parsed.max_depth
    num_leaves: int = parsed.num_leaves
    learning_rate: float = parsed.learning_rate
    max_bins: int = parsed.max_bins
    min_samples_leaf: int = parsed.min_samples_leaf
    early_stopping_rounds: int = parsed.early_stopping
    config = RegressionBenchConfig(
        dataset=dataset,
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        early_stopping_rounds=early_stopping_rounds,
    )

    manifest = run_regression_benchmark(config, seeds, external_dir)
    split_kind = "grouped" if manifest["grouped"] else "row"
    _write(f"regression corpus: {dataset} ({split_kind} split), seeds {seeds}\n")
    for result in manifest["results"]:
        quality = result["quality"]
        _write(
            f"  {result['model']:>18} seed={result['seed']} rmse={quality['rmse']:.6f} "
            f"mae={quality['mae']:.6f} r2={quality['r_squared']:.6f} "
            f"fit={result['fit_seconds']:.3f}s\n"
        )

    if out_path is not None:
        out_path.write_text(
            dump_json_str(encode_regression_manifest(manifest), indent=1),
            encoding="utf-8",
        )
        _write(f"manifest -> {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
