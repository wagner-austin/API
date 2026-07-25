"""Measure ClearGBM against LightGBM on the bankruptcy dataset.

Thin entry point: argument parsing, wiring, and output. All measurement logic
lives in :mod:`covenant_ml.benchmarking`, where it is unit tested.

Usage:
    poetry run python -m scripts.benchmark_cleargbm_vs_lightgbm
    poetry run python -m scripts.benchmark_cleargbm_vs_lightgbm --repeats 5 \
        --out docs/BENCHMARK_MANIFEST.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.json_utils import dump_json_str

from covenant_ml.benchmarking import (
    DEFAULT_REPEATS,
    DEFAULT_SEEDS,
    DEFAULT_WARMUPS,
    encode_benchmark_manifest,
    load_bankruptcy_dataset,
    make_benchmark_config,
    make_split_factory,
    make_trainers,
    render_report,
    run_benchmark,
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
    parser = argparse.ArgumentParser(description="Benchmark ClearGBM against LightGBM.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV path.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=default_seeds,
        help="Seeds to measure.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Timed fits per model per seed.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=DEFAULT_WARMUPS,
        help="Discarded fits before timing begins.",
    )
    parser.add_argument("--trees", type=int, default=200, help="Boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--max-bins", type=int, default=64, help="Histogram bin count.")
    parser.add_argument("--num-leaves", type=int, default=31, help="LightGBM leaf cap.")
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
    # argparse yields untyped attributes; bind each to a typed name once so
    # every use below is precisely typed.
    csv_path: Path = parsed.csv
    seeds: list[int] = parsed.seeds
    out_path: Path | None = parsed.out
    n_estimators: int = parsed.trees
    max_depth: int = parsed.max_depth
    max_bins: int = parsed.max_bins
    num_leaves: int = parsed.num_leaves
    repeats: int = parsed.repeats
    warmups: int = parsed.warmups

    config = make_benchmark_config(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_bins=max_bins,
        num_leaves=num_leaves,
        repeats=repeats,
        warmups=warmups,
    )

    _write(f"loading {csv_path} ...\n")
    dataset = load_bankruptcy_dataset(csv_path)
    _write(f"  rows={dataset.info['n_rows']} features={dataset.info['n_features']}\n\n")

    cleargbm_trainer, lightgbm_trainer = make_trainers(config)
    split_factory = make_split_factory(
        dataset.features,
        dataset.labels,
        dataset.company_codes,
    )

    manifest = run_benchmark(
        cleargbm_trainer,
        lightgbm_trainer,
        split_factory,
        seeds,
        config,
        dataset.info,
    )

    _write(render_report(manifest))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        document = dump_json_str(encode_benchmark_manifest(manifest), indent=1)
        out_path.write_text(document, encoding="utf-8")
        _write(f"\nmanifest -> {out_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
