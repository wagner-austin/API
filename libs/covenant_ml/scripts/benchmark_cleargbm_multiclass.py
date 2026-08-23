"""Measure ClearGBM's multiclass softmax against LightGBM.

Thin entry point: argument parsing, wiring, and output. All measurement
logic lives in :mod:`covenant_ml.benchmarking.multiclass_quality`, where it
is unit tested.

Usage:
    poetry run python -m scripts.benchmark_cleargbm_multiclass
    poetry run python -m scripts.benchmark_cleargbm_multiclass --seeds 42 43 \
        --out docs/BENCHMARK_MANIFEST_multiclass.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from platform_core.json_utils import dump_json_str

from covenant_ml.benchmarking.multiclass_quality import (
    MulticlassBenchConfig,
    encode_multiclass_manifest,
    run_multiclass_benchmark,
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
    default_seeds: list[int] = [42, 43, 44, 45]
    parser = argparse.ArgumentParser(
        description="Benchmark ClearGBM multiclass softmax against LightGBM."
    )
    parser.add_argument("--samples", type=int, default=6000, help="Corpus rows per seed.")
    parser.add_argument("--features", type=int, default=8, help="Corpus feature count.")
    parser.add_argument("--classes", type=int, default=5, help="Class count.")
    parser.add_argument("--trees", type=int, default=100, help="Boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Shrinkage.")
    parser.add_argument("--max-bins", type=int, default=64, help="Histogram bin count.")
    parser.add_argument("--min-samples-leaf", type=int, default=20, help="Minimum rows per leaf.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=default_seeds, help="Seeds to measure."
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
    # argparse yields untyped attributes; bind each to a typed name once so
    # every use below is precisely typed.
    seeds: list[int] = parsed.seeds
    out_path: Path | None = parsed.out
    n_samples: int = parsed.samples
    n_features: int = parsed.features
    n_classes: int = parsed.classes
    n_estimators: int = parsed.trees
    max_depth: int = parsed.max_depth
    learning_rate: float = parsed.learning_rate
    max_bins: int = parsed.max_bins
    min_samples_leaf: int = parsed.min_samples_leaf
    config = MulticlassBenchConfig(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
    )

    _write(
        f"multiclass corpus: {config['n_samples']} rows x {config['n_features']} features, "
        f"{config['n_classes']} classes, seeds {seeds}\n"
    )
    manifest = run_multiclass_benchmark(config, seeds)
    for result in manifest["results"]:
        quality = result["quality"]
        _write(
            f"  {result['model']:>9} seed={result['seed']} "
            f"log_loss={quality['log_loss']:.6f} accuracy={quality['accuracy']:.4f}\n"
        )

    if out_path is not None:
        out_path.write_text(
            dump_json_str(encode_multiclass_manifest(manifest), indent=1),
            encoding="utf-8",
        )
        _write(f"manifest -> {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
