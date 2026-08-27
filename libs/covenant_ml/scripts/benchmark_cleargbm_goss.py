"""Measure GOSS quality cost for ClearGBM and LightGBM side by side.

Thin entry point: argument parsing, wiring, and output. All measurement
logic lives in :mod:`covenant_ml.benchmarking.goss_quality`, where it is
unit tested.

Usage:
    poetry run python -m scripts.benchmark_cleargbm_goss
    poetry run python -m scripts.benchmark_cleargbm_goss --seeds 42 43 \
        --out docs/BENCHMARK_MANIFEST_goss.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Protocol

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import apply_cpu_determinism
from platform_core.determinism_env import SINGLE_THREAD
from platform_core.determinism_record import DeterminismRecord
from platform_core.json_utils import dump_json_str

# NOTHING FROM covenant_ml IS IMPORTED AT MODULE SCOPE, and that is a
# correctness requirement rather than a preference. `covenant_ml/__init__`
# pulls numpy, the BLAS thread variables are read when numpy loads, and a pin
# after that point writes variables nobody reads. `apply_cpu_determinism`
# refuses in that case instead of reporting a posture the run does not have.
#
# This entry point pinned NOTHING until 2026-08-27: its numbers were not
# reproducible against themselves, let alone comparable with another
# machine's. The imports it needs live inside `main`, after the pin.


class PinProtocol(Protocol):
    """Protocol for pinning this process's CPU reduction order."""

    def __call__(self) -> DeterminismRecord:
        """Pin the thread count and report what was pinned.

        Returns:
            The posture the process now has.
        """
        ...


def _real_pin() -> DeterminismRecord:
    """Pin the BLAS thread count to one and report it.

    Returns:
        The record naming every thread variable that was set.

    Raises:
        NativeLibrariesAlreadyLoadedError: When a native numeric library is
            already imported, so the write cannot take effect.
    """
    return apply_cpu_determinism(os.putenv, SINGLE_THREAD)


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
        description="Measure GOSS quality cost for ClearGBM and LightGBM."
    )
    parser.add_argument("--samples", type=int, default=20000, help="Corpus rows per seed.")
    parser.add_argument("--features", type=int, default=8, help="Corpus feature count.")
    parser.add_argument("--trees", type=int, default=200, help="Boosting rounds.")
    parser.add_argument("--max-depth", type=int, default=4, help="Maximum tree depth.")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Shrinkage.")
    parser.add_argument("--max-bins", type=int, default=64, help="Histogram bin count.")
    parser.add_argument("--min-samples-leaf", type=int, default=20, help="Minimum rows per leaf.")
    parser.add_argument("--top-rate", type=float, default=0.2, help="GOSS top rate.")
    parser.add_argument("--other-rate", type=float, default=0.1, help="GOSS other rate.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=default_seeds, help="Seeds to measure."
    )
    parser.add_argument("--out", type=Path, default=None, help="Manifest JSON output path.")
    return parser


def main(argv: list[str] | None = None, pin: PinProtocol = _real_pin) -> int:
    """Run the benchmark and report.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
        pin: How to pin CPU determinism, defaulting to the real pin. A test
            supplies a stand-in for one reason only: the real pin refuses
            once a native numeric library is loaded, and a numpy test suite
            has numpy loaded before collection begins. Substituting it does
            NOT excuse this module from being pinnable -- that property is
            asserted directly, by importing this file and checking nothing
            numeric arrived with it.

    Returns:
        Process exit code.
    """
    # PIN FIRST, THEN IMPORT. The thread count decides how a BLAS partitions
    # a reduction and floating-point addition is not associative, so the count
    # is an input to every number below -- measured at 865,498 of 16,777,216
    # matmul elements changing between 1, 8 and 24 threads.
    determinism = pin()

    # Imported after the pin, with everything else from covenant_ml: it reads
    # installed metadata and builds a host record, neither of which may
    # happen above the line that writes the thread variables.
    from covenant_ml.benchmarking.goss_quality import (
        GossBenchConfig,
        encode_goss_manifest,
        run_goss_benchmark,
    )
    from covenant_ml.benchmarking.provenance import benchmark_fingerprint

    # Read through the config layer, not os.environ. Writing a variable a
    # native library requires is a different act from reading configuration,
    # and only the first is this script's business.
    fingerprint = benchmark_fingerprint(determinism, config_test_hooks.get_env)

    parsed = build_parser().parse_args(argv)
    # argparse yields untyped attributes; bind each to a typed name once so
    # every use below is precisely typed.
    seeds: list[int] = parsed.seeds
    out_path: Path | None = parsed.out
    n_samples: int = parsed.samples
    n_features: int = parsed.features
    n_estimators: int = parsed.trees
    max_depth: int = parsed.max_depth
    learning_rate: float = parsed.learning_rate
    max_bins: int = parsed.max_bins
    min_samples_leaf: int = parsed.min_samples_leaf
    top_rate: float = parsed.top_rate
    other_rate: float = parsed.other_rate
    config = GossBenchConfig(
        n_samples=n_samples,
        n_features=n_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        top_rate=top_rate,
        other_rate=other_rate,
    )

    _write(
        f"goss corpus: {config['n_samples']} rows x {config['n_features']} features, "
        f"top {config['top_rate']} / other {config['other_rate']}, seeds {seeds}\n"
    )
    manifest = run_goss_benchmark(config, seeds, fingerprint)
    for result in manifest["results"]:
        quality = result["quality"]
        _write(
            f"  {result['model']:>9}/{result['sampling']:<4} seed={result['seed']} "
            f"auc={quality['auc']:.6f} log_loss={quality['log_loss']:.6f}\n"
        )

    if out_path is not None:
        out_path.write_text(
            dump_json_str(encode_goss_manifest(manifest), indent=1),
            encoding="utf-8",
        )
        _write(f"manifest -> {out_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
