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
# This entry point pinned NOTHING until 2026-08-27 -- and it is the one whose
# manifests carry the headline TIMING claim against LightGBM, so its fit times
# were being taken at whatever thread count the shell happened to inherit.
# The imports it needs live inside `main`, after the pin.


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

    The three defaults come from ``covenant_ml``, imported INSIDE this
    function rather than at module scope. Importing them at the top would
    pull numpy before the pin, which is the exact defect the module comment
    above describes; this function is only ever called from ``main`` after
    the pin has taken, so the import is safe here and nowhere else.

    Returns:
        The configured parser.
    """
    from covenant_ml.benchmarking import (
        DEFAULT_REPEATS,
        DEFAULT_SEEDS,
        DEFAULT_WARMUPS,
    )

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
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="Leaf cap for every leaf-wise arm (LightGBM and cleargbm@leaf_wise).",
    )
    parser.add_argument(
        "--variants",
        action="store_true",
        help="Include ClearGBM variant arms (adds cleargbm@leaf_wise).",
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
    # a reduction, and for THIS benchmark it also decides the fit times that
    # are its whole output.
    determinism = pin()

    # Imported after the pin, with everything else from covenant_ml: building
    # a fingerprint reads installed metadata, which must not happen above the
    # line that writes the thread variables.
    from covenant_ml.benchmarking import (
        encode_benchmark_manifest,
        load_bankruptcy_dataset,
        make_baseline_trainers,
        make_benchmark_config,
        make_split_factory,
        make_trainers,
        render_report,
        run_benchmark,
    )
    from covenant_ml.benchmarking.provenance import benchmark_fingerprint

    # Read through the config layer, not os.environ. Writing a variable a
    # native library requires is a different act from reading configuration,
    # and only the first is this script's business.
    fingerprint = benchmark_fingerprint(determinism, config_test_hooks.get_env)

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

    include_variants: bool = parsed.variants
    trainers = make_trainers(config) if include_variants else make_baseline_trainers(config)
    split_factory = make_split_factory(
        dataset.features,
        dataset.labels,
        dataset.company_codes,
    )

    manifest = run_benchmark(
        trainers,
        split_factory,
        seeds,
        config,
        dataset.info,
        fingerprint,
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
