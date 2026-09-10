"""Benchmark ClearGBM fit time against LightGBM on the bankruptcy corpus.

A DECLARATION AND THE WORK THIS FAMILY ACTUALLY DOES. What this benchmark IS
lives in :data:`covenant_ml.benchmarking.declarations.VS_LIGHTGBM`; the
manifest write and the run record are
:mod:`covenant_ml.benchmarking.harness`'s job, for every family.

THIS FAMILY IS THE ONE THAT DOES NOT FIT THE DECLARATION MODEL, and its
declaration says so by carrying no arm axes and no metric names. The other
five report per-arm means of per-seed quality. This one's headline is a
fit-time RATIO BETWEEN ARMS -- raw, per-leaf and normalized -- which is not
the mean of anything a result row carries. Those observations are built by
:func:`~covenant_ml.benchmarking.provenance.benchmark_observations` and handed
to the harness as ``derived``, and the label carries the corpus digest because
this benchmark measures a REAL dataset rather than one generated from its
seeds: a filename is not the bytes.

Usage:
    poetry run python -m scripts.benchmark_cleargbm_vs_lightgbm \\
        --csv data/bankruptcy.csv --seeds 42 43 44 --trees 200 \\
        --max-depth 6 --max-bins 64 --num-leaves 31 --repeats 5 --warmups 2 \\
        --out docs/BENCHMARK_MANIFEST.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread

# NOTHING FROM covenant_ml IS IMPORTED AT MODULE SCOPE, and that is a
# correctness requirement rather than a preference. `covenant_ml/__init__`
# pulls numpy, the BLAS thread variables are read when numpy loads, and a pin
# after that point writes variables nobody reads. `apply_cpu_determinism`
# refuses in that case instead of reporting a posture the run does not have.
# The harness is subject to the same rule and is imported below, after the pin.


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Run the fit-time benchmark and write its manifest and record.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
        pin: How to pin CPU determinism, defaulting to the real pin. A test
            supplies a stand-in for one reason only: the real pin refuses once
            a native numeric library is loaded, and a numpy test suite has
            numpy loaded before collection begins. Substituting it does NOT
            excuse this module from being pinnable -- that property is
            asserted directly, by importing this file and checking nothing
            numeric arrived with it.

    Returns:
        Process exit code.
    """
    determinism = pin()

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
    from covenant_ml.benchmarking.declarations import VS_LIGHTGBM
    from covenant_ml.benchmarking.harness import shared_parser, write_manifest_and_record
    from covenant_ml.benchmarking.provenance import (
        benchmark_fingerprint,
        benchmark_observations,
    )

    parsed = shared_parser(VS_LIGHTGBM).parse_args(argv)

    # Annotated at assignment: argparse hands back Any, and the annotation is
    # what makes these typed rather than the value's shape at run time.
    csv_path: Path = parsed.csv
    seeds: list[int] = parsed.seeds
    out_path: Path = parsed.out
    include_variants: bool = parsed.variants
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

    # Read through the config layer, not os.environ. Writing a variable a
    # native library requires is a different act from reading configuration,
    # and only the first is this script's business.
    fingerprint = benchmark_fingerprint(determinism, config_test_hooks.get_env)

    sys.stdout.write(f"loading {csv_path} ...\n")
    dataset = load_bankruptcy_dataset(csv_path)
    sys.stdout.write(f"  rows={dataset.info['n_rows']} features={dataset.info['n_features']}\n\n")
    trainers = make_trainers(config) if include_variants else make_baseline_trainers(config)
    manifest = run_benchmark(
        trainers,
        make_split_factory(dataset.features, dataset.labels, dataset.company_codes),
        seeds,
        config,
        dataset.info,
        fingerprint,
    )
    sys.stdout.write(render_report(manifest))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    record_path = write_manifest_and_record(
        VS_LIGHTGBM,
        encode_benchmark_manifest(manifest),
        fingerprint,
        out_path,
        seeds,
        derived=benchmark_observations(manifest),
        label_prefix=f"{manifest['dataset']['sha256'][:12]}-{manifest['estimator']}-",
    )
    sys.stdout.write(f"\nmanifest -> {out_path}\nrun record -> {record_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
