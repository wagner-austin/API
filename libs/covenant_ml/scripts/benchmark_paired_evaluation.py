"""Compare every arm with a reference on the same held-out rows of a real corpus.

What :data:`covenant_ml.benchmarking.declarations.PAIRED_EVALUATION` IS lives
there; the manifest and run-record writes are the harness's. This script
loads the corpus, builds the arms the timing benchmark times, and hands them
to :func:`covenant_ml.evaluation.runner.run_paired_evaluation`.

Its headline is not any arm's mean but each arm's DIFFERENCE from the
reference with an interval, pooled over seeds, so those are passed to the
harness as derived observations beside the declared per-arm means.

Usage:
    poetry run python -m scripts.benchmark_paired_evaluation \\
        --csv ../../services/covenant-radar-api/data/external/us_data/american_bankruptcy.csv \\
        --seeds 42 43 44 45 46 --trees 200 --max-depth 6 --max-bins 64 --num-leaves 31 \\
        --reference lightgbm --unit cluster --block-length 1 --n-resamples 2000 \\
        --resample-seed 7 --confidence 0.95 --true-alert-value 10 \\
        --false-alert-cost 1 --missed-cost 5 --out docs/PAIRED_EVALUATION.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from platform_core.config import config_test_hooks
from platform_core.determinism_cpu import PinProtocol, pin_single_thread

# NOTHING FROM covenant_ml IS IMPORTED AT MODULE SCOPE: `covenant_ml/__init__`
# pulls numpy, and the BLAS thread pin must be written before numpy loads.
# `benchmark_cleargbm_vs_lightgbm` carries the measurement behind the rule.


def main(argv: list[str] | None = None, pin: PinProtocol = pin_single_thread) -> int:
    """Run the paired evaluation and write its manifest and record.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
        pin: How to pin CPU determinism, defaulting to the real pin. A test
            supplies a stand-in because a numpy test suite has numpy loaded
            before collection, where the real pin refuses.

    Returns:
        Process exit code.
    """
    determinism = pin()

    from covenant_ml.benchmarking import load_bankruptcy_dataset, make_baseline_trainers
    from covenant_ml.benchmarking.declarations import PAIRED_EVALUATION, PAIRED_SOURCES
    from covenant_ml.benchmarking.factory import make_benchmark_config
    from covenant_ml.benchmarking.harness import (
        shared_parser,
        summary_lines,
        write_manifest_and_record,
    )
    from covenant_ml.benchmarking.provenance import benchmark_fingerprint
    from covenant_ml.benchmarking.types import DatasetInfo
    from covenant_ml.evaluation.arguments import parse_unit, resampling_settings
    from covenant_ml.evaluation.codec_manifests import encode_evaluation_manifest
    from covenant_ml.evaluation.provenance import policy_observations, pooled_observations
    from covenant_ml.evaluation.runner import run_paired_evaluation
    from covenant_ml.evaluation.splits import CompanyDisjointGroupedSplits, GroupedSplitFactoryProto
    from covenant_ml.evaluation.types import AlertCosts
    from covenant_ml.evaluation.wrds import load_wrds_evaluation

    parsed = shared_parser(PAIRED_EVALUATION, PAIRED_SOURCES).parse_args(argv)

    # Annotated at assignment: argparse hands back Any.
    wrds_path: Path | None = parsed.wrds_manifest
    seeds: list[int] = parsed.seeds
    out_path: Path = parsed.out
    n_estimators: int = parsed.trees
    max_depth: int = parsed.max_depth
    max_bins: int = parsed.max_bins
    num_leaves: int = parsed.num_leaves
    reference: str = parsed.reference
    unit_flag: str = parsed.unit
    block_length: int = parsed.block_length
    n_resamples: int = parsed.n_resamples
    resample_seed: int = parsed.resample_seed
    confidence: float = parsed.confidence
    true_alert_value: float = parsed.true_alert_value
    false_alert_cost: float = parsed.false_alert_cost
    missed_cost: float = parsed.missed_cost
    costs: AlertCosts = {
        "true_alert_value": true_alert_value,
        "false_alert_cost": false_alert_cost,
        "missed_cost": missed_cost,
    }

    # One fit per arm per seed: a held-out prediction does not change with
    # repetition, so the timing benchmark's repeats and warm-ups are 1 and 0.
    config = make_benchmark_config(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_bins=max_bins,
        num_leaves=num_leaves,
        repeats=1,
        warmups=0,
    )
    settings = resampling_settings(
        parse_unit(unit_flag), block_length, n_resamples, resample_seed, confidence
    )
    fingerprint = benchmark_fingerprint(determinism, config_test_hooks.get_env)

    split_factory: GroupedSplitFactoryProto
    dataset_info: DatasetInfo
    if wrds_path is not None:
        if unit_flag != "block":
            raise ValueError("WRDS_DATA_UNIT: time-series evaluation requires --unit block")
        wrds_data = load_wrds_evaluation(wrds_path)
        split_factory, dataset_info = wrds_data.splits, wrds_data.info
    else:
        csv_path: Path = parsed.csv
        dataset = load_bankruptcy_dataset(csv_path)
        split_factory = CompanyDisjointGroupedSplits(
            dataset.features, dataset.labels, dataset.company_codes
        )
        dataset_info = dataset.info
    manifest = run_paired_evaluation(
        make_baseline_trainers(config),
        split_factory,
        seeds,
        reference,
        config,
        dataset_info,
        settings,
        costs,
        fingerprint,
    )
    encoded = encode_evaluation_manifest(manifest)
    derived = pooled_observations(manifest["pooled"]) + policy_observations(
        manifest["seed_comparisons"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record_path = write_manifest_and_record(
        PAIRED_EVALUATION,
        encoded,
        fingerprint,
        out_path,
        seeds,
        derived=derived,
        label_prefix=f"{dataset_info['sha256'][:12]}-{settings['unit'].value}-",
    )
    for line in summary_lines(PAIRED_EVALUATION, encoded, derived):
        sys.stdout.write(f"{line}\n")
    sys.stdout.write(f"\nmanifest -> {out_path}\nrun record -> {record_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
