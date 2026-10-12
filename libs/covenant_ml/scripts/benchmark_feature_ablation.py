"""Ablate feature groups from one learner and trace its learning curve.

What :data:`covenant_ml.benchmarking.declarations.FEATURE_ABLATION` IS lives
there; the manifest and run-record writes are the harness's. This script
loads the corpus, picks the learner ``--model`` names from the arms the
timing benchmark times, and hands it to
:func:`covenant_ml.evaluation.ablation.run_feature_ablation`.

Usage:
    poetry run python -m scripts.benchmark_feature_ablation \\
        --csv ../../services/covenant-radar-api/data/external/us_data/american_bankruptcy.csv \\
        --seeds 42 43 44 45 46 --trees 200 --max-depth 6 --max-bins 64 --num-leaves 31 \\
        --model lightgbm --groups "X1;X2;X3,X4" --fractions 0.1,0.25,0.5,1.0 \\
        --unit cluster --block-length 1 --n-resamples 2000 --resample-seed 7 \\
        --confidence 0.95 --out docs/FEATURE_ABLATION.json
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
    """Run the ablation and learning curve and write the manifest and record.

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
    from covenant_ml.benchmarking.declarations import FEATURE_ABLATION
    from covenant_ml.benchmarking.factory import make_benchmark_config
    from covenant_ml.benchmarking.harness import (
        shared_parser,
        summary_lines,
        write_manifest_and_record,
    )
    from covenant_ml.benchmarking.provenance import benchmark_fingerprint
    from covenant_ml.evaluation.ablation import run_feature_ablation
    from covenant_ml.evaluation.arguments import (
        parse_fractions,
        parse_groups,
        parse_unit,
        resampling_settings,
        select_trainer,
    )
    from covenant_ml.evaluation.codec_manifests import encode_ablation_manifest
    from covenant_ml.evaluation.provenance import pooled_observations
    from covenant_ml.evaluation.splits import CompanyDisjointGroupedSplits

    parsed = shared_parser(FEATURE_ABLATION).parse_args(argv)

    # Annotated at assignment: argparse hands back Any.
    csv_path: Path = parsed.csv
    seeds: list[int] = parsed.seeds
    out_path: Path = parsed.out
    n_estimators: int = parsed.trees
    max_depth: int = parsed.max_depth
    max_bins: int = parsed.max_bins
    num_leaves: int = parsed.num_leaves
    model: str = parsed.model
    groups_flag: str = parsed.groups
    fractions_flag: str = parsed.fractions
    unit_flag: str = parsed.unit
    block_length: int = parsed.block_length
    n_resamples: int = parsed.n_resamples
    resample_seed: int = parsed.resample_seed
    confidence: float = parsed.confidence

    config = make_benchmark_config(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_bins=max_bins,
        num_leaves=num_leaves,
        repeats=1,
        warmups=0,
    )
    trainer = select_trainer(make_baseline_trainers(config), model)
    groups = parse_groups(groups_flag)
    fractions = parse_fractions(fractions_flag)
    settings = resampling_settings(
        parse_unit(unit_flag), block_length, n_resamples, resample_seed, confidence
    )
    fingerprint = benchmark_fingerprint(determinism, config_test_hooks.get_env)

    sys.stdout.write(f"loading {csv_path} ...\n")
    dataset = load_bankruptcy_dataset(csv_path)
    manifest = run_feature_ablation(
        trainer,
        CompanyDisjointGroupedSplits(dataset.features, dataset.labels, dataset.company_codes),
        seeds,
        dataset.feature_names,
        groups,
        fractions,
        config,
        dataset.info,
        settings,
        fingerprint,
    )
    encoded = encode_ablation_manifest(manifest)
    derived = pooled_observations(manifest["pooled"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record_path = write_manifest_and_record(
        FEATURE_ABLATION,
        encoded,
        fingerprint,
        out_path,
        seeds,
        derived=derived,
        label_prefix=f"{dataset.info['sha256'][:12]}-{model}-{settings['unit'].value}-",
    )
    for line in summary_lines(FEATURE_ABLATION, encoded, derived):
        sys.stdout.write(f"{line}\n")
    sys.stdout.write(f"\nmanifest -> {out_path}\nrun record -> {record_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(None))
