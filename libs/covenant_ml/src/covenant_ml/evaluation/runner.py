"""The paired evaluation: every arm against a reference, on shared rows, per seed.

At each seed every arm is fitted on the same partition and scored on the same
held-out rows. Each non-reference arm is then compared with the reference on
AUC, log loss, Brier score and the realised value of its decisions, and the
seeds are pooled. The trainers are the benchmark's own
:class:`~covenant_ml.benchmarking.protocols.TrainerProto`, so the arms
compared here are exactly the arms the timing benchmark times.

What the timing runner does that this one deliberately does not: no repeats,
no warm-ups, no power-throttling opt-out, no rotation. None of those touch a
held-out prediction, and repeating a deterministic fit to score it again
would only cost time.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from platform_core.comparability import RunFingerprint

from ..benchmarking.protocols import TrainerProto
from ..benchmarking.quality import compute_quality
from ..benchmarking.types import BenchmarkConfig, DatasetInfo
from .paired import compare_per_row, compare_probabilities
from .policy import ALERT_ACTIONS, alert_payoffs, evaluate_actions, evaluate_policy
from .pooling import pool_seed_comparisons
from .resampling import build_resampling
from .splits import GroupedSplit, GroupedSplitFactoryProto
from .types import (
    ERR_REFERENCE_ARM,
    ERR_TOO_FEW_SEEDS,
    EVALUATION_SCHEMA_VERSION,
    AlertCosts,
    ArmResult,
    ComparisonMetric,
    EvaluationManifest,
    PairedComparison,
    PolicyOutcome,
    ResamplingSettings,
    SeedComparisons,
)

#: Name of the fixed policy that never alerts, reported beside every model's.
#: It is the baseline every alerting model has to beat to be worth running.
NEVER_ALERT = "never_alert"


def require_arms(names: Sequence[str], reference: str) -> None:
    """Refuse an arm set that cannot be compared against its reference.

    Args:
        names: Every arm's name.
        reference: The arm the others are compared against.

    Raises:
        ValueError: If fewer than two arms are given, two share a name, or the
            reference is not among them.
    """
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError(
            f"[{ERR_REFERENCE_ARM}] at least two distinctly named arms are needed, got {names}"
        )
    if reference not in names:
        raise ValueError(f"[{ERR_REFERENCE_ARM}] reference '{reference}' is not an arm of {names}")


def require_seeds(seeds: Sequence[int]) -> None:
    """Refuse a seed list too short to pool.

    Args:
        seeds: Seeds to evaluate.

    Raises:
        ValueError: If fewer than two seeds are given.
    """
    if len(seeds) < 2:
        raise ValueError(
            f"[{ERR_TOO_FEW_SEEDS}] at least 2 seeds are needed to pool, got {len(seeds)}"
        )


def evaluate_seed(
    trainers: Sequence[TrainerProto],
    grouped: GroupedSplit,
    seed: int,
    reference: str,
    settings: ResamplingSettings,
    costs: AlertCosts,
) -> tuple[list[ArmResult], SeedComparisons]:
    """Fit every arm on one partition and compare each with the reference.

    Args:
        trainers: The arms, distinctly named, the reference among them.
        grouped: The partition and its held-out unit codes.
        seed: Model seed; also offsets the bootstrap seed, so each seed's
            resamples are independent draws.
        reference: The arm the others are compared against.
        settings: How intervals treat dependence.
        costs: The alerting decision's payoffs.

    Returns:
        Every arm's quality, and the seed's comparisons and policies.

    Raises:
        ValueError: If the held-out fold has too few units or classes for an
            interval.
    """
    split = grouped.split
    n_test = len(split.y_test)
    resampling = build_resampling(
        settings["unit"], n_test, grouped.test_groups, settings["block_length"]
    )
    table = alert_payoffs(n_test, costs)
    probabilities = {
        trainer.model_name.value: trainer.fit(split, seed).predict_positive_proba(split.x_test)
        for trainer in trainers
    }
    results: list[ArmResult] = [
        {"arm": arm, "seed": seed, "quality": compute_quality(split.y_test, proba)}
        for arm, proba in probabilities.items()
    ]
    never = evaluate_actions(
        NEVER_ALERT,
        np.zeros(n_test, dtype=np.int64),
        split.y_test,
        table,
        resampling,
        settings["confidence"],
    )
    reference_policy = evaluate_policy(
        reference,
        probabilities[reference],
        split.y_test,
        table,
        resampling,
        settings["confidence"],
    )
    comparisons: list[PairedComparison] = []
    policies: list[PolicyOutcome] = [never.outcome, reference_policy.outcome]
    for arm, proba in probabilities.items():
        if arm == reference:
            continue
        comparisons.extend(
            compare_probabilities(
                split.y_test,
                proba,
                probabilities[reference],
                arm,
                reference,
                resampling,
                settings["n_resamples"],
                settings["resample_seed"] + seed,
                settings["confidence"],
            )
        )
        policy = evaluate_policy(
            arm, proba, split.y_test, table, resampling, settings["confidence"]
        )
        policies.append(policy.outcome)
        comparisons.append(
            compare_per_row(
                ComparisonMetric.UTILITY,
                arm,
                reference,
                policy.utilities,
                reference_policy.utilities,
                resampling,
                settings["confidence"],
            )
        )
    return results, {
        "seed": seed,
        "n_train": len(split.y_train),
        "n_test": n_test,
        "n_test_units": resampling.n_units,
        "comparisons": comparisons,
        "policies": policies,
    }


def run_paired_evaluation(
    trainers: Sequence[TrainerProto],
    build_split: GroupedSplitFactoryProto,
    seeds: Sequence[int],
    reference: str,
    config: BenchmarkConfig,
    dataset: DatasetInfo,
    settings: ResamplingSettings,
    costs: AlertCosts,
    fingerprint: RunFingerprint,
) -> EvaluationManifest:
    """Evaluate every arm against the reference at every seed, then pool.

    Args:
        trainers: The arms, distinctly named, the reference among them.
        build_split: Produces the grouped partition for a seed.
        seeds: Seeds to evaluate, at least two.
        reference: The arm the others are compared against.
        config: Hyperparameters every arm was built with.
        dataset: Identity of the input data.
        settings: How intervals treat dependence.
        costs: The alerting decision's payoffs.
        fingerprint: The configuration this runs under.

    Returns:
        The complete manifest.

    Raises:
        ValueError: If the arms or seeds cannot be compared, or a held-out
            fold cannot support an interval.
    """
    require_arms([trainer.model_name.value for trainer in trainers], reference)
    require_seeds(seeds)
    results: list[ArmResult] = []
    seed_comparisons: list[SeedComparisons] = []
    for seed in seeds:
        seed_results, comparisons = evaluate_seed(
            trainers, build_split(seed), seed, reference, settings, costs
        )
        results.extend(seed_results)
        seed_comparisons.append(comparisons)
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "dataset": dataset,
        "config": config,
        "resampling": settings,
        "alert_costs": costs,
        "action_names": list(ALERT_ACTIONS),
        "reference": reference,
        "seeds": list(seeds),
        "results": results,
        "seed_comparisons": seed_comparisons,
        "pooled": pool_seed_comparisons(seed_comparisons, settings["confidence"]),
        "fingerprint": fingerprint,
    }


__all__ = [
    "NEVER_ALERT",
    "evaluate_seed",
    "require_arms",
    "require_seeds",
    "run_paired_evaluation",
]
