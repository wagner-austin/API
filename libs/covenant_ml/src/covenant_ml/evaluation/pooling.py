"""Carrying within-seed comparisons across seeds.

Each seed's comparison answers "on THESE held-out companies, did the arm beat
the reference". Pooling asks whether it still does when the companies that
trained and tested the models change, which is the question a reader of the
scoreboard is actually asking. One difference per seed goes into a corrected
t interval over seeds (:func:`.intervals.pool_across_seeds`); three seeds
give two degrees of freedom, and the interval says so by being wide.
"""

from __future__ import annotations

from collections.abc import Sequence

from .intervals import pool_across_seeds
from .types import ERR_TOO_FEW_SEEDS, ComparisonMetric, PooledComparison, SeedComparisons


def held_out_ratio(seed_comparisons: Sequence[SeedComparisons]) -> float:
    """Held-out rows over training rows, averaged over seeds.

    Args:
        seed_comparisons: One entry per seed, at least two.

    Returns:
        The mean of ``n_test / n_train``.

    Raises:
        ValueError: If fewer than two seeds are given, where there is nothing
            to pool.
    """
    if len(seed_comparisons) < 2:
        raise ValueError(
            f"[{ERR_TOO_FEW_SEEDS}] pooling needs at least 2 seeds, got {len(seed_comparisons)}"
        )
    ratios = [entry["n_test"] / entry["n_train"] for entry in seed_comparisons]
    return sum(ratios) / len(ratios)


def pool_seed_comparisons(
    seed_comparisons: Sequence[SeedComparisons],
    confidence: float,
) -> list[PooledComparison]:
    """Pool every (arm, reference, metric) difference across seeds.

    Args:
        seed_comparisons: One entry per seed, each holding the same set of
            comparisons.
        confidence: Coverage of the pooled intervals.

    Returns:
        One pooled comparison per (arm, reference, metric), in first-seen
        order.

    Raises:
        ValueError: If fewer than two seeds are given, which leaves no
            seed-to-seed spread to measure.
    """
    ratio = held_out_ratio(seed_comparisons)
    keys: list[tuple[str, str, ComparisonMetric]] = []
    estimates: dict[tuple[str, str, ComparisonMetric], list[float]] = {}
    errors: dict[tuple[str, str, ComparisonMetric], list[float]] = {}
    for entry in seed_comparisons:
        for comparison in entry["comparisons"]:
            key = (comparison["arm"], comparison["reference"], comparison["metric"])
            if key not in estimates:
                keys.append(key)
                estimates[key] = []
                errors[key] = []
            estimates[key].append(comparison["difference"]["estimate"])
            errors[key].append(comparison["difference"]["std_error"])
    return [
        {
            "metric": metric,
            "arm": arm,
            "reference": reference,
            "per_seed": estimates[(arm, reference, metric)],
            "difference": pool_across_seeds(
                estimates[(arm, reference, metric)],
                errors[(arm, reference, metric)],
                ratio,
                confidence,
            ),
        }
        for arm, reference, metric in keys
    ]


__all__ = ["held_out_ratio", "pool_seed_comparisons"]
