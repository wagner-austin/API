"""The evaluation's headline numbers, as run-record observations.

The harness already averages every declared quality metric within every arm.
What it cannot build is the evaluation's own claim: each pooled difference
with its interval, and each policy's mean value. Those are handed to it as
derived observations, the same path the fit-time ratios take.

Interval bounds are recorded as observations of their own, so two records of
this experiment can be read side by side bound by bound, not only by point.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.run_record import Observation

from .types import PooledComparison, SeedComparisons


def pooled_observations(pooled: Sequence[PooledComparison]) -> tuple[Observation, ...]:
    """Name each pooled difference, its bounds and its p-value.

    Args:
        pooled: The across-seed comparisons.

    Returns:
        Four observations per comparison, named
        ``pooled_<arm>_vs_<reference>.<metric>.<part>``.
    """
    observations: list[Observation] = []
    for comparison in pooled:
        base = (
            f"pooled_{comparison['arm']}_vs_{comparison['reference']}.{comparison['metric'].value}"
        )
        interval = comparison["difference"]
        observations.extend(
            [
                Observation(name=f"{base}.difference", value=interval["estimate"]),
                Observation(name=f"{base}.ci_low", value=interval["ci_low"]),
                Observation(name=f"{base}.ci_high", value=interval["ci_high"]),
                Observation(name=f"{base}.p_value", value=interval["p_value"]),
            ]
        )
    return tuple(observations)


def policy_observations(seed_comparisons: Sequence[SeedComparisons]) -> tuple[Observation, ...]:
    """Average each policy's per-row value across seeds.

    Args:
        seed_comparisons: Every seed's policies.

    Returns:
        One observation per policy, ``mean_policy_value.<arm>``, in first-seen
        order; empty when no seed carried a policy.
    """
    totals: dict[str, list[float]] = {}
    for entry in seed_comparisons:
        for policy in entry["policies"]:
            totals.setdefault(policy["arm"], []).append(policy["value"]["estimate"])
    return tuple(
        Observation(name=f"mean_policy_value.{arm}", value=sum(values) / len(values))
        for arm, values in totals.items()
    )


__all__ = ["policy_observations", "pooled_observations"]
