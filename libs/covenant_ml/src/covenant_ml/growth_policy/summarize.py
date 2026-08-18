"""Reduction of per-seed results to the per-arm means a table reports.

Pure functions over a result list, with no clock, no I/O and no vendor, so the
aggregation is testable from fixed records rather than from a real experiment.

Arms keep first-appearance order rather than being sorted. The report's reading
order is the order the arms were specified in, which is what makes a table read
as a progression -- depth-wise, then each leaf budget in turn -- instead of
alphabetically.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from .types import ERR_NO_RESULTS, ArmResult, ArmSummary


def summarize_arms(results: Sequence[ArmResult]) -> list[ArmSummary]:
    """Average every arm's results across the seeds it was measured at.

    Args:
        results: Every arm at every seed, in any order.

    Returns:
        One summary per arm, in the order the arms first appear.

    Raises:
        ValueError: If there are no results, which would produce an empty table
            presented as a completed experiment.
    """
    if len(results) == 0:
        raise ValueError(f"[{ERR_NO_RESULTS}] Cannot summarise an empty result set")
    order: list[str] = []
    grouped: dict[str, list[ArmResult]] = {}
    for result in results:
        arm = result["arm"]
        if arm not in grouped:
            grouped[arm] = []
            order.append(arm)
        grouped[arm].append(result)
    return [_summarize_one(arm, grouped[arm]) for arm in order]


def _summarize_one(arm: str, results: Sequence[ArmResult]) -> ArmSummary:
    """Average one arm's results across seeds.

    Args:
        arm: The arm's display name.
        results: That arm's results, one per seed.

    Returns:
        The arm's summary.
    """
    return {
        "arm": arm,
        "seed_count": len(results),
        "fit_seconds": statistics.fmean(result["fit_seconds"] for result in results),
        "auc_roc": statistics.fmean(result["auc_roc"] for result in results),
        "auc_pr": statistics.fmean(result["auc_pr"] for result in results),
        "log_loss": statistics.fmean(result["log_loss"] for result in results),
        "mean_leaves": statistics.fmean(result["mean_leaves"] for result in results),
    }


__all__ = ["summarize_arms"]
