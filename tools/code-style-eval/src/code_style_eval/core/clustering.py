"""Which held-out files count as one unit, and the difference series per item.

THE DOMAIN HALF OF A CORRECTION WHOSE ARITHMETIC LIVES ELSEWHERE.
:mod:`platform_core.clustering` computes the intracluster correlation and the
design effect from a grouping. It deliberately does not choose the grouping,
because that choice is a claim about which files share whatever makes them
correlated, and an instrument that picked one would be picking a p-value.
This module is where that claim is made for THIS corpus, and it makes three
of them rather than one.

WHY THREE UNITS AND NO DEFAULT. Over the same 875 items the number of
clusters is 14, 64 or 336 depending on whether a cluster is a top-level
category, a package or a containing directory, and the design effect moves
with it. Files in one directory plainly share an author and a shape; files
under ``libs/`` share much less. Nobody has established which level the
correlation actually lives at, so all three are reported and the reader sees
the spread. Picking the one that keeps a p-value under 0.05 is exactly the
move this reporting exists to make visible.

WHAT THE DIFFERENCE SERIES IS, AND THE MISTAKE IT EXISTS TO PREVENT. The
correlated quantity for a PAIRED comparison is the per-item difference --
candidate minus baseline, in ``{-1, 0, +1}`` -- and not either arm's raw
pass indicator. On this corpus the two disagree by up to five times and, at
the aggregate, in sign. :func:`paired_differences` is the only thing here
that produces a series, and it produces that one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum

from code_style_eval.contracts.outcomes import CHECKERS, ItemOutcome


class ClusteringUnit(StrEnum):
    """What one cluster is, in the corpus's own path structure.

    The values are the words the wiki table uses, so a record and a table
    row name the same thing rather than requiring a reader to map between
    two vocabularies.
    """

    TOP_LEVEL_CATEGORY = "top-level category"
    PACKAGE = "package"
    CONTAINING_DIRECTORY = "containing directory"


def cluster_key(item_id: str, unit: ClusteringUnit) -> str:
    """Project a held-out file's path onto its cluster.

    Args:
        item_id: Repository-relative path, ``/``-separated, as written in
            the outcome rows.
        unit: Which grouping to project onto.

    Returns:
        The cluster key. For a path with fewer segments than the unit asks
        for, the whole path is its own cluster -- a top-level file belongs
        to no package and grouping it with unrelated top-level files would
        assert a correlation nothing supports. Under
        :attr:`ClusteringUnit.CONTAINING_DIRECTORY` such a file yields the
        empty string, which is a real and shared cluster: the repository
        root.
    """
    segments = item_id.split("/")
    if unit is ClusteringUnit.CONTAINING_DIRECTORY:
        return "/".join(segments[:-1])
    depth = 1 if unit is ClusteringUnit.TOP_LEVEL_CATEGORY else 2
    return "/".join(segments[:depth])


def paired_differences(
    baseline: Mapping[str, ItemOutcome],
    candidate: Mapping[str, ItemOutcome],
    checker: str,
) -> dict[str, int]:
    """Per-item candidate-minus-baseline difference, over the shared items.

    Args:
        baseline: The baseline arm's outcomes, keyed by item id.
        candidate: The candidate arm's outcomes, keyed by item id.
        checker: A name from
            :data:`~code_style_eval.contracts.outcomes.CHECKERS` to score
            one checker, or ``"all"`` for the combined
            :attr:`ItemOutcome.all_passed`.

    Returns:
        A difference in ``{-1, 0, +1}`` per shared item. Items only one arm
        was scored on are absent, matching
        :func:`~code_style_eval.core.scoring.paired_counts` -- an item one
        arm never produced is not evidence about the other.

    Raises:
        ValueError: If ``checker`` names neither a known checker nor
            ``"all"``, or if a shared item lacks a row for the named
            checker. A missing checker row silently scored as a failure
            would move items between the discordant cells.
    """
    if checker != "all" and checker not in CHECKERS:
        raise ValueError(f"checker must be 'all' or one of {CHECKERS}; got {checker!r}")
    return {
        item_id: _passed(candidate[item_id], checker) - _passed(baseline[item_id], checker)
        for item_id in sorted(set(baseline) & set(candidate))
    }


def _passed(outcome: ItemOutcome, checker: str) -> int:
    """Read one item's pass indicator as 0 or 1.

    Args:
        outcome: The item's row.
        checker: A checker name, or ``"all"``.

    Returns:
        1 when it passed, 0 when it did not.

    Raises:
        ValueError: When the row carries no entry for the named checker.
    """
    if checker == "all":
        return int(outcome["all_passed"])
    for check in outcome["checks"]:
        if check["checker"] == checker:
            return int(check["passed"])
    raise ValueError(
        f"item {outcome['item_id']!r} has no {checker!r} row; scoring a missing "
        "checker as a failure would move the item between discordant cells"
    )


def grouped_differences(
    differences: Mapping[str, int], unit: ClusteringUnit
) -> dict[str, list[float]]:
    """Group a difference series by clustering unit.

    Args:
        differences: Per-item differences from :func:`paired_differences`.
        unit: Which grouping to apply.

    Returns:
        The mapping :func:`platform_core.clustering.clustered_paired_power`
        takes: cluster key to that cluster's differences.
    """
    grouped: dict[str, list[float]] = {}
    for item_id, difference in differences.items():
        grouped.setdefault(cluster_key(item_id, unit), []).append(float(difference))
    return grouped


def both_finished(
    baseline: Sequence[str], candidate: Sequence[str], scored: Mapping[str, int]
) -> dict[str, int]:
    """Restrict a difference series to items BOTH arms ran to a stop token.

    The stratum every guard-pass figure on the wiki page is reported over.
    Restricting to it separates "the code was wrong" from "the decode was
    cut off", which score identically and mean different things.

    Args:
        baseline: Item ids the baseline arm finished.
        candidate: Item ids the candidate arm finished.
        scored: The difference series to restrict.

    Returns:
        The subset of ``scored`` whose ids both arms finished. Order follows
        ``scored``.
    """
    finished = set(baseline) & set(candidate)
    return {item_id: value for item_id, value in scored.items() if item_id in finished}


__all__ = [
    "ClusteringUnit",
    "both_finished",
    "cluster_key",
    "grouped_differences",
    "paired_differences",
]
