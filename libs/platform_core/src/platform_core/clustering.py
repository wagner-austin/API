"""How much of a sample size is real when the units are not independent.

WHAT THIS ANSWERS, AND WHY EVERY OTHER INSTRUMENT IN THIS PACKAGE NEEDED IT.
:mod:`platform_core.minimum_detectable_effect` asks what a design of *n*
units could have resolved. Every one of its instruments takes *n* as given
and independent. When the units are held-out files from a handful of
packages, sentences from a handful of documents, or trials from a handful of
match seeds, they are not independent, and the *n* handed to those
instruments is larger than the amount of information present. Correlation
between units inflates significance, and nothing downstream can detect it
from the counts alone.

THE ONE THAT MUST BE MEASURED ON THE DIFFERENCE, NOT THE OUTCOME. For a
PAIRED comparison the correlated quantity is the per-unit DIFFERENCE --
``candidate`` minus ``baseline``, in ``{-1, 0, +1}`` for a binary outcome --
and not the raw outcome of either arm. The two are different numbers and the
gap is not small. Measured on ``code-style``'s 875 committed items on
2026-09-09:

    stratum, clustering unit          ICC of difference   ICC of raw outcome
    guards, containing directory              +0.0279              +0.1405
    guards, top-level category                +0.0442              +0.0814
    aggregate, containing directory           +0.0585              -0.0308

At the containing-directory unit the raw series is five times the one that
applies, and at the aggregate it has the OPPOSITE SIGN. A correction built
from the raw series would have deflated a real result by five times too much
in one row and applied a backwards correction in another. The difference
series is what :func:`clustered_paired_power` takes, and the parameter is
named for it so a caller passing pass-rates has to rename their variable to
do so.

WHY ``m0`` AND NOT A MEAN CLUSTER SIZE. Killip, Mahfoud and Pearce give
``DE = 1 + rho * (m - 1)`` for the special case of EQUAL cluster sizes. Real
corpora are not equal: code-style's largest package holds 165 of 875 files
against a median of 5. The standard unequal-size replacement is

    m0 = (N - sum(n_j^2) / N) / (k - 1)

which is smaller than the arithmetic mean whenever the sizes vary, and it is
what this module uses. Substituting the arithmetic mean would overstate the
design effect on unequal clusters, so this is the direction that costs the
author rather than flatters them.

THE REFUSAL THAT MATTERS MOST, AND IT IS A FLOOR RATHER THAN AN ERROR.
One-way ICC is a variance-ratio estimate and goes NEGATIVE whenever the
within-cluster spread exceeds the between-cluster spread -- which is common,
and on code-style's aggregate happens at two of three clustering units. Taken
literally a negative ICC yields ``DE < 1`` and an effective *n* LARGER than
the sample. Clustering cannot manufacture information, and a number that
says it did is exactly the kind that gets copied into a table because it
favours whoever computed it. A first draft of this arithmetic, run as a
throwaway script, printed ``effective n = 1718.6`` for 875 items.

So :attr:`~platform_core.power_types.ClusteredPairedPower.design_effect` is
floored at 1.0 and the effective *n* can never exceed the real one. The raw
estimate is carried UNFLOORED beside it in
:attr:`~platform_core.power_types.ClusteredPairedPower.intracluster_correlation`,
so a reader sees that the floor was applied instead of seeing a suspiciously
round 1.000 with no explanation. A negative ICC means NO POSITIVE CLUSTERING
WAS DETECTED AT THIS UNIT -- it does not mean the units are anti-correlated
in any way worth banking.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not choose the clustering
unit. On the same 875 items *k* ranges from 14 to 336 depending on whether a
"cluster" is a top-level category, a package or a directory, and the design
effect moves with it. That choice is a claim about which units share
whatever makes them correlated, it belongs to the study, and an instrument
that picked one would be picking a p-value. Callers pass the grouping they
have argued for, and the record carries *k* and ``m0`` so a reader can see
which one was chosen.

It also carries no :class:`~platform_core.power_types.PowerVerdict`, for the
reason :class:`~platform_core.power_types.McNemarPower` and
:class:`~platform_core.power_types.NetDifferencePower` carry none: "is this
design effect large?" has no threshold that is not a judgement about the
study, and a ``TESTED`` here would be a classification nobody made.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from platform_core.error_codes import StatisticalPowerErrorCode
from platform_core.errors import AppError
from platform_core.power_types import ClusteredPairedPower, PowerInstrument

MINIMUM_CLUSTERS = 2


def _require_estimable(cluster_sizes: Sequence[int]) -> None:
    """Refuse a grouping that cannot support a variance ratio.

    Args:
        cluster_sizes: Number of units in each non-empty cluster.

    Raises:
        AppError: With ``POWER_CLUSTERS_INSUFFICIENT`` when fewer than two
            clusters carry units, or when every cluster holds exactly one.
            The first leaves no between-cluster term, the second leaves no
            within-cluster term, and in both cases the ratio is undefined
            rather than zero. Returning 0.0 for either would publish "no
            clustering detected" where the honest answer is that this
            grouping cannot answer the question.
    """
    clusters = len(cluster_sizes)
    total = sum(cluster_sizes)
    if clusters < MINIMUM_CLUSTERS:
        raise AppError(
            StatisticalPowerErrorCode.POWER_CLUSTERS_INSUFFICIENT,
            f"an intracluster correlation needs at least {MINIMUM_CLUSTERS} non-empty "
            f"clusters and this grouping has {clusters}; with one cluster there is no "
            "between-cluster variance to compare the within-cluster variance against, "
            "so the ratio is undefined rather than zero",
        )
    if total <= clusters:
        raise AppError(
            StatisticalPowerErrorCode.POWER_CLUSTERS_INSUFFICIENT,
            f"{total} unit(s) across {clusters} cluster(s) leaves no within-cluster "
            "variance to estimate: every cluster holds at most one unit, which is the "
            "unclustered case wearing a grouping. Group by a coarser unit, or use an "
            "instrument that does not correct for clustering",
        )


def _group_sizes(differences_by_cluster: Mapping[str, Sequence[float]]) -> list[int]:
    """List the sizes of the clusters that actually hold units.

    Args:
        differences_by_cluster: Per-unit differences, grouped.

    Returns:
        The size of each NON-EMPTY cluster. Empty clusters are dropped
        rather than counted as zero-size: an empty group contributes no
        units and no variance, and counting it would deflate ``m0`` through
        ``k`` while adding nothing to ``N``.
    """
    return [len(values) for values in differences_by_cluster.values() if values]


def average_cluster_size(cluster_sizes: Sequence[int]) -> float:
    """Killip's ``m0``, the cluster size an unequal design behaves as.

    Equal to the arithmetic mean exactly when every cluster is the same
    size, and strictly smaller otherwise.

    Args:
        cluster_sizes: Number of units in each non-empty cluster.

    Returns:
        ``(N - sum(n_j^2) / N) / (k - 1)``.

    Raises:
        AppError: With ``POWER_CLUSTERS_INSUFFICIENT``, via
            :func:`_require_estimable`.
    """
    _require_estimable(cluster_sizes)
    total = sum(cluster_sizes)
    sum_of_squares = sum(size * size for size in cluster_sizes)
    return (total - sum_of_squares / total) / (len(cluster_sizes) - 1)


def intracluster_correlation(differences_by_cluster: Mapping[str, Sequence[float]]) -> float:
    """One-way random-effects ICC of the per-unit paired differences.

    The variance ratio ``(MSB - MSW) / (MSB + (m0 - 1) * MSW)``, with ``m0``
    from :func:`average_cluster_size` so unequal clusters are handled rather
    than assumed away.

    THE RETURN IS UNFLOORED and may be negative. That is deliberate: the
    floor belongs on the design effect, where its consequence is visible,
    and a caller reading this function's name expects the estimate rather
    than a policy about it.

    Args:
        differences_by_cluster: Per-unit differences of the PAIRED outcome
            -- candidate minus baseline -- grouped by clustering unit.
            Values in ``{-1, 0, +1}`` for a binary outcome, but any real
            differences work. Passing one arm's raw outcomes here answers a
            different question; see the module docstring.

    Returns:
        The estimate, in ``[-1, 1]``. Exactly 0.0 when both mean squares are
        zero -- every unit identical, so no variance is partitionable and
        there is nothing for clusters to explain.

    Raises:
        AppError: With ``POWER_CLUSTERS_INSUFFICIENT``, via
            :func:`average_cluster_size`.
    """
    groups = [list(values) for values in differences_by_cluster.values() if values]
    sizes = [len(group) for group in groups]
    m0 = average_cluster_size(sizes)
    total = sum(sizes)
    grand_mean = sum(sum(group) for group in groups) / total
    between = sum(len(group) * (sum(group) / len(group) - grand_mean) ** 2 for group in groups) / (
        len(groups) - 1
    )
    within = sum((value - sum(group) / len(group)) ** 2 for group in groups for value in group) / (
        total - len(groups)
    )
    denominator = between + (m0 - 1) * within
    if denominator == 0:
        return 0.0
    return (between - within) / denominator


def clustered_paired_power(
    differences_by_cluster: Mapping[str, Sequence[float]], unit: str
) -> ClusteredPairedPower:
    """How many independent units a clustered paired comparison really had.

    Args:
        differences_by_cluster: Per-unit differences of the paired outcome,
            grouped by clustering unit. See
            :func:`intracluster_correlation` for what belongs here and what
            does not.
        unit: What one cluster IS, in the study's own words -- ``"package"``,
            ``"containing directory"``, ``"match seed"``. Carried on the
            record because the design effect is meaningless without it: the
            same items grouped three ways give three different answers, and
            a number quoted without its unit will be read as though the
            grouping were forced.

    Returns:
        The record. ``effective_sample_size`` is never greater than
        ``total_units``.

    Raises:
        AppError: With ``POWER_CLUSTERS_INSUFFICIENT`` when the grouping
            cannot support the estimate, via
            :func:`intracluster_correlation`. With
            ``POWER_SAMPLE_SIZE_INVALID`` when ``unit`` is blank -- a record
            that cannot say what it grouped by is the one failure mode this
            record's own docstring calls out, and an empty string would
            serialise and publish as happily as a real answer.
    """
    if not unit.strip():
        raise AppError(
            StatisticalPowerErrorCode.POWER_SAMPLE_SIZE_INVALID,
            "a clustering unit must be named: the same units grouped by directory "
            "and by package give different design effects, so a record that does "
            "not say which grouping produced its number cannot be checked by anyone",
        )
    sizes = _group_sizes(differences_by_cluster)
    correlation = intracluster_correlation(differences_by_cluster)
    m0 = average_cluster_size(sizes)
    total = sum(sizes)
    design_effect = max(1.0, 1.0 + correlation * (m0 - 1.0))
    return ClusteredPairedPower(
        instrument=PowerInstrument.CLUSTERED_PAIRED.value,
        unit=unit,
        clusters=len(sizes),
        total_units=total,
        largest_cluster=max(sizes),
        average_cluster_size=m0,
        intracluster_correlation=correlation,
        design_effect=design_effect,
        effective_sample_size=total / design_effect,
    )


__all__ = [
    "MINIMUM_CLUSTERS",
    "average_cluster_size",
    "clustered_paired_power",
    "intracluster_correlation",
]
