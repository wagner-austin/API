"""AUC as a rank statistic: DeLong's paired test and a weighted form for resampling.

THE AUC HERE IS THE MANN-WHITNEY ONE, ties counted half. That is the quantity
DeLong's variance is derived for, and it differs from
:func:`covenant_ml.metrics.compute_auc` only where two rows share a score:
that function walks rows in ``argsort`` order and so splits a tie in whatever
order the sort left it. On untied scores the two agree to rounding, which the
tests assert.

DELONG (DeLong, DeLong and Clarke-Pearson 1988, in the midrank form of Sun
and Xu 2014). Each positive row gets a placement, the share of negatives it
outranks; each negative gets one too. An AUC is the mean placement, so two
models' AUC difference is the mean of per-row placement DIFFERENCES, and its
variance is the variance of those differences over positives plus that over
negatives. The pairing is what makes it powerful: a row both models find hard
costs neither of them anything in the difference.

DeLong assumes the rows are independent. Where they are not (company-years,
trading days), :func:`bootstrap_auc_difference` resamples whole units
instead, using :func:`weighted_auc` so no resampled array is ever built.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .inputs import require_observations
from .intervals import normal_interval, percentile_interval
from .resampling import Resampling, bootstrap_weights
from .types import (
    ERR_DEGENERATE_RESAMPLE,
    ERR_INVALID_ARGUMENT,
    ERR_LENGTH_MISMATCH,
    ERR_SINGLE_CLASS,
    IntervalEstimate,
    IntervalMethod,
)


def _require_aligned(labels: NDArray[np.int64], *scores: NDArray[np.float64]) -> None:
    """Refuse score arrays that do not cover the labels.

    Args:
        labels: Binary labels.
        *scores: Score arrays that must each have one entry per label.

    Raises:
        ValueError: If any score array differs in length from the labels.
    """
    if labels.ndim != 1 or not bool(np.all((labels == 0) | (labels == 1))):
        raise ValueError(f"[{ERR_INVALID_ARGUMENT}] labels must be a vector of binary values")
    for score in scores:
        require_observations(score, "scores")
    lengths = [len(score) for score in scores]
    if any(length != len(labels) for length in lengths):
        raise ValueError(
            f"[{ERR_LENGTH_MISMATCH}] scores must cover {len(labels)} labels, got {lengths}"
        )


def midranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rank values from 1, giving tied values the mean of their ranks.

    Args:
        values: Values to rank.

    Returns:
        One rank per value, in the input's order.
    """
    order: NDArray[np.intp] = np.argsort(values, kind="mergesort")
    ordered: NDArray[np.float64] = values[order]
    unique_parts = np.unique(ordered, return_index=True, return_counts=True)
    first: NDArray[np.intp] = unique_parts[1]
    counts: NDArray[np.intp] = unique_parts[2]
    mean_rank: NDArray[np.float64] = first.astype(np.float64) + (counts + 1) / 2.0
    ranked_in_order: NDArray[np.float64] = np.repeat(mean_rank, counts)
    ranks: NDArray[np.float64] = np.empty(len(values), dtype=np.float64)
    ranks[order] = ranked_in_order
    return ranks


class _Placements(NamedTuple):
    """One model's AUC and its per-row placements.

    Args:
        auc: The Mann-Whitney AUC.
        positive: Share of negatives each positive outranks.
        negative: Share of positives each negative is outranked by.
    """

    auc: float
    positive: NDArray[np.float64]
    negative: NDArray[np.float64]


def _placements(labels: NDArray[np.int64], scores: NDArray[np.float64]) -> _Placements:
    """Compute one model's AUC and placements from midranks.

    Args:
        labels: Binary labels with at least two of each class.
        scores: Scores, one per label.

    Returns:
        The AUC and placements.
    """
    positive_mask: NDArray[np.bool_] = labels == 1
    negative_mask: NDArray[np.bool_] = labels == 0
    n_pos = int(np.sum(positive_mask))
    n_neg = int(np.sum(negative_mask))
    overall = midranks(scores)
    overall_pos: NDArray[np.float64] = overall[positive_mask]
    overall_neg: NDArray[np.float64] = overall[negative_mask]
    within_pos = midranks(scores[positive_mask])
    within_neg = midranks(scores[negative_mask])
    auc = (float(np.sum(overall_pos)) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return _Placements(
        auc=auc,
        positive=(overall_pos - within_pos) / n_neg,
        negative=1.0 - (overall_neg - within_neg) / n_pos,
    )


def _sample_variance(values: NDArray[np.float64]) -> float:
    """Sample variance with one degree of freedom removed.

    Args:
        values: At least two values.

    Returns:
        The unbiased variance.
    """
    mean = float(np.sum(values)) / len(values)
    centred: NDArray[np.float64] = values - mean
    return float(np.sum(centred * centred)) / (len(values) - 1)


class AucDifference(NamedTuple):
    """Two models' AUCs on shared rows and the interval on their difference.

    Args:
        auc_arm: The arm's AUC.
        auc_reference: The reference's AUC.
        difference: ``auc_arm - auc_reference`` with its interval.
    """

    auc_arm: float
    auc_reference: float
    difference: IntervalEstimate


def delong_auc_difference(
    labels: NDArray[np.int64],
    scores_arm: NDArray[np.float64],
    scores_reference: NDArray[np.float64],
    confidence: float,
) -> AucDifference:
    """Test two models' AUC difference on the same independent rows.

    Args:
        labels: Binary labels (0 or 1).
        scores_arm: The arm's scores, one per label.
        scores_reference: The reference's scores, one per label.
        confidence: Coverage of the interval.

    Returns:
        Both AUCs and a normal interval on their difference.

    Raises:
        ValueError: If the arrays differ in length, or either class has fewer
            than two rows, which leaves its placement variance undefined.
    """
    _require_aligned(labels, scores_arm, scores_reference)
    n_pos = int(np.sum(labels))
    n_neg = len(labels) - n_pos
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            f"[{ERR_SINGLE_CLASS}] DeLong needs at least 2 rows of each class, "
            f"got {n_pos} positive and {n_neg} negative"
        )
    arm = _placements(labels, scores_arm)
    reference = _placements(labels, scores_reference)
    variance = (
        _sample_variance(arm.positive - reference.positive) / n_pos
        + _sample_variance(arm.negative - reference.negative) / n_neg
    )
    return AucDifference(
        auc_arm=arm.auc,
        auc_reference=reference.auc,
        difference=normal_interval(
            IntervalMethod.DELONG,
            arm.auc - reference.auc,
            math.sqrt(variance),
            confidence,
            n_pos + n_neg,
        ),
    )


class ScoreBins(NamedTuple):
    """Scores reduced to their distinct values, ascending.

    Args:
        bins: Index of each row's score among the distinct scores.
        n_bins: Number of distinct scores.
    """

    bins: NDArray[np.int64]
    n_bins: int


def score_bins(scores: NDArray[np.float64]) -> ScoreBins:
    """Index every row by its score's rank among the distinct scores.

    Computed once per model; every resample's AUC then costs two weighted
    counts and a cumulative sum, with no sort.

    Args:
        scores: Scores, one per row.

    Returns:
        The binning.
    """
    unique_parts = np.unique(scores, return_inverse=True)
    distinct: NDArray[np.float64] = unique_parts[0]
    inverse: NDArray[np.intp] = unique_parts[1]
    return ScoreBins(bins=inverse.astype(np.int64), n_bins=len(distinct))


def weighted_auc(
    binned: ScoreBins,
    labels: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> float:
    """The Mann-Whitney AUC with each row counted ``weight`` times.

    Args:
        binned: The rows' score bins.
        labels: Binary labels.
        weights: Non-negative row weights. All ones gives the plain AUC.

    Returns:
        The weighted AUC, ties counted half.

    Raises:
        ValueError: If the weights leave either class empty, where the AUC is
            undefined.
    """
    positive: NDArray[np.float64] = weights * labels
    negative: NDArray[np.float64] = weights * (1 - labels)
    # Weighted bincounts are float64 at run time; numpy's stubs type them as
    # integer, so astype states the real dtype.
    pos_by_bin: NDArray[np.float64] = np.bincount(
        binned.bins, weights=positive, minlength=binned.n_bins
    ).astype(np.float64)
    neg_by_bin: NDArray[np.float64] = np.bincount(
        binned.bins, weights=negative, minlength=binned.n_bins
    ).astype(np.float64)
    total_pos = float(np.sum(pos_by_bin))
    total_neg = float(np.sum(neg_by_bin))
    if total_pos == 0.0 or total_neg == 0.0:
        raise ValueError(
            f"[{ERR_SINGLE_CLASS}] the AUC needs weight on both classes, got "
            f"{total_pos} positive and {total_neg} negative"
        )
    neg_below: NDArray[np.float64] = np.cumsum(neg_by_bin) - neg_by_bin
    outranked: NDArray[np.float64] = pos_by_bin * (neg_below + 0.5 * neg_by_bin)
    return float(np.sum(outranked)) / (total_pos * total_neg)


def bootstrap_auc_difference(
    labels: NDArray[np.int64],
    scores_arm: NDArray[np.float64],
    scores_reference: NDArray[np.float64],
    resampling: Resampling,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> AucDifference:
    """Interval two models' AUC difference by resampling whole units.

    Args:
        labels: Binary labels (0 or 1).
        scores_arm: The arm's scores, one per label.
        scores_reference: The reference's scores, one per label.
        resampling: Which rows form one independent unit.
        n_resamples: Bootstrap resamples.
        seed: Seed of the draws.
        confidence: Coverage of the interval.

    Returns:
        Both AUCs and a percentile interval on their difference.

    Raises:
        ValueError: If the arrays differ in length or the sample has one
            class, or a resample draws units of one class only. The last is
            refused rather than skipped: dropping such resamples conditions
            the interval on an event, and a class that rare is better
            served by more units than by a quietly biased interval.
    """
    _require_aligned(labels, scores_arm, scores_reference)
    arm_bins = score_bins(scores_arm)
    reference_bins = score_bins(scores_reference)
    ones: NDArray[np.float64] = np.ones(len(labels), dtype=np.float64)
    auc_arm = weighted_auc(arm_bins, labels, ones)
    auc_reference = weighted_auc(reference_bins, labels, ones)

    draws: list[float] = []
    for index, weights in enumerate(bootstrap_weights(resampling, n_resamples, seed)):
        drawn_positive = float(np.sum(weights * labels))
        drawn_total = float(np.sum(weights))
        if drawn_positive == 0.0 or drawn_positive == drawn_total:
            raise ValueError(
                f"[{ERR_DEGENERATE_RESAMPLE}] resample {index} drew units of one class "
                f"only ({drawn_positive} of {drawn_total} weighted rows positive)"
            )
        draws.append(
            weighted_auc(arm_bins, labels, weights) - weighted_auc(reference_bins, labels, weights)
        )
    return AucDifference(
        auc_arm=auc_arm,
        auc_reference=auc_reference,
        difference=percentile_interval(
            auc_arm - auc_reference,
            np.asarray(draws, dtype=np.float64),
            confidence,
            resampling.n_units,
        ),
    )


__all__ = [
    "AucDifference",
    "ScoreBins",
    "bootstrap_auc_difference",
    "delong_auc_difference",
    "midranks",
    "score_bins",
    "weighted_auc",
]
