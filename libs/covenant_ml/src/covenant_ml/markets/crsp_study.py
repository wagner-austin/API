"""The frozen monthly CRSP study on its validation period, never its final period.

Rows are (decision, eligible security) pairs from ``build_crsp_panel``. A row's
label is whether the security's window value multiplier, terminal value
included once, beats the eligible universe's equal-weight multiplier; that
label becomes available at the window's exit close. The existing binary
trainers rank securities by predicted probability of beating the universe.

Validation decisions are scored in consecutive expanding folds. Each fold fits
on every earlier decision except the declared stopping months immediately
before it, which the trainers use for early stopping, and
``purge_label_overlap`` drops every training and stopping label not available
strictly before the next boundary. Every arm then runs through the same
self-financing holdings accounting at every frozen cost, and each candidate
and the equal-weight universe are compared with fixed momentum on identical
consecutive months by the paired moving-block bootstrap the design freezes.

Decisions on or after the final-period start are never constructed. Before
attaching outcomes, discard every window whose exit is on or after that start.
Thus validation selection consumes no final-period outcomes, including those
of positions entered before the boundary. This is a temporal rule, not a
process-level restriction on access to the source files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from fractions import Fraction
from math import ceil
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..benchmarking.protocols import TrainerProto
from ..benchmarking.splitting import SplitMasks, apply_split_masks
from ..evaluation.paired import compare_moving_block
from ..evaluation.temporal import purge_label_overlap
from ..evaluation.types import PairedComparison
from .crsp_calendar import crsp_month
from .crsp_extract import CrspSources
from .crsp_holdings import (
    CrspOutcomes,
    CrspPeriod,
    crsp_window_path,
    evaluate_crsp_holdings,
    index_crsp_outcomes,
)
from .crsp_panel import CrspDecision, CrspFeature, build_crsp_panel
from .crsp_protocol import CrspProtocol

#: The fixed baseline every candidate is measured against.
MOMENTUM = "momentum"
#: The eligible universe held equally, reported beside every candidate.
EQUAL_WEIGHT = "equal_weight"
#: Decision-time features the panel constructs, in their canonical order.
CRSP_FEATURES = ("momentum", "liquidity")


class CrspRows(NamedTuple):
    """One row per decision and eligible security, in decision then liquidity order.

    Args:
        features: Requested decision-time features, shape (n_rows, n_features).
        labels: One when the security's window multiplier beats the universe's.
        times: Decision close as a proleptic ordinal day.
        available: Exit close, when the label is first known, as an ordinal day.
        decisions: Index of each row's decision.
    """

    features: NDArray[np.float64]
    labels: NDArray[np.int64]
    times: NDArray[np.int64]
    available: NDArray[np.int64]
    decisions: NDArray[np.int64]


class CrspFold(NamedTuple):
    """One expanding validation fold's boundaries and purged row counts."""

    first_stopping: str
    first_scored: str
    n_train: int
    n_stopping: int
    n_purged: int
    n_scored: int


class CrspSeries(NamedTuple):
    """One arm's monthly accounting at one frozen cost."""

    arm: str
    cost_bps: int
    periods: tuple[CrspPeriod, ...]


class CrspScored(NamedTuple):
    """Scored decisions, their folds, every series and paired comparisons.

    The validation study scores expanding folds; the final study one fit.
    """

    decisions: tuple[CrspDecision, ...]
    folds: tuple[CrspFold, ...]
    series: tuple[CrspSeries, ...]
    comparisons: tuple[PairedComparison, ...]


def _ordinal(day: str) -> int:
    """Map a canonical date to its proleptic Gregorian ordinal day."""
    return date.fromisoformat(day).toordinal()


def _column(feature: CrspFeature, name: str) -> float:
    """Read one decision-time feature by its canonical name."""
    return feature.momentum if name == "momentum" else feature.liquidity


def require_crsp_features(features: tuple[str, ...]) -> None:
    """Refuse a feature the monthly panel does not construct at decision time.

    Args:
        features: Requested feature names.

    Raises:
        ValueError: CRSP-009 unless the names are a nonempty subset of
            ``CRSP_FEATURES``.
    """
    if not features or not set(features) <= set(CRSP_FEATURES):
        raise ValueError(f"[CRSP-009] features must be a nonempty subset of {CRSP_FEATURES}")


def crsp_study_rows(
    outcomes: CrspOutcomes, decisions: tuple[CrspDecision, ...], features: tuple[str, ...]
) -> CrspRows:
    """Attach dated universe-relative labels to every decision's eligible securities.

    Args:
        outcomes: Indexed verified evidence.
        decisions: Consecutive panel decisions.
        features: Requested names, a nonempty subset of ``CRSP_FEATURES``.

    Returns:
        Model rows; no security is dropped for missing outcome evidence.

    Raises:
        ValueError: CRSP-009 for an unsupported feature, an empty eligible
            universe or incomplete outcome evidence for any eligible security.
    """
    require_crsp_features(features)
    matrix: list[list[float]] = []
    labels: list[int] = []
    times: list[int] = []
    available: list[int] = []
    owners: list[int] = []
    for index, decision in enumerate(decisions):
        if not decision.features:
            raise ValueError(f"[CRSP-009] no eligible securities at {decision.window.decision}")
        values: list[float] = []
        for feature in decision.features:
            path = crsp_window_path(outcomes, feature.permno, decision.window)
            values.append(path.growth * (1.0 if path.terminal is None else path.terminal[1]))
        benchmark = sum(values) / len(values)
        for feature, value in zip(decision.features, values, strict=True):
            matrix.append([_column(feature, name) for name in features])
            labels.append(1 if value > benchmark else 0)
            times.append(_ordinal(decision.window.decision))
            available.append(_ordinal(decision.window.exit))
            owners.append(index)
    return CrspRows(
        np.asarray(matrix, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(times, dtype=np.int64),
        np.asarray(available, dtype=np.int64),
        np.asarray(owners, dtype=np.int64),
    )


def crsp_top(
    features: tuple[CrspFeature, ...], scores: Sequence[float], count: int
) -> tuple[int, ...]:
    """Select the highest scores, ties broken by ascending PERMNO.

    Args:
        features: One decision's eligible securities.
        scores: One score per security, in the same order.
        count: Securities to select.

    Returns:
        The selected PERMNOs in ascending order.
    """

    def rank(item: tuple[float, CrspFeature]) -> tuple[float, int]:
        """Order greatest score first, breaking ties by permanent identifier."""
        return -item[0], item[1].permno

    ranked = sorted(zip(scores, features, strict=True), key=rank)
    return tuple(sorted(feature.permno for _, feature in ranked[:count]))


def crsp_selected_count(fraction: float, eligible: int) -> int:
    """Round the declared decimal fraction of the universe up, without float error.

    Args:
        fraction: The frozen selection fraction, e.g. 0.1 for the top decile.
        eligible: Eligible securities at the decision.

    Returns:
        ``ceil(fraction * eligible)`` computed on the declared decimal.
    """
    return ceil(Fraction(repr(fraction)) * eligible)


def _fold_masks(rows: CrspRows, start: int, end: int, stopping_months: int) -> SplitMasks:
    """Assign earlier decisions to training and stopping and one fold to scoring.

    Args:
        rows: Model rows with their decision indexes.
        start: First scored decision index.
        end: Decision index after the last scored one.
        stopping_months: Decisions immediately before ``start`` used for stopping.

    Returns:
        Unpurged chronological fold membership.

    Raises:
        ValueError: CRSP-009 when no training decision precedes the stopping months.
    """
    stop = start - stopping_months
    if stop <= 0:
        raise ValueError("[CRSP-009] require training decisions before the stopping months")
    return SplitMasks(
        train=rows.decisions < stop,
        val=(rows.decisions >= stop) & (rows.decisions < start),
        test=(rows.decisions >= start) & (rows.decisions < end),
    )


def crsp_fold(
    rows: CrspRows,
    panel: tuple[CrspDecision, ...],
    scored: range,
    stopping_months: int,
    trainers: Sequence[TrainerProto],
    seed: int,
) -> tuple[CrspFold, dict[str, tuple[float, ...]]]:
    """Fit every trainer on earlier purged decisions and score one decision range.

    Args:
        rows: Model rows for ``panel`` with their decision indexes.
        panel: Decisions in calendar order.
        scored: Consecutive decision indexes scored by this fold.
        stopping_months: Decisions immediately before the range used for stopping.
        trainers: Distinctly named candidate trainers.
        seed: Model seed shared by every trainer.

    Returns:
        The fold's boundaries and purged counts, and each trainer's predicted
        probability per scored row in row order.

    Raises:
        ValueError: CRSP-009 when no training decision precedes the stopping
            months; label-overlap refusals propagate.
    """
    masks = _fold_masks(rows, scored.start, scored.stop, stopping_months)
    purged = purge_label_overlap(rows.times, rows.available, masks)
    split = apply_split_masks(rows.features, rows.labels, purged)
    scores: dict[str, tuple[float, ...]] = {}
    for trainer in trainers:
        proba = trainer.fit(split, seed).predict_positive_proba(split.x_test)
        scores[trainer.model_name.value] = tuple(
            float(proba.flat[index]) for index in range(len(proba))
        )
    train, stopping = int(np.count_nonzero(purged.train)), int(np.count_nonzero(purged.val))
    candidates = int(np.count_nonzero(masks.train)) + int(np.count_nonzero(masks.val))
    fold = CrspFold(
        panel[scored.start - stopping_months].window.decision,
        panel[scored.start].window.decision,
        train,
        stopping,
        candidates - train - stopping,
        int(np.count_nonzero(purged.test)),
    )
    return fold, scores


def _validation_start(decisions: tuple[CrspDecision, ...], first: str) -> int:
    """Locate the first validation decision.

    Raises:
        ValueError: CRSP-009 when the panel holds no validation decision.
    """
    for index, decision in enumerate(decisions):
        if decision.window.decision >= first:
            return index
    raise ValueError("[CRSP-009] the panel holds no validation decision")


def run_crsp_validation(
    sources: CrspSources,
    trainers: Sequence[TrainerProto],
    seed: int,
    *,
    features: tuple[str, ...],
    stopping_months: int,
    fold_months: int,
    confidence: float,
) -> CrspScored:
    """Score the validation period with purged expanding folds and paired holdings.

    Args:
        sources: Verified CRSP evidence carrying the frozen design.
        trainers: Distinctly named candidate trainers.
        seed: Model seed shared by every candidate.
        features: Requested decision-time features.
        stopping_months: Decision months before each fold used for early stopping.
        fold_months: Validation decision months scored per fold.
        confidence: Coverage of each moving-block interval.

    Returns:
        Every arm's accounting at every frozen cost and, at the primary cost,
        each candidate's and the universe's comparison with momentum. All scored
        exits are strictly before the frozen holdout start.

    Raises:
        ValueError: CRSP-009 for an unsupported schedule or incomplete evidence;
            fold, holdings and moving-block refusals propagate.
    """
    protocol = sources.protocol
    if stopping_months < 1 or fold_months < 1:
        raise ValueError("[CRSP-009] stopping and fold months must be positive")
    panel = crsp_known_panel(sources)
    outcomes = index_crsp_outcomes(sources)
    rows = crsp_study_rows(outcomes, panel, features)
    first = _validation_start(panel, protocol["validation_start"])
    scores: dict[str, list[float]] = {trainer.model_name.value: [] for trainer in trainers}
    folds: list[CrspFold] = []
    for start in range(first, len(panel), fold_months):
        scored_range = range(start, min(start + fold_months, len(panel)))
        fold, fold_scores = crsp_fold(rows, panel, scored_range, stopping_months, trainers, seed)
        for arm, values in fold_scores.items():
            scores[arm].extend(values)
        folds.append(fold)
    scored = panel[first:]
    series, comparisons = crsp_arms(outcomes, scored, scores, protocol, confidence)
    return CrspScored(scored, tuple(folds), series, comparisons)


def crsp_known_panel(sources: CrspSources) -> tuple[CrspDecision, ...]:
    """Build every decision whose outcome is known before the final period starts.

    Args:
        sources: Verified CRSP evidence carrying the frozen design.

    Returns:
        Decisions from the training start whose window exits strictly before
        ``holdout_start``; a window exiting on or after it is never labelled.
    """
    protocol = sources.protocol
    last = date.fromisoformat(protocol["holdout_start"]) - timedelta(days=1)
    return tuple(
        decision
        for decision in build_crsp_panel(sources, protocol["train_start"], last.isoformat())
        if decision.window.exit < protocol["holdout_start"]
    )


def crsp_arms(
    outcomes: CrspOutcomes,
    decisions: tuple[CrspDecision, ...],
    scores: Mapping[str, Sequence[float]],
    protocol: CrspProtocol,
    confidence: float,
) -> tuple[tuple[CrspSeries, ...], tuple[PairedComparison, ...]]:
    """Select, account and pair every arm on the same scored decisions.

    Args:
        outcomes: Indexed verified evidence covering every scored window.
        decisions: Scored decisions in calendar order.
        scores: Each candidate's score per eligible security, decisions concatenated.
        protocol: Frozen selection fraction, costs and moving-block settings.
        confidence: Coverage of each moving-block interval.

    Returns:
        Momentum, the equal-weight universe and each candidate at every frozen
        cost, and each non-momentum arm paired with momentum at the primary cost.

    Raises:
        ValueError: Holdings and moving-block refusals propagate.
    """
    selections: dict[str, list[tuple[int, ...]]] = {MOMENTUM: [], EQUAL_WEIGHT: []}
    selections.update({arm: [] for arm in scores})
    offset = 0
    for decision in decisions:
        members = decision.features
        count = crsp_selected_count(protocol["selection_fraction"], len(members))
        selections[MOMENTUM].append(crsp_top(members, [item.momentum for item in members], count))
        selections[EQUAL_WEIGHT].append(tuple(sorted(item.permno for item in members)))
        for arm, values in scores.items():
            selections[arm].append(crsp_top(members, values[offset : offset + len(members)], count))
        offset += len(members)
    windows = tuple(decision.window for decision in decisions)
    series = tuple(
        CrspSeries(arm, cost, evaluate_crsp_holdings(outcomes, windows, tuple(chosen), cost))
        for arm, chosen in selections.items()
        for cost in protocol["cost_bps"]
    )
    return series, _comparisons(series, protocol, confidence)


def _net_returns(periods: tuple[CrspPeriod, ...]) -> NDArray[np.float64]:
    """Collect one series' monthly net returns in window order."""
    values: list[float] = [period.net_return for period in periods]
    return np.asarray(values, dtype=np.float64)


def _comparisons(
    series: tuple[CrspSeries, ...], protocol: CrspProtocol, confidence: float
) -> tuple[PairedComparison, ...]:
    """Compare every non-momentum arm with momentum at the primary frozen cost.

    Args:
        series: Every arm at every cost, momentum included.
        protocol: Frozen primary cost and moving-block block length, draws and seed.
        confidence: Interval coverage.

    Returns:
        One paired moving-block comparison per non-momentum arm.
    """
    primary = protocol["cost_bps"][0]
    at_primary = {item.arm: item.periods for item in series if item.cost_bps == primary}
    reference = at_primary[MOMENTUM]
    ordinals: list[int] = [crsp_month(period.window.decision) for period in reference]
    months: NDArray[np.int64] = np.asarray(ordinals, dtype=np.int64)
    baseline = _net_returns(reference)
    return tuple(
        compare_moving_block(
            arm,
            MOMENTUM,
            _net_returns(periods),
            baseline,
            months,
            months,
            block_length=protocol["bootstrap_block_months"],
            n_resamples=protocol["bootstrap_draws"],
            seed=protocol["bootstrap_seed"],
            confidence=confidence,
        )
        for arm, periods in at_primary.items()
        if arm != MOMENTUM
    )


__all__ = [
    "CRSP_FEATURES",
    "EQUAL_WEIGHT",
    "MOMENTUM",
    "CrspFold",
    "CrspRows",
    "CrspScored",
    "CrspSeries",
    "crsp_arms",
    "crsp_fold",
    "crsp_known_panel",
    "crsp_selected_count",
    "crsp_study_rows",
    "crsp_top",
    "require_crsp_features",
    "run_crsp_validation",
]
