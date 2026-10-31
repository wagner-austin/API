"""Select from complete paired monthly observations in the shared RunRecord.

Monthly observations contain portfolio returns and calendar boundaries, never
security identifiers or holdings. The native verifier binds the originating
record and runtime before the journal accepts completion. Selection uses that
same durable record; no independent result store or caller-supplied score exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import fsum, isfinite
from typing import NamedTuple

from platform_core.run_record import Observation, RunRecord

from .crsp_calendar import CrspWindow, crsp_month
from .crsp_study import CrspSeries
from .research_limits import policy_refusal
from .research_records import ResearchPolicy, ResearchProposal, candidate_digest
from .research_selection_records import ResearchSelection
from .validation import require_day


class SeedResult(NamedTuple):
    """One completed journal attempt, its admitted request and native record."""

    number: int
    proposal: ResearchProposal
    record: RunRecord


def monthly_selection_observations(
    series: tuple[CrspSeries, ...], selection: ResearchSelection
) -> tuple[Observation, ...]:
    """Publish the two primary series using the existing named-scalar record.

    Args:
        series: All computed arms and costs from the actual evaluator.
        selection: Frozen primary arms and cost.

    Returns:
        Returns identified by arm, cost and all three calendar boundaries.
    """
    return tuple(
        Observation(
            name=(
                f"monthly_net_return.{item.arm}.{item.cost_bps}bps."
                f"{period.window.decision}.{period.window.entry}.{period.window.exit}"
            ),
            value=period.net_return,
        )
        for item in series
        if item.cost_bps == selection["cost_bps"]
        and item.arm in {selection["primary"], selection["reference"]}
        for period in item.periods
    )


def _series(
    record: RunRecord, selection: ResearchSelection, arm: str
) -> tuple[tuple[CrspWindow, float], ...]:
    """Require one finite observation per frozen month, without intersecting calendars."""
    prefix = f"monthly_net_return.{arm}.{selection['cost_bps']}bps."
    monthly: dict[int, tuple[CrspWindow, float]] = {}
    for observation in record["observations"]:
        name = observation["name"]
        if not name.startswith(prefix):
            continue
        fields = name[len(prefix) :].split(".")
        if len(fields) != 3:
            raise ValueError("RESEARCH_SELECTION_CALENDAR: require three calendar boundaries")
        decision, entry, exit_day = (require_day({"day": field}, "day") for field in fields)
        month = crsp_month(decision)
        value = observation["value"]
        if (
            month in monthly
            or crsp_month(entry) != month + 1
            or crsp_month(exit_day) != month + 2
            or not isfinite(value)
        ):
            raise ValueError("RESEARCH_SELECTION_CALENDAR: duplicate, invalid or nonfinite month")
        monthly[month] = (CrspWindow(decision, entry, exit_day), value)
    expected = tuple(range(selection["first_month"], selection["last_month"] + 1))
    if tuple(sorted(monthly)) != expected:
        raise ValueError("RESEARCH_SELECTION_CALENDAR: require every frozen month exactly once")
    ordered = tuple(monthly[month] for month in expected)
    if any(left[0].exit != right[0].entry for left, right in pairwise(ordered)):
        raise ValueError("RESEARCH_SELECTION_CALENDAR: consecutive windows do not share a close")
    return ordered


def _paired(
    item: SeedResult, policy: ResearchPolicy
) -> tuple[tuple[tuple[CrspWindow, float], ...], tuple[float, ...]]:
    """Validate lineage and pair both arms on identical decision, entry and exit dates."""
    record = item.record
    if (
        policy_refusal(policy, item.proposal)
        or record["experiment"] != policy["protocol_digest"]
        or record["label"] != str(item.number)
        or record["fingerprint"]["image_digest"] != policy["image_digest"]
        or not record["payload_digest"]
    ):
        raise ValueError("RESEARCH_SELECTION_LINEAGE: result differs from admitted study")
    selection = policy["selection"]
    reference = _series(record, selection, selection["reference"])
    primary = _series(record, selection, selection["primary"])
    if tuple(row[0] for row in primary) != tuple(row[0] for row in reference):
        raise ValueError("RESEARCH_SELECTION_CALENDAR: paired arm calendars differ")
    differences = tuple(left[1] - right[1] for left, right in zip(primary, reference, strict=True))
    if not all(isfinite(value) for value in differences):
        raise ValueError("RESEARCH_SELECTION_SCORE: paired differences must be finite")
    return reference, differences


def winning_attempt(policy: ResearchPolicy, completed: Sequence[SeedResult]) -> int:
    """Rank complete candidate/seed groups and return the winner's fixed final seed.

    Args:
        policy: Immutable journal policy, including required initial candidates.
        completed: All completed attempts, read under the existing journal lock.

    Returns:
        Canonical final-seed attempt of the highest-scoring eligible candidate.

    Raises:
        ValueError: RESEARCH_SELECTION_* for changed evidence, duplicate seeds,
            incomplete mandatory candidates, no eligible candidate or nonfinite scores.
        OverflowError: A finite-input sum exceeds the representable range.
    """
    selection = policy["selection"]
    grouped: dict[str, dict[int, tuple[SeedResult, tuple[float, ...]]]] = {}
    common: tuple[tuple[CrspWindow, float], ...] | None = None
    for item in completed:
        reference, differences = _paired(item, policy)
        if common is None:
            common = reference
        elif reference != common:
            raise ValueError("RESEARCH_SELECTION_CALENDAR: reference changed across attempts")
        candidate = candidate_digest(item.proposal)
        group = grouped.setdefault(candidate, {})
        seed = item.proposal["seed"]
        if seed in group:
            raise ValueError("RESEARCH_SELECTION_SEEDS: duplicate candidate seed")
        group[seed] = item, differences
    seeds = tuple(sorted(policy["seeds"]))
    eligible = {key: rows for key, rows in grouped.items() if tuple(sorted(rows)) == seeds}
    if not set(selection["required_candidates"]) <= set(eligible):
        raise ValueError("RESEARCH_SELECTION_SEEDS: required baselines lack complete seed evidence")
    ranked: list[tuple[float, int, str, int]] = []
    for candidate, rows in eligible.items():
        columns = tuple(rows[seed][1] for seed in seeds)
        means = tuple(
            fsum(column[month] for column in columns) / len(seeds)
            for month in range(len(columns[0]))
        )
        score = fsum(means) / len(means)
        chosen = rows[selection["final_seed"]][0]
        ranked.append((-score, chosen.proposal["config"]["num_leaves"], candidate, chosen.number))
    return min(ranked)[3]


__all__ = ["SeedResult", "monthly_selection_observations", "winning_attempt"]
