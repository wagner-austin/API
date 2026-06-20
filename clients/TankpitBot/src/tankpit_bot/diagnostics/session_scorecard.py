"""Per-run session scorecard: time budget, combat outcome, fuel, dot ledger.

This module owns the scorecard concern end to end -- accumulating the
relevant event records, distilling them into a
:class:`SessionScorecardDict`, rendering the report section, and
deriving the scorecard's top-level issues. The issue report composes
these functions; it does not reimplement them.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from typing_extensions import TypedDict

from tankpit_bot.diagnostics.issue_report_types import (
    InventoryCountsDict,
    SessionScorecardDict,
    StateBudgetRecordDict,
    TargetedTeleportRecordDict,
    make_unsampled_inventory_counts,
    make_zero_inventory_counts,
)
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    require_bool_field,
    require_int_field,
)

# A dot teleported to this many times in one session was never revealed
# or refuted by a scan -- the orbit class of bug from live run
# 20260612-062453 (fuel bled 151->119 around one in-viewport dot).
_DOT_ORBIT_REPEAT_THRESHOLD = 3

# An equipment container teleport-approached this many times never
# became collectable -- the unreachable-pocket orbit from live run
# 20260612-071918 ((128,126)/(129,127) re-approached 7x each).
_EQUIPMENT_ORBIT_REPEAT_THRESHOLD = 3

# Sessions that shoot this much without a single observed deactivation
# are chasing unkillable or repairing targets.
_COMBAT_FUTILITY_SHOT_THRESHOLD = 20

# The fuel-critical band: combat needs ~10 fuel per shot and teleports
# cost 6 per tile, so dipping below this means the session nearly
# stranded itself.
_FUEL_FLOOR_THRESHOLD = 100


class ScorecardAccumulatorDict(TypedDict):
    """Mutable scratch space for scorecard-relevant event records.

    Attributes:
        state_transitions: ``(timestamp, message)`` pairs from the
            ``STATE`` channel, in stream order.
        kills: Count of ``tank_deactivated`` DIAGNOSTIC events.
        shots: Count of ``WIRE`` events whose message starts with
            ``shoot(``.
        combat_misses: Count of ``combat_miss`` DIAGNOSTIC events
            (shot resolved with no tank at the target tile).
        combat_ghosts_blocked: Count of ``combat_ghost_detected``
            DIAGNOSTIC events (combat shot refused because
            ``last_wire_seen_ms`` was stale).
        combat_stale_positions_blocked: Count of
            ``combat_stale_position`` DIAGNOSTIC events (combat shot
            refused because ``last_position_update_ms`` was stale --
            the kill-shot gate added with the 2026-06-19 freshness
            refactor).
        tank_damage_changes: Count of ``tank_damage_changed``
            DIAGNOSTIC events (any tank's ``damage_state`` transitioned
            via wire), useful for sanity-checking shots against damage
            observations.
        fuel_samples: ``belief_fuel`` values from every
            ``self_alignment_sample`` event, in stream order.
        dot_hops: Every ``fuel_dot_hop`` event, in stream order.
        inventory_samples: Counts from every ``inventory_sample``
            event, in stream order.
        equipment_gain_events: Count of ``equipment_gain`` events.
        equipment_gained: Running per-type gain totals.
        scans_extra: ``radar_dispatch`` events with ``uses_extra``.
        scans_builtin: ``radar_dispatch`` events without
            ``uses_extra``.
        equipment_approaches: Every ``equipment_approach`` event, in
            stream order.
        first_timestamp: Timestamp of the first record, or ``""``.
        last_timestamp: Timestamp of the last record, or ``""``.
    """

    state_transitions: list[tuple[str, str]]
    kills: int
    shots: int
    combat_misses: int
    combat_ghosts_blocked: int
    combat_stale_positions_blocked: int
    tank_damage_changes: int
    fuel_samples: list[int]
    dot_hops: list[TargetedTeleportRecordDict]
    inventory_samples: list[InventoryCountsDict]
    equipment_gain_events: int
    equipment_gained: InventoryCountsDict
    scans_extra: int
    scans_builtin: int
    equipment_approaches: list[TargetedTeleportRecordDict]
    first_timestamp: str
    last_timestamp: str


def new_scorecard_accumulator() -> ScorecardAccumulatorDict:
    """Return a fresh :class:`ScorecardAccumulatorDict`.

    Returns:
        Accumulator with empty collections and zeroed counters.
    """
    return ScorecardAccumulatorDict(
        state_transitions=[],
        kills=0,
        shots=0,
        combat_misses=0,
        combat_ghosts_blocked=0,
        combat_stale_positions_blocked=0,
        tank_damage_changes=0,
        fuel_samples=[],
        dot_hops=[],
        inventory_samples=[],
        equipment_gain_events=0,
        equipment_gained=make_zero_inventory_counts(),
        scans_extra=0,
        scans_builtin=0,
        equipment_approaches=[],
        first_timestamp="",
        last_timestamp="",
    )


def _classify_targeted_teleport(record: RuntimeEventRecordDict) -> TargetedTeleportRecordDict:
    """Build a typed targeted-teleport row from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``fuel_dot_hop`` or ``equipment_approach``.

    Returns:
        Strict-typed targeted-teleport row.
    """
    fields = record["fields"]
    return TargetedTeleportRecordDict(
        target_x=require_int_field(fields, "target_x"),
        target_y=require_int_field(fields, "target_y"),
        fuel=require_int_field(fields, "fuel"),
        timestamp=record["timestamp"],
    )


def _classify_inventory_counts(record: RuntimeEventRecordDict) -> InventoryCountsDict:
    """Build typed inventory counts from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``inventory_sample`` or ``equipment_gain``.

    Returns:
        Strict-typed inventory counts.
    """
    fields = record["fields"]
    return InventoryCountsDict(
        armor=require_int_field(fields, "armor"),
        dual=require_int_field(fields, "dual"),
        missile=require_int_field(fields, "missile"),
        homing=require_int_field(fields, "homing"),
        radar=require_int_field(fields, "radar"),
    )


def _add_inventory_counts(
    totals: InventoryCountsDict,
    gained: InventoryCountsDict,
) -> InventoryCountsDict:
    """Return per-type totals with one gain event added.

    Args:
        totals: Running totals.
        gained: Gain amounts from one ``equipment_gain`` event.

    Returns:
        New totals.
    """
    return InventoryCountsDict(
        armor=totals["armor"] + gained["armor"],
        dual=totals["dual"] + gained["dual"],
        missile=totals["missile"] + gained["missile"],
        homing=totals["homing"] + gained["homing"],
        radar=totals["radar"] + gained["radar"],
    )


def route_scorecard_record(
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> None:
    """Route one event record into the scorecard accumulator.

    Every record contributes to the session duration bounds; only the
    scorecard-relevant channels and diagnostic kinds populate the
    other buckets.

    Args:
        record: Decoded runtime event record.
        accumulator: Scorecard accumulator to update in place.
    """
    if not accumulator["first_timestamp"]:
        accumulator["first_timestamp"] = record["timestamp"]
    accumulator["last_timestamp"] = record["timestamp"]
    channel = record["channel"]
    if channel == "STATE":
        accumulator["state_transitions"].append((record["timestamp"], record["message"]))
        return
    if channel == "WIRE" and record["message"].startswith("shoot("):
        accumulator["shots"] += 1
        return
    if channel != "DIAGNOSTIC":
        return
    _route_scorecard_diagnostic(record, accumulator)


def _route_scorecard_diagnostic(
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> None:
    """Route one DIAGNOSTIC record into the scorecard accumulator.

    Args:
        record: Decoded DIAGNOSTIC-channel event record.
        accumulator: Scorecard accumulator to update in place.
    """
    kind = record["fields"].get("diagnostic_kind")
    if _route_combat_diagnostic(kind, accumulator):
        return
    if kind == "self_alignment_sample":
        accumulator["fuel_samples"].append(require_int_field(record["fields"], "belief_fuel"))
    elif kind == "fuel_dot_hop":
        accumulator["dot_hops"].append(_classify_targeted_teleport(record))
    elif kind == "equipment_approach":
        accumulator["equipment_approaches"].append(_classify_targeted_teleport(record))
    elif kind == "inventory_sample":
        accumulator["inventory_samples"].append(_classify_inventory_counts(record))
    elif kind == "equipment_gain":
        accumulator["equipment_gain_events"] += 1
        accumulator["equipment_gained"] = _add_inventory_counts(
            accumulator["equipment_gained"],
            _classify_inventory_counts(record),
        )
    elif kind == "radar_dispatch":
        if require_bool_field(record["fields"], "uses_extra"):
            accumulator["scans_extra"] += 1
        else:
            accumulator["scans_builtin"] += 1


def _route_combat_diagnostic(
    kind: str | int | float | bool | None,
    accumulator: ScorecardAccumulatorDict,
) -> bool:
    """Increment the combat counters that the freshness gates emit.

    Args:
        kind: ``diagnostic_kind`` field value pulled from the record's
            ``fields`` dict. The dict is typed
            ``dict[str, str | int | float | bool]`` so the value
            received here is one of those primitives or ``None`` when
            the field is absent. Non-string values can never match the
            string literals tested below, so they fall through and the
            caller routes the record onward.
        accumulator: Scorecard accumulator to update in place.

    Returns:
        True when ``kind`` matched a combat counter and was applied,
        False otherwise. The caller routes non-matching kinds onward.
    """
    if kind == "tank_deactivated":
        accumulator["kills"] += 1
        return True
    if kind == "combat_miss":
        accumulator["combat_misses"] += 1
        return True
    if kind == "combat_ghost_detected":
        accumulator["combat_ghosts_blocked"] += 1
        return True
    if kind == "combat_stale_position":
        accumulator["combat_stale_positions_blocked"] += 1
        return True
    if kind == "tank_damage_changed":
        accumulator["tank_damage_changes"] += 1
        return True
    return False


def _budget_sort_key(record: StateBudgetRecordDict) -> tuple[int, str]:
    """Sort key for the state budget: descending seconds, then name.

    Args:
        record: State budget record to key.

    Returns:
        Tuple of ``(-seconds, state)``.
    """
    return (-record["seconds"], record["state"])


def _build_state_budget(transitions: list[tuple[str, str]]) -> list[StateBudgetRecordDict]:
    """Sum seconds spent in each bot state from STATE-channel transitions.

    The interval between consecutive ``A -> B`` transitions is credited
    to the EARLIER transition's destination -- the state the bot was
    actually in during that interval. Non-transition STATE lines (the
    initial bare state announcement) carry no interval and are skipped.

    Args:
        transitions: ``(timestamp, message)`` pairs in stream order.

    Returns:
        Per-state totals sorted by descending seconds then state name.
    """
    totals: Counter[str] = Counter()
    previous_state = ""
    previous_moment: datetime | None = None
    for timestamp, message in transitions:
        if " -> " not in message:
            continue
        _, _, destination = message.partition(" -> ")
        moment = datetime.fromisoformat(timestamp)
        if previous_moment is not None:
            totals[previous_state] += int((moment - previous_moment).total_seconds())
        previous_state = destination
        previous_moment = moment
    records = [
        StateBudgetRecordDict(state=state, seconds=seconds) for state, seconds in totals.items()
    ]
    records.sort(key=_budget_sort_key)
    return records


def build_session_scorecard(accumulator: ScorecardAccumulatorDict) -> SessionScorecardDict:
    """Distill the per-run outcome scorecard from the accumulator.

    Args:
        accumulator: Fully routed scorecard accumulator.

    Returns:
        Session scorecard.
    """
    duration_seconds = 0
    if accumulator["first_timestamp"] and accumulator["last_timestamp"]:
        first = datetime.fromisoformat(accumulator["first_timestamp"])
        last = datetime.fromisoformat(accumulator["last_timestamp"])
        duration_seconds = int((last - first).total_seconds())
    fuel_samples = accumulator["fuel_samples"]
    dot_hops = accumulator["dot_hops"]
    hop_counts = Counter((hop["target_x"], hop["target_y"]) for hop in dot_hops)
    inventory_samples = accumulator["inventory_samples"]
    approaches = accumulator["equipment_approaches"]
    approach_counts = Counter((row["target_x"], row["target_y"]) for row in approaches)
    return SessionScorecardDict(
        duration_seconds=duration_seconds,
        state_budget=_build_state_budget(accumulator["state_transitions"]),
        kills=accumulator["kills"],
        shots=accumulator["shots"],
        combat_misses=accumulator["combat_misses"],
        combat_ghosts_blocked=accumulator["combat_ghosts_blocked"],
        combat_stale_positions_blocked=accumulator["combat_stale_positions_blocked"],
        tank_damage_changes=accumulator["tank_damage_changes"],
        fuel_min=min(fuel_samples) if fuel_samples else -1,
        fuel_last=fuel_samples[-1] if fuel_samples else -1,
        fuel_sample_count=len(fuel_samples),
        dot_hops=dot_hops,
        dot_hop_distinct_targets=len(hop_counts),
        dot_hop_max_repeats=max(hop_counts.values()) if hop_counts else 0,
        inventory_first=(
            inventory_samples[0] if inventory_samples else make_unsampled_inventory_counts()
        ),
        inventory_last=(
            inventory_samples[-1] if inventory_samples else make_unsampled_inventory_counts()
        ),
        inventory_sample_count=len(inventory_samples),
        equipment_gain_events=accumulator["equipment_gain_events"],
        equipment_gained=accumulator["equipment_gained"],
        scans_extra=accumulator["scans_extra"],
        scans_builtin=accumulator["scans_builtin"],
        equipment_approaches=approaches,
        equipment_approach_distinct_targets=len(approach_counts),
        equipment_approach_max_repeats=(max(approach_counts.values()) if approach_counts else 0),
    )


def render_scorecard_section(scorecard: SessionScorecardDict) -> list[str]:
    """Return the session scorecard section lines for the report.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        Section lines including the trailing blank separator.
    """
    fuel_text = (
        "no samples"
        if scorecard["fuel_sample_count"] == 0
        else f"min={scorecard['fuel_min']} last={scorecard['fuel_last']} "
        f"samples={scorecard['fuel_sample_count']}"
    )
    first = scorecard["inventory_first"]
    last = scorecard["inventory_last"]
    inventory_text = (
        "no samples"
        if scorecard["inventory_sample_count"] == 0
        else f"dual {first['dual']}->{last['dual']} "
        f"homing {first['homing']}->{last['homing']} "
        f"radar {first['radar']}->{last['radar']} "
        f"samples={scorecard['inventory_sample_count']}"
    )
    gained = scorecard["equipment_gained"]
    lines = [
        "=== SESSION SCORECARD ===",
        f"  duration={scorecard['duration_seconds']}s "
        f"kills={scorecard['kills']} shots={scorecard['shots']}",
        f"  fuel: {fuel_text}",
        f"  inventory: {inventory_text}",
        f"  equipment gains: events={scorecard['equipment_gain_events']} "
        f"armor={gained['armor']} dual={gained['dual']} missile={gained['missile']} "
        f"homing={gained['homing']} radar={gained['radar']}",
        f"  scans: extra={scorecard['scans_extra']} builtin={scorecard['scans_builtin']}",
        f"  dot hops: events={len(scorecard['dot_hops'])} "
        f"distinct={scorecard['dot_hop_distinct_targets']} "
        f"max_repeats={scorecard['dot_hop_max_repeats']}",
        f"  equipment approaches: events={len(scorecard['equipment_approaches'])} "
        f"distinct={scorecard['equipment_approach_distinct_targets']} "
        f"max_repeats={scorecard['equipment_approach_max_repeats']}",
    ]
    if not scorecard["state_budget"]:
        lines.append("  state budget: (no transitions)")
    else:
        for record in scorecard["state_budget"]:
            lines.append(f"  {record['state']:>22}: {record['seconds']}s")
    lines.append("")
    return lines


def collect_scorecard_issues(scorecard: SessionScorecardDict) -> list[str]:
    """Return top-level issue lines derived from the session scorecard.

    Args:
        scorecard: Session scorecard to inspect.

    Returns:
        Human-readable issue lines (possibly empty).
    """
    issues: list[str] = []
    if scorecard["dot_hop_max_repeats"] >= _DOT_ORBIT_REPEAT_THRESHOLD:
        issues.append(
            f"fuel-dot orbit: one dot targeted {scorecard['dot_hop_max_repeats']} times "
            "without being revealed or refuted"
        )
    if scorecard["equipment_approach_max_repeats"] >= _EQUIPMENT_ORBIT_REPEAT_THRESHOLD:
        issues.append(
            "equipment-approach orbit: one container teleport-approached "
            f"{scorecard['equipment_approach_max_repeats']} times without completing a pickup"
        )
    if 0 <= scorecard["fuel_min"] < _FUEL_FLOOR_THRESHOLD:
        issues.append(
            f"fuel floor critical: belief fuel dipped to {scorecard['fuel_min']} "
            f"(below {_FUEL_FLOOR_THRESHOLD})"
        )
    if scorecard["shots"] >= _COMBAT_FUTILITY_SHOT_THRESHOLD and scorecard["kills"] == 0:
        issues.append(f"combat futility: {scorecard['shots']} shots produced 0 observed kills")
    if scorecard["inventory_sample_count"] > 0 and scorecard["inventory_last"]["radar"] == 0:
        issues.append(
            "extra radars exhausted: run ended with 0 extra radars "
            "(scans degrade to the 5x5 built-in and equipment discovery stalls)"
        )
    return issues


__all__ = [
    "ScorecardAccumulatorDict",
    "build_session_scorecard",
    "collect_scorecard_issues",
    "new_scorecard_accumulator",
    "render_scorecard_section",
    "route_scorecard_record",
]
