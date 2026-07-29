"""Per-run session scorecard: time budget, combat outcome, fuel, dot ledger.

This module owns the scorecard concern end to end -- accumulating the
relevant event records, distilling them into a
:class:`SessionScorecardDict`, rendering the report section, and
deriving the scorecard's top-level issues. The issue report composes
these functions; it does not reimplement them.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

from typing_extensions import TypedDict

from tankpit_bot.diagnostics.issue_report_types import (
    FuelLowWaterEpisodeDict,
    InventoryCountsDict,
    SessionScorecardDict,
    StateBudgetRecordDict,
    TargetedTeleportRecordDict,
    TeleportSpendRecordDict,
    make_unsampled_inventory_counts,
    make_zero_inventory_counts,
)
from tankpit_bot.runtime_logging import (
    RuntimeEventRecordDict,
    require_bool_field,
    require_int_field,
    require_str_field,
)

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

# Low-water episodes listed in full in the rendered report; beyond this
# many, the tail is summarized as a count to keep the section readable.
_LOW_WATER_RENDER_CAP = 10

# WORLD-channel fuel transition receipts, e.g. "Fuel: 1090 -> 823 (-267)".
_WORLD_FUEL_PATTERN = re.compile(r"^Fuel: (-?\d+) -> (-?\d+) ")


class FuelSampleRecordDict(TypedDict):
    """One ``self_alignment_sample`` fuel reading with its tick context.

    The bare fuel integer cannot explain a low-water dip; the ambient
    ``bot_state`` / ``in_flight_action_kind`` context stamped on every
    runtime event is what attributes each sample-to-sample drop to the
    action that paid it.

    Attributes:
        timestamp: ISO timestamp of the sample record.
        fuel: ``belief_fuel`` value.
        bot_state: Ambient ``MODE/STATE`` context, ``""`` on artifacts
            predating the context fields.
        in_flight: Ambient ``in_flight_action_kind`` context,
            ``"none"`` when absent.
    """

    timestamp: str
    fuel: int
    bot_state: str
    in_flight: str


def _optional_int_field(
    fields: dict[str, str | int | float | bool],
    key: str,
    default: int,
) -> int:
    """Extract an int field, tolerating absence on older artifacts.

    Args:
        fields: Decoded structured payload from a runtime event record.
        key: Field name to extract.
        default: Value returned when the field is absent or not an int.

    Returns:
        The int value, or ``default``.
    """
    value = fields.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _optional_str_field(
    fields: dict[str, str | int | float | bool],
    key: str,
    default: str,
) -> str:
    """Extract a str field, tolerating absence on older artifacts.

    Args:
        fields: Decoded structured payload from a runtime event record.
        key: Field name to extract.
        default: Value returned when the field is absent or not a str.

    Returns:
        The str value, or ``default``.
    """
    value = fields.get(key)
    if not isinstance(value, str):
        return default
    return value


class ScorecardAccumulatorDict(TypedDict):
    """Mutable scratch space for scorecard-relevant event records.

    Attributes:
        state_transitions: ``(timestamp, message)`` pairs from the
            ``STATE`` channel, in stream order.
        kills: Count of ``tank_deactivated`` events. Since the DOM
            game-log kill channel was retired (2026-07-19), the wire
            ``0x41 Deactivation`` is the single emitter -- exactly one
            event per kill, so the raw count is the kill count and a
            respawned victim killed again counts again.
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
        fuel_samples: Context-stamped ``belief_fuel`` readings from
            every ``self_alignment_sample`` event, in stream order.
        max_escape_floor: Highest ``escape_floor`` any
            ``engagement_break`` event computed, ``0`` when combat
            never projected one. This is the session's own danger
            line, used as the low-water episode threshold.
        teleport_spend_fuel: Per-``bot_state`` fuel totals from
            WORLD-channel fuel debits billed while a teleport was in
            flight.
        teleport_spend_drops: Per-``bot_state`` receipt counts for
            those debits.
        ledger_teleport_spend_min: Least possible teleport spend from
            the fuel book's feasibility bounds, ``-1`` without a
            ``damage_ledger`` event.
        ledger_teleport_spend_max: Greatest possible teleport spend,
            ``-1`` without one.
        ledger_shot_singles: ``shot_single_count`` from the ledger,
            ``-1`` without one.
        ledger_shot_duals: ``shot_dual_count`` from the ledger,
            ``-1`` without one.
        ledger_shot_homings: ``shot_homing_count`` from the ledger,
            ``-1`` without one.
        inventory_samples: Counts from every ``inventory_sample``
            event, in stream order.
        equipment_gain_events: Count of ``equipment_gain`` events.
        equipment_gained: Running per-type gain totals.
        scans_extra: ``radar_dispatch`` events with ``uses_extra``.
        scans_builtin: ``radar_dispatch`` events without
            ``uses_extra``.
        equipment_approaches: Every ``equipment_approach`` event, in
            stream order.
        action_outcome_counts: Per ``"kind:outcome"`` tallies from the
            unified ``action_outcome`` fabric -- the ledger-grade view
            of every attempt resolution (hits, stalls, rejections,
            executor discards, superseded re-dispatches).
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
    fuel_samples: list[FuelSampleRecordDict]
    max_escape_floor: int
    teleport_spend_fuel: dict[str, int]
    teleport_spend_drops: dict[str, int]
    ledger_teleport_spend_min: int
    ledger_teleport_spend_max: int
    ledger_shot_singles: int
    ledger_shot_duals: int
    ledger_shot_homings: int
    inventory_samples: list[InventoryCountsDict]
    equipment_gain_events: int
    equipment_gained: InventoryCountsDict
    scans_extra: int
    scans_builtin: int
    physics_divergences: int
    equipment_approaches: list[TargetedTeleportRecordDict]
    action_outcome_counts: dict[str, int]
    first_timestamp: str
    last_timestamp: str
    # Career totals from the wire's 0x56 broadcast (latest seen this
    # run; ``-1`` when never sent during the session) and per-record
    # container pickup tallies. These mirror the fields tracked by
    # :class:`tankpit_bot.diagnostics.issue_report._ReportAccumulatorDict`
    # so both accumulator paths populate the same scorecard fields
    # from the same diagnostic kinds.
    career_destroyed_last: int
    career_deactivated_last: int
    career_score_last: int
    career_playtime_seconds_last: int
    container_pickups_full: int
    container_pickups_partial: int


def new_scorecard_accumulator() -> ScorecardAccumulatorDict:
    """Return a fresh :class:`ScorecardAccumulatorDict`.

    Returns:
        Accumulator with empty collections and zeroed counters.
    """
    # First six career/pickup fields are zero/sentinel until the wire's
    # 0x56 Statistics or a 0x43 ContainerPickup fires during the run.
    # See :class:`ScorecardAccumulatorDict` for the contract.
    return ScorecardAccumulatorDict(
        state_transitions=[],
        kills=0,
        shots=0,
        combat_misses=0,
        combat_ghosts_blocked=0,
        combat_stale_positions_blocked=0,
        tank_damage_changes=0,
        fuel_samples=[],
        max_escape_floor=0,
        teleport_spend_fuel={},
        teleport_spend_drops={},
        ledger_teleport_spend_min=-1,
        ledger_teleport_spend_max=-1,
        ledger_shot_singles=-1,
        ledger_shot_duals=-1,
        ledger_shot_homings=-1,
        inventory_samples=[],
        equipment_gain_events=0,
        equipment_gained=make_zero_inventory_counts(),
        scans_extra=0,
        physics_divergences=0,
        scans_builtin=0,
        equipment_approaches=[],
        action_outcome_counts={},
        first_timestamp="",
        last_timestamp="",
        career_destroyed_last=-1,
        career_deactivated_last=-1,
        career_score_last=-1,
        career_playtime_seconds_last=-1,
        container_pickups_full=0,
        container_pickups_partial=0,
    )


def _classify_targeted_teleport(record: RuntimeEventRecordDict) -> TargetedTeleportRecordDict:
    """Build a typed targeted-teleport row from a DIAGNOSTIC event.

    Args:
        record: Decoded event record whose ``diagnostic_kind`` is
            ``equipment_approach``.

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
    if channel == "WORLD":
        _route_world_fuel_receipt(record, accumulator)
        return
    if channel != "DIAGNOSTIC":
        return
    _route_scorecard_diagnostic(record, accumulator)


def _route_world_fuel_receipt(
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> None:
    """Attribute an in-flight teleport fuel debit to its bot state.

    WORLD-channel fuel transitions are the per-receipt view of the
    fuel book; a debit billed while ``in_flight_action_kind`` is
    ``teleport`` is teleport spend, attributed to the ambient
    ``bot_state``. Measured on run 20260729-105325: 15592 across 104
    receipts, inside the ledger's 11993..19290 feasibility bound
    (sample-to-sample fuel deltas undercounted at 10972 because
    pickup credits mask debits inside one tick window).

    Args:
        record: Decoded WORLD-channel event record.
        accumulator: Scorecard accumulator to update in place.
    """
    if _optional_str_field(record["fields"], "in_flight_action_kind", "none") != "teleport":
        return
    match = _WORLD_FUEL_PATTERN.match(record["message"])
    if match is None:
        return
    delta = int(match.group(2)) - int(match.group(1))
    if delta >= 0:
        return
    state = _optional_str_field(record["fields"], "bot_state", "")
    accumulator["teleport_spend_fuel"][state] = (
        accumulator["teleport_spend_fuel"].get(state, 0) - delta
    )
    accumulator["teleport_spend_drops"][state] = (
        accumulator["teleport_spend_drops"].get(state, 0) + 1
    )


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
    if _route_fuel_diagnostic(kind, record, accumulator):
        return
    if kind == "equipment_approach":
        accumulator["equipment_approaches"].append(_classify_targeted_teleport(record))
    elif kind == "inventory_sample":
        accumulator["inventory_samples"].append(_classify_inventory_counts(record))
    elif kind == "equipment_gain":
        accumulator["equipment_gain_events"] += 1
        accumulator["equipment_gained"] = _add_inventory_counts(
            accumulator["equipment_gained"],
            _classify_inventory_counts(record),
        )
    elif kind == "physics_divergence":
        accumulator["physics_divergences"] += 1
    elif kind == "radar_dispatch":
        if require_bool_field(record["fields"], "uses_extra"):
            accumulator["scans_extra"] += 1
        else:
            accumulator["scans_builtin"] += 1
    elif kind == "action_outcome":
        counter_key = (
            f"{require_str_field(record['fields'], 'action_kind')}:"
            f"{require_str_field(record['fields'], 'outcome')}"
        )
        counts = accumulator["action_outcome_counts"]
        counts[counter_key] = counts.get(counter_key, 0) + 1
    else:
        _route_metrics_diagnostic(kind, record, accumulator)


def _route_fuel_diagnostic(
    kind: str | int | float | bool | None,
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> bool:
    """Route the fuel-trajectory diagnostics into the accumulator.

    Covers the three kinds the low-water / teleport-spend analysis
    reads: context-stamped fuel samples, the engagement-break escape
    floors (the session's own danger line), and the end-of-run damage
    ledger's authoritative billing totals.

    Args:
        kind: ``diagnostic_kind`` field value.
        record: Decoded event record carrying the structured payload.
        accumulator: Scorecard accumulator to update in place.

    Returns:
        True when ``kind`` matched and was applied, False otherwise.
    """
    fields = record["fields"]
    if kind == "self_alignment_sample":
        accumulator["fuel_samples"].append(
            FuelSampleRecordDict(
                timestamp=record["timestamp"],
                fuel=require_int_field(fields, "belief_fuel"),
                bot_state=_optional_str_field(fields, "bot_state", ""),
                in_flight=_optional_str_field(fields, "in_flight_action_kind", "none"),
            )
        )
        return True
    if kind == "engagement_break":
        accumulator["max_escape_floor"] = max(
            accumulator["max_escape_floor"],
            _optional_int_field(fields, "escape_floor", 0),
        )
        return True
    if kind == "damage_ledger":
        # The fuel book's ``[lo, hi]`` sums are feasibility bounds on
        # spend, both negative; negate into a positive min..max spend
        # interval. Absent fields (pre-ledger artifacts) keep the -1
        # sentinels from the accumulator factory.
        lo = _optional_int_field(fields, "teleport_fuel_lo", 1)
        hi = _optional_int_field(fields, "teleport_fuel_hi", 1)
        accumulator["ledger_teleport_spend_max"] = -lo if lo <= 0 else -1
        accumulator["ledger_teleport_spend_min"] = -hi if hi <= 0 else -1
        accumulator["ledger_shot_singles"] = _optional_int_field(fields, "shot_single_count", -1)
        accumulator["ledger_shot_duals"] = _optional_int_field(fields, "shot_dual_count", -1)
        accumulator["ledger_shot_homings"] = _optional_int_field(fields, "shot_homing_count", -1)
        return True
    return False


def _route_metrics_diagnostic(
    kind: str | int | float | bool | None,
    record: RuntimeEventRecordDict,
    accumulator: ScorecardAccumulatorDict,
) -> None:
    """Route career-stats and pickup-tally diagnostics into the scorecard accumulator.

    Split out of :func:`_route_scorecard_diagnostic` to keep the
    primary router under the C901 complexity ceiling.

    Args:
        kind: ``diagnostic_kind`` field value.
        record: Decoded event record carrying the structured payload.
        accumulator: Scorecard accumulator to update in place.
    """
    if kind == "self_statistics":
        accumulator["career_destroyed_last"] = require_int_field(record["fields"], "destroyed")
        accumulator["career_deactivated_last"] = require_int_field(record["fields"], "deactivated")
        accumulator["career_score_last"] = require_int_field(record["fields"], "score")
        accumulator["career_playtime_seconds_last"] = require_int_field(
            record["fields"], "playtime_seconds_total"
        )
        return
    if kind == "container_pickup_dispatched":
        if record["fields"].get("is_partial") is True:
            accumulator["container_pickups_partial"] += 1
        else:
            accumulator["container_pickups_full"] += 1


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
    Each interval is also one VISIT to its state, so the per-state
    stretch count and longest single visit fall out of the same walk --
    that pair distinguishes tick-boundary residue (many short visits)
    from a stall (one long visit) at no extra cost.

    Args:
        transitions: ``(timestamp, message)`` pairs in stream order.

    Returns:
        Per-state totals sorted by descending seconds then state name.
    """
    totals: Counter[str] = Counter()
    visits: Counter[str] = Counter()
    longest: dict[str, int] = {}
    previous_state = ""
    previous_moment: datetime | None = None
    for timestamp, message in transitions:
        if " -> " not in message:
            continue
        _, _, destination = message.partition(" -> ")
        moment = datetime.fromisoformat(timestamp)
        if previous_moment is not None:
            interval = int((moment - previous_moment).total_seconds())
            totals[previous_state] += interval
            visits[previous_state] += 1
            longest[previous_state] = max(longest.get(previous_state, 0), interval)
        previous_state = destination
        previous_moment = moment
    records = [
        StateBudgetRecordDict(
            state=state,
            seconds=seconds,
            stretches=visits[state],
            max_seconds=longest[state],
        )
        for state, seconds in totals.items()
    ]
    records.sort(key=_budget_sort_key)
    return records


def _build_teleport_spend(
    spent: dict[str, int],
    drops: dict[str, int],
) -> tuple[list[TeleportSpendRecordDict], int]:
    """Shape the accumulated per-state teleport spend into sorted rows.

    Args:
        spent: Per-``bot_state`` fuel totals from the WORLD receipts.
        drops: Per-``bot_state`` receipt counts.

    Returns:
        Tuple of per-state spend rows (descending fuel, then state)
        and the total spend.
    """
    records = [
        TeleportSpendRecordDict(bot_state=state, drops=drops[state], fuel_spent=fuel)
        for state, fuel in spent.items()
    ]
    records.sort(key=lambda record: (-record["fuel_spent"], record["bot_state"]))
    return records, sum(spent.values())


def _episode_cause(
    samples: list[FuelSampleRecordDict],
    entry_index: int,
    min_index: int,
) -> tuple[str, int, str]:
    """Find the largest fuel drop on the way down into one episode.

    Args:
        samples: Context-stamped fuel samples in stream order.
        entry_index: Index of the first below-threshold sample.
        min_index: Index of the episode's minimum-fuel sample.

    Returns:
        Tuple of ``(cause_kind, cause_drop, cause_state)``. When no
        positive drop exists in the window (a session that STARTED
        below threshold at its minimum), the entry sample's own
        context is returned with a drop of 0.
    """
    best_drop = 0
    best_index = entry_index
    for index in range(max(entry_index, 1), min_index + 1):
        drop = samples[index - 1]["fuel"] - samples[index]["fuel"]
        if drop > best_drop:
            best_drop = drop
            best_index = index
    chosen = samples[best_index]
    return chosen["in_flight"], best_drop, chosen["bot_state"]


def _build_low_water_episodes(
    samples: list[FuelSampleRecordDict],
    threshold: int,
) -> list[FuelLowWaterEpisodeDict]:
    """Split the fuel trajectory into below-threshold episodes.

    Args:
        samples: Context-stamped fuel samples in stream order.
        threshold: Danger line; samples strictly below it are "low".

    Returns:
        One record per maximal contiguous below-threshold run, in
        stream order.
    """
    episodes: list[FuelLowWaterEpisodeDict] = []
    index = 0
    while index < len(samples):
        if samples[index]["fuel"] >= threshold:
            index += 1
            continue
        end = index
        min_index = index
        while end + 1 < len(samples) and samples[end + 1]["fuel"] < threshold:
            end += 1
            if samples[end]["fuel"] < samples[min_index]["fuel"]:
                min_index = end
        cause_kind, cause_drop, cause_state = _episode_cause(samples, index, min_index)
        first = datetime.fromisoformat(samples[index]["timestamp"])
        last = datetime.fromisoformat(samples[end]["timestamp"])
        recovery = samples[end + 1] if end + 1 < len(samples) else None
        episodes.append(
            FuelLowWaterEpisodeDict(
                start_timestamp=samples[index]["timestamp"],
                end_timestamp=samples[end]["timestamp"],
                duration_seconds=int((last - first).total_seconds()),
                entry_fuel=samples[index - 1]["fuel"] if index > 0 else -1,
                min_fuel=samples[min_index]["fuel"],
                cause_kind=cause_kind,
                cause_drop=cause_drop,
                cause_state=cause_state,
                recovery_fuel=recovery["fuel"] if recovery is not None else -1,
                recovery_kind=recovery["in_flight"] if recovery is not None else "",
            )
        )
        index = end + 1
    return episodes


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
    fuel_values = [sample["fuel"] for sample in fuel_samples]
    low_water_threshold = (
        accumulator["max_escape_floor"]
        if accumulator["max_escape_floor"] > 0
        else _FUEL_FLOOR_THRESHOLD
    )
    teleport_spend, teleport_spend_total = _build_teleport_spend(
        accumulator["teleport_spend_fuel"],
        accumulator["teleport_spend_drops"],
    )
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
        fuel_min=min(fuel_values) if fuel_values else -1,
        fuel_last=fuel_values[-1] if fuel_values else -1,
        fuel_sample_count=len(fuel_values),
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
        physics_divergences=accumulator["physics_divergences"],
        equipment_approaches=approaches,
        equipment_approach_distinct_targets=len(approach_counts),
        equipment_approach_max_repeats=(max(approach_counts.values()) if approach_counts else 0),
        action_outcome_counts=dict(sorted(accumulator["action_outcome_counts"].items())),
        fuel_low_water_threshold=low_water_threshold,
        fuel_low_water_episodes=_build_low_water_episodes(fuel_samples, low_water_threshold),
        teleport_spend=teleport_spend,
        teleport_spend_total=teleport_spend_total,
        ledger_teleport_spend_min=accumulator["ledger_teleport_spend_min"],
        ledger_teleport_spend_max=accumulator["ledger_teleport_spend_max"],
        ledger_shot_singles=accumulator["ledger_shot_singles"],
        ledger_shot_duals=accumulator["ledger_shot_duals"],
        ledger_shot_homings=accumulator["ledger_shot_homings"],
        career_destroyed_last=accumulator["career_destroyed_last"],
        career_deactivated_last=accumulator["career_deactivated_last"],
        career_score_last=accumulator["career_score_last"],
        career_playtime_seconds_last=accumulator["career_playtime_seconds_last"],
        container_pickups_full=accumulator["container_pickups_full"],
        container_pickups_partial=accumulator["container_pickups_partial"],
    )


def render_state_budget_lines(scorecard: SessionScorecardDict) -> list[str]:
    """Render the per-state time budget with stretch statistics.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        One line per state (or the no-transitions placeholder).
    """
    if not scorecard["state_budget"]:
        return ["  state budget: (no transitions)"]
    return [
        f"  {record['state']:>22}: {record['seconds']}s "
        f"({record['stretches']}x, max {record['max_seconds']}s)"
        for record in scorecard["state_budget"]
    ]


def render_shot_billing_lines(scorecard: SessionScorecardDict) -> list[str]:
    """Render the ledger's shot billing with the singles reconciliation.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        A single billing line, or no lines when the run ended without
        a ``damage_ledger`` event.
    """
    if scorecard["ledger_shot_singles"] < 0:
        return []
    return [
        f"  shot billing (ledger): dual={scorecard['ledger_shot_duals']} "
        f"homing={scorecard['ledger_shot_homings']} "
        f"single={scorecard['ledger_shot_singles']} "
        "-- singles are server-billed non-connects (weapon=0 misses/clips), "
        "not loadout drift"
    ]


def render_fuel_low_water_lines(scorecard: SessionScorecardDict) -> list[str]:
    """Render the fuel low-water episode narrative.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        Header plus one line per episode (capped), or the all-clear
        line when fuel never dipped below the threshold.
    """
    threshold = scorecard["fuel_low_water_threshold"]
    episodes = scorecard["fuel_low_water_episodes"]
    if not episodes:
        return [f"  fuel low-water: none (never below {threshold})"]
    lines = [f"  fuel low-water (below {threshold}): {len(episodes)} episode(s)"]
    for episode in episodes[:_LOW_WATER_RENDER_CAP]:
        entry = "start" if episode["entry_fuel"] < 0 else str(episode["entry_fuel"])
        recovery = (
            "session end"
            if episode["recovery_fuel"] < 0
            else f"{episode['recovery_fuel']} via {episode['recovery_kind']}"
        )
        lines.append(
            f"    {episode['start_timestamp']} ({episode['duration_seconds']}s) "
            f"entry={entry} min={episode['min_fuel']} "
            f"cause={episode['cause_kind']} -{episode['cause_drop']} "
            f"in {episode['cause_state']} recovery={recovery}"
        )
    hidden = len(episodes) - _LOW_WATER_RENDER_CAP
    if hidden > 0:
        lines.append(f"    ... and {hidden} more episode(s)")
    return lines


def render_teleport_spend_lines(scorecard: SessionScorecardDict) -> list[str]:
    """Render the teleport fuel spend grouped by paying bot state.

    Args:
        scorecard: Session scorecard to render.

    Returns:
        Header plus one line per bot-state group, or the no-spend
        line when no in-flight teleport drops were observed.
    """
    spend_min = scorecard["ledger_teleport_spend_min"]
    spend_max = scorecard["ledger_teleport_spend_max"]
    ledger_text = f" (ledger bound {spend_min}..{spend_max})" if spend_max >= 0 else ""
    if not scorecard["teleport_spend"]:
        return [f"  teleport spend: none observed{ledger_text}"]
    lines = [f"  teleport spend: {scorecard['teleport_spend_total']} fuel{ledger_text}"]
    lines.extend(
        f"    {record['bot_state'] or '(no context)'}: {record['fuel_spent']} "
        f"over {record['drops']} drop(s)"
        for record in scorecard["teleport_spend"]
    )
    return lines


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
        f"  physics divergences: {scorecard['physics_divergences']}",
        f"  equipment approaches: events={len(scorecard['equipment_approaches'])} "
        f"distinct={scorecard['equipment_approach_distinct_targets']} "
        f"max_repeats={scorecard['equipment_approach_max_repeats']}",
    ]
    lines.extend(render_shot_billing_lines(scorecard))
    lines.extend(render_fuel_low_water_lines(scorecard))
    lines.extend(render_teleport_spend_lines(scorecard))
    lines.extend(render_state_budget_lines(scorecard))
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
    if scorecard["physics_divergences"] > 0:
        issues.append(
            f"physics divergences: {scorecard['physics_divergences']} fuel window(s) "
            "outside the physics-predicted feasibility interval -- each is a candidate "
            "wiki claim (new mechanic or drifted constant); query "
            "diagnostic_kind=physics_divergence in the events log"
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
    "FuelSampleRecordDict",
    "ScorecardAccumulatorDict",
    "build_session_scorecard",
    "collect_scorecard_issues",
    "new_scorecard_accumulator",
    "render_fuel_low_water_lines",
    "render_scorecard_section",
    "render_shot_billing_lines",
    "render_state_budget_lines",
    "render_teleport_spend_lines",
    "route_scorecard_record",
]
