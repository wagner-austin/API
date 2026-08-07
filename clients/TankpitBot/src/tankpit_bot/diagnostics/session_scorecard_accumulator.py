"""Scorecard accumulation: fold raw event records into one accumulator.

The ingest half of the scorecard concern -- :func:`route_scorecard_record`
and its per-channel routers decide what each runtime event contributes.
Distillation into a :class:`SessionScorecardDict` is
:mod:`tankpit_bot.diagnostics.session_scorecard`; rendering is
:mod:`tankpit_bot.diagnostics.session_scorecard_render`.
"""

from __future__ import annotations

import re

from typing_extensions import TypedDict

from tankpit_bot.diagnostics.issue_report_types import (
    InventoryCountsDict,
    TargetedTeleportRecordDict,
    make_zero_inventory_counts,
)
from tankpit_bot.runtime_records import (
    RuntimeEventRecordDict,
    require_bool_field,
    require_int_field,
    require_str_field,
)

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


__all__ = [
    "FuelSampleRecordDict",
    "ScorecardAccumulatorDict",
    "new_scorecard_accumulator",
    "route_scorecard_record",
]
