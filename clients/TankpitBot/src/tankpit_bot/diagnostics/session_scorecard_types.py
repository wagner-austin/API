"""Scorecard accumulator shape: the scratch TypedDict, factory, helpers.

Split from :mod:`tankpit_bot.diagnostics.session_scorecard_accumulator`
(2026-08-20, at the file-size bar): the channel router stays there and
the DIAGNOSTIC-kind arms live in
:mod:`tankpit_bot.diagnostics.session_scorecard_routes`; both consume
this module, so it holds everything they share.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.diagnostics.issue_report_types import (
    InventoryCountsDict,
    TargetedTeleportRecordDict,
    make_zero_inventory_counts,
)


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


def optional_int_field(
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


def optional_str_field(
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
        scope_shift_sends_at: Timestamps of dispatched ``scope_shift``
            commands. Same reason as the map opens below: a scope
            shift is a command, not a state, so the tick that sends
            one transitions nowhere and its seconds land in IDLE.
        map_open_completions_at: Timestamps of completed ``map_open``
            actions. A map open is dispatched FROM the IDLE state and
            has no state of its own, so without these the seconds it
            spends waiting for MAP_DATA are credited to IDLE and the
            budget reads as though the bot sat doing nothing.
        state_transitions: ``(timestamp, message)`` pairs from the
            ``STATE`` channel, in stream order.
        self_tank_id: This session's own wire tank id from the first
            ``tank_identity`` diagnostic, ``-1`` until one arrives.
            Kill attribution reads it: a fleet sibling's 0x41 lands in
            the same event stream, so the raw event count stopped
            being the kill count the day two bots shared a room.
        kills: Count of ``tank_deactivated`` events whose ``killer_id``
            names this session's own tank. The pre-fleet counter took
            the raw event count -- correct while every 0x41 in view
            was our own kill, falsified by the first fleet firefight
            (2026-08-14) and caught in this counter on the first
            gatherer run (2026-08-20: the analyzer read kills=2 with
            shots=0). Unattributable events (no ``killer_id`` on
            pre-fleet artifacts, or no identity seen yet) never count.
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

    map_open_completions_at: list[str]
    scope_shift_sends_at: list[str]
    state_transitions: list[tuple[str, str]]
    self_tank_id: int
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
        map_open_completions_at=[],
        scope_shift_sends_at=[],
        state_transitions=[],
        self_tank_id=-1,
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


__all__ = [
    "FuelSampleRecordDict",
    "ScorecardAccumulatorDict",
    "new_scorecard_accumulator",
    "optional_int_field",
    "optional_str_field",
]
