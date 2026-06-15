"""Strict TypedDict payloads for the post-run issue report.

Each section of an :class:`IssueReportDict` is its own TypedDict with
the explicit fields that the report renderer and consumers can rely on.
Every section follows the project's encode / decode / ``require_*``
pattern so the report can be persisted, replayed, and compared across
runs without ambiguity.
"""

from __future__ import annotations

from typing_extensions import TypedDict


class TeleportAttemptRecordDict(TypedDict):
    """One teleport attempt observed in the event stream.

    Attributes:
        target_x: Teleport target X coordinate as recorded in the
            ``teleport_attempt`` diagnostic.
        target_y: Teleport target Y coordinate.
        teleport_cycle_id: Cycle ID stamped on the diagnostic.
        status: Terminal status string (``landed_exact``,
            ``teleport_timeout``, ``map_sync_timeout``, ...).
        timestamp: ISO timestamp from the event record.
        sent_window: Compact text rendering of the sent message window
            covering this attempt (already produced by the action_lab
            diagnostic emitter).
        received_window: Compact text rendering of the received window.
        page_snapshot_count: Number of teleport-phase page snapshots
            captured for the attempt.
    """

    target_x: int
    target_y: int
    teleport_cycle_id: int
    status: str
    timestamp: str
    sent_window: str
    received_window: str
    page_snapshot_count: int


class MapOpenSkippedRecordDict(TypedDict):
    """One ``map_open_skipped_already_open`` event.

    Attributes:
        origin: Code site that emitted the skip
            (``acquisition_phase`` or ``executor.dispatch_command.*``).
        timestamp: ISO timestamp from the event record.
    """

    origin: str
    timestamp: str


class FuelTargetSelectionRecordDict(TypedDict):
    """One ``fuel_target_selection`` event from a probe radar cycle.

    Attributes:
        radar_cycle_id: Radar cycle ID stamped on the diagnostic.
        target_present: Whether a fuel target was selected.
        target_x: Selected target X coordinate (``-1`` when none).
        target_y: Selected target Y coordinate (``-1`` when none).
        summary: Compact ``describe_container_search`` summary string.
        decision_basis: Compact decision-basis breakdown string.
        timestamp: ISO timestamp from the event record.
    """

    radar_cycle_id: int
    target_present: bool
    target_x: int
    target_y: int
    summary: str
    decision_basis: str
    timestamp: str


class WireCompleteRecordDict(TypedDict):
    """One ``WIRE_COMPLETE`` event.

    Attributes:
        action_kind: Kind of action that completed (``map_open``,
            ``move``, ``teleport``, ``collect``, ``scan``).
        duration_ms: Wall-clock milliseconds between dispatch and the
            observed completion signal.
        signal: Authoritative completion signal name.
        timestamp: ISO timestamp from the event record.
    """

    action_kind: str
    duration_ms: int
    signal: str
    timestamp: str


class SessionRoomRecordDict(TypedDict):
    """The single ``session_room_joined`` event for the run.

    Attributes:
        room_id: Room ID joined for the session.
        field_image: Field image name reported by the server, or
            ``unknown`` when the room image cache was empty.
        timestamp: ISO timestamp from the event record.
    """

    room_id: str
    field_image: str
    timestamp: str


class StateBudgetRecordDict(TypedDict):
    """Seconds spent in one bot state across the session.

    Attributes:
        state: Bot state name (``COMBAT``, ``MOVING``, ``IDLE``, ...).
        seconds: Whole seconds attributed to the state, summed across
            every visit (event timestamps have second granularity).
    """

    state: str
    seconds: int


class TargetedTeleportRecordDict(TypedDict):
    """One targeted-teleport DIAGNOSTIC event.

    Shared row shape for the ``fuel_dot_hop`` and
    ``equipment_approach`` diagnostics -- both record a deliberate
    teleport at a known coordinate with the fuel level at dispatch.

    Attributes:
        target_x: Target X coordinate.
        target_y: Target Y coordinate.
        fuel: Fuel level when the teleport was planned.
        timestamp: ISO timestamp from the event record.
    """

    target_x: int
    target_y: int
    fuel: int
    timestamp: str


class InventoryCountsDict(TypedDict):
    """Absolute counts for the five inventory item types.

    Attributes:
        armor: Armor shield count.
        dual: Dual shot count.
        missile: Missile shot count.
        homing: Homing shot count.
        radar: Extra radar count.
    """

    armor: int
    dual: int
    missile: int
    homing: int
    radar: int


def make_zero_inventory_counts() -> InventoryCountsDict:
    """Return inventory counts with every item at zero.

    Returns:
        All-zero inventory counts, used as the gain-total accumulator
        seed.
    """
    return InventoryCountsDict(armor=0, dual=0, missile=0, homing=0, radar=0)


def make_unsampled_inventory_counts() -> InventoryCountsDict:
    """Return the sentinel inventory counts for a run with no samples.

    Returns:
        Inventory counts with every item at ``-1``, mirroring the
        ``fuel_min == -1`` no-samples convention.
    """
    return InventoryCountsDict(armor=-1, dual=-1, missile=-1, homing=-1, radar=-1)


class SessionScorecardDict(TypedDict):
    """Per-run outcome scorecard distilled from the event stream.

    This is the audit every live run gets compared on: where the time
    went, what combat produced, how low fuel dipped, how the inventory
    moved, what each radar press actually consumed, and whether the
    dot-atlas refuels or equipment approaches show pathological
    repetition (the orbit class of bug from live runs 20260612-062453
    and 20260612-071918).

    Attributes:
        duration_seconds: Whole seconds between the first and last
            event record.
        state_budget: Seconds per bot state, sorted by descending
            seconds then state name.
        kills: Count of ``tank_deactivated`` DIAGNOSTIC events.
        shots: Count of ``WIRE`` events whose message starts with
            ``shoot(``.
        fuel_min: Lowest ``belief_fuel`` across
            ``self_alignment_sample`` events, or ``-1`` with no samples.
        fuel_last: Final ``belief_fuel`` sample, or ``-1`` with no
            samples.
        fuel_sample_count: Number of fuel samples observed.
        dot_hops: Every ``fuel_dot_hop`` event in order.
        dot_hop_distinct_targets: Number of distinct dot coordinates
            targeted.
        dot_hop_max_repeats: Highest event count for any single dot
            coordinate, ``0`` with no hops.
        inventory_first: First ``inventory_sample`` counts, or the
            all ``-1`` sentinel with no samples.
        inventory_last: Final ``inventory_sample`` counts, or the
            all ``-1`` sentinel with no samples.
        inventory_sample_count: Number of inventory samples observed.
        equipment_gain_events: Count of ``equipment_gain`` events
            (0x67 messages -- one per equipment container collected).
        equipment_gained: Per-type totals summed across every
            ``equipment_gain`` event.
        scans_extra: Radar dispatches that consumed an extra radar.
        scans_builtin: Radar dispatches that used the free 5x5 scan.
        equipment_approaches: Every ``equipment_approach`` event in
            order.
        equipment_approach_distinct_targets: Number of distinct
            equipment coordinates teleport-approached.
        equipment_approach_max_repeats: Highest event count for any
            single equipment coordinate, ``0`` with no approaches.
    """

    duration_seconds: int
    state_budget: list[StateBudgetRecordDict]
    kills: int
    shots: int
    fuel_min: int
    fuel_last: int
    fuel_sample_count: int
    dot_hops: list[TargetedTeleportRecordDict]
    dot_hop_distinct_targets: int
    dot_hop_max_repeats: int
    inventory_first: InventoryCountsDict
    inventory_last: InventoryCountsDict
    inventory_sample_count: int
    equipment_gain_events: int
    equipment_gained: InventoryCountsDict
    scans_extra: int
    scans_builtin: int
    equipment_approaches: list[TargetedTeleportRecordDict]
    equipment_approach_distinct_targets: int
    equipment_approach_max_repeats: int


class IssueReportDict(TypedDict):
    """Aggregated post-run analysis of a JSONL event artifact.

    Attributes:
        source_path: JSONL path the report was built from.
        mode: Runtime mode string from the events (``bot``, ``sniff``,
            ``probe:<name>``).
        event_count: Total event records in the artifact.
        session_room: Recorded room/field for this session, or ``None``
            when the artifact does not include a ``session_room_joined``
            diagnostic.
        teleport_attempts: Every teleport attempt observed.
        map_open_skipped: Every ``map_open_skipped_already_open`` event.
        fuel_target_selections: Every fuel target selection (selected
            and rejected).
        wire_completes: Every ``WIRE_COMPLETE`` event.
        teleport_success_count: Count of attempts whose status is
            ``landed_exact`` or ``landed_inexact``.
        teleport_failure_count: Count of attempts whose status is not a
            "landed" status (timeouts, etc.).
        fuel_selected_count: Count of fuel target selections where
            ``target_present`` is True.
        fuel_rejected_count: Count of fuel target selections where
            ``target_present`` is False.
        map_open_dispatches: Count of ``WIRE`` events whose first
            message starts with ``map_open`` (i.e. successful sends).
        map_open_completions: Count of ``WIRE_COMPLETE`` events whose
            ``action_kind`` is ``map_open``.
        scorecard: Per-run outcome scorecard (time budget, combat,
            fuel trajectory, dot-hop ledger).
    """

    source_path: str
    mode: str
    event_count: int
    session_room: SessionRoomRecordDict | None
    teleport_attempts: list[TeleportAttemptRecordDict]
    map_open_skipped: list[MapOpenSkippedRecordDict]
    fuel_target_selections: list[FuelTargetSelectionRecordDict]
    wire_completes: list[WireCompleteRecordDict]
    teleport_success_count: int
    teleport_failure_count: int
    fuel_selected_count: int
    fuel_rejected_count: int
    map_open_dispatches: int
    map_open_completions: int
    scorecard: SessionScorecardDict
    recovery_boxed_in_count: int


__all__ = [
    "FuelTargetSelectionRecordDict",
    "InventoryCountsDict",
    "IssueReportDict",
    "MapOpenSkippedRecordDict",
    "SessionRoomRecordDict",
    "SessionScorecardDict",
    "StateBudgetRecordDict",
    "TargetedTeleportRecordDict",
    "TeleportAttemptRecordDict",
    "WireCompleteRecordDict",
    "make_unsampled_inventory_counts",
    "make_zero_inventory_counts",
]
