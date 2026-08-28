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


class ActionOutcomeRowDict(TypedDict):
    """One ``action_outcome`` DIAGNOSTIC event (the unified fabric).

    Attributes:
        action_kind: Kind of action that resolved (``map_open``,
            ``move``, ``teleport``, ``collect``, ``scan``, ``shoot``).
        outcome: Outcome label from the kind's outcome union
            (``landed_exact``, ``stall_timeout``, ``hit``,
            ``discarded_no_container``, ...).
        event_id: Process-wide monotonic event id.
        attempt_id: Per-kind monotonic attempt counter.
        duration_ms: Wall-clock milliseconds between dispatch and the
            resolution (0 for pre-dispatch executor discards, -1 when
            the gate fired with no recorded dispatch time).
        dispatched: Whether the decision's command reached the wire.
            True by definition for every genuine resolution (the wire
            answered, so something was sent); for ``superseded`` rows
            it is the executor's dispatch mark — False on artifacts
            predating the mark (2026-08-21), where a dispatched
            supersede is indistinguishable from a vetoed one.
        timestamp: ISO timestamp from the event record.
    """

    action_kind: str
    outcome: str
    event_id: int
    attempt_id: int
    duration_ms: int
    dispatched: bool
    timestamp: str


class SuppressedDispatchRecordDict(TypedDict):
    """Per-target tally of ``dispatch_suppressed`` DIAGNOSTIC events.

    A suppression is the executor's refusal prediction sparing one
    wasted server call -- designed behavior, once. A TALLY is the
    planner failing to consume that veto: nothing was dispatched, so
    no failure mark exists, and the same belief keeps winning the next
    plan (the 2026-08-20 gatherer livelock re-planned one suppressed
    pickup 93 consecutive ticks while the report read healthy).

    Attributes:
        command_name: Suppressed command kind (``pickup_equipment``,
            ``pickup_fuel``, ``teleport``).
        target_x: Target tile X.
        target_y: Target tile Y.
        predicted_error_code: The 0x52 code the belief predicted (the
            last one seen for this target).
        count: How many times this exact target was suppressed.
    """

    command_name: str
    target_x: int
    target_y: int
    predicted_error_code: int
    count: int


class DisplacedTeleportRecordDict(TypedDict):
    """Per-destination tally of ``teleport_displacement`` events.

    A displaced teleport resolves as a SUCCESS (``landed_inexact``),
    so repetition at one destination accumulates in no failure
    machinery — the third liveness flavor (successful actions, no
    progress): the 08-05 ancestor ran 534 bounces at one tile over 43
    minutes, the 2026-08-21 marooning ran 4-in-10-s escape and
    harvest loops, and none of it surfaced anywhere.

    Attributes:
        requested_x: The repeatedly requested landing X.
        requested_y: The repeatedly requested landing Y.
        count: How many teleports at this destination displaced.
        max_displacement: Largest Manhattan bounce observed.
    """

    requested_x: int
    requested_y: int
    count: int
    max_displacement: int


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
        stretches: Number of distinct visits to the state. A large
            total made of many short visits (tick-boundary residue) is
            healthy; the same total in one visit is a stall.
        max_seconds: Longest single visit in whole seconds. The
            stall detector: run 20260729-105325 spent 285s IDLE across
            283 visits with ``max_seconds=6`` -- pure residue, no stall.
    """

    state: str
    seconds: int
    stretches: int
    max_seconds: int


class FuelLowWaterEpisodeDict(TypedDict):
    """One contiguous dip of belief fuel below the session's danger line.

    The scorecard's bare ``fuel_min`` says how low the session got but
    not why; each episode narrates one dip -- what spent the fuel, how
    low it went, and what refilled it (run 20260729-105325: a 158-fuel
    chase teleport explained the session min of 140, invisible in the
    old report).

    Attributes:
        start_timestamp: ISO timestamp of the first below-threshold
            fuel sample.
        end_timestamp: ISO timestamp of the last below-threshold fuel
            sample in the episode.
        duration_seconds: Whole seconds between the first and last
            below-threshold samples (0 for a single-sample dip).
        entry_fuel: Last at-or-above-threshold sample before the
            episode, or ``-1`` when the session started below.
        min_fuel: Lowest sample inside the episode.
        cause_kind: ``in_flight_action_kind`` of the largest
            sample-to-sample fuel drop between entry and the episode
            minimum (``none`` when nothing was in flight).
        cause_drop: Size of that largest drop in fuel units (positive).
        cause_state: ``bot_state`` at the sample where the largest
            drop landed.
        recovery_fuel: First at-or-above-threshold sample after the
            episode, or ``-1`` when the session ended below.
        recovery_kind: ``in_flight_action_kind`` at the recovery
            sample, or ``""`` when the session ended below.
    """

    start_timestamp: str
    end_timestamp: str
    duration_seconds: int
    entry_fuel: int
    min_fuel: int
    cause_kind: str
    cause_drop: int
    cause_state: str
    recovery_fuel: int
    recovery_kind: str


class TeleportSpendRecordDict(TypedDict):
    """Fuel spent on teleports, grouped by the bot state paying it.

    Teleports were the dominant fuel expense of run 20260729-105325
    (measured 15592, ledger feasibility bound 11993..19290) but the
    fuel book's totals cannot say whether chases or forage hops paid
    it; this row attributes each WORLD-channel fuel debit billed while
    a teleport was in flight to the ``bot_state`` that dispatched the
    jump.

    Attributes:
        bot_state: ``bot_state`` context (``MODE/STATE``) stamped on
            the WORLD fuel-transition record.
        drops: Number of in-flight debit receipts observed
            (approximately the jump count -- a jump billed across two
            receipts counts twice, with the fuel split between them).
        fuel_spent: Total fuel across those receipts (positive).
    """

    bot_state: str
    drops: int
    fuel_spent: int


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
    moved, and what each radar press actually consumed.

    Attributes:
        duration_seconds: Whole seconds between the first and last
            event record.
        state_budget: Seconds per bot state, sorted by descending
            seconds then state name.
        kills: Count of ``tank_deactivated`` DIAGNOSTIC events.
        shots: Count of ``WIRE`` events whose message starts with
            ``shoot(``.
        combat_misses: Count of ``combat_miss`` DIAGNOSTIC events
            (shot resolved with no tank at the target tile).
        tank_damage_changes: Count of ``tank_damage_changed``
            DIAGNOSTIC events.
        fuel_min: Lowest ``belief_fuel`` across
            ``self_alignment_sample`` events, or ``-1`` with no samples.
        fuel_last: Final ``belief_fuel`` sample, or ``-1`` with no
            samples.
        fuel_sample_count: Number of fuel samples observed.
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
        physics_divergences: Count of ``physics_divergence``
            DIAGNOSTIC events -- fuel windows outside the physics
            book's feasibility interval.
        action_outcome_counts: Per ``"kind:outcome"`` tallies from the
            unified ``action_outcome`` fabric, sorted by key.
        fuel_low_water_threshold: The danger line used for episode
            detection -- the highest ``escape_floor`` any
            ``engagement_break`` event computed this session, falling
            back to the static critical floor when combat never
            projected one.
        fuel_low_water_episodes: Every contiguous dip below the
            threshold, in stream order.
        teleport_spend: In-flight teleport fuel debits grouped by
            paying ``bot_state``, sorted by descending spend.
        teleport_spend_total: Sum of ``fuel_spent`` across the groups.
        ledger_teleport_spend_min: Least possible total teleport
            spend from the fuel book's summed feasibility intervals
            (the negated ``teleport_fuel_hi``), or ``-1`` when the run
            ended without a ``damage_ledger`` event. The measured
            ``teleport_spend_total`` must fall inside
            ``[min, max]`` -- outside means the attribution drifted.
        ledger_teleport_spend_max: Greatest possible total teleport
            spend (the negated ``teleport_fuel_lo``), or ``-1``.
        ledger_shot_singles: ``shot_single_count`` from the
            ``damage_ledger`` event, or ``-1`` without a ledger.
            Nonzero singles under a dual+homing loadout are
            server-billed non-connects (weapon=0 misses/clips), not
            loadout drift -- see wiki weapon-selection.
        ledger_shot_duals: ``shot_dual_count`` from the ledger, or
            ``-1``.
        ledger_shot_homings: ``shot_homing_count`` from the ledger,
            or ``-1``.
    """

    duration_seconds: int
    state_budget: list[StateBudgetRecordDict]
    kills: int
    shots: int
    combat_misses: int
    tank_damage_changes: int
    fuel_min: int
    fuel_last: int
    fuel_sample_count: int
    inventory_first: InventoryCountsDict
    inventory_last: InventoryCountsDict
    inventory_sample_count: int
    equipment_gain_events: int
    equipment_gained: InventoryCountsDict
    scans_extra: int
    scans_builtin: int
    physics_divergences: int
    action_outcome_counts: dict[str, int]
    fuel_low_water_threshold: int
    fuel_low_water_episodes: list[FuelLowWaterEpisodeDict]
    teleport_spend: list[TeleportSpendRecordDict]
    teleport_spend_total: int
    ledger_teleport_spend_min: int
    ledger_teleport_spend_max: int
    ledger_shot_singles: int
    ledger_shot_duals: int
    ledger_shot_homings: int
    # Career-totals snapshot taken at the LAST 0x56 Statistics broadcast
    # the run saw. -1 means the wire never sent one (very short Practice
    # runs sometimes finish before the cadence fires).
    career_destroyed_last: int
    career_deactivated_last: int
    career_score_last: int
    career_playtime_seconds_last: int
    # Per-record container pickup tallies (each multi-record 0x43 body
    # contributes N events). ``container_pickups_partial`` is records
    # where the picker hit the fuel cap and left some fuel; everything
    # else counts as ``container_pickups_full``.
    container_pickups_full: int
    container_pickups_partial: int


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
        teleport_attempts: Every action-lab ``teleport_attempt``
            diagnostic observed (probe runs; the live bot records
            teleports as ``action_outcome`` events instead).
        map_open_skipped: Every ``map_open_skipped_already_open`` event.
        fuel_target_selections: Every fuel target selection (selected
            and rejected).
        action_outcomes: Every ``action_outcome`` event (the unified
            per-attempt fabric from the ledger).
        teleport_success_count: Landed teleports -- action-lab attempts
            with a landed status plus bot outcomes ``landed_exact`` /
            ``landed_inexact``.
        teleport_failure_count: Non-landed teleport resolutions from
            both sources (timeouts, rejections, discards).
        fuel_selected_count: Count of fuel target selections where
            ``target_present`` is True.
        fuel_rejected_count: Count of fuel target selections where
            ``target_present`` is False.
        map_open_dispatches: Count of ``WIRE`` events whose first
            message starts with ``map_open`` (i.e. successful sends).
        map_open_completions: Count of ``action_outcome`` events with
            ``action_kind == "map_open"`` and
            ``outcome == "map_data_processed"``.
        suppressed_dispatches: Per-target ``dispatch_suppressed``
            tallies, highest count first -- the executor's belief-veto
            refusals the planner failed to consume.
        displaced_teleports: Per-destination ``teleport_displacement``
            tallies, highest count first -- bounced landings that
            resolve as successes and therefore hide from every
            failure counter.
        wire_dispatches_by_kind: Count of ``WIRE`` command sends per
            LEDGER action kind (the wire event's ``action_kind`` field
            mapped through the command-to-kind table). The completion
            audit compares this against recorded completions: a kind
            that dispatches but never completes is a ledger modeling
            gap, and every outcome-derived rule is blind to it
            (2026-08-21: shoot ran 13 dispatches / 0 completions and
            the liveness scan read the silence as a livelock).
        scorecard: Per-run outcome scorecard (time budget, combat,
            fuel trajectory, equipment-approach ledger).
    """

    source_path: str
    mode: str
    event_count: int
    session_room: SessionRoomRecordDict | None
    teleport_attempts: list[TeleportAttemptRecordDict]
    map_open_skipped: list[MapOpenSkippedRecordDict]
    fuel_target_selections: list[FuelTargetSelectionRecordDict]
    action_outcomes: list[ActionOutcomeRowDict]
    teleport_success_count: int
    teleport_failure_count: int
    fuel_selected_count: int
    fuel_rejected_count: int
    map_open_dispatches: int
    map_open_completions: int
    suppressed_dispatches: list[SuppressedDispatchRecordDict]
    displaced_teleports: list[DisplacedTeleportRecordDict]
    wire_dispatches_by_kind: dict[str, int]
    scorecard: SessionScorecardDict


__all__ = [
    "ActionOutcomeRowDict",
    "DisplacedTeleportRecordDict",
    "FuelLowWaterEpisodeDict",
    "FuelTargetSelectionRecordDict",
    "InventoryCountsDict",
    "IssueReportDict",
    "MapOpenSkippedRecordDict",
    "SessionRoomRecordDict",
    "SessionScorecardDict",
    "StateBudgetRecordDict",
    "SuppressedDispatchRecordDict",
    "TeleportAttemptRecordDict",
    "TeleportSpendRecordDict",
    "make_unsampled_inventory_counts",
    "make_zero_inventory_counts",
]
