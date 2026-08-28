"""Row and table shapes for the per-run digest.

Split from :mod:`tankpit_bot.diagnostics.run_digest` (2026-08-28, at
the file-size bar) when the wasted-tick census landed: the builder and
CLI stay there, the renderer lives in
:mod:`tankpit_bot.diagnostics.run_digest_render`, and both import the
shapes from here.
"""

from __future__ import annotations

from typing_extensions import TypedDict


class DisplacementRowDict(TypedDict):
    """One repeated-displacement histogram row.

    Attributes:
        requested_x: Aimed landing X.
        requested_y: Aimed landing Y.
        count: How many teleports at this tile displaced.
    """

    requested_x: int
    requested_y: int
    count: int


class ClearanceShotRowDict(TypedDict):
    """One mine-clearance shot and whether it converted.

    Attributes:
        timestamp: Shot decision timestamp.
        x: Aim tile X.
        y: Aim tile Y.
        pickup_followed: A pickup dispatched within the convert window.
    """

    timestamp: str
    x: int
    y: int
    pickup_followed: bool


class TimelineRowDict(TypedDict):
    """Activity counts for one five-minute bucket.

    Attributes:
        minute: Bucket start offset from session start, in minutes.
        kills: Kills registered in the bucket.
        shots: Shoot dispatches in the bucket.
        teleports: Teleport dispatches in the bucket.
        pickups: Pickup dispatches in the bucket.
    """

    minute: int
    kills: int
    shots: int
    teleports: int
    pickups: int


class RunDigestDict(TypedDict):
    """The whole-run digest table.

    Attributes:
        source: Events artifact the digest was computed from.
        started_at: First event timestamp.
        ended_at: Last event timestamp.
        duration_s: Wall seconds between first and last event.
        clean_exit: A session scorecard event was present (teardown ran).
        exit_reason: Scorecard exit reason, empty when the run crashed.
        room_id: Last joined room.
        self_tank_id: Wire id of our own tank (-1 when never identified).
        kills: Count of ``tank_deactivated`` events whose ``killer_id``
            names this session's own tank -- the same attribution rule
            the scorecard uses. The old free-text "kill registered"
            count missed coordinate-aimed kills (the 2026-08-26
            arterial run: 44 wire kills, 43 registered lines -- the
            missing one hit orange-7 with ``on_intended_target=false``).
        deaths: Own deactivations observed.
        rank_changes: Wire-observed own rank changes, in order (e.g.
            ``"promoted to captain (rank 5)"``) — the 2026-08-27
            Captain promotion went unnoticed for a session because
            nothing surfaced it.
        shots: Shoot dispatches.
        hits: Server-confirmed shot hits (``action_outcome`` hit).
        misses: Server-confirmed shot misses.
        zero_yield_radars: Radar dispatches followed by no container
            pickup before the next radar (or session end) — scans
            that bought nothing collectible.
        damage_dealt: Fuel-confirmed damage dealt (``damage_ledger``).
        damage_taken: Fuel-confirmed damage taken.
        teleports: Teleport dispatches.
        pickups: Pickup dispatches.
        displacements: Total displaced teleports.
        displacement_top: Most-displaced request tiles, descending.
        clearance_shots: Every mine-clearance shot with conversion.
        releases_by_reason: ``plan_released`` reason counts.
        liveness_stalls: ``liveness_stall`` diagnostics -- the live
            livelock detector crossed its zero-dispatch streak. The
            detector fired unread for a week (2026-08-21 gatherer
            livelock era) because no report consumed it.
        superseded_undispatched: ``action_outcome`` superseded closes
            whose decision never reached the wire -- planner churn,
            the wasted-tick shape the livelock detector streaks on.
        superseded_dispatched: Superseded closes of decisions that DID
            reach the wire -- re-aims, not waste.
        max_wire_gap_s: Longest silence between consecutive WIRE
            dispatches, in whole seconds. A live session's tick loop
            dispatches every ~2 s; a triple-digit gap is a stall the
            timeline buckets smooth over.
        wire_gaps_over_30s: How many inter-dispatch gaps exceeded 30 s.
        rank_name: Account rank label at startup, empty when unscraped.
        rank_number: Countdown rank number at startup (-1 unscraped).
        promotion_points: Account promotion points (-1 unscraped).
        inventory_first: First sampled (armor,dual,missile,homing,radar).
        inventory_last: Last sampled (armor,dual,missile,homing,radar).
        timeline: Five-minute activity buckets.
    """

    source: str
    started_at: str
    ended_at: str
    duration_s: int
    clean_exit: bool
    exit_reason: str
    room_id: str
    self_tank_id: int
    kills: int
    deaths: int
    rank_changes: list[str]
    shots: int
    hits: int
    misses: int
    zero_yield_radars: int
    damage_dealt: int
    damage_taken: int
    teleports: int
    pickups: int
    displacements: int
    displacement_top: list[DisplacementRowDict]
    clearance_shots: list[ClearanceShotRowDict]
    releases_by_reason: dict[str, int]
    liveness_stalls: int
    superseded_undispatched: int
    superseded_dispatched: int
    max_wire_gap_s: int
    wire_gaps_over_30s: int
    rank_name: str
    rank_number: int
    promotion_points: int
    inventory_first: list[int]
    inventory_last: list[int]
    timeline: list[TimelineRowDict]


__all__ = [
    "ClearanceShotRowDict",
    "DisplacementRowDict",
    "RunDigestDict",
    "TimelineRowDict",
]
