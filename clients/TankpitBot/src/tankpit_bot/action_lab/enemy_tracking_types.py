"""TypedDicts for the enemy-tracking probe session.

The record shapes only. Their encode/decode pairs are
:mod:`tankpit_bot.action_lab.enemy_tracking_codecs`.
"""

from __future__ import annotations

from typing import TypedDict

from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.browser.page_client_snapshot import PageClientSnapshotDict


class OurTankBeliefDict(TypedDict):
    """Snapshot of our wire-derived belief about one tank at one tick.

    Mirrors :class:`tankpit_bot.state.types.TankStateDict` but flattens
    to the exact fields needed for tracking analysis. Capturing this
    row-by-row gives a frame-by-frame view of how our world state
    evolves -- specifically, the moment wire-presence ages out or the
    registry zeroes a position.

    Attributes:
        tank_id: Tank id in our registry.
        present: Whether ``world.tanks[id]`` had an entry at sample time.
        x: Last-known tile X (0 when ``present`` is False).
        y: Last-known tile Y (0 when ``present`` is False).
        liveness: Liveness label (``"alive"``, ``"deactivated"``, ``""`` when absent).
        last_wire_seen_ms: Timestamp of the tank's most recent wire packet.
        last_position_update_ms: Timestamp of the tank's most recent
            position-bearing wire packet.
        wire_age_ms: ``sample_timestamp_ms - last_wire_seen_ms``.
        position_age_ms: ``sample_timestamp_ms - last_position_update_ms``.
        is_in_threats: Whether the tank was included by
            ``analyze_threats`` for this sample.
        would_locked_target_return: True when, given a hypothetical
            ``combat_target_id == tank_id``, ``get_locked_target`` would
            return a non-None value. This is the field that pinpoints
            whether ``_decide_hunt_engage`` would drop the lock.
        locked_target_source: Where the would-be lock came from
            (``"threats"`` -- in current threat list,
            ``"world_fallback"`` -- synthesised from world.tanks,
            ``"none"`` -- dropped).
    """

    tank_id: int
    present: bool
    x: int
    y: int
    liveness: str
    last_wire_seen_ms: int
    last_position_update_ms: int
    wire_age_ms: int
    position_age_ms: int
    is_in_threats: bool
    would_locked_target_return: bool
    locked_target_source: str


class JSTankBeliefDict(TypedDict):
    """Snapshot of the JS client's own belief about one tank at one tick.

    Captured from ``world_collections`` keyed by ``P.j`` in the
    page-client snapshot (the live JS tank registry per
    :mod:`page_client_snapshot`). When the JS client still lists a
    tank but our wire-presence is dead, the JS side wins -- the JS
    client is the official source of truth for what's visible.

    Attributes:
        present: Whether *any* entry in ``P.j`` matched at sample time.
            Matching uses ``tracked_js_key`` (see
            :class:`TrackedEnemyDict`) for stable identity across ticks
            even if the position changes.
        fields: The matched entry's primitive fields verbatim, minified
            key names from the JS client. Empty when ``present`` is
            False.
    """

    present: bool
    fields: dict[str, int | float | bool | str | None]


class TrackingObservationDict(TypedDict):
    """One per-tank sample row from the tracking probe.

    Pairs our-side and JS-side belief at the same wall-clock instant
    so divergence is unambiguous in the output. ``tracked_label`` is
    the human-readable tank name captured at acquisition time so
    rows from later samples remain readable after the tank has left
    the threat list.

    Attributes:
        sample_index: Zero-based index of the sample within the probe.
        sample_timestamp_ms: Wall-clock time the sample was captured.
        tank_id: Stable tank id from our world state at acquisition.
        tracked_label: Human-readable tank name at acquisition time.
        our_belief: Our wire-derived view of the tank for this sample.
        js_belief: The JS client's view of the same tank for this sample.
        bot_combat_target_id: The bot's ``ai_state.combat_target_id``
            at sample time -- ``-1`` when no lock; ``tank_id`` when
            this tank is the lock; some other id when the bot has
            already moved on.
        bot_mode_state: The bot's ``ai_state.mode_state`` at sample
            time. Reading this tells you whether the bot was in
            ``ENGAGE`` (post-shot, about to drop lock) or some other
            state.
    """

    sample_index: int
    sample_timestamp_ms: int
    tank_id: int
    tracked_label: str
    our_belief: OurTankBeliefDict
    js_belief: JSTankBeliefDict
    bot_combat_target_id: int
    bot_mode_state: str


class TrackedEnemyDict(TypedDict):
    """One enemy the probe locked on to at acquisition time.

    Recording the JS-side key (the minified field name and value used
    to identify the entry in ``P.j``) makes the cross-tick join
    stable even if the position changes between samples -- a position
    match alone would lose identity the moment the enemy moves.

    Attributes:
        tank_id: Stable tank id from our world state.
        name: Human-readable tank name at acquisition.
        team: Team id at acquisition.
        rank: Rank at acquisition.
        acquired_x: Tile X at acquisition.
        acquired_y: Tile Y at acquisition.
        tracked_js_key: The minified field name within a ``P.j`` item
            used to identify the matching JS entry across samples.
            Empty when no JS-side entry could be paired at acquisition
            -- which is itself a data point.
        tracked_js_value: The value of ``tracked_js_key`` in the
            matched ``P.j`` entry. Encoded as a string so the field
            map's mixed primitive value type round-trips cleanly.
    """

    tank_id: int
    name: str
    team: int
    rank: int
    acquired_x: int
    acquired_y: int
    tracked_js_key: str
    tracked_js_value: str


class ShotEventDict(TypedDict):
    """One shot fired by the probe, with the server's response.

    The probe fires ONE shot at the closest enemy after teleporting
    adjacent -- that is the user-reported failure scenario ("fires
    one shot then finds a new one"). Recording the shot's
    boundaries gives the analysis script the line between
    pre-shot and post-shot samples.

    Attributes:
        target_tank_id: Tank id of the shot target.
        target_x: Target tile X at the moment of the shot.
        target_y: Target tile Y at the moment of the shot.
        self_x: Our tank's tile X at the moment of the shot.
        self_y: Our tank's tile Y at the moment of the shot.
        sent_ms: Wall-clock time the shot command was sent.
        responded_ms: Wall-clock time the shot response arrived
            (``-1`` when no response within timeout).
        outcome: ``"hit"``, ``"miss"``, or ``"timeout"``.
    """

    target_tank_id: int
    target_x: int
    target_y: int
    self_x: int
    self_y: int
    sent_ms: int
    responded_ms: int
    outcome: str


class EnemyTrackingProbeSessionDict(TypedDict):
    """Complete tracking-probe session payload.

    The file the analysis tools read to find the divergence row.
    Carries enough metadata to reproduce the run end-to-end:
    bootstrap timing, the enemies under track, the shot fired, and
    the per-tank sample stream.

    Attributes:
        session_id: Stable probe session id.
        start_timestamp_ms: Wall-clock time the probe entered bootstrap.
        end_timestamp_ms: Wall-clock time the probe finished sampling.
        base_url: Probe target URL.
        spawn_x: Tile X of our spawn point.
        spawn_y: Tile Y of our spawn point.
        capture_session_path: On-disk path of the raw wire capture.
        initial_sync_timeout_ms: Bootstrap timeout used.
        startup_timing: Standard startup timing payload.
        acquisition_timeout_ms: Map-open / acquisition timeout used.
        teleport_timeout_ms: Combat teleport timeout used.
        shot_feedback_timeout_ms: Per-shot feedback timeout used.
        sample_interval_ms: Sampling cadence used by the probe.
        sample_duration_ms: Total sampling window after the shot.
        tracked: Enemies the probe locked on to at acquisition.
        shot: The shot the probe fired (or ``None`` when acquisition
            never reached the engage tile).
        snapshot_at_acquisition: Page-client snapshot captured the
            tick the shot fired (or just after acquisition when no
            shot fired).
        observations: All per-tank, per-sample rows produced by the
            sampling loop. Ordered by ``sample_index`` then ``tank_id``.
    """

    session_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int
    base_url: str
    spawn_x: int
    spawn_y: int
    capture_session_path: str
    initial_sync_timeout_ms: int
    startup_timing: TeleportStartupTimingDict
    acquisition_timeout_ms: int
    teleport_timeout_ms: int
    shot_feedback_timeout_ms: int
    sample_interval_ms: int
    sample_duration_ms: int
    tracked: list[TrackedEnemyDict]
    shot: ShotEventDict | None
    snapshot_at_acquisition: PageClientSnapshotDict
    observations: list[TrackingObservationDict]


__all__ = [
    "EnemyTrackingProbeSessionDict",
    "JSTankBeliefDict",
    "OurTankBeliefDict",
    "ShotEventDict",
    "TrackedEnemyDict",
    "TrackingObservationDict",
]
