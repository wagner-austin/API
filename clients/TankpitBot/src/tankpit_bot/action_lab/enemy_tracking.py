"""Live enemy-tracking probe.

Validates the wire-presence heuristic against the JS client's own
tank registry. The probe acquires every visible enemy, teleports
adjacent to the closest, fires ONE shot at it, then samples for a
fixed duration -- recording, for each tracked enemy at each tick,
both our wire-derived belief AND the JS client's belief from
``activeGame.P.j``.

The output JSON is what the analysis script reads to answer "is the
wire-presence TTL accurate" and "did the bot drop the lock when JS
truth still had the tank visible". This is the action-lab counterpart
to the synthetic scenario tests in :mod:`tests.scenarios`; the
scenario tests assert HFSM logic against wire we wrote ourselves,
this probe measures the wire heuristic itself against ground truth.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.enemy_tracking_types import (
    EnemyTrackingProbeSessionDict,
    ShotEventDict,
    TrackedEnemyDict,
    TrackingObservationDict,
    encode_enemy_tracking_probe_session,
)
from tankpit_bot.action_lab.page_client_snapshot import (
    PageClientSnapshotDict,
    capture_page_client_snapshot,
)
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.probe_entrypoint import (
    run_and_save_standard_probe_session,
)
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.teleport_acquisition import run_tracked_acquisition_phase
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    _wait_for_teleport_outcome,
)
from tankpit_bot.action_lab.teleport_phase import run_tracked_teleport_command
from tankpit_bot.action_lab.tracking_observation import (
    build_tracking_observation,
    find_js_entry_by_position,
    select_js_identity_key,
)
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.bot.ai.combat_landing import choose_combat_landing_tile
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.types import EnemyThreatDict
from tankpit_bot.sniffer.world_state import (
    get_terrain_map,
    get_world_service,
)
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_combat_hit,
    check_and_clear_our_shot_response,
)
from tankpit_bot.state.types import WorldStateDict

log = get_logger(__name__)

#: Polling interval used by the shot-feedback wait. Short enough not
#: to add noticeable latency, long enough to keep CPU use reasonable.
_SHOT_POLL_INTERVAL_MS = 100.0


def _wait_for_shot_feedback(
    page: action_session.WaitPageProtocol,
    probe: ProbeBase,
    *,
    timeout_ms: int,
) -> tuple[bool, bool]:
    """Wait for the server's response to our shot.

    Mirrors the combat-probe shot wait so both probes interpret hit
    / miss / timeout the same way.

    Args:
        page: Active page handle used for cadence sleeps.
        probe: Probe instance whose buffered messages to drain.
        timeout_ms: Maximum wait before giving up.

    Returns:
        ``(got_response, was_hit)`` -- ``got_response`` is False when
        the wait timed out.
    """
    ws = get_world_service()
    started = action_hooks.get_current_time_ms()
    while action_hooks.get_current_time_ms() - started < timeout_ms:
        action_hooks.drain_buffered_messages(probe)
        if ws.got_our_shot_response:
            was_hit = check_and_clear_combat_hit(ws)
            check_and_clear_our_shot_response(ws)
            return (True, was_hit)
        page.wait_for_timeout(_SHOT_POLL_INTERVAL_MS)
    return (False, False)


def _build_tracked_records(
    threats: list[EnemyThreatDict],
    snapshot: PageClientSnapshotDict,
) -> list[TrackedEnemyDict]:
    """Build :class:`TrackedEnemyDict` records for every visible enemy.

    Resolves the JS-side identity for each tank by position-matching
    a ``P.j`` entry against the wire-derived ``(x, y)`` at
    acquisition time. The selected JS field becomes the cross-tick
    join key for the sampling loop.

    Args:
        threats: Visible enemies returned by ``analyze_threats``.
        snapshot: Page-client snapshot captured at acquisition time.

    Returns:
        One record per visible enemy. Records carry empty
        ``tracked_js_key`` and ``tracked_js_value`` when no JS entry
        could be paired -- the row still records the wire-side view
        so divergence stays visible.
    """
    records: list[TrackedEnemyDict] = []
    for threat in threats:
        js_entry = find_js_entry_by_position(
            snapshot["world_collections"],
            threat["x"],
            threat["y"],
        )
        if js_entry is None:
            tracked_key = ""
            tracked_value = ""
        else:
            from tankpit_bot.bot.ai.types import make_enemy_threat as _unused

            del _unused
            tracked_key, tracked_value = _resolve_identity(js_entry, threat["tank_id"])
        records.append(
            TrackedEnemyDict(
                tank_id=threat["tank_id"],
                name=threat["name"],
                team=threat["team"],
                rank=threat["rank"],
                acquired_x=threat["x"],
                acquired_y=threat["y"],
                tracked_js_key=tracked_key,
                tracked_js_value=tracked_value,
            ),
        )
    return records


def _resolve_identity(
    js_entry: dict[str, int | float | bool | str | None],
    tank_id: int,
) -> tuple[str, str]:
    """Pair our tank id with a stable JS-side identity field.

    The JS registry hands us minified field names whose semantics
    we do not statically know. We pair against ``tank_id`` because
    we just confirmed this entry by position -- whichever field
    holds an integer equal to our ``tank_id`` is the JS-side tank
    id.

    Args:
        js_entry: Registry entry that matched our tank by position.
        tank_id: Tank id from our world state.

    Returns:
        ``(key, str(value))`` -- empty strings when no field equals
        the tank id.
    """
    from tankpit_bot.state.types.tank import make_tank_state

    surrogate = make_tank_state(
        tank_id=tank_id,
        x=0,
        y=0,
        team=0,
        rank=0,
        damage_state=0,
        name="",
        is_bot=False,
        is_self=False,
    )
    return select_js_identity_key(js_entry, surrogate)


def _build_sample_observations(
    *,
    sample_index: int,
    sample_timestamp_ms: int,
    tracked: list[TrackedEnemyDict],
    world: WorldStateDict,
    threats: list[EnemyThreatDict],
    snapshot: PageClientSnapshotDict,
    bot_combat_target_id: int,
    bot_mode_state: str,
) -> list[TrackingObservationDict]:
    """Build one observation row per tracked tank for one sample.

    Args:
        sample_index: Zero-based sample number.
        sample_timestamp_ms: Wall-clock time of this sample.
        tracked: Enemies the probe locked on to at acquisition.
        world: World state captured at sample time.
        threats: ``analyze_threats`` output at sample time.
        snapshot: Page-client snapshot at sample time.
        bot_combat_target_id: ``ai_state.combat_target_id`` at sample time.
        bot_mode_state: ``ai_state.mode_state`` at sample time.

    Returns:
        One observation row per tracked tank.
    """
    return [
        build_tracking_observation(
            sample_index=sample_index,
            sample_timestamp_ms=sample_timestamp_ms,
            tank_id=record["tank_id"],
            tracked_label=record["name"],
            tracked_js_key=record["tracked_js_key"],
            tracked_js_value=record["tracked_js_value"],
            world=world,
            threats=threats,
            world_collections=snapshot["world_collections"],
            bot_combat_target_id=bot_combat_target_id,
            bot_mode_state=bot_mode_state,
        )
        for record in tracked
    ]


def format_enemy_tracking_probe_summary(session: EnemyTrackingProbeSessionDict) -> str:
    """Format a compact human-readable summary for the tracking session.

    Highlights the divergence count -- the rows where our wire-side
    belief disagreed with the JS-side belief about a tank's
    presence. A non-zero divergence count is what the user wants
    to read in the terminal after the run.

    Args:
        session: Completed session payload.

    Returns:
        One-line summary string.
    """
    diverged = 0
    our_present_js_absent = 0
    js_present_our_absent = 0
    for observation in session["observations"]:
        our_present = observation["our_belief"]["would_locked_target_return"]
        js_present = observation["js_belief"]["present"]
        if our_present == js_present:
            continue
        diverged += 1
        if our_present:
            our_present_js_absent += 1
        else:
            js_present_our_absent += 1
    return (
        "Enemy tracking probe complete: "
        f"tracked={len(session['tracked'])} "
        f"samples={len(session['observations'])} "
        f"divergence={diverged} "
        f"our_present_js_absent={our_present_js_absent} "
        f"js_present_our_absent={js_present_our_absent}"
    )


class EnemyTrackingProbe(ProbeBase):
    """Live enemy-tracking probe.

    Acquires enemies, fires one shot at the closest, then samples
    wire-side and JS-side belief at a fixed cadence. Output is the
    typed session payload the analysis script reads.
    """

    def _capture_world_threats(
        self,
        cdp: CDPSessionProtocol,
    ) -> tuple[WorldStateDict, list[EnemyThreatDict], PageClientSnapshotDict, int]:
        """Snapshot world state, threats, JS client, and wall time."""
        action_hooks.drain_buffered_messages(self)
        world = self.get_world_state()
        self_state = self._require_self_state()
        sample_ms = action_hooks.get_current_time_ms()
        threats = analyze_threats(world, self_state, sample_ms)
        snapshot = capture_page_client_snapshot(cdp)
        return world, threats, snapshot, sample_ms

    def _acquire_enemies(
        self,
        *,
        cdp: CDPSessionProtocol,
        acquisition_timeout_ms: int,
    ) -> tuple[list[TrackedEnemyDict], PageClientSnapshotDict, list[EnemyThreatDict]]:
        """Open the map, then build :class:`TrackedEnemyDict` records.

        Args:
            cdp: Active CDP session.
            acquisition_timeout_ms: Maximum wait for the map sync.

        Returns:
            Tracked records, the snapshot captured at acquisition,
            and the threat list at acquisition.

        Raises:
            TeleportProbeError: When acquisition does not sync.
        """
        page = self._require_page()
        _started_ms, sync_ms, _snapshots, _capture = run_tracked_acquisition_phase(
            page,
            self,
            cdp=cdp,
            send_command=self.open_map,
            command_name="tracking_acquisition",
            capture_before_map_open=True,
            wait_for_sync=True,
            sync_timeout_ms=acquisition_timeout_ms,
            dispatch_failure_error=TeleportProbeError,
            dispatch_failure_message="acquisition command dispatch failed",
            unavailable_error=TeleportProbeError,
            unavailable_message="cdp session is unavailable",
        )
        if sync_ms is None:
            raise TeleportProbeError("map sync did not complete within timeout")
        _world, threats, snapshot, _ = self._capture_world_threats(cdp)
        tracked = _build_tracked_records(threats, snapshot)
        return tracked, snapshot, threats

    def _teleport_to_closest_enemy(
        self,
        *,
        cdp: CDPSessionProtocol,
        threats: list[EnemyThreatDict],
        teleport_timeout_ms: int,
        message_start_index: int,
    ) -> EnemyThreatDict | None:
        """Teleport adjacent to the closest visible enemy.

        Returns ``None`` when there are no viable landings -- the
        probe still finishes the sampling phase from the spawn point
        in that case so the operator sees what the bot would see.

        Args:
            cdp: Active CDP session.
            threats: Visible enemies at acquisition.
            teleport_timeout_ms: Maximum wait for the teleport outcome.
            message_start_index: Wire-message index at acquisition.

        Returns:
            The teleport target's enemy threat record, or ``None``
            when no landing is possible.
        """
        page = self._require_page()
        if not threats:
            return None
        target_enemy = threats[0]
        self_state = self._require_self_state()
        landing_x, landing_y = choose_combat_landing_tile(
            self.get_world_state(),
            self_state,
            target_enemy,
            get_terrain_map(),
            action_hooks.get_current_time_ms(),
        )
        if landing_x == -1 and landing_y == -1:
            return None
        landing = TeleportTargetDict(
            label=f"track_{target_enemy['tank_id']}_{target_enemy['x']}_{target_enemy['y']}",
            x=landing_x,
            y=landing_y,
        )
        teleport_cycle = self._start_action_phase("teleport", attempt_label=landing["label"])
        teleport_result, _started_ms = run_tracked_teleport_command(
            page,
            self,
            landing,
            teleport_cycle=teleport_cycle,
            message_start_index=message_start_index,
            map_open_started_ms=action_hooks.get_current_time_ms(),
            map_sync_timestamp_ms=action_hooks.get_current_time_ms(),
            fuel_before=self_state["fuel"],
            world_timestamp_before=self.get_world_state()["timestamp_ms"],
            timeout_ms=teleport_timeout_ms,
            page_snapshots=[],
            capture_page_snapshot=lambda phase: action_hooks.capture_teleport_page_snapshot(
                cdp,
                phase,
            ),
            wait_for_outcome=_wait_for_teleport_outcome,
            dispatch_failure_error=TeleportProbeError,
        )
        if teleport_result["status"] == "teleport_timeout":
            return None
        return target_enemy

    def _fire_one_shot(
        self,
        *,
        target: EnemyThreatDict,
        shot_feedback_timeout_ms: int,
    ) -> ShotEventDict:
        """Fire one shot at the locked target and record the outcome.

        Args:
            target: Enemy to shoot.
            shot_feedback_timeout_ms: Maximum wait for the response.

        Returns:
            Shot event record.
        """
        page = self._require_page()
        self_state = self._require_self_state()
        sent_ms = action_hooks.get_current_time_ms()
        self.shoot(target["x"], target["y"], target["tank_id"])
        got_response, was_hit = _wait_for_shot_feedback(
            page,
            self,
            timeout_ms=shot_feedback_timeout_ms,
        )
        responded_ms = action_hooks.get_current_time_ms() if got_response else -1
        if not got_response:
            outcome = "timeout"
        elif was_hit:
            outcome = "hit"
        else:
            outcome = "miss"
        return ShotEventDict(
            target_tank_id=target["tank_id"],
            target_x=target["x"],
            target_y=target["y"],
            self_x=self_state["x"],
            self_y=self_state["y"],
            sent_ms=sent_ms,
            responded_ms=responded_ms,
            outcome=outcome,
        )

    def _sample_loop(
        self,
        *,
        cdp: CDPSessionProtocol,
        tracked: list[TrackedEnemyDict],
        sample_interval_ms: int,
        sample_duration_ms: int,
    ) -> list[TrackingObservationDict]:
        """Sample wire-side and JS-side belief on a fixed cadence.

        Loop terminates after ``sample_duration_ms`` of wall-clock
        has elapsed. Per-tick we drain wire messages, run
        ``analyze_threats`` against fresh world state, capture a
        page-client snapshot, and emit one observation row per
        tracked tank.

        Args:
            cdp: Active CDP session.
            tracked: Enemies under track.
            sample_interval_ms: Time between samples.
            sample_duration_ms: Total sampling window length.

        Returns:
            One row per tracked tank per sample, ordered by
            sample then tank id.
        """
        page = self._require_page()
        observations: list[TrackingObservationDict] = []
        started_ms = action_hooks.get_current_time_ms()
        sample_index = 0
        while action_hooks.get_current_time_ms() - started_ms < sample_duration_ms:
            world, threats, snapshot, sample_ms = self._capture_world_threats(cdp)
            observations.extend(
                _build_sample_observations(
                    sample_index=sample_index,
                    sample_timestamp_ms=sample_ms,
                    tracked=tracked,
                    world=world,
                    threats=threats,
                    snapshot=snapshot,
                    bot_combat_target_id=-1,
                    bot_mode_state="OBSERVE",
                ),
            )
            sample_index += 1
            page.wait_for_timeout(float(sample_interval_ms))
        return observations

    def execute_probe(
        self,
        *,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        shot_feedback_timeout_ms: int,
        sample_interval_ms: int,
        sample_duration_ms: int,
    ) -> EnemyTrackingProbeSessionDict:
        """Run the full tracking-probe session.

        Args:
            initial_sync_timeout_ms: Bootstrap world-sync timeout.
            acquisition_timeout_ms: Map-open / acquisition timeout.
            teleport_timeout_ms: Teleport outcome wait.
            shot_feedback_timeout_ms: Shot-response wait.
            sample_interval_ms: Sampling cadence.
            sample_duration_ms: Total sampling window.

        Returns:
            Completed session payload ready for persistence.
        """

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> EnemyTrackingProbeSessionDict:
            cdp = self._cdp
            if cdp is None:
                raise TeleportProbeError("cdp session is unavailable")
            message_start_index = len(self.messages)
            tracked, acq_snapshot, threats = self._acquire_enemies(
                cdp=cdp,
                acquisition_timeout_ms=acquisition_timeout_ms,
            )
            target = self._teleport_to_closest_enemy(
                cdp=cdp,
                threats=threats,
                teleport_timeout_ms=teleport_timeout_ms,
                message_start_index=message_start_index,
            )
            shot: ShotEventDict | None = None
            if target is not None:
                shot = self._fire_one_shot(
                    target=target,
                    shot_feedback_timeout_ms=shot_feedback_timeout_ms,
                )
            observations = self._sample_loop(
                cdp=cdp,
                tracked=tracked,
                sample_interval_ms=sample_interval_ms,
                sample_duration_ms=sample_duration_ms,
            )
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=None if shot is None else shot["sent_ms"],
            )
            return EnemyTrackingProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=envelope.startup_timing,
                acquisition_timeout_ms=acquisition_timeout_ms,
                teleport_timeout_ms=teleport_timeout_ms,
                shot_feedback_timeout_ms=shot_feedback_timeout_ms,
                sample_interval_ms=sample_interval_ms,
                sample_duration_ms=sample_duration_ms,
                tracked=tracked,
                shot=shot,
                snapshot_at_acquisition=acq_snapshot,
                observations=observations,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def _create_enemy_tracking_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> EnemyTrackingProbe:
    """Factory for :class:`EnemyTrackingProbe` with injected services.

    Args:
        target_url: Game URL to navigate to.
        headless: Whether to run the browser headlessly.
        prefer_account: Whether to prefer account-based login.

    Returns:
        Ready-to-bootstrap probe instance.
    """
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        EnemyTrackingProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, EnemyTrackingProbe)
    return probe


def run_enemy_tracking_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    initial_sync_timeout_ms: int = 10000,
    acquisition_timeout_ms: int = 5000,
    teleport_timeout_ms: int = 10000,
    shot_feedback_timeout_ms: int = 4000,
    sample_interval_ms: int = 1000,
    sample_duration_ms: int = 120000,
) -> EnemyTrackingProbeSessionDict:
    """Run a live enemy-tracking probe and persist its artifacts.

    Args:
        target_url: Game URL to navigate to.
        output_path: JSON output path for the structured session.
        headless: Whether the browser should run headless.
        prefer_account: Whether to prefer account-based login.
        initial_sync_timeout_ms: Bootstrap world-sync timeout.
        acquisition_timeout_ms: Map-open / acquisition timeout.
        teleport_timeout_ms: Teleport outcome wait.
        shot_feedback_timeout_ms: Shot-response wait.
        sample_interval_ms: Sampling cadence.
        sample_duration_ms: Total sampling window.

    Returns:
        Completed and persisted session payload.
    """

    def _run_session(probe: EnemyTrackingProbe) -> EnemyTrackingProbeSessionDict:
        return probe.execute_probe(
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            shot_feedback_timeout_ms=shot_feedback_timeout_ms,
            sample_interval_ms=sample_interval_ms,
            sample_duration_ms=sample_duration_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_enemy_tracking_probe,
        run_session=_run_session,
        encoder=encode_enemy_tracking_probe_session,
        summary_formatter=format_enemy_tracking_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "EnemyTrackingProbe",
    "format_enemy_tracking_probe_summary",
    "run_enemy_tracking_probe",
]
