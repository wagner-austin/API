"""The enemy-tracking probe: compare wire beliefs against JS truth.

Holds :class:`EnemyTrackingProbe` and its entry points. The record
builders and run summary are
:mod:`tankpit_bot.action_lab.enemy_tracking_records`.
"""

from __future__ import annotations

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.enemy_tracking_codecs import encode_enemy_tracking_probe_session
from tankpit_bot.action_lab.enemy_tracking_records import (
    _build_sample_observations,
    _build_tracked_records,
    _wait_for_shot_feedback,
    format_enemy_tracking_probe_summary,
)
from tankpit_bot.action_lab.enemy_tracking_types import (
    EnemyTrackingProbeSessionDict,
    ShotEventDict,
    TrackedEnemyDict,
    TrackingObservationDict,
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
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.bot.ai.combat_landing import choose_combat_landing_tile
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.browser.page_client_snapshot import (
    PageClientSnapshotDict,
    capture_page_client_snapshot,
)
from tankpit_bot.sniffer.world_state import (
    get_terrain_map,
)
from tankpit_bot.state.types import WorldStateDict


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
        action_hooks.drain_buffered_messages(self, self.world)
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
    "run_enemy_tracking_probe",
]
