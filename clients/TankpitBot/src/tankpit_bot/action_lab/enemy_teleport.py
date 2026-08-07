"""Live enemy-directed teleport probe harness."""

from __future__ import annotations

from typing import Literal

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.enemy_teleport_types import (
    EnemyTeleportAttemptResultDict,
    EnemyTeleportProbeSessionDict,
    encode_enemy_teleport_probe_session,
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
from tankpit_bot.action_lab.teleport_phase import (
    run_tracked_teleport_command,
)
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.bot.ai.combat_landing import (
    choose_combat_landing_tile,
    has_cardinal_enemy_adjacency,
)
from tankpit_bot.bot.ai.threat_primitives import find_closest_threat
from tankpit_bot.bot.ai.threats import analyze_threats
from tankpit_bot.bot.ai.world_types import EnemyThreatDict
from tankpit_bot.sniffer.world_state import get_terrain_map

log = get_logger(__name__)


def _require_fresh_enemy_threat(
    probe: ProbeBase,
    started_ms: int,
    excluded_tank_ids: frozenset[int],
) -> EnemyThreatDict | None:
    """Return the closest enemy threat confirmed after a probe action."""
    self_state = probe.get_self_state()
    if self_state is None:
        return None
    threats = analyze_threats(
        probe.get_world_state(), self_state, action_hooks.get_current_time_ms()
    )
    fresh = [
        threat
        for threat in threats
        if threat["timestamp_ms"] > started_ms and threat["tank_id"] not in excluded_tank_ids
    ]
    return find_closest_threat(fresh)


def _enemy_by_id(probe: ProbeBase, tank_id: int) -> EnemyThreatDict | None:
    """Return the current threat snapshot for a specific tank id."""
    self_state = probe.get_self_state()
    if self_state is None:
        return None
    for threat in analyze_threats(
        probe.get_world_state(), self_state, action_hooks.get_current_time_ms()
    ):
        if threat["tank_id"] == tank_id:
            return threat
    return None


def _format_enemy_label(enemy: EnemyThreatDict) -> str:
    """Return a deterministic teleport target label for an enemy landing."""
    return f"enemy_{enemy['tank_id']}_{enemy['x']}_{enemy['y']}"


def _make_terminal_result(
    *,
    acquisition_strategy: Literal["map_open", "nearest_enemy"],
    status: Literal["no_enemy", "no_landing_tile", "acquisition_timeout"],
    acquisition_started_ms: int,
    acquisition_sync_timestamp_ms: int | None,
    fuel_before: int,
    world_timestamp_before: int,
    completion_timestamp_ms: int,
    fuel_after: int,
    world_timestamp_after: int,
    enemy: EnemyThreatDict | None,
    landing_target: TeleportTargetDict | None,
    landed_x: int,
    landed_y: int,
    message_start_index: int,
    message_end_index: int,
    snapshot_before: PageClientSnapshotDict,
    snapshot_after: PageClientSnapshotDict,
) -> EnemyTeleportAttemptResultDict:
    """Build a non-teleport terminal enemy-teleport result."""
    return EnemyTeleportAttemptResultDict(
        acquisition_strategy=acquisition_strategy,
        status=status,
        acquisition_started_ms=acquisition_started_ms,
        acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
        teleport_started_ms=None,
        completion_timestamp_ms=completion_timestamp_ms,
        acquisition_elapsed_ms=(
            None
            if acquisition_sync_timestamp_ms is None
            else acquisition_sync_timestamp_ms - acquisition_started_ms
        ),
        teleport_elapsed_ms=None,
        fuel_before=fuel_before,
        fuel_after=fuel_after,
        world_timestamp_before=world_timestamp_before,
        world_timestamp_after=world_timestamp_after,
        enemy=enemy,
        landing_target=landing_target,
        landed_signal_received=False,
        landed_x=landed_x,
        landed_y=landed_y,
        enemy_still_visible=False,
        enemy_distance_after=None,
        enemy_x_after=None,
        enemy_y_after=None,
        message_start_index=message_start_index,
        message_end_index=message_end_index,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
    )


def format_enemy_teleport_probe_summary(session: EnemyTeleportProbeSessionDict) -> str:
    """Format a compact human-readable summary line for the session."""
    landed_adjacent = 0
    landed_not_adjacent = 0
    no_enemy = 0
    no_landing_tile = 0
    acquisition_timeout = 0
    teleport_timeout = 0
    for attempt in session["attempts"]:
        if attempt["status"] == "landed_adjacent":
            landed_adjacent += 1
        elif attempt["status"] == "landed_not_adjacent":
            landed_not_adjacent += 1
        elif attempt["status"] == "no_enemy":
            no_enemy += 1
        elif attempt["status"] == "no_landing_tile":
            no_landing_tile += 1
        elif attempt["status"] == "acquisition_timeout":
            acquisition_timeout += 1
        else:
            teleport_timeout += 1
    startup_timing = session["startup_timing"]
    bootstrap_ms = (
        startup_timing["command_ready_timestamp_ms"] - startup_timing["initial_sync_started_ms"]
    )
    return (
        "Enemy teleport probe complete: "
        f"strategy={session['acquisition_strategy']} "
        f"attempts={len(session['attempts'])} "
        f"landed_adjacent={landed_adjacent} "
        f"landed_not_adjacent={landed_not_adjacent} "
        f"no_enemy={no_enemy} "
        f"no_landing_tile={no_landing_tile} "
        f"acquisition_timeout={acquisition_timeout} "
        f"teleport_timeout={teleport_timeout} "
        "session_to_initial_sync_ms="
        f"{startup_timing['initial_sync_started_ms'] - session['start_timestamp_ms']} "
        f"initial_sync_to_command_ready_ms={bootstrap_ms}"
    )


class EnemyTeleportProbe(ProbeBase):
    """Live enemy-directed teleport probe for combat acquisition timing."""

    def _heartbeat_action(self, beat: int) -> None:
        """Fire one dwell heartbeat: a 1-tile walk shuffle.

        Query heartbeats were falsified 2026-07-24 (366 inventory
        requests held only their own responses; the broadcast stream
        mutes for any non-PLAYING client), so the heartbeat is a real
        gameplay action: alternate one tile east / back west (1 fuel
        per beat). Falls back to the inventory query when self state
        is unknown (cannot aim a walk).

        Args:
            beat: Zero-based heartbeat counter (parity picks the
                shuffle direction).
        """
        self_state = self.get_self_state()
        if self_state is None:
            self.request_inventory()
            return
        step = 1 if beat % 2 == 0 else -1
        self.move_to(self_state["x"] + step, self_state["y"])

    def _settle_dwell(
        self,
        page: action_session.WaitPageProtocol,
        settle_delay_ms: int,
        heartbeat_interval_ms: int,
    ) -> None:
        """Wait out the settle window, optionally holding the stream open.

        With a positive heartbeat interval, one heartbeat action fires
        at the start of the dwell and every interval after — the
        broadcast stream mutes for non-playing clients (wiki log
        2026-07-24), so a silent dwell observes nothing. Each beat
        drains the CDP buffer first so the shuffle tracks the tank's
        true position: the 2026-07-24 decisive run walked against a
        frozen landing position and half its moves were supervisor-
        rejected (154 CANT_GO into the watched bot's tile). Zero
        interval preserves the historical silent-settle behavior.

        Args:
            page: Playwright page driving the wait.
            settle_delay_ms: Total dwell duration.
            heartbeat_interval_ms: Heartbeat period (0 = no heartbeat).
        """
        if settle_delay_ms <= 0:
            return
        if heartbeat_interval_ms <= 0:
            page.wait_for_timeout(float(settle_delay_ms))
            return
        remaining = settle_delay_ms
        beat = 0
        while remaining > 0:
            action_hooks.drain_buffered_messages(self)
            self._heartbeat_action(beat)
            beat += 1
            step = min(heartbeat_interval_ms, remaining)
            page.wait_for_timeout(float(step))
            remaining -= step

    def _send_enemy_acquisition(
        self,
        acquisition_strategy: Literal["map_open", "nearest_enemy"],
    ) -> bool:
        """Send the acquisition command for one enemy-teleport attempt."""
        if acquisition_strategy == "map_open":
            return self.open_map()
        return self.request_nearest_enemy()

    def _finish_non_teleport_attempt(
        self,
        *,
        page: action_session.WaitPageProtocol,
        cdp: CDPSessionProtocol,
        acquisition_strategy: Literal["map_open", "nearest_enemy"],
        status: Literal["no_enemy", "no_landing_tile", "acquisition_timeout"],
        acquisition_started_ms: int,
        acquisition_sync_timestamp_ms: int | None,
        fuel_before: int,
        world_timestamp_before: int,
        enemy: EnemyThreatDict | None,
        landing_target: TeleportTargetDict | None,
        message_start_index: int,
        settle_delay_ms: int,
        heartbeat_interval_ms: int,
        snapshot_before: PageClientSnapshotDict,
    ) -> EnemyTeleportAttemptResultDict:
        """Build and finalize a non-teleport terminal attempt.

        Captures the page-client ``snapshot_after`` from the live JS
        client immediately after the probe transitions back to IDLE and
        before the optional settle delay. The result therefore carries
        a side-by-side view of the live client state at attempt entry
        (``snapshot_before``) and at the terminal boundary.
        """
        completion_timestamp_ms = action_hooks.get_current_time_ms()
        self._reset_probe_state_to_idle()
        self_state_after = self._require_self_state()
        snapshot_after = capture_page_client_snapshot(cdp)
        result = _make_terminal_result(
            acquisition_strategy=acquisition_strategy,
            status=status,
            acquisition_started_ms=acquisition_started_ms,
            acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
            fuel_before=fuel_before,
            world_timestamp_before=world_timestamp_before,
            completion_timestamp_ms=completion_timestamp_ms,
            fuel_after=self_state_after["fuel"],
            world_timestamp_after=self.get_world_state()["timestamp_ms"],
            enemy=enemy,
            landing_target=landing_target,
            landed_x=self_state_after["x"],
            landed_y=self_state_after["y"],
            message_start_index=message_start_index,
            message_end_index=len(self.messages),
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )
        self._settle_dwell(page, settle_delay_ms, heartbeat_interval_ms)
        return result

    def _probe_single_enemy_attempt(
        self,
        *,
        acquisition_strategy: Literal["map_open", "nearest_enemy"],
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
        heartbeat_interval_ms: int,
        excluded_tank_ids: frozenset[int],
    ) -> EnemyTeleportAttemptResultDict:
        """Run one enemy-directed teleport attempt against the live server.

        Captures a page-client snapshot immediately before the acquisition
        command dispatches and again immediately before each terminal
        return point; these snapshots provide a side-by-side view of the
        live JS client's belief about tank state at each boundary.
        """
        page = self._require_page()
        cdp = self._cdp
        if cdp is None:
            raise TeleportProbeError("cdp session is unavailable")
        world_before = self.get_world_state()
        self_state_before = self._require_self_state()
        fuel_before = self_state_before["fuel"]
        world_timestamp_before = world_before["timestamp_ms"]
        snapshot_before = capture_page_client_snapshot(cdp)

        self._reset_probe_state_to_idle()
        message_start_index = len(self.messages)
        (
            acquisition_started_ms,
            acquisition_sync_timestamp_ms,
            page_snapshots,
            capture_page_snapshot,
        ) = run_tracked_acquisition_phase(
            page,
            self,
            cdp=cdp,
            send_command=lambda: self._send_enemy_acquisition(acquisition_strategy),
            command_name="enemy_acquisition",
            capture_before_map_open=acquisition_strategy == "map_open",
            wait_for_sync=True,
            sync_timeout_ms=acquisition_timeout_ms,
            dispatch_failure_error=TeleportProbeError,
            dispatch_failure_message="enemy acquisition command dispatch failed",
            unavailable_error=TeleportProbeError,
            unavailable_message="cdp session is unavailable",
        )
        if acquisition_sync_timestamp_ms is None:
            return self._finish_non_teleport_attempt(
                page=page,
                cdp=cdp,
                acquisition_strategy=acquisition_strategy,
                status="acquisition_timeout",
                acquisition_started_ms=acquisition_started_ms,
                acquisition_sync_timestamp_ms=None,
                fuel_before=fuel_before,
                world_timestamp_before=world_timestamp_before,
                enemy=None,
                landing_target=None,
                message_start_index=message_start_index,
                settle_delay_ms=settle_delay_ms,
                heartbeat_interval_ms=heartbeat_interval_ms,
                snapshot_before=snapshot_before,
            )

        enemy = _require_fresh_enemy_threat(self, acquisition_started_ms, excluded_tank_ids)
        if enemy is None:
            return self._finish_non_teleport_attempt(
                page=page,
                cdp=cdp,
                acquisition_strategy=acquisition_strategy,
                status="no_enemy",
                acquisition_started_ms=acquisition_started_ms,
                acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
                fuel_before=fuel_before,
                world_timestamp_before=world_timestamp_before,
                enemy=None,
                landing_target=None,
                message_start_index=message_start_index,
                settle_delay_ms=settle_delay_ms,
                heartbeat_interval_ms=heartbeat_interval_ms,
                snapshot_before=snapshot_before,
            )

        landing_x, landing_y = choose_combat_landing_tile(
            self.get_world_state(),
            self._require_self_state(),
            enemy,
            get_terrain_map(),
            action_hooks.get_current_time_ms(),
        )
        if landing_x == -1 and landing_y == -1:
            return self._finish_non_teleport_attempt(
                page=page,
                cdp=cdp,
                acquisition_strategy=acquisition_strategy,
                status="no_landing_tile",
                acquisition_started_ms=acquisition_started_ms,
                acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
                fuel_before=fuel_before,
                world_timestamp_before=world_timestamp_before,
                enemy=enemy,
                landing_target=None,
                message_start_index=message_start_index,
                settle_delay_ms=settle_delay_ms,
                heartbeat_interval_ms=heartbeat_interval_ms,
                snapshot_before=snapshot_before,
            )

        landing_target = TeleportTargetDict(
            label=_format_enemy_label(enemy),
            x=landing_x,
            y=landing_y,
        )
        teleport_cycle = self._start_action_phase(
            "teleport",
            attempt_label=landing_target["label"],
        )
        teleport_result, teleport_started_ms = run_tracked_teleport_command(
            page,
            self,
            landing_target,
            teleport_cycle=teleport_cycle,
            message_start_index=message_start_index,
            map_open_started_ms=acquisition_started_ms,
            map_sync_timestamp_ms=acquisition_sync_timestamp_ms,
            fuel_before=fuel_before,
            world_timestamp_before=world_timestamp_before,
            timeout_ms=teleport_timeout_ms,
            page_snapshots=page_snapshots,
            capture_page_snapshot=capture_page_snapshot,
            wait_for_outcome=_wait_for_teleport_outcome,
            dispatch_failure_error=TeleportProbeError,
        )
        current_enemy = _enemy_by_id(self, enemy["tank_id"])
        self_state_after = self._require_self_state()
        if teleport_result["status"] == "teleport_timeout":
            status: Literal[
                "landed_adjacent",
                "landed_not_adjacent",
                "no_enemy",
                "no_landing_tile",
                "acquisition_timeout",
                "teleport_timeout",
            ] = "teleport_timeout"
        elif current_enemy is not None and has_cardinal_enemy_adjacency(
            self_state_after,
            current_enemy,
        ):
            status = "landed_adjacent"
        else:
            status = "landed_not_adjacent"
        snapshot_after = capture_page_client_snapshot(cdp)
        result = EnemyTeleportAttemptResultDict(
            acquisition_strategy=acquisition_strategy,
            status=status,
            acquisition_started_ms=acquisition_started_ms,
            acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            completion_timestamp_ms=teleport_result["completion_timestamp_ms"],
            acquisition_elapsed_ms=acquisition_sync_timestamp_ms - acquisition_started_ms,
            teleport_elapsed_ms=teleport_result["teleport_elapsed_ms"],
            fuel_before=fuel_before,
            fuel_after=teleport_result["fuel_after"],
            world_timestamp_before=world_timestamp_before,
            world_timestamp_after=teleport_result["world_timestamp_after"],
            enemy=enemy,
            landing_target=landing_target,
            landed_signal_received=teleport_result["landed_signal_received"],
            landed_x=teleport_result["landed_x"],
            landed_y=teleport_result["landed_y"],
            enemy_still_visible=current_enemy is not None,
            enemy_distance_after=(
                None
                if current_enemy is None
                else abs(self_state_after["x"] - current_enemy["x"])
                + abs(self_state_after["y"] - current_enemy["y"])
            ),
            enemy_x_after=None if current_enemy is None else current_enemy["x"],
            enemy_y_after=None if current_enemy is None else current_enemy["y"],
            message_start_index=message_start_index,
            message_end_index=len(self.messages),
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )
        self._reset_probe_state_to_idle()
        self._post_landing_phase(page, enemy, settle_delay_ms, heartbeat_interval_ms)
        return result

    def _post_landing_phase(
        self,
        page: action_session.WaitPageProtocol,
        enemy: EnemyThreatDict,
        settle_delay_ms: int,
        heartbeat_interval_ms: int,
    ) -> None:
        """Hold the post-landing observation phase.

        The base probe dwells (optionally walking a heartbeat, see
        ``_settle_dwell``). Subclasses override this to act on the
        landed-adjacent enemy — the respawn-watch probe replaces the
        dwell with an engage-then-map-poll choreography.

        Args:
            page: Playwright page driving the wait.
            enemy: The enemy this attempt teleported to.
            settle_delay_ms: Total dwell duration.
            heartbeat_interval_ms: Heartbeat period (0 = no heartbeat).
        """
        del enemy
        self._settle_dwell(page, settle_delay_ms, heartbeat_interval_ms)

    def execute_probe(
        self,
        *,
        acquisition_strategy: Literal["map_open", "nearest_enemy"],
        max_attempts: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        settle_delay_ms: int,
        heartbeat_interval_ms: int,
    ) -> EnemyTeleportProbeSessionDict:
        """Run the live enemy-directed teleport probe session."""
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> EnemyTeleportProbeSessionDict:
            attempts: list[EnemyTeleportAttemptResultDict] = []
            targeted_enemy_ids: set[int] = set()
            for _ in range(max_attempts):
                attempt = self._probe_single_enemy_attempt(
                    acquisition_strategy=acquisition_strategy,
                    acquisition_timeout_ms=acquisition_timeout_ms,
                    teleport_timeout_ms=teleport_timeout_ms,
                    settle_delay_ms=settle_delay_ms,
                    heartbeat_interval_ms=heartbeat_interval_ms,
                    excluded_tank_ids=frozenset(targeted_enemy_ids),
                )
                attempts.append(attempt)
                enemy = attempt["enemy"]
                if enemy is not None:
                    targeted_enemy_ids.add(enemy["tank_id"])
            first_attempt_started_ms = attempts[0]["acquisition_started_ms"] if attempts else None
            session_envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_attempt_started_ms,
            )
            return EnemyTeleportProbeSessionDict(
                session_id=session_envelope.session_id,
                start_timestamp_ms=session_envelope.start_timestamp_ms,
                end_timestamp_ms=session_envelope.end_timestamp_ms,
                base_url=session_envelope.base_url,
                spawn_x=session_envelope.spawn_x,
                spawn_y=session_envelope.spawn_y,
                acquisition_strategy=acquisition_strategy,
                max_attempts=max_attempts,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=session_envelope.startup_timing,
                acquisition_timeout_ms=acquisition_timeout_ms,
                teleport_timeout_ms=teleport_timeout_ms,
                settle_delay_ms=settle_delay_ms,
                heartbeat_interval_ms=heartbeat_interval_ms,
                attempts=attempts,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def _create_enemy_teleport_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> EnemyTeleportProbe:
    """Factory for EnemyTeleportProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        EnemyTeleportProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, EnemyTeleportProbe)
    return probe


def run_enemy_teleport_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    acquisition_strategy: Literal["map_open", "nearest_enemy"] = "map_open",
    max_attempts: int = 3,
    initial_sync_timeout_ms: int = 10000,
    acquisition_timeout_ms: int = 3000,
    teleport_timeout_ms: int = 10000,
    settle_delay_ms: int = 500,
    heartbeat_interval_ms: int = 0,
) -> EnemyTeleportProbeSessionDict:
    """Run a live enemy teleport probe and save the session JSON."""

    def _run_session(probe: EnemyTeleportProbe) -> EnemyTeleportProbeSessionDict:
        return probe.execute_probe(
            acquisition_strategy=acquisition_strategy,
            max_attempts=max_attempts,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            heartbeat_interval_ms=heartbeat_interval_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_enemy_teleport_probe,
        run_session=_run_session,
        encoder=encode_enemy_teleport_probe_session,
        summary_formatter=format_enemy_teleport_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "EnemyTeleportProbe",
    "format_enemy_teleport_probe_summary",
    "run_enemy_teleport_probe",
]
