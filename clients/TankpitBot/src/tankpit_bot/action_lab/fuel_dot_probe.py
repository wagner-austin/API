"""Live fuel-dot verification probe harness.

Teleports to map fuel dots (the atlas decoded from the MAP_DATA dot
layer) and radars at each landing to record the ground truth: is a
fuel container actually sitting on the dot tile, and at what volume?
"""

from __future__ import annotations

from typing import Literal

from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.fuel_dot_probe_types import (
    DotContainerObservationDict,
    FuelDotAttemptResultDict,
    FuelDotProbeSessionDict,
    encode_fuel_dot_probe_session,
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
from tankpit_bot.action_lab.radar_phase import run_tracked_radar_phase
from tankpit_bot.action_lab.teleport_acquisition import run_tracked_acquisition_phase
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
    _wait_for_teleport_outcome,
)
from tankpit_bot.action_lab.teleport_phase import run_tracked_teleport_command
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.state.types import WorldStateDict, coord_key, parse_coord_key
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

log = get_logger(__name__)

# A dot on or cardinally beside the tank is already visible ground; a
# teleport there is degenerate. Anything two or more tiles out is a
# legitimate verification target.
_MIN_DOT_DISTANCE = 2


def select_next_dot(
    world: WorldStateDict,
    self_x: int,
    self_y: int,
    visited: frozenset[str],
) -> tuple[int, int] | None:
    """Pick the nearest unvisited fuel dot worth teleporting to.

    Args:
        world: Current world state carrying the fuel-dot atlas.
        self_x: Tank X coordinate.
        self_y: Tank Y coordinate.
        visited: Coordinate keys of dots already visited this session.

    Returns:
        ``(x, y)`` of the chosen dot, or ``None`` when no unvisited dot
        at least ``_MIN_DOT_DISTANCE`` tiles away remains.
    """
    candidates = sorted(
        (abs(x - self_x) + abs(y - self_y), y, x)
        for x, y in (parse_coord_key(key) for key in world["map_fuel_dots"] if key not in visited)
    )
    for distance, y, x in candidates:
        if distance >= _MIN_DOT_DISTANCE:
            return (x, y)
    return None


def _observation_sort_key(observation: DotContainerObservationDict) -> tuple[int, int]:
    """Return the deterministic (row, column) sort key for an observation.

    Args:
        observation: Observation to key.

    Returns:
        Tuple of ``(y, x)``.
    """
    return (observation["y"], observation["x"])


def observe_dot_containers(
    world: WorldStateDict,
    dot_x: int,
    dot_y: int,
) -> tuple[DotContainerObservationDict | None, list[DotContainerObservationDict]]:
    """Read radar truth at a dot tile and across the visible viewport.

    Args:
        world: Current world state after the landing radar.
        dot_x: Dot tile X coordinate.
        dot_y: Dot tile Y coordinate.

    Returns:
        Tuple of ``(container_on_dot, viewport_fuel_containers)`` where
        the first element is the container sitting exactly on the dot
        tile (or ``None``) and the second lists every visible fuel
        container sorted by coordinate.
    """
    on_dot = world["containers"].get(coord_key(dot_x, dot_y))
    container_on_dot: DotContainerObservationDict | None = None
    if on_dot is not None:
        container_on_dot = DotContainerObservationDict(
            x=on_dot["x"],
            y=on_dot["y"],
            is_fuel=on_dot["is_fuel"],
            volume=on_dot["volume"],
        )
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    viewport_fuel = sorted(
        (
            DotContainerObservationDict(
                x=container["x"],
                y=container["y"],
                is_fuel=True,
                volume=container["volume"],
            )
            for container in world["containers"].values()
            if container["is_fuel"]
            and left <= container["x"] <= right
            and top <= container["y"] <= bottom
        ),
        key=_observation_sort_key,
    )
    return (container_on_dot, viewport_fuel)


def format_fuel_dot_probe_summary(session: FuelDotProbeSessionDict) -> str:
    """Format a compact human-readable summary line for the session.

    Args:
        session: Completed fuel-dot probe session.

    Returns:
        Single-line summary of dot verification outcomes.
    """
    fuel_on_dot = 0
    equipment_on_dot = 0
    empty_dot = 0
    timeouts = 0
    volumes: list[int] = []
    for attempt in session["attempts"]:
        if attempt["status"] == "fuel_on_dot":
            fuel_on_dot += 1
            container = attempt["container_on_dot"]
            if container is not None:
                volumes.append(container["volume"])
        elif attempt["status"] == "equipment_on_dot":
            equipment_on_dot += 1
        elif attempt["status"] == "empty_dot":
            empty_dot += 1
        else:
            timeouts += 1
    dots_in_atlas = session["attempts"][0]["dots_in_atlas"] if session["attempts"] else 0
    volume_text = ",".join(str(volume) for volume in volumes) if volumes else "-"
    return (
        "Fuel dot probe complete: "
        f"dots_in_atlas={dots_in_atlas} "
        f"attempts={len(session['attempts'])} "
        f"fuel_on_dot={fuel_on_dot} "
        f"equipment_on_dot={equipment_on_dot} "
        f"empty_dot={empty_dot} "
        f"timeouts={timeouts} "
        f"fuel_volumes={volume_text}"
    )


class FuelDotProbe(ProbeBase):
    """Live fuel-dot verification probe."""

    def _finish_dot_attempt_without_radar(
        self,
        *,
        page: action_session.WaitPageProtocol,
        cdp: CDPSessionProtocol,
        status: Literal["acquisition_timeout", "teleport_timeout"],
        acquisition_started_ms: int,
        acquisition_sync_timestamp_ms: int | None,
        dots_in_atlas: int,
        dot_x: int | None,
        dot_y: int | None,
        dot_distance: int | None,
        fuel_before: int,
        message_start_index: int,
        settle_delay_ms: int,
        snapshot_before: PageClientSnapshotDict,
        teleport_started_ms: int | None,
        landed_signal_received: bool,
        landed_x: int | None,
        landed_y: int | None,
    ) -> FuelDotAttemptResultDict:
        """Build and finalize a terminal attempt that never reached radar.

        Args:
            page: Page used for the optional settle wait.
            cdp: Active CDP session for the closing snapshot.
            status: Terminal attempt status.
            acquisition_started_ms: Map-open dispatch timestamp.
            acquisition_sync_timestamp_ms: Map-sync timestamp or None.
            dots_in_atlas: Atlas size after the map refresh.
            dot_x: Selected dot X coordinate, when one was selected.
            dot_y: Selected dot Y coordinate, when one was selected.
            dot_distance: Manhattan distance to the selected dot.
            fuel_before: Fuel level at attempt entry.
            message_start_index: Message buffer index at attempt entry.
            settle_delay_ms: Post-attempt settle delay.
            snapshot_before: Page-client snapshot at attempt entry.
            teleport_started_ms: Teleport dispatch timestamp or None.
            landed_signal_received: Whether the landed signal arrived.
            landed_x: Landing X coordinate or None.
            landed_y: Landing Y coordinate or None.

        Returns:
            Finalized terminal attempt result.
        """
        completion_timestamp_ms = action_hooks.get_current_time_ms()
        self._reset_probe_state_to_idle()
        self_state_after = self._require_self_state()
        snapshot_after = capture_page_client_snapshot(cdp)
        result = FuelDotAttemptResultDict(
            status=status,
            acquisition_started_ms=acquisition_started_ms,
            acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
            dots_in_atlas=dots_in_atlas,
            dot_x=dot_x,
            dot_y=dot_y,
            dot_distance=dot_distance,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=None,
            radar_sync_timestamp_ms=None,
            completion_timestamp_ms=completion_timestamp_ms,
            fuel_before=fuel_before,
            fuel_after=self_state_after["fuel"],
            landed_signal_received=landed_signal_received,
            landed_x=landed_x,
            landed_y=landed_y,
            container_on_dot=None,
            viewport_fuel_containers=[],
            message_start_index=message_start_index,
            message_end_index=len(self.messages),
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )
        if settle_delay_ms > 0:
            page.wait_for_timeout(float(settle_delay_ms))
        return result

    def _probe_single_dot_attempt(
        self,
        *,
        visited: frozenset[str],
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelDotAttemptResultDict | None:
        """Run one fuel-dot verification attempt against the live server.

        Opens the map (refreshing the atlas), teleports to the nearest
        unvisited dot, radars at the landing, and reads the container
        truth at the dot tile.

        Args:
            visited: Coordinate keys of dots already visited.
            acquisition_timeout_ms: Maximum map-sync wait.
            teleport_timeout_ms: Maximum teleport-landing wait.
            radar_timeout_ms: Maximum radar-sync wait.
            settle_delay_ms: Post-attempt settle delay.

        Returns:
            Attempt result, or ``None`` when no unvisited dot remains.

        Raises:
            TeleportProbeError: If a command fails to dispatch or the
                CDP session is unavailable.
        """
        page = self._require_page()
        cdp = self._cdp
        if cdp is None:
            raise TeleportProbeError("cdp session is unavailable")
        self_state_before = self._require_self_state()
        fuel_before = self_state_before["fuel"]
        world_timestamp_before = self.get_world_state()["timestamp_ms"]
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
            send_command=self.open_map,
            command_name="map_open",
            capture_before_map_open=True,
            wait_for_sync=True,
            sync_timeout_ms=acquisition_timeout_ms,
            dispatch_failure_error=TeleportProbeError,
            dispatch_failure_message="map_open command dispatch failed",
            unavailable_error=TeleportProbeError,
            unavailable_message="cdp session is unavailable",
        )
        world = self.get_world_state()
        dots_in_atlas = len(world["map_fuel_dots"])
        if acquisition_sync_timestamp_ms is None:
            return self._finish_dot_attempt_without_radar(
                page=page,
                cdp=cdp,
                status="acquisition_timeout",
                acquisition_started_ms=acquisition_started_ms,
                acquisition_sync_timestamp_ms=None,
                dots_in_atlas=dots_in_atlas,
                dot_x=None,
                dot_y=None,
                dot_distance=None,
                fuel_before=fuel_before,
                message_start_index=message_start_index,
                settle_delay_ms=settle_delay_ms,
                snapshot_before=snapshot_before,
                teleport_started_ms=None,
                landed_signal_received=False,
                landed_x=None,
                landed_y=None,
            )

        dot = select_next_dot(
            world,
            self_state_before["x"],
            self_state_before["y"],
            visited,
        )
        if dot is None:
            return None
        dot_x, dot_y = dot
        dot_distance = abs(dot_x - self_state_before["x"]) + abs(dot_y - self_state_before["y"])

        landing_target = TeleportTargetDict(
            label=f"fuel_dot_{dot_x}_{dot_y}",
            x=dot_x,
            y=dot_y,
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
        if teleport_result["status"] == "teleport_timeout":
            return self._finish_dot_attempt_without_radar(
                page=page,
                cdp=cdp,
                status="teleport_timeout",
                acquisition_started_ms=acquisition_started_ms,
                acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
                dots_in_atlas=dots_in_atlas,
                dot_x=dot_x,
                dot_y=dot_y,
                dot_distance=dot_distance,
                fuel_before=fuel_before,
                message_start_index=message_start_index,
                settle_delay_ms=settle_delay_ms,
                snapshot_before=snapshot_before,
                teleport_started_ms=teleport_started_ms,
                landed_signal_received=teleport_result["landed_signal_received"],
                landed_x=teleport_result["landed_x"],
                landed_y=teleport_result["landed_y"],
            )

        (_radar_cycle, radar_started_ms, radar_sync_timestamp_ms) = run_tracked_radar_phase(
            page,
            self,
            attempt_label=landing_target["label"],
            timeout_ms=radar_timeout_ms,
            dispatch_failure_error=TeleportProbeError,
        )
        world_after = self.get_world_state()
        container_on_dot, viewport_fuel = observe_dot_containers(world_after, dot_x, dot_y)
        status: Literal["fuel_on_dot", "equipment_on_dot", "empty_dot", "radar_timeout"]
        if radar_sync_timestamp_ms is None:
            status = "radar_timeout"
        elif container_on_dot is None:
            status = "empty_dot"
        elif container_on_dot["is_fuel"]:
            status = "fuel_on_dot"
        else:
            status = "equipment_on_dot"
        completion_timestamp_ms = action_hooks.get_current_time_ms()
        self_state_after = self._require_self_state()
        snapshot_after = capture_page_client_snapshot(cdp)
        result = FuelDotAttemptResultDict(
            status=status,
            acquisition_started_ms=acquisition_started_ms,
            acquisition_sync_timestamp_ms=acquisition_sync_timestamp_ms,
            dots_in_atlas=dots_in_atlas,
            dot_x=dot_x,
            dot_y=dot_y,
            dot_distance=dot_distance,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            completion_timestamp_ms=completion_timestamp_ms,
            fuel_before=fuel_before,
            fuel_after=self_state_after["fuel"],
            landed_signal_received=teleport_result["landed_signal_received"],
            landed_x=teleport_result["landed_x"],
            landed_y=teleport_result["landed_y"],
            container_on_dot=container_on_dot,
            viewport_fuel_containers=viewport_fuel,
            message_start_index=message_start_index,
            message_end_index=len(self.messages),
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )
        self._reset_probe_state_to_idle()
        if settle_delay_ms > 0:
            page.wait_for_timeout(float(settle_delay_ms))
        return result

    def execute_probe(
        self,
        *,
        max_dots: int,
        initial_sync_timeout_ms: int,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelDotProbeSessionDict:
        """Run the live fuel-dot verification probe session.

        Args:
            max_dots: Maximum number of dots to visit.
            initial_sync_timeout_ms: Maximum initial world-sync wait.
            acquisition_timeout_ms: Maximum map-sync wait per attempt.
            teleport_timeout_ms: Maximum teleport-landing wait.
            radar_timeout_ms: Maximum radar-sync wait.
            settle_delay_ms: Post-attempt settle delay.

        Returns:
            Completed fuel-dot probe session.

        Raises:
            ValueError: If ``max_dots`` is not positive.
        """
        if max_dots <= 0:
            raise ValueError("max_dots must be positive")

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> FuelDotProbeSessionDict:
            attempts: list[FuelDotAttemptResultDict] = []
            visited: set[str] = set()
            for _ in range(max_dots):
                attempt = self._probe_single_dot_attempt(
                    visited=frozenset(visited),
                    acquisition_timeout_ms=acquisition_timeout_ms,
                    teleport_timeout_ms=teleport_timeout_ms,
                    radar_timeout_ms=radar_timeout_ms,
                    settle_delay_ms=settle_delay_ms,
                )
                if attempt is None:
                    break
                attempts.append(attempt)
                if attempt["dot_x"] is not None and attempt["dot_y"] is not None:
                    visited.add(coord_key(attempt["dot_x"], attempt["dot_y"]))
            first_attempt_started_ms = attempts[0]["acquisition_started_ms"] if attempts else None
            session_envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_attempt_started_ms,
            )
            return FuelDotProbeSessionDict(
                session_id=session_envelope.session_id,
                start_timestamp_ms=session_envelope.start_timestamp_ms,
                end_timestamp_ms=session_envelope.end_timestamp_ms,
                base_url=session_envelope.base_url,
                spawn_x=session_envelope.spawn_x,
                spawn_y=session_envelope.spawn_y,
                max_dots=max_dots,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                startup_timing=session_envelope.startup_timing,
                acquisition_timeout_ms=acquisition_timeout_ms,
                teleport_timeout_ms=teleport_timeout_ms,
                radar_timeout_ms=radar_timeout_ms,
                settle_delay_ms=settle_delay_ms,
                attempts=attempts,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def _create_fuel_dot_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> FuelDotProbe:
    """Factory for FuelDotProbe with injected services."""
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        FuelDotProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, FuelDotProbe)
    return probe


def run_fuel_dot_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    max_dots: int = 6,
    initial_sync_timeout_ms: int = 10000,
    acquisition_timeout_ms: int = 3000,
    teleport_timeout_ms: int = 10000,
    radar_timeout_ms: int = 5000,
    settle_delay_ms: int = 500,
) -> FuelDotProbeSessionDict:
    """Run a live fuel-dot verification probe and save the session JSON.

    Args:
        target_url: Game URL to open.
        output_path: Session JSON output path.
        headless: Whether to run the browser headless.
        prefer_account: Whether to prefer account login.
        max_dots: Maximum number of dots to visit.
        initial_sync_timeout_ms: Maximum initial world-sync wait.
        acquisition_timeout_ms: Maximum map-sync wait per attempt.
        teleport_timeout_ms: Maximum teleport-landing wait.
        radar_timeout_ms: Maximum radar-sync wait.
        settle_delay_ms: Post-attempt settle delay.

    Returns:
        Completed fuel-dot probe session.
    """

    def _run_session(probe: FuelDotProbe) -> FuelDotProbeSessionDict:
        return probe.execute_probe(
            max_dots=max_dots,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            acquisition_timeout_ms=acquisition_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            radar_timeout_ms=radar_timeout_ms,
            settle_delay_ms=settle_delay_ms,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_fuel_dot_probe,
        run_session=_run_session,
        encoder=encode_fuel_dot_probe_session,
        summary_formatter=format_fuel_dot_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "FuelDotProbe",
    "format_fuel_dot_probe_summary",
    "observe_dot_containers",
    "run_fuel_dot_probe",
    "select_next_dot",
]
