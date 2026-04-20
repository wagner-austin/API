"""Shared session-runner helpers for the live fuel probe."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab.fuel_locations import build_distinct_ground_targets
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
)
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    ProbeSessionRunnerProtocol,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.types import TeleportTargetDict
from tankpit_bot.state import SelfStateDict


class FuelProbeSessionRunnerProtocol(ProbeSessionRunnerProtocol, Protocol):
    """Protocol required to execute one live fuel-probe session."""

    _target_url: str

    @property
    def session_id(self) -> str:
        """Return the session identifier."""

    def _require_self_state(self) -> SelfStateDict:
        """Return the required current self state."""

    def _probe_single_fuel_target(
        self,
        *,
        target: TeleportTargetDict,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
        teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
    ) -> FuelProbeAttemptResultDict:
        """Run one teleport-radar-fuel attempt."""


def execute_fuel_probe_session(
    probe: FuelProbeSessionRunnerProtocol,
    *,
    target_pickups: int,
    max_attempts: int,
    initial_sync_timeout_ms: int,
    map_sync_timeout_ms: int,
    teleport_timeout_ms: int,
    radar_timeout_ms: int,
    pickup_timeout_ms: int,
    settle_delay_ms: int,
    target_step: int,
    target_max_radius: int,
    teleport_strategy: Literal["sync_before_teleport", "immediate_after_map_open"],
    terrain_provider: Callable[[], TerrainMapProtocol | None],
    terrain_unavailable_error: type[Exception],
    terrain_unavailable_message: str,
) -> FuelProbeSessionDict:
    """Execute one live fuel-probe session.

    Args:
        probe: Probe instance that owns the session.
        target_pickups: Number of successful pickups required before exit.
        max_attempts: Maximum number of attempts to run.
        initial_sync_timeout_ms: Initial sync timeout in milliseconds.
        map_sync_timeout_ms: Map sync timeout in milliseconds.
        teleport_timeout_ms: Teleport timeout in milliseconds.
        radar_timeout_ms: Radar timeout in milliseconds.
        pickup_timeout_ms: Pickup timeout in milliseconds.
        settle_delay_ms: Optional per-attempt settle delay in milliseconds.
        target_step: Distance step used for target generation.
        target_max_radius: Maximum target radius around the current tank.
        teleport_strategy: Teleport acquisition strategy.
        terrain_provider: Terrain map lookup for target generation.
        terrain_unavailable_error: Error type raised if terrain is unavailable.
        terrain_unavailable_message: Error text for missing terrain.

    Returns:
        Completed session payload.

    Raises:
        ValueError: If the target/attempt limits are invalid.
        Exception: Raised via ``terrain_unavailable_error`` when terrain is missing.
    """
    if target_pickups <= 0:
        raise ValueError("target_pickups must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if max_attempts < target_pickups:
        raise ValueError("max_attempts must be at least target_pickups")

    def _run_ready_session(context: ProbeCommandReadyContextDict) -> FuelProbeSessionDict:
        terrain = terrain_provider()
        if terrain is None:
            raise terrain_unavailable_error(terrain_unavailable_message)
        used_targets: set[tuple[int, int]] = set()
        attempts: list[FuelProbeAttemptResultDict] = []
        successful_pickups = 0
        while len(attempts) < max_attempts and successful_pickups < target_pickups:
            self_state = probe._require_self_state()
            targets = build_distinct_ground_targets(
                self_state["x"],
                self_state["y"],
                terrain,
                count=1,
                step=target_step,
                max_radius=target_max_radius,
                excluded=frozenset(used_targets),
            )
            target = targets[0]
            used_targets.add((target["x"], target["y"]))
            attempt_result = probe._probe_single_fuel_target(
                target=target,
                map_sync_timeout_ms=map_sync_timeout_ms,
                teleport_timeout_ms=teleport_timeout_ms,
                radar_timeout_ms=radar_timeout_ms,
                pickup_timeout_ms=pickup_timeout_ms,
                settle_delay_ms=settle_delay_ms,
                teleport_strategy=teleport_strategy,
            )
            attempts.append(attempt_result)
            if attempt_result["status"] == "picked_up_fuel":
                successful_pickups += 1
        first_attempt_started_ms = attempts[0]["map_open_started_ms"] if attempts else None
        session_envelope = build_probe_session_envelope(
            probe,
            context=context,
            first_attempt_started_ms=first_attempt_started_ms,
        )
        return FuelProbeSessionDict(
            session_id=session_envelope.session_id,
            start_timestamp_ms=session_envelope.start_timestamp_ms,
            end_timestamp_ms=session_envelope.end_timestamp_ms,
            base_url=session_envelope.base_url,
            spawn_x=session_envelope.spawn_x,
            spawn_y=session_envelope.spawn_y,
            target_pickups=target_pickups,
            max_attempts=max_attempts,
            capture_session_path="",
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            startup_timing=session_envelope.startup_timing,
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            radar_timeout_ms=radar_timeout_ms,
            pickup_timeout_ms=pickup_timeout_ms,
            settle_delay_ms=settle_delay_ms,
            attempts=attempts,
        )

    return execute_live_probe_bootstrap(
        probe,
        initial_sync_timeout_ms=initial_sync_timeout_ms,
        run_ready_session=_run_ready_session,
    )


__all__ = ["FuelProbeSessionRunnerProtocol", "execute_fuel_probe_session"]
