"""Live teleport-radar-fuel action probe harness."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.capture import save_capture_session
from tankpit_bot.action_lab.fuel_locations import build_distinct_ground_targets
from tankpit_bot.action_lab.fuel_probe_types import (
    FuelProbeAttemptResultDict,
    FuelProbeSessionDict,
    encode_fuel_probe_session,
)
from tankpit_bot.action_lab.fuel_targeting import (
    FuelTargetingError,
    find_visible_fuel_landing_tile,
    visible_fuel_requires_reposition,
)
from tankpit_bot.action_lab.teleport import (
    TeleportProbe,
    TeleportProbeError,
    _wait_for_teleport_outcome,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportStartupTimingDict,
    TeleportTargetDict,
)
from tankpit_bot.bot.ai.equipment import find_best_fuel
from tankpit_bot.browser import PlaywrightNotInstalledError, reset_cdp_time_offset
from tankpit_bot.sniffer import reset_all_trackers, reset_world_state
from tankpit_bot.sniffer.viewport import reset_viewport_tracking
from tankpit_bot.sniffer.world_state import get_terrain_map
from tankpit_bot.state.types import ContainerStateDict

log = get_logger(__name__)
_FUEL_PROBE_TARGET_STEP = 16
_FUEL_PROBE_TARGET_MAX_RADIUS = 48


class FuelProbeError(Exception):
    """Raised when the fuel probe cannot proceed."""


def _find_visible_fuel_target(
    probe: TeleportProbe,
    allow_unreachable: bool = False,
) -> ContainerStateDict | None:
    """Return the best currently visible fuel container."""
    terrain = get_terrain_map()
    if terrain is None:
        raise FuelProbeError("terrain map is unavailable")
    self_state = probe.get_self_state()
    if self_state is None:
        raise FuelProbeError("self state is unavailable")
    world = probe.get_world_state()
    return find_best_fuel(
        world,
        self_state,
        terrain,
        allow_unreachable=allow_unreachable,
        now_ms=world["timestamp_ms"],
        minimum_volume=1,
    )


def _visible_fuel_requires_reposition(
    probe: TeleportProbe,
    fuel_target: ContainerStateDict,
) -> bool:
    """Return whether a visible fuel target needs a reposition teleport."""
    try:
        return visible_fuel_requires_reposition(probe, fuel_target)
    except FuelTargetingError as exc:
        raise FuelProbeError(str(exc)) from exc


def _find_visible_fuel_landing_tile(
    probe: TeleportProbe,
    fuel_target: ContainerStateDict,
) -> tuple[int, int] | None:
    """Return the landing tile for a blocked visible fuel target."""
    try:
        return find_visible_fuel_landing_tile(probe, fuel_target)
    except FuelTargetingError as exc:
        raise FuelProbeError(str(exc)) from exc


def _make_reposition_target(target_x: int, target_y: int) -> TeleportTargetDict:
    """Return a typed target label for a fuel reposition teleport."""
    return TeleportTargetDict(
        label=f"fuel_reposition_{target_x}_{target_y}",
        x=target_x,
        y=target_y,
    )


def _wait_for_pickup_outcome(
    page: action_session.WaitPageProtocol,
    probe: TeleportProbe,
    *,
    target_x: int,
    target_y: int,
    pickup_started_ms: int,
    fuel_before: int,
    timeout_ms: int,
) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
    """Wait for a fuel pickup to complete or time out."""
    while action_hooks.get_current_time_ms() - pickup_started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(probe)
        pickup_outcome = _get_completed_pickup_outcome(
            probe,
            target_x=target_x,
            target_y=target_y,
            fuel_before=fuel_before,
        )
        if pickup_outcome is not None:
            return pickup_outcome
        page.wait_for_timeout(100.0)
    self_state = probe.get_world_state()["self_state"]
    if self_state is None:
        raise FuelProbeError("self state disappeared after fuel pickup timeout")
    return ("pickup_timeout", action_hooks.get_current_time_ms(), self_state["fuel"])


def _get_completed_pickup_outcome(
    probe: TeleportProbe,
    *,
    target_x: int,
    target_y: int,
    fuel_before: int,
) -> tuple[Literal["picked_up_fuel"], int, int] | None:
    """Return a completed pickup outcome when fuel is already collected."""
    world = probe.get_world_state()
    self_state = world["self_state"]
    if self_state is None:
        raise FuelProbeError("self state disappeared while waiting for fuel pickup")
    if self_state["fuel"] > fuel_before or f"{target_x},{target_y}" not in world["containers"]:
        return ("picked_up_fuel", action_hooks.get_current_time_ms(), self_state["fuel"])
    return None


def format_fuel_probe_summary(session: FuelProbeSessionDict) -> str:
    """Format a compact summary for a fuel probe session."""
    picked_up_fuel = 0
    no_fuel_visible = 0
    radar_timeout = 0
    map_sync_timeout = 0
    reposition_map_sync_timeout = 0
    teleport_timeout = 0
    reposition_teleport_timeout = 0
    pickup_timeout = 0
    for attempt in session["attempts"]:
        if attempt["status"] == "picked_up_fuel":
            picked_up_fuel += 1
        elif attempt["status"] == "no_fuel_visible":
            no_fuel_visible += 1
        elif attempt["status"] == "radar_timeout":
            radar_timeout += 1
        elif attempt["status"] == "map_sync_timeout":
            map_sync_timeout += 1
        elif attempt["status"] == "reposition_map_sync_timeout":
            reposition_map_sync_timeout += 1
        elif attempt["status"] == "teleport_timeout":
            teleport_timeout += 1
        elif attempt["status"] == "reposition_teleport_timeout":
            reposition_teleport_timeout += 1
        else:
            pickup_timeout += 1
    startup_timing = session["startup_timing"]
    bootstrap_ms = (
        startup_timing["command_ready_timestamp_ms"] - startup_timing["initial_sync_started_ms"]
    )
    return (
        "Fuel probe complete: "
        f"attempts={len(session['attempts'])} "
        f"target_pickups={session['target_pickups']} "
        f"picked_up_fuel={picked_up_fuel} "
        f"no_fuel_visible={no_fuel_visible} "
        f"radar_timeout={radar_timeout} "
        f"map_sync_timeout={map_sync_timeout} "
        f"reposition_map_sync_timeout={reposition_map_sync_timeout} "
        f"teleport_timeout={teleport_timeout} "
        f"reposition_teleport_timeout={reposition_teleport_timeout} "
        f"pickup_timeout={pickup_timeout} "
        "session_to_initial_sync_ms="
        f"{startup_timing['initial_sync_started_ms'] - session['start_timestamp_ms']} "
        f"initial_sync_to_command_ready_ms={bootstrap_ms}"
    )


class FuelProbe(TeleportProbe):
    """Live teleport-radar-fuel probe."""

    def _build_attempt_result(
        self,
        *,
        target: TeleportTargetDict,
        status: Literal[
            "picked_up_fuel",
            "no_fuel_visible",
            "radar_timeout",
            "map_sync_timeout",
            "reposition_map_sync_timeout",
            "teleport_timeout",
            "reposition_teleport_timeout",
            "pickup_timeout",
        ],
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int | None,
        radar_started_ms: int | None,
        radar_sync_timestamp_ms: int | None,
        pickup_started_ms: int | None,
        completion_timestamp_ms: int,
        fuel_before: int,
        fuel_after: int | None,
        landed_signal_received: bool,
        landed_x: int | None,
        landed_y: int | None,
        fuel_target: ContainerStateDict | None,
        message_start_index: int,
        reposition_map_open_started_ms: int | None = None,
        reposition_map_sync_timestamp_ms: int | None = None,
        reposition_teleport_started_ms: int | None = None,
    ) -> FuelProbeAttemptResultDict:
        """Create a typed attempt result payload."""
        return FuelProbeAttemptResultDict(
            target=target,
            status=status,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
            pickup_started_ms=pickup_started_ms,
            completion_timestamp_ms=completion_timestamp_ms,
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            landed_signal_received=landed_signal_received,
            landed_x=landed_x,
            landed_y=landed_y,
            fuel_target_x=None if fuel_target is None else fuel_target["x"],
            fuel_target_y=None if fuel_target is None else fuel_target["y"],
            fuel_target_volume=None if fuel_target is None else fuel_target["volume"],
            message_start_index=message_start_index,
            message_end_index=len(self.messages),
        )

    def _build_terminal_attempt(
        self,
        *,
        target: TeleportTargetDict,
        status: Literal[
            "no_fuel_visible",
            "radar_timeout",
            "map_sync_timeout",
            "reposition_map_sync_timeout",
            "teleport_timeout",
            "reposition_teleport_timeout",
        ],
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int | None,
        radar_started_ms: int | None,
        radar_sync_timestamp_ms: int | None,
        fuel_before: int,
        fuel_after: int | None,
        landed_signal_received: bool,
        landed_x: int | None,
        landed_y: int | None,
        message_start_index: int,
        reposition_map_open_started_ms: int | None = None,
        reposition_map_sync_timestamp_ms: int | None = None,
        reposition_teleport_started_ms: int | None = None,
    ) -> FuelProbeAttemptResultDict:
        """Build a non-pickup terminal attempt result."""
        return self._build_attempt_result(
            target=target,
            status=status,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
            pickup_started_ms=None,
            completion_timestamp_ms=action_hooks.get_current_time_ms(),
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            landed_signal_received=landed_signal_received,
            landed_x=landed_x,
            landed_y=landed_y,
            fuel_target=None,
            message_start_index=message_start_index,
        )

    def _finalize_attempt_delay(
        self,
        page: action_session.WaitPageProtocol,
        *,
        settle_delay_ms: int,
    ) -> None:
        """Apply optional settle delay after an attempt."""
        if settle_delay_ms > 0:
            page.wait_for_timeout(float(settle_delay_ms))

    def _build_map_sync_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        fuel_before: int,
        message_start_index: int,
    ) -> FuelProbeAttemptResultDict:
        """Build a map-sync-timeout result."""
        self_state = self._require_self_state()
        return self._build_terminal_attempt(
            target=target,
            status="map_sync_timeout",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=None,
            teleport_started_ms=None,
            radar_started_ms=None,
            radar_sync_timestamp_ms=None,
            fuel_before=fuel_before,
            fuel_after=self_state["fuel"],
            landed_signal_received=False,
            landed_x=self_state["x"],
            landed_y=self_state["y"],
            message_start_index=message_start_index,
        )

    def _build_teleport_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
    ) -> FuelProbeAttemptResultDict:
        """Build a teleport-timeout result."""
        return self._build_attempt_result(
            target=target,
            status="teleport_timeout",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=None,
            radar_sync_timestamp_ms=None,
            pickup_started_ms=None,
            completion_timestamp_ms=teleport_result["completion_timestamp_ms"],
            fuel_before=fuel_before,
            fuel_after=teleport_result["fuel_after"],
            landed_signal_received=teleport_result["landed_signal_received"],
            landed_x=teleport_result["landed_x"],
            landed_y=teleport_result["landed_y"],
            fuel_target=None,
            message_start_index=message_start_index,
        )

    def _build_reposition_map_sync_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        reposition_map_open_started_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        fuel_target: ContainerStateDict,
        message_start_index: int,
    ) -> FuelProbeAttemptResultDict:
        """Build a reposition map-sync-timeout result."""
        self_state = self._require_self_state()
        return self._build_attempt_result(
            target=target,
            status="reposition_map_sync_timeout",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=None,
            reposition_teleport_started_ms=None,
            pickup_started_ms=None,
            completion_timestamp_ms=action_hooks.get_current_time_ms(),
            fuel_before=fuel_before,
            fuel_after=self_state["fuel"],
            landed_signal_received=teleport_result["landed_signal_received"],
            landed_x=teleport_result["landed_x"],
            landed_y=teleport_result["landed_y"],
            fuel_target=fuel_target,
            message_start_index=message_start_index,
        )

    def _build_reposition_teleport_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        reposition_map_open_started_ms: int,
        reposition_map_sync_timestamp_ms: int,
        reposition_teleport_started_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        fuel_target: ContainerStateDict,
        message_start_index: int,
    ) -> FuelProbeAttemptResultDict:
        """Build a reposition teleport-timeout result."""
        return self._build_attempt_result(
            target=target,
            status="reposition_teleport_timeout",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
            pickup_started_ms=None,
            completion_timestamp_ms=teleport_result["completion_timestamp_ms"],
            fuel_before=fuel_before,
            fuel_after=teleport_result["fuel_after"],
            landed_signal_received=teleport_result["landed_signal_received"],
            landed_x=teleport_result["landed_x"],
            landed_y=teleport_result["landed_y"],
            fuel_target=fuel_target,
            message_start_index=message_start_index,
        )

    def _build_radar_timeout_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        radar_started_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
    ) -> FuelProbeAttemptResultDict:
        """Build a radar-timeout result."""
        self_state = self._require_self_state()
        return self._build_terminal_attempt(
            target=target,
            status="radar_timeout",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=None,
            fuel_before=fuel_before,
            fuel_after=self_state["fuel"],
            landed_signal_received=teleport_result["landed_signal_received"],
            landed_x=teleport_result["landed_x"],
            landed_y=teleport_result["landed_y"],
            message_start_index=message_start_index,
        )

    def _build_no_fuel_visible_result(
        self,
        *,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
    ) -> FuelProbeAttemptResultDict:
        """Build a no-fuel-visible result."""
        self_state = self._require_self_state()
        return self._build_terminal_attempt(
            target=target,
            status="no_fuel_visible",
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            fuel_before=fuel_before,
            fuel_after=self_state["fuel"],
            landed_signal_received=teleport_result["landed_signal_received"],
            landed_x=teleport_result["landed_x"],
            landed_y=teleport_result["landed_y"],
            message_start_index=message_start_index,
        )

    def _run_pickup_attempt(
        self,
        *,
        page: action_session.WaitPageProtocol,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        reposition_map_open_started_ms: int | None,
        reposition_map_sync_timestamp_ms: int | None,
        reposition_teleport_started_ms: int | None,
        pickup_timeout_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        fuel_target: ContainerStateDict,
        message_start_index: int,
    ) -> FuelProbeAttemptResultDict:
        """Run the pickup portion of a fuel attempt."""
        pickup_started_ms = action_hooks.get_current_time_ms()
        pickup_fuel_before = self._require_self_state()["fuel"]
        action_hooks.drain_buffered_messages(self)
        immediate_pickup_outcome = _get_completed_pickup_outcome(
            self,
            target_x=fuel_target["x"],
            target_y=fuel_target["y"],
            fuel_before=pickup_fuel_before,
        )
        if immediate_pickup_outcome is not None:
            (
                immediate_pickup_status,
                immediate_completion_timestamp_ms,
                immediate_fuel_after,
            ) = immediate_pickup_outcome
            return self._build_attempt_result(
                target=target,
                status=immediate_pickup_status,
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                radar_started_ms=radar_started_ms,
                radar_sync_timestamp_ms=radar_sync_timestamp_ms,
                reposition_map_open_started_ms=reposition_map_open_started_ms,
                reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
                reposition_teleport_started_ms=reposition_teleport_started_ms,
                pickup_started_ms=pickup_started_ms,
                completion_timestamp_ms=immediate_completion_timestamp_ms,
                fuel_before=fuel_before,
                fuel_after=immediate_fuel_after,
                landed_signal_received=teleport_result["landed_signal_received"],
                landed_x=teleport_result["landed_x"],
                landed_y=teleport_result["landed_y"],
                fuel_target=fuel_target,
                message_start_index=message_start_index,
            )
        if not self.move_to(fuel_target["x"], fuel_target["y"]):
            raise FuelProbeError("move_to command dispatch failed during fuel collection")
        pickup_status, completion_timestamp_ms, fuel_after = _wait_for_pickup_outcome(
            page,
            self,
            target_x=fuel_target["x"],
            target_y=fuel_target["y"],
            pickup_started_ms=pickup_started_ms,
            fuel_before=pickup_fuel_before,
            timeout_ms=pickup_timeout_ms,
        )
        self._reset_probe_state_to_idle()
        return self._build_attempt_result(
            target=target,
            status=pickup_status,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
            pickup_started_ms=pickup_started_ms,
            completion_timestamp_ms=completion_timestamp_ms,
            fuel_before=fuel_before,
            fuel_after=fuel_after,
            landed_signal_received=teleport_result["landed_signal_received"],
            landed_x=teleport_result["landed_x"],
            landed_y=teleport_result["landed_y"],
            fuel_target=fuel_target,
            message_start_index=message_start_index,
        )

    def _reposition_for_blocked_fuel(
        self,
        *,
        page: action_session.WaitPageProtocol,
        target: TeleportTargetDict,
        fuel_target: ContainerStateDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
    ) -> tuple[
        TeleportAttemptResultDict | None,
        FuelProbeAttemptResultDict | None,
        int | None,
        int | None,
        int | None,
    ]:
        """Reposition to a blocked visible fuel target when needed."""
        if not _visible_fuel_requires_reposition(self, fuel_target):
            return (teleport_result, None, None, None, None)

        landing_tile = _find_visible_fuel_landing_tile(self, fuel_target)
        if landing_tile is None:
            raise FuelProbeError("visible fuel target has no teleport landing tile")
        reposition_target = _make_reposition_target(landing_tile[0], landing_tile[1])
        reposition_map_open_started_ms = action_hooks.get_current_time_ms()
        if not self.open_map():
            raise FuelProbeError("map_open command dispatch failed during fuel reposition")
        reposition_map_sync_timestamp_ms = action_session.wait_for_world_sync(
            page,
            self,
            reposition_map_open_started_ms,
            map_sync_timeout_ms,
        )
        if reposition_map_sync_timestamp_ms is None:
            return (
                None,
                self._build_reposition_map_sync_timeout_result(
                    target=target,
                    map_open_started_ms=map_open_started_ms,
                    map_sync_timestamp_ms=map_sync_timestamp_ms,
                    teleport_started_ms=teleport_started_ms,
                    radar_started_ms=radar_started_ms,
                    radar_sync_timestamp_ms=radar_sync_timestamp_ms,
                    reposition_map_open_started_ms=reposition_map_open_started_ms,
                    fuel_before=fuel_before,
                    teleport_result=teleport_result,
                    fuel_target=fuel_target,
                    message_start_index=message_start_index,
                ),
                reposition_map_open_started_ms,
                None,
                None,
            )

        reposition_teleport_started_ms = action_hooks.get_current_time_ms()
        if not self.teleport_to(reposition_target["x"], reposition_target["y"]):
            raise FuelProbeError("teleport command dispatch failed during fuel reposition")
        reposition_result = _wait_for_teleport_outcome(
            page,
            self,
            reposition_target,
            map_open_started_ms=reposition_map_open_started_ms,
            map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            teleport_started_ms=reposition_teleport_started_ms,
            fuel_before=self._require_self_state()["fuel"],
            world_timestamp_before=self.get_world_state()["timestamp_ms"],
            timeout_ms=teleport_timeout_ms,
        )
        self._reset_probe_state_to_idle()
        if reposition_result["status"] == "map_sync_timeout":
            raise TeleportProbeError(
                "teleport outcome reported impossible map_sync_timeout during fuel reposition"
            )
        if reposition_result["status"] == "teleport_timeout":
            return (
                None,
                self._build_reposition_teleport_timeout_result(
                    target=target,
                    map_open_started_ms=map_open_started_ms,
                    map_sync_timestamp_ms=map_sync_timestamp_ms,
                    teleport_started_ms=teleport_started_ms,
                    radar_started_ms=radar_started_ms,
                    radar_sync_timestamp_ms=radar_sync_timestamp_ms,
                    reposition_map_open_started_ms=reposition_map_open_started_ms,
                    reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
                    reposition_teleport_started_ms=reposition_teleport_started_ms,
                    fuel_before=fuel_before,
                    teleport_result=reposition_result,
                    fuel_target=fuel_target,
                    message_start_index=message_start_index,
                ),
                reposition_map_open_started_ms,
                reposition_map_sync_timestamp_ms,
                reposition_teleport_started_ms,
            )
        return (
            reposition_result,
            None,
            reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms,
        )

    def _resolve_fuel_target_after_radar(
        self,
        *,
        page: action_session.WaitPageProtocol,
        target: TeleportTargetDict,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int,
        teleport_started_ms: int,
        radar_started_ms: int,
        radar_sync_timestamp_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        fuel_before: int,
        teleport_result: TeleportAttemptResultDict,
        message_start_index: int,
    ) -> tuple[
        ContainerStateDict | None,
        TeleportAttemptResultDict,
        FuelProbeAttemptResultDict | None,
        int | None,
        int | None,
        int | None,
    ]:
        """Resolve the visible fuel target and optional blocked-fuel reposition."""
        fuel_target = _find_visible_fuel_target(self, True)
        if fuel_target is None:
            return (
                None,
                teleport_result,
                self._build_no_fuel_visible_result(
                    target=target,
                    map_open_started_ms=map_open_started_ms,
                    map_sync_timestamp_ms=map_sync_timestamp_ms,
                    teleport_started_ms=teleport_started_ms,
                    radar_started_ms=radar_started_ms,
                    radar_sync_timestamp_ms=radar_sync_timestamp_ms,
                    fuel_before=fuel_before,
                    teleport_result=teleport_result,
                    message_start_index=message_start_index,
                ),
                None,
                None,
                None,
            )
        (
            reposition_result,
            terminal_result,
            reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms,
        ) = self._reposition_for_blocked_fuel(
            page=page,
            target=target,
            fuel_target=fuel_target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            fuel_before=fuel_before,
            teleport_result=teleport_result,
            message_start_index=message_start_index,
        )
        if reposition_result is not None:
            teleport_result = reposition_result
        return (
            fuel_target,
            teleport_result,
            terminal_result,
            reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms,
        )

    def _probe_single_fuel_target(
        self,
        *,
        target: TeleportTargetDict,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelProbeAttemptResultDict:
        """Run one teleport-radar-fuel attempt."""
        page = self._require_page()
        self_state_before = self._require_self_state()
        fuel_before = self_state_before["fuel"]
        message_start_index = len(self.messages)
        self._reset_probe_state_to_idle()

        map_open_started_ms = action_hooks.get_current_time_ms()
        if not self.open_map():
            raise FuelProbeError("map_open command dispatch failed")
        map_sync_timestamp_ms = action_session.wait_for_world_sync(
            page,
            self,
            map_open_started_ms,
            map_sync_timeout_ms,
        )
        if map_sync_timestamp_ms is None:
            result = self._build_map_sync_timeout_result(
                target=target,
                map_open_started_ms=map_open_started_ms,
                fuel_before=fuel_before,
                message_start_index=message_start_index,
            )
            self._reset_probe_state_to_idle()
            self._finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)
            return result

        teleport_started_ms = action_hooks.get_current_time_ms()
        if not self.teleport_to(target["x"], target["y"]):
            raise FuelProbeError("teleport command dispatch failed")
        teleport_result = _wait_for_teleport_outcome(
            page,
            self,
            target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            fuel_before=fuel_before,
            world_timestamp_before=self.get_world_state()["timestamp_ms"],
            timeout_ms=teleport_timeout_ms,
        )
        self._reset_probe_state_to_idle()
        if teleport_result["status"] == "map_sync_timeout":
            raise TeleportProbeError("teleport outcome reported impossible map_sync_timeout")
        if teleport_result["status"] == "teleport_timeout":
            result = self._build_teleport_timeout_result(
                target=target,
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                fuel_before=fuel_before,
                teleport_result=teleport_result,
                message_start_index=message_start_index,
            )
            self._finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)
            return result

        radar_started_ms = action_hooks.get_current_time_ms()
        if not self.use_radar():
            raise FuelProbeError("radar command dispatch failed")
        radar_sync_timestamp_ms = action_session.wait_for_world_sync(
            page,
            self,
            radar_started_ms,
            radar_timeout_ms,
        )
        self._reset_probe_state_to_idle()
        if radar_sync_timestamp_ms is None:
            result = self._build_radar_timeout_result(
                target=target,
                map_open_started_ms=map_open_started_ms,
                map_sync_timestamp_ms=map_sync_timestamp_ms,
                teleport_started_ms=teleport_started_ms,
                radar_started_ms=radar_started_ms,
                fuel_before=fuel_before,
                teleport_result=teleport_result,
                message_start_index=message_start_index,
            )
            self._finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)
            return result

        (
            fuel_target,
            teleport_result,
            terminal_result,
            reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms,
        ) = self._resolve_fuel_target_after_radar(
            page=page,
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            map_sync_timeout_ms=map_sync_timeout_ms,
            teleport_timeout_ms=teleport_timeout_ms,
            fuel_before=fuel_before,
            teleport_result=teleport_result,
            message_start_index=message_start_index,
        )
        if terminal_result is not None:
            self._finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)
            return terminal_result
        if fuel_target is None:
            raise FuelProbeError("visible fuel target disappeared unexpectedly")

        result = self._run_pickup_attempt(
            page=page,
            target=target,
            map_open_started_ms=map_open_started_ms,
            map_sync_timestamp_ms=map_sync_timestamp_ms,
            teleport_started_ms=teleport_started_ms,
            radar_started_ms=radar_started_ms,
            radar_sync_timestamp_ms=radar_sync_timestamp_ms,
            reposition_map_open_started_ms=reposition_map_open_started_ms,
            reposition_map_sync_timestamp_ms=reposition_map_sync_timestamp_ms,
            reposition_teleport_started_ms=reposition_teleport_started_ms,
            pickup_timeout_ms=pickup_timeout_ms,
            fuel_before=fuel_before,
            teleport_result=teleport_result,
            fuel_target=fuel_target,
            message_start_index=message_start_index,
        )
        self._finalize_attempt_delay(page, settle_delay_ms=settle_delay_ms)
        return result

    def execute_probe(
        self,
        *,
        target_pickups: int,
        max_attempts: int,
        initial_sync_timeout_ms: int,
        map_sync_timeout_ms: int,
        teleport_timeout_ms: int,
        radar_timeout_ms: int,
        pickup_timeout_ms: int,
        settle_delay_ms: int,
    ) -> FuelProbeSessionDict:
        """Run the live fuel probe session."""
        if target_pickups <= 0:
            raise ValueError("target_pickups must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if max_attempts < target_pickups:
            raise ValueError("max_attempts must be at least target_pickups")
        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError("Playwright is not installed.")

        self._start_timestamp_ms = action_hooks.get_current_time_ms()
        self._messages = []
        self._ws_urls = {}
        self._magic = None
        self._cdp_message_buffer = []

        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=self._headless)
            context = browser.new_context()
            page = context.new_page()
            cdp = context.new_cdp_session(page)

            self._cdp = cdp
            self._page = page

            reset_world_state()
            reset_all_trackers()
            reset_cdp_time_offset()
            reset_viewport_tracking()

            self._setup_console_listener(cdp)
            self._setup_cdp_handlers(cdp)
            self._navigate_and_login(page, cdp, tank_name_prefix="TP", auto_join_room=True)
            self._wait_for_game_ready(page)
            game_ready_timestamp_ms = action_hooks.get_current_time_ms()
            self._gather_intel(page, cdp)
            intel_ready_timestamp_ms = action_hooks.get_current_time_ms()

            try:
                initial_sync_started_ms = action_hooks.get_current_time_ms()
                initial_world_timestamp_ms, spawn = action_session.wait_for_initial_self_state(
                    page,
                    self,
                    initial_sync_started_ms,
                    initial_sync_timeout_ms,
                )
                action_session.advance_startup_state(self)
                command_ready_timestamp_ms = action_hooks.get_current_time_ms()
                terrain = get_terrain_map()
                if terrain is None:
                    raise FuelProbeError("terrain map is unavailable")
                used_targets: set[tuple[int, int]] = set()
                attempts: list[FuelProbeAttemptResultDict] = []
                successful_pickups = 0
                while len(attempts) < max_attempts and successful_pickups < target_pickups:
                    self_state = self._require_self_state()
                    targets = build_distinct_ground_targets(
                        self_state["x"],
                        self_state["y"],
                        terrain,
                        count=1,
                        step=_FUEL_PROBE_TARGET_STEP,
                        max_radius=_FUEL_PROBE_TARGET_MAX_RADIUS,
                        excluded=frozenset(used_targets),
                    )
                    target = targets[0]
                    used_targets.add((target["x"], target["y"]))
                    attempt_result = self._probe_single_fuel_target(
                        target=target,
                        map_sync_timeout_ms=map_sync_timeout_ms,
                        teleport_timeout_ms=teleport_timeout_ms,
                        radar_timeout_ms=radar_timeout_ms,
                        pickup_timeout_ms=pickup_timeout_ms,
                        settle_delay_ms=settle_delay_ms,
                    )
                    attempts.append(attempt_result)
                    if attempt_result["status"] == "picked_up_fuel":
                        successful_pickups += 1
                first_attempt_started_ms = attempts[0]["map_open_started_ms"] if attempts else None
                startup_timing = TeleportStartupTimingDict(
                    game_ready_timestamp_ms=game_ready_timestamp_ms,
                    intel_ready_timestamp_ms=intel_ready_timestamp_ms,
                    initial_sync_started_ms=initial_sync_started_ms,
                    initial_world_timestamp_ms=initial_world_timestamp_ms,
                    command_ready_timestamp_ms=command_ready_timestamp_ms,
                    first_attempt_started_ms=first_attempt_started_ms,
                    game_ready_to_intel_ready_ms=intel_ready_timestamp_ms - game_ready_timestamp_ms,
                    intel_ready_to_initial_world_ms=(
                        initial_world_timestamp_ms - intel_ready_timestamp_ms
                    ),
                    initial_world_to_command_ready_ms=(
                        command_ready_timestamp_ms - initial_world_timestamp_ms
                    ),
                    command_ready_to_first_attempt_ms=(
                        None
                        if first_attempt_started_ms is None
                        else first_attempt_started_ms - command_ready_timestamp_ms
                    ),
                )
                return FuelProbeSessionDict(
                    session_id=self.session_id,
                    start_timestamp_ms=self._start_timestamp_ms,
                    end_timestamp_ms=action_hooks.get_current_time_ms(),
                    base_url=self._target_url,
                    spawn_x=spawn["x"],
                    spawn_y=spawn["y"],
                    target_pickups=target_pickups,
                    max_attempts=max_attempts,
                    capture_session_path="",
                    initial_sync_timeout_ms=initial_sync_timeout_ms,
                    startup_timing=startup_timing,
                    map_sync_timeout_ms=map_sync_timeout_ms,
                    teleport_timeout_ms=teleport_timeout_ms,
                    radar_timeout_ms=radar_timeout_ms,
                    pickup_timeout_ms=pickup_timeout_ms,
                    settle_delay_ms=settle_delay_ms,
                    attempts=attempts,
                )
            finally:
                self._cdp = None
                self._page = None
                self._cleanup(cdp, page, context, browser)


def run_fuel_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    target_pickups: int = 3,
    max_attempts: int = 9,
    initial_sync_timeout_ms: int = 10000,
    map_sync_timeout_ms: int = 3000,
    teleport_timeout_ms: int = 10000,
    radar_timeout_ms: int = 3000,
    pickup_timeout_ms: int = 3000,
    settle_delay_ms: int = 500,
) -> FuelProbeSessionDict:
    """Run a live fuel probe and save the session JSON."""
    probe = FuelProbe(target_url, headless=headless, prefer_account=prefer_account)
    session = probe.execute_probe(
        target_pickups=target_pickups,
        max_attempts=max_attempts,
        initial_sync_timeout_ms=initial_sync_timeout_ms,
        map_sync_timeout_ms=map_sync_timeout_ms,
        teleport_timeout_ms=teleport_timeout_ms,
        radar_timeout_ms=radar_timeout_ms,
        pickup_timeout_ms=pickup_timeout_ms,
        settle_delay_ms=settle_delay_ms,
    )
    capture_session_path = save_capture_session(
        session_id=session["session_id"],
        start_timestamp_ms=session["start_timestamp_ms"],
        end_timestamp_ms=session["end_timestamp_ms"],
        base_url=session["base_url"],
        messages=probe.messages,
        magic=probe.magic,
        output_path=output_path,
    )
    session["capture_session_path"] = capture_session_path
    encoded = encode_fuel_probe_session(session)
    json_str = dump_json_str(encoded, compact=False, indent=2)
    _test_hooks.write_text(Path(output_path), json_str)
    log.info(format_fuel_probe_summary(session))
    return session


__all__ = ["FuelProbe", "FuelProbeError", "format_fuel_probe_summary", "run_fuel_probe"]
