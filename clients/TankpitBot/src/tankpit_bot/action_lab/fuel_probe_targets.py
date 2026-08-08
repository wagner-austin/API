"""Fuel-target selection and pickup-outcome helpers.

Everything the probe needs BEFORE and AFTER a pickup attempt: finding a
visible fuel container, deciding whether it needs a reposition, and
waiting on the pickup outcome.

The probe reaches these through the module rather than by name, so the
tests that swap them keep a working injection seam.
"""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.fuel_probe_diagnostics import (
    format_visible_fuel_entries as _shared_format_visible_fuel_entries,
)
from tankpit_bot.action_lab.fuel_probe_diagnostics import (
    log_fuel_target_diagnostic as _shared_log_fuel_target_diagnostic,
)
from tankpit_bot.action_lab.fuel_target_phase import (
    FuelTargetPhaseProbeProtocol,
)
from tankpit_bot.action_lab.fuel_targeting import (
    FuelTargetingError,
    find_visible_fuel_landing_tile,
    visible_fuel_requires_reposition,
)
from tankpit_bot.action_lab.pickup_phase import (
    PickupPhaseError,
)
from tankpit_bot.action_lab.pickup_phase import (
    get_completed_pickup_outcome as _shared_get_completed_pickup_outcome,
)
from tankpit_bot.action_lab.pickup_phase import (
    wait_for_pickup_outcome as _shared_wait_for_pickup_outcome,
)
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.types import (
    TeleportTargetDict,
)
from tankpit_bot.bot.ai.equipment_search import find_best_fuel
from tankpit_bot.state.types import ContainerStateDict


class FuelProbeError(Exception):
    """Raised when the fuel probe cannot proceed."""


def _log_fuel_target_diagnostic(
    probe: ProbeBase,
    *,
    radar_cycle_id: int,
    fuel_target: ContainerStateDict | None,
) -> None:
    """Emit one structured diagnostic line after radar target resolution."""
    _shared_log_fuel_target_diagnostic(
        probe,
        radar_cycle_id=radar_cycle_id,
        fuel_target=fuel_target,
        terrain_provider=probe.world.get_terrain_map,
    )


def _find_visible_fuel_target(
    probe: FuelTargetPhaseProbeProtocol,
) -> ContainerStateDict | None:
    """Return the best currently visible walk-reachable fuel container."""
    terrain = probe.world.get_terrain_map()
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
        minimum_volume=1,
    )


def _format_visible_fuel_entries(
    probe: FuelTargetPhaseProbeProtocol,
    *,
    fuel_target: ContainerStateDict | None,
) -> str:
    """Format the currently visible fuel candidates for diagnostics.

    Args:
        probe: Probe exposing current world and self state.
        fuel_target: Selected target for the current decision, if any.

    Returns:
        ``"unavailable"`` when terrain or self state is missing, ``"none"``
        when no visible fuel is tracked, or a compact candidate summary.
    """
    return _shared_format_visible_fuel_entries(
        probe,
        fuel_target=fuel_target,
        terrain_provider=probe.world.get_terrain_map,
    )


def _visible_fuel_requires_reposition(
    probe: FuelTargetPhaseProbeProtocol,
    fuel_target: ContainerStateDict,
) -> bool:
    """Return whether a visible fuel target needs a reposition teleport."""
    try:
        return visible_fuel_requires_reposition(probe, fuel_target)
    except FuelTargetingError as exc:
        raise FuelProbeError(str(exc)) from exc


def _find_visible_fuel_landing_tile(
    probe: FuelTargetPhaseProbeProtocol,
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
    probe: action_session.BufferedWorldStateProviderProtocol,
    *,
    target_x: int,
    target_y: int,
    pickup_started_ms: int,
    fuel_before: int,
    timeout_ms: int,
) -> tuple[Literal["picked_up_fuel", "pickup_timeout"], int, int]:
    """Wait for a fuel pickup to complete or time out."""
    try:
        return _shared_wait_for_pickup_outcome(
            page,
            probe,
            target_x=target_x,
            target_y=target_y,
            pickup_started_ms=pickup_started_ms,
            fuel_before=fuel_before,
            timeout_ms=timeout_ms,
        )
    except PickupPhaseError as exc:
        raise FuelProbeError(str(exc)) from exc


def _get_completed_pickup_outcome(
    probe: action_session.WorldStateProviderProtocol,
    *,
    target_x: int,
    target_y: int,
    fuel_before: int,
) -> tuple[Literal["picked_up_fuel"], int, int] | None:
    """Return a completed pickup outcome once the fuel credit is observed."""
    try:
        return _shared_get_completed_pickup_outcome(
            probe,
            target_x=target_x,
            target_y=target_y,
            fuel_before=fuel_before,
        )
    except PickupPhaseError as exc:
        raise FuelProbeError(str(exc)) from exc


__all__ = [
    "FuelProbeError",
]
