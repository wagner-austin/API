"""Visible fuel targeting helpers for live action-lab probes."""

from __future__ import annotations

from typing import Protocol

from tankpit_bot.bot.ai.equipment import hostile_mines
from tankpit_bot.bot.ai.equipment_search import find_best_fuel, find_teleport_landing_tile
from tankpit_bot.bot.ai.reachability import is_collection_reachable_in_viewport
from tankpit_bot.sniffer.world_state import get_terrain_map
from tankpit_bot.state.types import ContainerStateDict, SelfStateDict, WorldStateDict


class FuelTargetingError(Exception):
    """Raised when visible fuel targeting prerequisites are unavailable."""


class VisibleFuelTargetingProbeProtocol(Protocol):
    """Minimal probe interface required for visible-fuel targeting helpers."""

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""

    def get_self_state(self) -> SelfStateDict | None:
        """Return the current self state when available."""


def find_visible_fuel_target(
    probe: VisibleFuelTargetingProbeProtocol,
) -> ContainerStateDict | None:
    """Return the best walk-reachable visible fuel target for the current viewport.

    Args:
        probe: Live probe providing world and self state.

    Returns:
        Best walk-reachable visible fuel container, or ``None`` when none are actionable.

    Raises:
        FuelTargetingError: If terrain or self state is unavailable.
    """
    terrain = get_terrain_map()
    if terrain is None:
        raise FuelTargetingError("terrain map is unavailable")
    self_state = probe.get_self_state()
    if self_state is None:
        raise FuelTargetingError("self state is unavailable")
    world = probe.get_world_state()
    return find_best_fuel(
        world,
        self_state,
        terrain,
        minimum_volume=1,
    )


def visible_fuel_requires_reposition(
    probe: VisibleFuelTargetingProbeProtocol,
    fuel_target: ContainerStateDict,
) -> bool:
    """Return whether a visible fuel target needs a reposition teleport.

    Args:
        probe: Live probe providing world and self state.
        fuel_target: Visible fuel container under consideration.

    Returns:
        True when the container cannot be collected by walking inside the
        current viewport.

    Raises:
        FuelTargetingError: If terrain or self state is unavailable.
    """
    terrain = get_terrain_map()
    if terrain is None:
        raise FuelTargetingError("terrain map is unavailable")
    self_state = probe.get_self_state()
    if self_state is None:
        raise FuelTargetingError("self state is unavailable")
    world = probe.get_world_state()
    return not is_collection_reachable_in_viewport(
        world,
        terrain,
        self_state["x"],
        self_state["y"],
        fuel_target["x"],
        fuel_target["y"],
        hostile_mines(world),
    )


def find_visible_fuel_landing_tile(
    probe: VisibleFuelTargetingProbeProtocol,
    fuel_target: ContainerStateDict,
) -> tuple[int, int] | None:
    """Return the teleport landing tile for a blocked visible fuel target.

    Args:
        probe: Live probe providing world and self state.
        fuel_target: Visible fuel container under consideration.

    Returns:
        Safe teleport landing tile, or ``None`` when none exists.

    Raises:
        FuelTargetingError: If terrain or self state is unavailable.
    """
    terrain = get_terrain_map()
    if terrain is None:
        raise FuelTargetingError("terrain map is unavailable")
    self_state = probe.get_self_state()
    if self_state is None:
        raise FuelTargetingError("self state is unavailable")
    return find_teleport_landing_tile(
        terrain,
        fuel_target["x"],
        fuel_target["y"],
    )


__all__ = [
    "FuelTargetingError",
    "VisibleFuelTargetingProbeProtocol",
    "find_visible_fuel_landing_tile",
    "find_visible_fuel_target",
    "visible_fuel_requires_reposition",
]
