"""Visible equipment targeting helpers for live action-lab probes.

Mirrors the fuel-side targeting helpers but selects equipment containers via
nearest-first reachability rather than fuel's volume-distance scoring.
"""

from __future__ import annotations

from typing import Protocol

from tankpit_bot.bot.ai.equipment_search import find_nearest_equipment, find_teleport_landing_tile
from tankpit_bot.bot.ai.reachability import is_collection_reachable_in_viewport
from tankpit_bot.sniffer.world_state import get_terrain_map
from tankpit_bot.state.types import ContainerStateDict, SelfStateDict, WorldStateDict


class EquipmentTargetingError(Exception):
    """Raised when visible equipment targeting prerequisites are unavailable."""


class VisibleEquipmentTargetingProbeProtocol(Protocol):
    """Minimal probe interface required for visible-equipment targeting helpers."""

    def get_world_state(self) -> WorldStateDict:
        """Return the current world state."""

    def get_self_state(self) -> SelfStateDict | None:
        """Return the current self state when available."""


def find_visible_equipment_target(
    probe: VisibleEquipmentTargetingProbeProtocol,
) -> ContainerStateDict | None:
    """Return the nearest walk-reachable visible equipment target for the current viewport.

    Args:
        probe: Live probe providing world and self state.

    Returns:
        Nearest walk-reachable visible equipment container, or ``None``
        when none are actionable.

    Raises:
        EquipmentTargetingError: If terrain or self state is unavailable.
    """
    terrain = get_terrain_map()
    if terrain is None:
        raise EquipmentTargetingError("terrain map is unavailable")
    self_state = probe.get_self_state()
    if self_state is None:
        raise EquipmentTargetingError("self state is unavailable")
    world = probe.get_world_state()
    return find_nearest_equipment(
        world,
        self_state,
        terrain,
        now_ms=world["timestamp_ms"],
    )


def visible_equipment_requires_reposition(
    probe: VisibleEquipmentTargetingProbeProtocol,
    equipment_target: ContainerStateDict,
) -> bool:
    """Return whether a visible equipment target needs a reposition teleport.

    Args:
        probe: Live probe providing world and self state.
        equipment_target: Visible equipment container under consideration.

    Returns:
        True when the container cannot be collected by walking inside the
        current viewport.

    Raises:
        EquipmentTargetingError: If terrain or self state is unavailable.
    """
    terrain = get_terrain_map()
    if terrain is None:
        raise EquipmentTargetingError("terrain map is unavailable")
    self_state = probe.get_self_state()
    if self_state is None:
        raise EquipmentTargetingError("self state is unavailable")
    world = probe.get_world_state()
    return not is_collection_reachable_in_viewport(
        world,
        terrain,
        self_state["x"],
        self_state["y"],
        equipment_target["x"],
        equipment_target["y"],
        world["mines"],
    )


def find_visible_equipment_landing_tile(
    probe: VisibleEquipmentTargetingProbeProtocol,
    equipment_target: ContainerStateDict,
) -> tuple[int, int] | None:
    """Return the teleport landing tile for a blocked visible equipment target.

    Args:
        probe: Live probe providing world and self state.
        equipment_target: Visible equipment container under consideration.

    Returns:
        Safe teleport landing tile, or ``None`` when none exists.

    Raises:
        EquipmentTargetingError: If terrain or self state is unavailable.
    """
    terrain = get_terrain_map()
    if terrain is None:
        raise EquipmentTargetingError("terrain map is unavailable")
    self_state = probe.get_self_state()
    if self_state is None:
        raise EquipmentTargetingError("self state is unavailable")
    world = probe.get_world_state()
    return find_teleport_landing_tile(
        terrain,
        self_state["x"],
        self_state["y"],
        equipment_target["x"],
        equipment_target["y"],
        world["mines"],
    )


__all__ = [
    "EquipmentTargetingError",
    "VisibleEquipmentTargetingProbeProtocol",
    "find_visible_equipment_landing_tile",
    "find_visible_equipment_target",
    "visible_equipment_requires_reposition",
]
