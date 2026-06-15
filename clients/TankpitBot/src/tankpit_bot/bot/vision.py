"""Bot vision module -- mutations, merging, and ASCII rendering.

Provides immutable mutation functions for the vision state caches and
rendering utilities. TypedDicts and codecs are in vision_types.py.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.bot.vision_types import (
    ContainerEntryDict,
    PositionEntryDict,
    TankRegistryEntryDict,
    VisionStateDict,
    decode_container_entry,
    decode_position_entry,
    decode_tank_registry_entry,
    decode_vision_state,
    encode_container_entry,
    encode_position_entry,
    encode_tank_registry_entry,
    encode_vision_state,
    make_container_entry,
    make_empty_vision_state,
    make_position_entry,
    make_tank_registry_entry,
)
from tankpit_bot.sniffer.world_state import get_world_service, get_world_state
from tankpit_bot.sniffer.world_state_tiles import render_world_state_ascii
from tankpit_bot.state import ContainerStateDict, coord_key, make_container_state

log = get_logger(__name__)


def add_tank_to_registry(
    state: VisionStateDict,
    tank_id: int,
    name: str,
    team: int,
) -> VisionStateDict:
    """Add or update tank in registry.

    Args:
        state: Current vision state.
        tank_id: Unique tank identifier.
        name: Player name.
        team: Team ID (0-3).

    Returns:
        New VisionStateDict with updated registry.
    """
    new_registry = dict(state["tank_registry"])
    key = str(tank_id)
    new_registry[key] = make_tank_registry_entry(tank_id, name, team)
    return VisionStateDict(
        tank_registry=new_registry,
        position_cache=state["position_cache"],
        container_cache=state["container_cache"],
        self_fuel=state["self_fuel"],
        self_tank_id=state["self_tank_id"],
    )


def update_tank_position(
    state: VisionStateDict,
    tank_id: int,
    x: int,
    y: int,
) -> VisionStateDict:
    """Update tank position in cache.

    Args:
        state: Current vision state.
        tank_id: Unique tank identifier.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).

    Returns:
        New VisionStateDict with updated position.
    """
    new_cache = dict(state["position_cache"])
    key = str(tank_id)
    new_cache[key] = make_position_entry(tank_id, x, y)
    return VisionStateDict(
        tank_registry=state["tank_registry"],
        position_cache=new_cache,
        container_cache=state["container_cache"],
        self_fuel=state["self_fuel"],
        self_tank_id=state["self_tank_id"],
    )


def update_container(
    state: VisionStateDict,
    x: int,
    y: int,
    volume: int,
) -> VisionStateDict:
    """Update container in cache.

    Args:
        state: Current vision state.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        volume: Fuel volume (0 for equipment).

    Returns:
        New VisionStateDict with updated container.
    """
    new_cache = dict(state["container_cache"])
    key = coord_key(x, y)
    new_cache[key] = make_container_entry(x, y, volume)
    return VisionStateDict(
        tank_registry=state["tank_registry"],
        position_cache=state["position_cache"],
        container_cache=new_cache,
        self_fuel=state["self_fuel"],
        self_tank_id=state["self_tank_id"],
    )


def remove_container(
    state: VisionStateDict,
    x: int,
    y: int,
) -> VisionStateDict:
    """Remove container from cache.

    Args:
        state: Current vision state.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).

    Returns:
        New VisionStateDict with container removed.
    """
    new_cache = dict(state["container_cache"])
    key = coord_key(x, y)
    new_cache.pop(key, None)
    return VisionStateDict(
        tank_registry=state["tank_registry"],
        position_cache=state["position_cache"],
        container_cache=new_cache,
        self_fuel=state["self_fuel"],
        self_tank_id=state["self_tank_id"],
    )


def update_self_fuel_vision(
    state: VisionStateDict,
    fuel: int,
) -> VisionStateDict:
    """Update self fuel in vision state.

    Args:
        state: Current vision state.
        fuel: New fuel value.

    Returns:
        New VisionStateDict with updated fuel.
    """
    return VisionStateDict(
        tank_registry=state["tank_registry"],
        position_cache=state["position_cache"],
        container_cache=state["container_cache"],
        self_fuel=fuel,
        self_tank_id=state["self_tank_id"],
    )


def add_fuel_delta(
    state: VisionStateDict,
    delta: int,
) -> VisionStateDict:
    """Add fuel delta to self fuel.

    Args:
        state: Current vision state.
        delta: Fuel amount to add (can be negative).

    Returns:
        New VisionStateDict with updated fuel.
    """
    new_fuel = state["self_fuel"] + delta
    return update_self_fuel_vision(state, new_fuel)


def set_self_tank_id(
    state: VisionStateDict,
    tank_id: int,
) -> VisionStateDict:
    """Set self tank ID.

    Args:
        state: Current vision state.
        tank_id: Self tank ID.

    Returns:
        New VisionStateDict with updated self tank ID.
    """
    return VisionStateDict(
        tank_registry=state["tank_registry"],
        position_cache=state["position_cache"],
        container_cache=state["container_cache"],
        self_fuel=state["self_fuel"],
        self_tank_id=tank_id,
    )


def pickup_container_vision(
    state: VisionStateDict,
    x: int,
    y: int,
) -> VisionStateDict:
    """Process container pickup - remove container and add fuel.

    Args:
        state: Current vision state.
        x: Container X coordinate.
        y: Container Y coordinate.

    Returns:
        New VisionStateDict with container removed and fuel added.
    """
    key = coord_key(x, y)
    container = state["container_cache"].get(key)
    volume = container["volume"] if container else 0

    # Remove container and add fuel
    state_without_container = remove_container(state, x, y)
    if volume > 0:
        return add_fuel_delta(state_without_container, volume)
    return state_without_container


# =============================================================================
# Merge Functions
# =============================================================================


def get_merged_fuel_containers(vision_state: VisionStateDict) -> list[ContainerStateDict]:
    """Get fuel containers merged from vision cache and world state.

    Combines containers from both sources, preferring world state when
    both have the same location (world state has more accurate data).

    Args:
        vision_state: Vision state with fallback container cache.

    Returns:
        List of ContainerStateDict for fuel containers (volume > 0).
    """
    world_state = get_world_state()
    world_containers = world_state["containers"]

    # Start with world state containers
    merged: dict[str, ContainerStateDict] = dict(world_containers)

    # Add vision cache containers if not in world state
    for key, entry in vision_state["container_cache"].items():
        if key not in merged:
            is_fuel = entry["volume"] > 0
            merged[key] = make_container_state(
                x=entry["x"],
                y=entry["y"],
                is_fuel=is_fuel,
                volume=entry["volume"],
                source="world_state",
                refresh_kind="world_state",
                timestamp_ms=0,
                failed_pickups=0,
            )

    # Filter to fuel containers only
    return [c for c in merged.values() if c["is_fuel"] and c["volume"] > 0]


def get_merged_fuel(vision_state: VisionStateDict) -> int:
    """Get fuel from world state, falling back to vision cache.

    Prefers world state self_state.fuel when available, otherwise
    uses the vision cache tracked fuel.

    Args:
        vision_state: Vision state with fallback fuel tracking.

    Returns:
        Current fuel amount.
    """
    world_state = get_world_state()
    self_state = world_state["self_state"]
    if self_state is not None:
        return self_state["fuel"]
    return vision_state["self_fuel"]


# =============================================================================
# ASCII Rendering
# =============================================================================


def render_vision_ascii() -> str | None:
    """Render current world state as ASCII viewport.

    Uses the sniffer's world state and terrain map to generate
    an ASCII representation of the visible game area.

    Returns:
        Multi-line ASCII string, or None if terrain map not loaded.
    """
    return render_world_state_ascii(get_world_service())


def render_vision_debug(vision_state: VisionStateDict) -> str:
    """Render vision state debug info.

    Provides a summary of what the vision system is tracking
    independent of the world state.

    Args:
        vision_state: Current vision state.

    Returns:
        Multi-line debug string.
    """
    lines: list[str] = []
    lines.append("=== Vision Cache Debug ===")
    lines.append(f"Self tank ID: {vision_state['self_tank_id']}")
    lines.append(f"Self fuel: {vision_state['self_fuel']}")
    lines.append(f"Tanks registered: {len(vision_state['tank_registry'])}")
    lines.append(f"Positions cached: {len(vision_state['position_cache'])}")
    lines.append(f"Containers cached: {len(vision_state['container_cache'])}")

    # List containers by type
    fuel_count = 0
    equip_count = 0
    for entry in vision_state["container_cache"].values():
        if entry["volume"] > 0:
            fuel_count += 1
        else:
            equip_count += 1

    lines.append(f"  Fuel containers: {fuel_count}")
    lines.append(f"  Equipment: {equip_count}")

    # Compare with world state
    world_state = get_world_state()
    world_containers = len(world_state["containers"])
    world_tanks = len(world_state["tanks"])
    lines.append("")
    lines.append("=== World State Comparison ===")
    lines.append(f"World containers: {world_containers}")
    lines.append(f"World tanks: {world_tanks}")

    # Show merged fuel containers
    merged = get_merged_fuel_containers(vision_state)
    lines.append(f"Merged fuel containers: {len(merged)}")

    return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ContainerEntryDict",
    "PositionEntryDict",
    "TankRegistryEntryDict",
    "VisionStateDict",
    "add_fuel_delta",
    "add_tank_to_registry",
    "decode_container_entry",
    "decode_position_entry",
    "decode_tank_registry_entry",
    "decode_vision_state",
    "encode_container_entry",
    "encode_position_entry",
    "encode_tank_registry_entry",
    "encode_vision_state",
    "get_merged_fuel",
    "get_merged_fuel_containers",
    "make_container_entry",
    "make_empty_vision_state",
    "make_position_entry",
    "make_tank_registry_entry",
    "pickup_container_vision",
    "remove_container",
    "render_vision_ascii",
    "render_vision_debug",
    "set_self_tank_id",
    "update_container",
    "update_self_fuel_vision",
    "update_tank_position",
]
