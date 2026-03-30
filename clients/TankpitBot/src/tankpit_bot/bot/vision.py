"""Bot vision module with fallback caches and ASCII rendering.

This module provides independent tracking of game entities as a fallback
when the primary world state tracking misses updates. It uses multiple
data sources (radar, tank registry, position updates) to maintain a
robust picture of the game world.

The vision system maintains:
- Tank registry: tank_id -> name mapping
- Tank teams: tank_id -> team mapping
- Position cache: tank_id -> (x, y) position
- Container cache: (x, y) -> volume mapping
- Self fuel tracking

These caches are updated from protocol messages and provide redundancy
when sniffer world state tracking has gaps.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_int,
    require_str,
)
from platform_core.logging import get_logger
from typing_extensions import TypedDict

from tankpit_bot.sniffer.world_state import get_world_state, render_world_state_ascii
from tankpit_bot.state import ContainerStateDict, coord_key

log = get_logger(__name__)


# =============================================================================
# TypedDicts
# =============================================================================


class TankRegistryEntryDict(TypedDict):
    """Tank registry entry for tracking known tanks.

    Attributes:
        tank_id: Unique tank identifier.
        name: Player name.
        team: Team ID (0=red, 1=purple, 2=blue, 3=orange).
    """

    tank_id: int
    name: str
    team: int


class PositionEntryDict(TypedDict):
    """Position cache entry for tracking tank positions.

    Attributes:
        tank_id: Unique tank identifier.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
    """

    tank_id: int
    x: int
    y: int


class ContainerEntryDict(TypedDict):
    """Container cache entry for tracking containers.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        volume: Fuel volume (0 for equipment).
    """

    x: int
    y: int
    volume: int


class VisionStateDict(TypedDict):
    """Complete vision state snapshot.

    Attributes:
        tank_registry: Known tanks by tank_id string key.
        position_cache: Tank positions by tank_id string key.
        container_cache: Containers by "x,y" string key.
        self_fuel: Tracked self fuel amount.
        self_tank_id: Self tank ID, or -1 if unknown.
    """

    tank_registry: dict[str, TankRegistryEntryDict]
    position_cache: dict[str, PositionEntryDict]
    container_cache: dict[str, ContainerEntryDict]
    self_fuel: int
    self_tank_id: int


# =============================================================================
# Factory Functions
# =============================================================================


def make_tank_registry_entry(tank_id: int, name: str, team: int) -> TankRegistryEntryDict:
    """Create a tank registry entry.

    Args:
        tank_id: Unique tank identifier.
        name: Player name.
        team: Team ID (0-3).

    Returns:
        TankRegistryEntryDict with the provided values.
    """
    return TankRegistryEntryDict(tank_id=tank_id, name=name, team=team)


def make_position_entry(tank_id: int, x: int, y: int) -> PositionEntryDict:
    """Create a position cache entry.

    Args:
        tank_id: Unique tank identifier.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).

    Returns:
        PositionEntryDict with the provided values.
    """
    return PositionEntryDict(tank_id=tank_id, x=x, y=y)


def make_container_entry(x: int, y: int, volume: int) -> ContainerEntryDict:
    """Create a container cache entry.

    Args:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        volume: Fuel volume (0 for equipment).

    Returns:
        ContainerEntryDict with the provided values.
    """
    return ContainerEntryDict(x=x, y=y, volume=volume)


def make_empty_vision_state() -> VisionStateDict:
    """Create an empty vision state.

    Returns:
        Empty VisionStateDict with default values.
    """
    return VisionStateDict(
        tank_registry={},
        position_cache={},
        container_cache={},
        self_fuel=1000,
        self_tank_id=-1,
    )


# =============================================================================
# Encode Functions
# =============================================================================


def encode_tank_registry_entry(entry: TankRegistryEntryDict) -> JSONObject:
    """Encode TankRegistryEntryDict to JSON.

    Args:
        entry: Entry to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "tank_id": entry["tank_id"],
        "name": entry["name"],
        "team": entry["team"],
    }


def encode_position_entry(entry: PositionEntryDict) -> JSONObject:
    """Encode PositionEntryDict to JSON.

    Args:
        entry: Entry to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "tank_id": entry["tank_id"],
        "x": entry["x"],
        "y": entry["y"],
    }


def encode_container_entry(entry: ContainerEntryDict) -> JSONObject:
    """Encode ContainerEntryDict to JSON.

    Args:
        entry: Entry to encode.

    Returns:
        JSON-serializable dict.
    """
    return {
        "x": entry["x"],
        "y": entry["y"],
        "volume": entry["volume"],
    }


def encode_vision_state(state: VisionStateDict) -> JSONObject:
    """Encode VisionStateDict to JSON.

    Args:
        state: State to encode.

    Returns:
        JSON-serializable dict.
    """
    tank_reg: JSONValue = {
        k: encode_tank_registry_entry(v) for k, v in state["tank_registry"].items()
    }
    pos_cache: JSONValue = {k: encode_position_entry(v) for k, v in state["position_cache"].items()}
    cont_cache: JSONValue = {
        k: encode_container_entry(v) for k, v in state["container_cache"].items()
    }
    return {
        "tank_registry": tank_reg,
        "position_cache": pos_cache,
        "container_cache": cont_cache,
        "self_fuel": state["self_fuel"],
        "self_tank_id": state["self_tank_id"],
    }


# =============================================================================
# Decode Functions
# =============================================================================


def decode_tank_registry_entry(data: JSONObject) -> TankRegistryEntryDict:
    """Decode TankRegistryEntryDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated TankRegistryEntryDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return TankRegistryEntryDict(
        tank_id=require_int(data, "tank_id"),
        name=require_str(data, "name"),
        team=require_int(data, "team"),
    )


def decode_position_entry(data: JSONObject) -> PositionEntryDict:
    """Decode PositionEntryDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated PositionEntryDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return PositionEntryDict(
        tank_id=require_int(data, "tank_id"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
    )


def decode_container_entry(data: JSONObject) -> ContainerEntryDict:
    """Decode ContainerEntryDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ContainerEntryDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return ContainerEntryDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        volume=require_int(data, "volume"),
    )


def _decode_dict_registry(raw: JSONValue) -> dict[str, TankRegistryEntryDict]:
    """Decode tank registry dict from raw JSON."""
    result: dict[str, TankRegistryEntryDict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                result[k] = decode_tank_registry_entry(v)
    return result


def _decode_dict_positions(raw: JSONValue) -> dict[str, PositionEntryDict]:
    """Decode position cache dict from raw JSON."""
    result: dict[str, PositionEntryDict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                result[k] = decode_position_entry(v)
    return result


def _decode_dict_containers(raw: JSONValue) -> dict[str, ContainerEntryDict]:
    """Decode container cache dict from raw JSON."""
    result: dict[str, ContainerEntryDict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                result[k] = decode_container_entry(v)
    return result


def decode_vision_state(data: JSONObject) -> VisionStateDict:
    """Decode VisionStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated VisionStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return VisionStateDict(
        tank_registry=_decode_dict_registry(data.get("tank_registry")),
        position_cache=_decode_dict_positions(data.get("position_cache")),
        container_cache=_decode_dict_containers(data.get("container_cache")),
        self_fuel=require_int(data, "self_fuel"),
        self_tank_id=require_int(data, "self_tank_id"),
    )


# =============================================================================
# Vision State Mutations (Immutable)
# =============================================================================


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
            merged[key] = ContainerStateDict(
                x=entry["x"],
                y=entry["y"],
                is_fuel=is_fuel,
                volume=entry["volume"],
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
    return render_world_state_ascii()


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
