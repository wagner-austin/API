"""TypedDicts, factories, and codecs for the bot vision system."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

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


__all__ = [
    "ContainerEntryDict",
    "PositionEntryDict",
    "TankRegistryEntryDict",
    "VisionStateDict",
    "decode_container_entry",
    "decode_position_entry",
    "decode_tank_registry_entry",
    "decode_vision_state",
    "encode_container_entry",
    "encode_position_entry",
    "encode_tank_registry_entry",
    "encode_vision_state",
    "make_container_entry",
    "make_empty_vision_state",
    "make_position_entry",
    "make_tank_registry_entry",
]
