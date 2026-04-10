"""World state TypedDicts, constants, and serialization.

This module provides TypedDicts for representing game world state elements,
along with factory functions and JSON encode/decode functions.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_dict,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

# =============================================================================
# Constants
# =============================================================================

# Terrain type values from ViewportUpdate
TERRAIN_GROUND = 0
TERRAIN_ROCK_A = 1
TERRAIN_ROCK_B = 2
TERRAIN_ROCK_AB = 3
TERRAIN_FERRY = 5
TERRAIN_FERRY_ROCK = 7

# Team IDs
TEAM_RED = 0
TEAM_PURPLE = 1
TEAM_BLUE = 2
TEAM_ORANGE = 3

# Damage states (0 = full HP, 3 = critical)
DAMAGE_FULL = 0
DAMAGE_LIGHT = 1
DAMAGE_MEDIUM = 2
DAMAGE_CRITICAL = 3

# ASCII representation characters
ASCII_GROUND = "."
ASCII_ROCK = "#"
ASCII_FERRY = "~"
ASCII_WATER = "W"
ASCII_FUEL = "F"
ASCII_EQUIPMENT = "E"
ASCII_MINE = "*"
ASCII_SELF = "@"
ASCII_ENEMY = "T"
ASCII_ALLY = "A"
ASCII_UNKNOWN = "?"

# Explicit entity observation sources.
ENTITY_SOURCES: tuple[str, ...] = (
    "viewport",
    "radar",
    "world_state",
)


def _require_entity_source(
    data: JSONObject,
    key: str,
) -> Literal["viewport", "radar", "world_state"]:
    """Validate and extract an entity source from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated entity source value.

    Raises:
        JSONTypeError: If the value is not a supported entity source.
    """
    raw = require_str(data, key)
    if raw == "viewport":
        return "viewport"
    if raw == "radar":
        return "radar"
    if raw == "world_state":
        return "world_state"
    raise JSONTypeError(f"{key} must be one of {ENTITY_SOURCES}, got {raw!r}")


# =============================================================================
# TypedDicts for World State Elements
# =============================================================================


class TankStateDict(TypedDict):
    """State of a single tank in the game world.

    Attributes:
        tank_id: Unique identifier for this tank.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0=red, 1=purple, 2=blue, 3=orange).
        rank: Military rank (0-7).
        damage_state: Health state (0=full, 1=light, 2=medium, 3=critical).
        name: Player name.
        is_bot: Whether this is a bot player.
        is_self: Whether this is the player's own tank.
        source: Which observed source most recently confirmed this tank.
        timestamp_ms: When this tank was last confirmed by the server
            (world state, movement response, or radar). Used for
            freshness-based combat target selection.
    """

    tank_id: int
    x: int
    y: int
    team: int
    rank: int
    damage_state: int
    name: str
    is_bot: bool
    is_self: bool
    source: Literal["viewport", "radar", "world_state"]
    timestamp_ms: int


class ContainerStateDict(TypedDict):
    """State of a fuel or equipment container.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        is_fuel: True if fuel container, False if equipment.
        volume: Fuel amount (0 for equipment).
        source: Which observed source most recently confirmed this container.
        timestamp_ms: When this container was last confirmed by the
            server (radar, viewport, or world state). Used for
            freshness-based target selection.
        failed_pickups: How many pickup attempts failed for this
            container. Incremented on stall timeout, reset when the
            container is re-confirmed by a fresh source.
    """

    x: int
    y: int
    is_fuel: bool
    volume: int
    source: Literal["viewport", "radar", "world_state"]
    timestamp_ms: int
    failed_pickups: int


class MineStateDict(TypedDict):
    """State of a placed mine.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        mine_type: Type of mine (from protocol). 0 if unknown (radar-discovered).
        tank_id: ID of tank that placed the mine. -1 if unknown (radar-discovered).
        team: Team that owns the mine (0=red, 1=purple, 2=blue, 3=orange).
        source: Which observed source most recently confirmed this mine.
        timestamp_ms: When this mine was last confirmed by the server.
    """

    x: int
    y: int
    mine_type: int
    tank_id: int
    team: int
    source: Literal["viewport", "radar", "world_state"]
    timestamp_ms: int


class TerrainTileDict(TypedDict):
    """State of a terrain tile.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        terrain_type: Terrain/structure type (0=ground, 1-3=rock variants, 5=ferry, 7=ferry+rock).
        cache_value: Tile cache value (0=none, -1=equipment, >0=fuel volume).
        overlay_value: Tile overlay value (255=clear, other values protocol-defined).
    """

    x: int
    y: int
    terrain_type: int
    cache_value: int
    overlay_value: int


class ViewportStateDict(TypedDict):
    """Current viewport state.

    Attributes:
        left: Left edge X coordinate of viewport.
        top: Top edge Y coordinate of viewport.
        width: Visible viewport width in tiles (typically 16).
        height: Visible viewport height in tiles (typically 16).
    """

    left: int
    top: int
    width: int
    height: int


class SelfStateDict(TypedDict):
    """State of the player's own tank.

    Attributes:
        tank_id: Player's tank ID.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0-7).
        fuel: Current fuel (also health).
        leaderboard_position: Position on leaderboard.
    """

    tank_id: int
    x: int
    y: int
    team: int
    rank: int
    fuel: int
    leaderboard_position: int


class WorldStateDict(TypedDict):
    """Complete world state aggregated from protocol messages.

    Attributes:
        self_state: Player's own tank state.
        tanks: All known tanks indexed by tank_id.
        containers: All known containers indexed by "x,y" string key.
        mines: All known mines indexed by "x,y" string key.
        terrain: Terrain tiles indexed by "x,y" string key.
        viewport: Current viewport bounds.
        scanned_viewports: Viewport origins confirmed by authoritative local
            resource data, indexed by "left,top" string key with
            timestamp_ms values. Confirmation can come from a radar response
            or a fresh visible viewport tile update. Used to distinguish
            authoritative local resource truth from stale remembered cache
            observations.
        timestamp_ms: Last update timestamp in milliseconds.
    """

    self_state: SelfStateDict | None
    tanks: dict[str, TankStateDict]
    containers: dict[str, ContainerStateDict]
    mines: dict[str, MineStateDict]
    terrain: dict[str, TerrainTileDict]
    viewport: ViewportStateDict
    scanned_viewports: dict[str, int]
    timestamp_ms: int


# =============================================================================
# Factory Functions
# =============================================================================


def make_empty_world_state() -> WorldStateDict:
    """Create an empty world state.

    Returns:
        Empty WorldStateDict with default viewport.
    """
    return WorldStateDict(
        self_state=None,
        tanks={},
        containers={},
        mines={},
        terrain={},
        viewport=ViewportStateDict(left=0, top=0, width=16, height=16),
        scanned_viewports={},
        timestamp_ms=0,
    )


def make_tank_state(
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
    damage_state: int,
    name: str,
    is_bot: bool,
    is_self: bool,
    source: Literal["viewport", "radar", "world_state"] = "viewport",
    timestamp_ms: int = 0,
) -> TankStateDict:
    """Create a tank state.

    Args:
        tank_id: Unique tank identifier.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0-7).
        damage_state: Health state (0-3).
        name: Player name.
        is_bot: Whether this is a bot.
        is_self: Whether this is the player's tank.
        source: Which observed source confirmed this tank.
        timestamp_ms: When this tank was last confirmed.

    Returns:
        TankStateDict with the provided values.
    """
    return TankStateDict(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        damage_state=damage_state,
        name=name,
        is_bot=is_bot,
        is_self=is_self,
        source=source,
        timestamp_ms=timestamp_ms,
    )


def make_container_state(
    x: int,
    y: int,
    is_fuel: bool,
    volume: int,
    source: Literal["viewport", "radar", "world_state"] = "radar",
    timestamp_ms: int = 0,
    failed_pickups: int = 0,
) -> ContainerStateDict:
    """Create a container state.

    Args:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        is_fuel: True if fuel, False if equipment.
        volume: Fuel amount (0 for equipment).
        source: Which observed source confirmed this container.
        timestamp_ms: When this container was confirmed.
        failed_pickups: How many pickup attempts failed.

    Returns:
        ContainerStateDict with the provided values.
    """
    return ContainerStateDict(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=volume,
        source=source,
        timestamp_ms=timestamp_ms,
        failed_pickups=failed_pickups,
    )


def make_mine_state(
    x: int,
    y: int,
    mine_type: int,
    tank_id: int,
    team: int,
    source: Literal["viewport", "radar", "world_state"] = "viewport",
    timestamp_ms: int = 0,
) -> MineStateDict:
    """Create a mine state.

    Args:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        mine_type: Type of mine. 0 if unknown (radar-discovered).
        tank_id: ID of placing tank. -1 if unknown (radar-discovered).
        team: Team that owns the mine (0=red, 1=purple, 2=blue, 3=orange).
        source: Which observed source confirmed this mine.
        timestamp_ms: When this mine was confirmed.

    Returns:
        MineStateDict with the provided values.
    """
    return MineStateDict(
        x=x,
        y=y,
        mine_type=mine_type,
        tank_id=tank_id,
        team=team,
        source=source,
        timestamp_ms=timestamp_ms,
    )


def make_terrain_tile(
    x: int,
    y: int,
    terrain_type: int,
    cache_value: int,
    overlay_value: int,
) -> TerrainTileDict:
    """Create a terrain tile.

    Args:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        terrain_type: Terrain type (0-7).
        cache_value: Tile cache value (0=none, -1=equipment, >0=fuel volume).
        overlay_value: Tile overlay value (255=clear).

    Returns:
        TerrainTileDict with the provided values.
    """
    return TerrainTileDict(
        x=x,
        y=y,
        terrain_type=terrain_type,
        cache_value=cache_value,
        overlay_value=overlay_value,
    )


def make_self_state(
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
    fuel: int,
    leaderboard_position: int,
) -> SelfStateDict:
    """Create self state.

    Args:
        tank_id: Player's tank ID.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0-7).
        fuel: Current fuel amount.
        leaderboard_position: Leaderboard position.

    Returns:
        SelfStateDict with the provided values.
    """
    return SelfStateDict(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        fuel=fuel,
        leaderboard_position=leaderboard_position,
    )


# =============================================================================
# Coordinate Key Helpers
# =============================================================================


def coord_key(x: int, y: int) -> str:
    """Create a coordinate key string for dict indexing.

    Args:
        x: X coordinate.
        y: Y coordinate.

    Returns:
        String key in format "x,y".
    """
    return f"{x},{y}"


def parse_coord_key(key: str) -> tuple[int, int]:
    """Parse a coordinate key string.

    Args:
        key: String key in format "x,y".

    Returns:
        Tuple of (x, y) coordinates.

    Raises:
        ValueError: If key format is invalid.
    """
    parts = key.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid coord key format: {key}")
    return int(parts[0]), int(parts[1])


def viewport_scan_key(left: int, top: int) -> str:
    """Create a viewport-origin key string for scan coverage indexing.

    Args:
        left: Viewport left X coordinate.
        top: Viewport top Y coordinate.

    Returns:
        String key in format "left,top".
    """
    return f"{left},{top}"


# =============================================================================
# Encode Functions
# =============================================================================


def encode_tank_state(state: TankStateDict) -> JSONObject:
    """Encode TankStateDict to JSON-serializable dict.

    Args:
        state: TankStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "tank_id": state["tank_id"],
        "x": state["x"],
        "y": state["y"],
        "team": state["team"],
        "rank": state["rank"],
        "damage_state": state["damage_state"],
        "name": state["name"],
        "is_bot": state["is_bot"],
        "is_self": state["is_self"],
        "source": state["source"],
        "timestamp_ms": state["timestamp_ms"],
    }


def encode_container_state(state: ContainerStateDict) -> JSONObject:
    """Encode ContainerStateDict to JSON-serializable dict.

    Args:
        state: ContainerStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "x": state["x"],
        "y": state["y"],
        "is_fuel": state["is_fuel"],
        "volume": state["volume"],
        "source": state["source"],
        "timestamp_ms": state["timestamp_ms"],
        "failed_pickups": state["failed_pickups"],
    }


def encode_mine_state(state: MineStateDict) -> JSONObject:
    """Encode MineStateDict to JSON-serializable dict.

    Args:
        state: MineStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "x": state["x"],
        "y": state["y"],
        "mine_type": state["mine_type"],
        "tank_id": state["tank_id"],
        "team": state["team"],
        "source": state["source"],
        "timestamp_ms": state["timestamp_ms"],
    }


def encode_terrain_tile(tile: TerrainTileDict) -> JSONObject:
    """Encode TerrainTileDict to JSON-serializable dict.

    Args:
        tile: TerrainTileDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "x": tile["x"],
        "y": tile["y"],
        "terrain_type": tile["terrain_type"],
        "cache_value": tile["cache_value"],
        "overlay_value": tile["overlay_value"],
    }


def encode_viewport_state(state: ViewportStateDict) -> JSONObject:
    """Encode ViewportStateDict to JSON-serializable dict.

    Args:
        state: ViewportStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "left": state["left"],
        "top": state["top"],
        "width": state["width"],
        "height": state["height"],
    }


def encode_self_state(state: SelfStateDict) -> JSONObject:
    """Encode SelfStateDict to JSON-serializable dict.

    Args:
        state: SelfStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "tank_id": state["tank_id"],
        "x": state["x"],
        "y": state["y"],
        "team": state["team"],
        "rank": state["rank"],
        "fuel": state["fuel"],
        "leaderboard_position": state["leaderboard_position"],
    }


def encode_world_state(state: WorldStateDict) -> JSONObject:
    """Encode WorldStateDict to JSON-serializable dict.

    Args:
        state: WorldStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "self_state": encode_self_state(state["self_state"]) if state["self_state"] else None,
        "tanks": {k: encode_tank_state(v) for k, v in state["tanks"].items()},
        "containers": {k: encode_container_state(v) for k, v in state["containers"].items()},
        "mines": {k: encode_mine_state(v) for k, v in state["mines"].items()},
        "terrain": {k: encode_terrain_tile(v) for k, v in state["terrain"].items()},
        "viewport": encode_viewport_state(state["viewport"]),
        "scanned_viewports": dict(state["scanned_viewports"]),
        "timestamp_ms": state["timestamp_ms"],
    }


# =============================================================================
# Decode Functions
# =============================================================================


def decode_tank_state(data: JSONObject) -> TankStateDict:
    """Decode TankStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated TankStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return TankStateDict(
        tank_id=require_int(data, "tank_id"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        team=require_int(data, "team"),
        rank=require_int(data, "rank"),
        damage_state=require_int(data, "damage_state"),
        name=require_str(data, "name"),
        is_bot=require_bool(data, "is_bot"),
        is_self=require_bool(data, "is_self"),
        source=_require_entity_source(data, "source"),
        timestamp_ms=require_int(data, "timestamp_ms"),
    )


def decode_container_state(data: JSONObject) -> ContainerStateDict:
    """Decode ContainerStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ContainerStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return ContainerStateDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        is_fuel=require_bool(data, "is_fuel"),
        volume=require_int(data, "volume"),
        source=_require_entity_source(data, "source"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        failed_pickups=require_int(data, "failed_pickups"),
    )


def decode_mine_state(data: JSONObject) -> MineStateDict:
    """Decode MineStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated MineStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return MineStateDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        mine_type=require_int(data, "mine_type"),
        tank_id=require_int(data, "tank_id"),
        team=require_int(data, "team"),
        source=_require_entity_source(data, "source"),
        timestamp_ms=require_int(data, "timestamp_ms"),
    )


def decode_terrain_tile(data: JSONObject) -> TerrainTileDict:
    """Decode TerrainTileDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated TerrainTileDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return TerrainTileDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        terrain_type=require_int(data, "terrain_type"),
        cache_value=require_int(data, "cache_value"),
        overlay_value=require_int(data, "overlay_value"),
    )


def decode_viewport_state(data: JSONObject) -> ViewportStateDict:
    """Decode ViewportStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ViewportStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return ViewportStateDict(
        left=require_int(data, "left"),
        top=require_int(data, "top"),
        width=require_int(data, "width"),
        height=require_int(data, "height"),
    )


def decode_self_state(data: JSONObject) -> SelfStateDict:
    """Decode SelfStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated SelfStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return SelfStateDict(
        tank_id=require_int(data, "tank_id"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        team=require_int(data, "team"),
        rank=require_int(data, "rank"),
        fuel=require_int(data, "fuel"),
        leaderboard_position=require_int(data, "leaderboard_position"),
    )


def _decode_dict_field_tanks(raw: JSONValue) -> dict[str, TankStateDict]:
    """Decode tanks dict from raw JSON value."""
    result: dict[str, TankStateDict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                result[k] = decode_tank_state(v)
    return result


def _decode_dict_field_containers(raw: JSONValue) -> dict[str, ContainerStateDict]:
    """Decode containers dict from raw JSON value."""
    result: dict[str, ContainerStateDict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                result[k] = decode_container_state(v)
    return result


def _decode_dict_field_mines(raw: JSONValue) -> dict[str, MineStateDict]:
    """Decode mines dict from raw JSON value."""
    result: dict[str, MineStateDict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                result[k] = decode_mine_state(v)
    return result


def _decode_dict_field_terrain(raw: JSONValue) -> dict[str, TerrainTileDict]:
    """Decode terrain dict from raw JSON value."""
    result: dict[str, TerrainTileDict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                result[k] = decode_terrain_tile(v)
    return result


def decode_world_state(data: JSONObject) -> WorldStateDict:
    """Decode WorldStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated WorldStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    self_state_raw = data.get("self_state")
    self_state: SelfStateDict | None = None
    if self_state_raw is not None and isinstance(self_state_raw, dict):
        self_state = decode_self_state(self_state_raw)

    viewport_raw = data.get("viewport")
    if not isinstance(viewport_raw, dict):
        raise JSONTypeError("viewport must be an object")
    scanned_viewports_raw = require_dict(data, "scanned_viewports")
    scanned_viewports: dict[str, int] = {}
    for key, value in scanned_viewports_raw.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise JSONTypeError(f"scanned_viewports.{key} must be an integer")
        scanned_viewports[key] = value

    return WorldStateDict(
        self_state=self_state,
        tanks=_decode_dict_field_tanks(data.get("tanks")),
        containers=_decode_dict_field_containers(data.get("containers")),
        mines=_decode_dict_field_mines(data.get("mines")),
        terrain=_decode_dict_field_terrain(data.get("terrain")),
        viewport=decode_viewport_state(viewport_raw),
        scanned_viewports=scanned_viewports,
        timestamp_ms=require_int(data, "timestamp_ms"),
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ASCII_ALLY",
    "ASCII_ENEMY",
    "ASCII_EQUIPMENT",
    "ASCII_FERRY",
    "ASCII_FUEL",
    "ASCII_GROUND",
    "ASCII_MINE",
    "ASCII_ROCK",
    "ASCII_SELF",
    "ASCII_UNKNOWN",
    "ASCII_WATER",
    "DAMAGE_CRITICAL",
    "DAMAGE_FULL",
    "DAMAGE_LIGHT",
    "DAMAGE_MEDIUM",
    "ENTITY_SOURCES",
    "TEAM_BLUE",
    "TEAM_ORANGE",
    "TEAM_PURPLE",
    "TEAM_RED",
    "TERRAIN_FERRY",
    "TERRAIN_FERRY_ROCK",
    "TERRAIN_GROUND",
    "TERRAIN_ROCK_A",
    "TERRAIN_ROCK_AB",
    "TERRAIN_ROCK_B",
    "ContainerStateDict",
    "MineStateDict",
    "SelfStateDict",
    "TankStateDict",
    "TerrainTileDict",
    "ViewportStateDict",
    "WorldStateDict",
    "coord_key",
    "decode_container_state",
    "decode_mine_state",
    "decode_self_state",
    "decode_tank_state",
    "decode_terrain_tile",
    "decode_viewport_state",
    "decode_world_state",
    "encode_container_state",
    "encode_mine_state",
    "encode_self_state",
    "encode_tank_state",
    "encode_terrain_tile",
    "encode_viewport_state",
    "encode_world_state",
    "make_container_state",
    "make_empty_world_state",
    "make_mine_state",
    "make_self_state",
    "make_tank_state",
    "make_terrain_tile",
    "parse_coord_key",
    "viewport_scan_key",
]
