"""Aggregate world-state TypedDict + factory + encode/decode.

The world state composes every other entity collection. Decoding uses
the lifted ``decode_entity_dict`` helper so the per-collection iteration
logic stays in one place.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_dict,
    require_int,
)
from typing_extensions import TypedDict

from tankpit_bot.state.types._helpers import decode_entity_dict
from tankpit_bot.state.types.container import (
    ContainerStateDict,
    decode_container_state,
    encode_container_state,
)
from tankpit_bot.state.types.mine import MineStateDict, decode_mine_state, encode_mine_state
from tankpit_bot.state.types.self_state import (
    SelfStateDict,
    decode_self_state,
    encode_self_state,
)
from tankpit_bot.state.types.tank import TankStateDict, decode_tank_state, encode_tank_state
from tankpit_bot.state.types.terrain import (
    TerrainTileDict,
    decode_terrain_tile,
    encode_terrain_tile,
)
from tankpit_bot.state.types.viewport import (
    ViewportStateDict,
    decode_viewport_state,
    encode_viewport_state,
)


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


def _decode_timestamp_dict(data: JSONObject, field: str) -> dict[str, int]:
    """Validate and decode a string-keyed timestamp mapping field.

    Used for ``scanned_viewports`` ("left,top" keys).

    Args:
        data: World-state JSON object.
        field: Field name holding the mapping.

    Returns:
        Mapping of string keys to timestamps.

    Raises:
        JSONTypeError: If any value is not an integer.
    """
    raw = require_dict(data, field)
    result: dict[str, int] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise JSONTypeError(f"{field}.{key} must be an integer")
        result[key] = value
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

    return WorldStateDict(
        self_state=self_state,
        tanks=decode_entity_dict(data.get("tanks"), decode_tank_state),
        containers=decode_entity_dict(data.get("containers"), decode_container_state),
        mines=decode_entity_dict(data.get("mines"), decode_mine_state),
        terrain=decode_entity_dict(data.get("terrain"), decode_terrain_tile),
        viewport=decode_viewport_state(viewport_raw),
        scanned_viewports=_decode_timestamp_dict(data, "scanned_viewports"),
        timestamp_ms=require_int(data, "timestamp_ms"),
    )


__all__ = [
    "WorldStateDict",
    "decode_world_state",
    "encode_world_state",
    "make_empty_world_state",
]
