"""Mine state TypedDict + factory + encode/decode."""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_int
from typing_extensions import TypedDict

from tankpit_bot.state.types.constants import EntitySource, require_entity_source


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
    source: EntitySource
    timestamp_ms: int


def make_mine_state(
    x: int,
    y: int,
    mine_type: int,
    tank_id: int,
    team: int,
    source: EntitySource = "viewport",
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
        source=require_entity_source(data, "source"),
        timestamp_ms=require_int(data, "timestamp_ms"),
    )


__all__ = [
    "MineStateDict",
    "decode_mine_state",
    "encode_mine_state",
    "make_mine_state",
]
