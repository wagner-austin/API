"""Tank state TypedDict + factory + encode/decode."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.state.types.constants import EntitySource, require_entity_source


class TankStateDict(TypedDict):
    """State of a single tank in the game world.

    Attributes:
        tank_id: Unique identifier for this tank.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0=red, 1=purple, 2=blue, 3=orange).
        rank: Military rank (0-7).
        damage_state: Health state (0=full, 1=light, 2=medium, 3=critical).
        direction: Sprite direction byte. Low nibble (0-15) = facing
            heading, high nibble carries state flags. Bit 5 (value 32)
            is the DEAD flag — the game client sets direction to 32 or
            33 on deactivation (tpclient.js ``Pg.prototype.h``). Check
            ``direction >= 32`` to detect dead/corpse tanks. Verified
            across 42 corpse transitions in capture data (2026-06-18).
        name: Player name.
        is_bot: Whether this is a bot player.
        is_self: Whether this is the player's own tank.
        source: Which observed source most recently confirmed this tank.
        timestamp_ms: When this tank was last confirmed by ANY source
            (map blob, movement response, viewport, or radar). Used for
            acquisition freshness — finding enemies to teleport toward.
            Advanced by map updates, so a departed tank the map still
            lists stays "fresh" here; that is intentional for
            navigation and is why it must NOT gate the kill shot.
        last_wire_seen_ms: When a WIRE-PRESENCE source last vouched for
            this tank actually being present (viewport, radar, movement
            response, enemy detection). Map blob updates deliberately do
            NOT advance it: a tank truly present talks on the wire
            (raw-capture 2026-06-13: a live tank emits a wire message
            every few seconds; a departed afterimage goes silent for
            minutes while the map keeps re-listing it). This is the
            kill-shot gate — only fire at a tank with recent wire
            presence, never at a map-only afterimage.
    """

    tank_id: int
    x: int
    y: int
    team: int
    rank: int
    damage_state: int
    direction: int
    name: str
    is_bot: bool
    is_self: bool
    source: EntitySource
    timestamp_ms: int
    last_wire_seen_ms: int


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
    source: EntitySource = "viewport",
    timestamp_ms: int = 0,
    last_wire_seen_ms: int = 0,
    direction: int = 0,
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
        timestamp_ms: When this tank was last confirmed by any source.
        last_wire_seen_ms: When a wire-presence source last vouched for
            this tank's presence. Zero means never wire-confirmed.
        direction: Sprite direction byte. 0-31 = alive facing,
            32-33 = dead corpse.

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
        direction=direction,
        name=name,
        is_bot=is_bot,
        is_self=is_self,
        source=source,
        timestamp_ms=timestamp_ms,
        last_wire_seen_ms=last_wire_seen_ms,
    )


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
        "direction": state["direction"],
        "name": state["name"],
        "is_bot": state["is_bot"],
        "is_self": state["is_self"],
        "source": state["source"],
        "timestamp_ms": state["timestamp_ms"],
        "last_wire_seen_ms": state["last_wire_seen_ms"],
    }


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
        direction=require_int(data, "direction"),
        name=require_str(data, "name"),
        is_bot=require_bool(data, "is_bot"),
        is_self=require_bool(data, "is_self"),
        source=require_entity_source(data, "source"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        last_wire_seen_ms=require_int(data, "last_wire_seen_ms"),
    )


__all__ = [
    "TankStateDict",
    "decode_tank_state",
    "encode_tank_state",
    "make_tank_state",
]
