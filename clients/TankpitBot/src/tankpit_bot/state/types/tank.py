"""Tank state TypedDict + factory + encode/decode."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from tankpit_bot.state.types.constants import (
    EntitySource,
    TankLiveness,
    require_entity_source,
    require_tank_liveness,
)


class TankStateDict(TypedDict):
    """State of a single tank in the game world.

    Three independent freshness timestamps lock the freshness model:

    * ``timestamp_ms`` advances on ANY observation source (wire OR map).
      Used to keep a tank in the registry as a HUNT acquisition
      candidate even when only the map snapshot has confirmed it.

    * ``last_wire_seen_ms`` advances only on WIRE-SOURCED observations
      (viewport, radar, movement response, enemy detection,
      TankStatusSync). Map snapshot updates deliberately do NOT advance
      it — a tank truly present talks on the wire; a departed
      afterimage goes silent on the wire while the map keeps re-listing
      it.

    * ``last_position_update_ms`` advances ONLY when an observation
      carries a fresh ``(x, y)`` value. Damage-only wire messages
      (TankStatusSync, TankStatusShort) refresh ``last_wire_seen_ms``
      but NOT this field. This is the kill-shot gate — only fire at a
      tank whose position is structurally proven recent, never at a
      stale registry entry being kept alive by status-only broadcasts.

    The three-timestamp model exists because the broadcast cadences
    differ by message kind. 0x2E TankStatusSync broadcasts globally
    every ~2 s for every active tank regardless of viewport, so a
    single "any wire activity" timestamp would never expire and the
    bot would keep firing at stale registry positions. Position-bearing
    messages (0x3D MovementResponse, 0x47 Movement, 0x28 TankEntry,
    container TankUpdate*) refresh on a slower viewport-bound cadence.

    Attributes:
        tank_id: Unique identifier for this tank.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0=red, 1=purple, 2=blue, 3=orange).
        rank: Military rank (0-7).
        damage_state: Health state (0=full, 1=light, 2=medium, 3=critical).
        direction: Sprite direction byte. Low nibble (0-15) = facing
            heading, high nibble carries state flags. Bit 5 (value 32)
            is the DEAD flag -- the game client sets direction to 32 or
            33 on deactivation (tpclient.js ``Pg.prototype.h``). Check
            ``direction >= 32`` to detect dead/corpse tanks. Verified
            across 42 corpse transitions in capture data (2026-06-18).
        name: Player name.
        is_bot: Whether this is a bot player.
        is_self: Whether this is the player's own tank.
        source: Which observed source most recently confirmed this tank.
        timestamp_ms: Wall-clock ms of the most recent observation by
            ANY source (wire OR map). Acquisition gate.
        last_wire_seen_ms: Wall-clock ms of the most recent
            wire-sourced observation. Wire-presence gate.
        last_position_update_ms: Wall-clock ms of the most recent
            wire-sourced observation that carried a fresh ``(x, y)``.
            Kill-shot gate.
        liveness: Three-state lifecycle gate. ``alive`` is the default.
            ``deactivated`` is set on 0x41 Deactivation -- the tank is a
            corpse on the tile for ~22 s until the server cleans it up
            with 0x58 TankRemove. ``removed`` is set on 0x58 -- the
            tile is empty and MapData entries for this id must be
            skipped (tombstone). Any per-tank wire (TankInfo,
            TankEntry, MovementResponse, TankStatusSync, Movement) flips
            a non-alive tank back to ``alive`` -- the respawn flow.
            ``analyze_threats`` filters to ``liveness == "alive"``;
            ``_combat_shoot`` thus cannot fire at a corpse or empty
            tile. Empirical capture 2026-06-20: bot used to shoot the
            corpse 3 times during the 22 s window because no 0x41
            handler updated the tank state.
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
    last_position_update_ms: int
    liveness: TankLiveness


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
    last_position_update_ms: int = 0,
    direction: int = 0,
    liveness: TankLiveness = "alive",
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
        timestamp_ms: Wall-clock ms of the most recent observation by
            ANY source. Zero means never observed.
        last_wire_seen_ms: Wall-clock ms of the most recent
            wire-sourced observation. Zero means never wire-confirmed.
        last_position_update_ms: Wall-clock ms of the most recent
            wire-sourced observation that carried fresh ``(x, y)``.
            Zero means the position has never been wire-confirmed.
        direction: Sprite direction byte. 0-31 = alive facing,
            32-33 = dead corpse.
        liveness: Three-state lifecycle gate. Defaults to ``alive``.
            See :class:`TankStateDict` for the full semantics.

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
        last_position_update_ms=last_position_update_ms,
        liveness=liveness,
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
        "last_position_update_ms": state["last_position_update_ms"],
        "liveness": state["liveness"],
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
        last_position_update_ms=require_int(data, "last_position_update_ms"),
        liveness=require_tank_liveness(data, "liveness"),
    )


__all__ = [
    "TankStateDict",
    "decode_tank_state",
    "encode_tank_state",
    "make_tank_state",
]
