"""Shared world-state constants and entity-source literals.

All numeric tile-type / team / damage / ASCII codes used throughout the
world-state, renderer, and decoder layers. Also hosts the strict
literal validators that translate string fields into TypedDict-friendly
``Literal`` types (entity source, container refresh kind).
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONObject, JSONTypeError, require_str

TERRAIN_GROUND = 0
TERRAIN_ROCK_A = 1
TERRAIN_ROCK_B = 2
TERRAIN_ROCK_AB = 3
TERRAIN_FERRY = 5
TERRAIN_FERRY_ROCK = 7

TEAM_RED = 0
TEAM_PURPLE = 1
TEAM_BLUE = 2
TEAM_ORANGE = 3

DAMAGE_FULL = 0
DAMAGE_LIGHT = 1
DAMAGE_MEDIUM = 2
DAMAGE_CRITICAL = 3

DIRECTION_DEAD_THRESHOLD = 32

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

ENTITY_SOURCES: tuple[str, ...] = (
    "viewport",
    "radar",
    "world_state",
)

TANK_LIVENESS_STATES: tuple[str, ...] = (
    "alive",
    "deactivated",
)

CONTAINER_REFRESH_KINDS: tuple[str, ...] = (
    "radar_response",
    "radar_cache_refresh",
    "radar_known_resources",
    "world_state",
)

EntitySource = Literal["viewport", "radar", "world_state"]
"""Coarse observed-source label attached to every entity TypedDict."""

TankLiveness = Literal["alive", "deactivated"]
"""Per-tank liveness state.

Two states, wire-driven:

* ``alive`` is the default. Any wire-sourced observation with a
  non-corpse direction flips a ``deactivated`` tank back to ``alive``
  (the respawn-then-move flow). MapData is always applied as a
  position update regardless of liveness.
* ``deactivated`` is the corpse window between 0x41 Deactivation and
  the tank being fully cleaned up. The bot must NOT acquire
  deactivated tanks; the tile renders a corpse for ~22 s
  (empirical, 2026-06-20). The wire path also flips a tank into
  ``deactivated`` when a 0x3D MovementResponse arrives with the
  corpse-direction sprite (direction >= 32, per JS Pg.prototype.h).

Note: 0x58 TankRemove does NOT set liveness="deactivated". 0x58 means
"server stopped broadcasting per-tank updates to this client" -- it
fires for actual deaths, but also when a tank simply leaves the
client's awareness radius (verified 2026-06-20: orange-5 got 5
TankRemove events across 2 actual kills, the other 3 were tracking
churn). The handler instead deletes the tank from the registry; the
next MapData / per-tank wire re-adds it at its current position.
"""

ContainerRefreshKind = Literal[
    "radar_response",
    "radar_cache_refresh",
    "radar_known_resources",
    "world_state",
]
"""Specific confirmation path that most recently refreshed a container."""


def require_entity_source(data: JSONObject, key: str) -> EntitySource:
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


def require_tank_liveness(data: JSONObject, key: str) -> TankLiveness:
    """Validate and extract a tank liveness value from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated tank liveness value.

    Raises:
        JSONTypeError: If the value is not a supported liveness state.
    """
    raw = require_str(data, key)
    if raw == "alive":
        return "alive"
    if raw == "deactivated":
        return "deactivated"
    raise JSONTypeError(f"{key} must be one of {TANK_LIVENESS_STATES}, got {raw!r}")


def decode_container_refresh_kind(data: JSONObject, key: str) -> ContainerRefreshKind:
    """Validate and extract a container refresh kind from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated container refresh kind.

    Raises:
        JSONTypeError: If the value is not a supported refresh kind.
    """
    raw = require_str(data, key)
    if raw == "radar_response":
        return "radar_response"
    if raw == "radar_cache_refresh":
        return "radar_cache_refresh"
    if raw == "radar_known_resources":
        return "radar_known_resources"
    if raw == "world_state":
        return "world_state"
    raise JSONTypeError(f"{key} must be one of {CONTAINER_REFRESH_KINDS}, got {raw!r}")


def encode_container_refresh_kind(kind: ContainerRefreshKind) -> str:
    """Encode a container refresh kind.

    Args:
        kind: Refresh kind to encode.

    Returns:
        JSON string value for the refresh kind.
    """
    return kind


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
    "CONTAINER_REFRESH_KINDS",
    "DAMAGE_CRITICAL",
    "DAMAGE_FULL",
    "DAMAGE_LIGHT",
    "DAMAGE_MEDIUM",
    "DIRECTION_DEAD_THRESHOLD",
    "ENTITY_SOURCES",
    "TANK_LIVENESS_STATES",
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
    "ContainerRefreshKind",
    "EntitySource",
    "TankLiveness",
    "decode_container_refresh_kind",
    "encode_container_refresh_kind",
    "require_entity_source",
    "require_tank_liveness",
]
