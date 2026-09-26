"""Shared world-state constants and entity vocabularies.

All numeric tile-type / team / damage / ASCII codes used throughout the
world-state, renderer, and decoder layers. Also hosts the closed
vocabularies entity TypedDicts carry (entity source, tank liveness,
container refresh kind) as StrEnums, which decoders narrow with
:func:`platform_core.members.require_member`.
"""

from __future__ import annotations

from enum import StrEnum

# Wire terrain vocabulary (0x42 obstacle_type ≡ 0x4A tile value ≡ 0x5A
# terrain_type — one enum, see wiki [[movable-blocks]]). Values 1-3 are
# movable concrete blocks, pinned 2026-07-20 by manual captures plus a
# 228-session archive sweep: value 1 appears ONLY over static water
# (walkable bridge), 2 ONLY over static ground (obstacle), 3 ONLY over
# static water (stacked, impassable). Formerly misnamed ROCK_A/B/AB.
TERRAIN_GROUND = 0
TERRAIN_BLOCK_BRIDGE = 1
TERRAIN_BLOCK_LAND = 2
TERRAIN_BLOCK_STACKED = 3
TERRAIN_FERRY = 5
TERRAIN_FERRY_ROCK = 7

TEAM_RED = 0
TEAM_PURPLE = 1
TEAM_BLUE = 2
TEAM_ORANGE = 3

TROOP_COLOR_NAMES: tuple[str, ...] = ("red", "purple", "blue", "orange")
"""Team colors indexed BY team id -- ``TROOP_COLOR_NAMES[TEAM_BLUE]``
is ``"blue"``. One home for the color<->team mapping that three
callers need: the practice-bot roster classification (join-roster
ground truth: red-1 arrives team 0, purple-2 team 1, blue-7 team 2,
orange-1 team 3), the ``TANKPIT_TROOP`` selector the join flow sends
as the room-entry troop byte, and the fleet control page's color
dropdown. Order is the wire's, not a display preference: the index IS
the team id, so the tuple must never be re-sorted."""

# Wire damage tier = fuel quartile (corpus-fitted 2026-07-23, 19,658
# samples, zero exceptions; [[deactivation-format]]): tier 3 is the
# TOP quartile (healthy, lightest shade), tier 0 the bottom (near
# death). Tanks do not heal — fuel IS the health pool.
DAMAGE_CRITICAL = 0
DAMAGE_MEDIUM = 1
DAMAGE_LIGHT = 2
DAMAGE_FULL = 3

DIRECTION_DEAD_THRESHOLD = 32

ASCII_GROUND = "."
ASCII_ROCK = "#"
ASCII_BRIDGE = "="
ASCII_FERRY = "~"
ASCII_WATER = "W"
ASCII_FUEL = "F"
ASCII_EQUIPMENT = "E"
ASCII_MINE = "*"
ASCII_SELF = "@"
ASCII_ENEMY = "T"
ASCII_ALLY = "A"
ASCII_UNKNOWN = "?"


class EntitySource(StrEnum):
    """Coarse observed-source label attached to every entity TypedDict."""

    VIEWPORT = "viewport"
    RADAR = "radar"
    WORLD_STATE = "world_state"


class TankLiveness(StrEnum):
    """Per-tank liveness state.

    Two states, wire-driven:

    * ``ALIVE`` is the default. Any wire-sourced observation with a
      non-corpse direction flips a ``DEACTIVATED`` tank back to
      ``ALIVE`` (the respawn-then-move flow). MapData is always applied
      as a position update regardless of liveness.
    * ``DEACTIVATED`` is the corpse window between 0x41 Deactivation
      and the tank being fully cleaned up. The bot must NOT acquire
      deactivated tanks; the tile renders a corpse for ~22 s
      (empirical, 2026-06-20). The wire path also flips a tank into
      ``DEACTIVATED`` when a 0x3D MovementResponse arrives with the
      corpse-direction sprite (direction >= 32, per JS Pg.prototype.h).

    Note: 0x58 TankRemove does NOT set liveness to ``DEACTIVATED``.
    0x58 means "server stopped broadcasting per-tank updates to this
    client" -- it fires for actual deaths, but also when a tank simply
    leaves the client's awareness radius (verified 2026-06-20: orange-5
    got 5 TankRemove events across 2 actual kills, the other 3 were
    tracking churn). The handler instead deletes the tank from the
    registry; the next MapData / per-tank wire re-adds it at its current
    position.
    """

    ALIVE = "alive"
    DEACTIVATED = "deactivated"


class ContainerRefreshKind(StrEnum):
    """Specific confirmation path that most recently refreshed a container."""

    RADAR_RESPONSE = "radar_response"
    RADAR_CACHE_REFRESH = "radar_cache_refresh"
    RADAR_KNOWN_RESOURCES = "radar_known_resources"
    VIEWPORT_PATCH = "viewport_patch"
    WORLD_STATE = "world_state"
    FLEET_REPORT = "fleet_report"


__all__ = [
    "ASCII_ALLY",
    "ASCII_BRIDGE",
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
    "DIRECTION_DEAD_THRESHOLD",
    "TEAM_BLUE",
    "TEAM_ORANGE",
    "TEAM_PURPLE",
    "TEAM_RED",
    "TERRAIN_BLOCK_BRIDGE",
    "TERRAIN_BLOCK_LAND",
    "TERRAIN_BLOCK_STACKED",
    "TERRAIN_FERRY",
    "TERRAIN_FERRY_ROCK",
    "TERRAIN_GROUND",
    "TROOP_COLOR_NAMES",
    "ContainerRefreshKind",
    "EntitySource",
    "TankLiveness",
]
