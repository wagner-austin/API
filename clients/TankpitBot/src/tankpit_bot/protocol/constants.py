"""Protocol constants and enumerations.

This module contains all protocol-level constants including message type codes,
enumerations for game concepts (rank, team, equipment), and shared values.
"""

from __future__ import annotations

from enum import IntEnum


class Rank(IntEnum):
    """Tank rank levels (0-8)."""

    RECRUIT = 0
    PRIVATE = 1
    CORPORAL = 2
    SERGEANT = 3
    LIEUTENANT = 4
    CAPTAIN = 5
    MAJOR = 6
    COLONEL = 7
    GENERAL = 8


# Starting fuel by rank (resets on death/respawn)
RANK_FUEL: dict[Rank, int] = {
    Rank.RECRUIT: 1000,
    Rank.PRIVATE: 1100,
    Rank.CORPORAL: 1200,
    Rank.SERGEANT: 1300,
    Rank.LIEUTENANT: 1400,
    Rank.CAPTAIN: 1500,
    Rank.MAJOR: 1600,
    Rank.COLONEL: 1700,
    Rank.GENERAL: 1800,
}


# Canonical team names indexed by team number (from JS client jb array)
TEAM_NAMES: tuple[str, ...] = ("red", "purple", "blue", "orange")

# Canonical rank names indexed by rank number (from JS client ec array)
RANK_NAMES: tuple[str, ...] = (
    "recruit",
    "private",
    "corporal",
    "sergeant",
    "lieutenant",
    "captain",
    "major",
    "colonel",
    "general",
)

# Damage state names indexed by damage value. The tier COUNTS DOWN
# toward deactivation: live run 20260610-231x recorded every fight as
# 0 -> 3 -> 2 -> 1 under sustained fire, and all five kills with tier
# data died from tier 1.
DAMAGE_NAMES: tuple[str, ...] = ("full", "critical", "medium", "light")


class Team(IntEnum):
    """Team colors (0-3)."""

    RED = 0
    PURPLE = 1
    BLUE = 2
    ORANGE = 3


class Equipment(IntEnum):
    """Equipment types (0-4)."""

    ARMOR_SHIELD = 0
    DUAL_SHOT = 1
    MISSILE_SHOT = 2
    HOMING_SHOT = 3
    EXTRA_RADAR = 4


class TerrainType(IntEnum):
    """Terrain types from JS rendering code."""

    GROUND = 0
    ROCK_A = 1
    ROCK_B = 2
    ROCK_AB = 3
    FERRY = 5
    FERRY_ROCK = 7


# Message type characters
MSG_TANK_STATS = ord(".")
MSG_TANK_INFO = ord("!")
MSG_TANK_POS = ord("=")
MSG_MOVEMENT = ord("G")
MSG_SHOOT = ord("S")
MSG_DEACTIVATE = ord("A")
MSG_FUEL_GAIN = ord("D")
MSG_FUEL_DEPOSIT = ord("d")
MSG_RADAR_RESULT = ord("F")
MSG_ENEMY_DETECT = ord("H")
MSG_INVENTORY = ord("I")
MSG_EQUIP_GAIN = ord("g")
MSG_EQUIP_TOGGLE = ord("t")
MSG_MINE_PLACE = ord("K")
MSG_MINE_DETONATE = ord("E")
MSG_CHAT = ord("M")
MSG_TANK_REMOVE = ord("X")
MSG_MAP_DATA = ord("L")
MSG_MAP_UPDATE = ord("Z")
MSG_TANK_ENTRY = ord("(")
MSG_TANK_EXIT = ord(")")
MSG_TANK_STATUS = ord(">")
MSG_PROMOTION = ord("+")
MSG_DECORATION = ord("N")
MSG_STATISTICS = ord("V")
MSG_ACTIVE_FORCES = ord("*")
MSG_ACTIVE_PLAYERS = ord("/")
MSG_TOP10 = ord("1")
MSG_CACHE_OVERLAY_UPDATE = ord("O")
MSG_BUILD_PICKUP = ord("B")
MSG_ACTION_DONE = ord("T")
MSG_OVERLAY_UPDATE = ord("@")
MSG_TERRAIN_UPDATE = ord("J")
MSG_PING = ord("`")
MSG_DISCONNECT = ord("~")
MSG_SUPERVISOR = ord("R")
MSG_TANK_STATUS_FULL = ord(">")
MSG_VIEWPORT = ord("Z")
MSG_SYNC = ord("?")
MSG_CACHE_UPDATE = ord("C")
MSG_MOVE_RESPONSE = ord("=")

# Supervisor error codes (0x52 'R' message data field).
# Named from tpclient.js Gb[] array. The message is the server's
# command failure response, NOT a promotion/kill signal.
SUPERVISOR_ERROR_CANT_DO = 0
SUPERVISOR_ERROR_CANT_GO = 1
SUPERVISOR_ERROR_UNCONTROLLABLE = 2
SUPERVISOR_ERROR_FRIENDLY_FIRE = 3
SUPERVISOR_ERROR_EMPTY_CONTAINER = 4
SUPERVISOR_ERROR_TANK_FULL = 5
SUPERVISOR_ERROR_ALREADY_THERE = 6
SUPERVISOR_ERROR_INVENTORY_FULL = 7
SUPERVISOR_ERROR_INSUFFICIENT_FUEL = 8
SUPERVISOR_ERROR_NO_ENEMIES = 9
SUPERVISOR_ERROR_CONGRATULATIONS = 10

# Text message types that don't use XOR encoding
TEXT_MSG_TYPES = frozenset(
    {
        ord("="),
        ord("+"),
        ord("%"),
        ord("*"),
        ord("$"),
        ord("-"),
        ord("~"),
        ord("`"),
        ord("R"),
    }
)


def is_text_message(msg_type: int) -> bool:
    """Check if a message type uses text format (not XOR encoded).

    Args:
        msg_type: Message type byte.

    Returns:
        True if message uses text format.
    """
    return msg_type in TEXT_MSG_TYPES


__all__ = [
    "DAMAGE_NAMES",
    "MSG_ACTION_DONE",
    "MSG_ACTIVE_FORCES",
    "MSG_ACTIVE_PLAYERS",
    "MSG_BUILD_PICKUP",
    "MSG_CACHE_OVERLAY_UPDATE",
    "MSG_CACHE_UPDATE",
    "MSG_CHAT",
    "MSG_DEACTIVATE",
    "MSG_DECORATION",
    "MSG_DISCONNECT",
    "MSG_ENEMY_DETECT",
    "MSG_EQUIP_GAIN",
    "MSG_EQUIP_TOGGLE",
    "MSG_FUEL_DEPOSIT",
    "MSG_FUEL_GAIN",
    "MSG_INVENTORY",
    "MSG_MAP_DATA",
    "MSG_MAP_UPDATE",
    "MSG_MINE_DETONATE",
    "MSG_MINE_PLACE",
    "MSG_MOVEMENT",
    "MSG_MOVE_RESPONSE",
    "MSG_OVERLAY_UPDATE",
    "MSG_PING",
    "MSG_PROMOTION",
    "MSG_RADAR_RESULT",
    "MSG_SHOOT",
    "MSG_STATISTICS",
    "MSG_SUPERVISOR",
    "MSG_SYNC",
    "MSG_TANK_ENTRY",
    "MSG_TANK_EXIT",
    "MSG_TANK_INFO",
    "MSG_TANK_POS",
    "MSG_TANK_REMOVE",
    "MSG_TANK_STATS",
    "MSG_TANK_STATUS",
    "MSG_TANK_STATUS_FULL",
    "MSG_TERRAIN_UPDATE",
    "MSG_TOP10",
    "MSG_VIEWPORT",
    "RANK_FUEL",
    "RANK_NAMES",
    "SUPERVISOR_ERROR_ALREADY_THERE",
    "SUPERVISOR_ERROR_CANT_DO",
    "SUPERVISOR_ERROR_CANT_GO",
    "SUPERVISOR_ERROR_CONGRATULATIONS",
    "SUPERVISOR_ERROR_EMPTY_CONTAINER",
    "SUPERVISOR_ERROR_FRIENDLY_FIRE",
    "SUPERVISOR_ERROR_INSUFFICIENT_FUEL",
    "SUPERVISOR_ERROR_INVENTORY_FULL",
    "SUPERVISOR_ERROR_NO_ENEMIES",
    "SUPERVISOR_ERROR_TANK_FULL",
    "SUPERVISOR_ERROR_UNCONTROLLABLE",
    "TEAM_NAMES",
    "TEXT_MSG_TYPES",
    "Equipment",
    "Rank",
    "Team",
    "TerrainType",
    "is_text_message",
]
