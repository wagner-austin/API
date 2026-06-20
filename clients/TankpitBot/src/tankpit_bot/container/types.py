"""Container message types and enumerations.

This module contains all TypedDict definitions for container messages and
enumerations for message types and decode levels.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import Literal, TypedDict


class ContainerMessageType(IntEnum):
    """Types of messages found inside 0x2E containers.

    Identified by structure, not subtype byte value.
    """

    UNKNOWN = 0
    MINE_DETONATION = auto()
    MINE_PLACEMENT = auto()
    TANK_REGISTRY = auto()
    POSITION_UPDATE = auto()
    TANK_STATUS_SHORT = auto()
    TANK_UPDATE_COMPACT = auto()
    TANK_UPDATE_FULL = auto()
    TANK_LEAVE = auto()
    PLAYER_LIST_SHORT = auto()
    PLAYER_LIST_EXTENDED = auto()
    DEACTIVATION_DEATH = auto()
    TELEPORT_LANDED = auto()
    CONTAINER_PICKUP = auto()
    TIP_NOTIFICATION = auto()
    CHUNK_DATA = auto()
    WORLD_STATE = auto()


class DecodeLevel(IntEnum):
    """Decode understanding level for message types.

    Used to calculate decode coverage percentage in stats.
    The integer value represents the weight for coverage calculation.

    Levels:
        UNKNOWN: Message type not recognized (0 points).
        IDENTIFIED: Type known but fields not decoded (25 points).
        PARTIAL: Key fields decoded but some unknown (50 points).
        FULL: All fields fully decoded and understood (100 points).
    """

    UNKNOWN = 0
    IDENTIFIED = 25
    PARTIAL = 50
    FULL = 100


# =============================================================================
# Decode Level Registry
# =============================================================================
# Maps each message type to its decode understanding level.
# This is the single source of truth for stats coverage calculation.

MESSAGE_TYPE_LEVELS: dict[ContainerMessageType, DecodeLevel] = {
    ContainerMessageType.UNKNOWN: DecodeLevel.UNKNOWN,
    # Fully decoded message types (all fields understood)
    ContainerMessageType.MINE_DETONATION: DecodeLevel.FULL,
    ContainerMessageType.MINE_PLACEMENT: DecodeLevel.FULL,
    ContainerMessageType.TANK_REGISTRY: DecodeLevel.FULL,
    ContainerMessageType.POSITION_UPDATE: DecodeLevel.FULL,
    ContainerMessageType.TANK_STATUS_SHORT: DecodeLevel.FULL,
    ContainerMessageType.TANK_UPDATE_COMPACT: DecodeLevel.FULL,
    ContainerMessageType.TANK_UPDATE_FULL: DecodeLevel.FULL,
    ContainerMessageType.TANK_LEAVE: DecodeLevel.FULL,
    ContainerMessageType.PLAYER_LIST_SHORT: DecodeLevel.FULL,
    ContainerMessageType.PLAYER_LIST_EXTENDED: DecodeLevel.FULL,
    ContainerMessageType.DEACTIVATION_DEATH: DecodeLevel.FULL,
    ContainerMessageType.TELEPORT_LANDED: DecodeLevel.FULL,
    ContainerMessageType.CONTAINER_PICKUP: DecodeLevel.FULL,
    # Identified but not fully decoded (type known, structure partial)
    ContainerMessageType.TIP_NOTIFICATION: DecodeLevel.IDENTIFIED,
    ContainerMessageType.CHUNK_DATA: DecodeLevel.IDENTIFIED,
    ContainerMessageType.WORLD_STATE: DecodeLevel.IDENTIFIED,
}


def get_decode_level(msg_type: ContainerMessageType) -> DecodeLevel:
    """Get the decode understanding level for a message type.

    Returns the level from MESSAGE_TYPE_LEVELS registry.
    This is used by stats calculation to determine coverage percentage.

    Args:
        msg_type: The container message type to look up.

    Returns:
        The decode level for the message type.
        Returns DecodeLevel.UNKNOWN if type not in registry.
    """
    return MESSAGE_TYPE_LEVELS.get(msg_type, DecodeLevel.UNKNOWN)


# =============================================================================
# Combat Messages
# =============================================================================
# 0x53 ShootEvent lives in tankpit_bot.protocol (ShootEventDict). The
# container path is intentionally not duplicated -- the protocol path
# is the single source of truth (re-verified 2026-06-19 against JS
# Gg.h and capture bot-20260619-050303).


class MineDetonationDict(TypedDict):
    """Mine detonation decoded from 0x2E container.

    Structure (proven from captures):
      [subtype:1] [positions: repeated (x, y) bytes]
    """

    msg_type: Literal[0x45]
    positions: list[tuple[int, int]]


# 0x41 Deactivation lives in tankpit_bot.protocol (DeactivationDict).
# The container path was deleted 2026-06-19; routing min_len fix in
# protocol/decoders/routing.py (7 -> 6) ensures the protocol path now
# fires for the wire 6-byte body.


class DeactivationDeathDict(TypedDict):
    """Deactivation event when you were killed by another tank.

    Structure (7 bytes, verified from captures):
      [subtype:1] [flags:1] [killer_id:2 LE] [extra:3]

    Subtype is 0x43 ('C') after XOR decode.
    Sent when you are deactivated by another tank.
    """

    msg_type: Literal["deactivation_death"]
    flags: int
    killer_id: int
    extra_data: bytes


class MinePlacementDict(TypedDict):
    """Mine placement decoded from 0x2E container.

    Structure (15 bytes, proven from capture):
      [subtype:1] [mine_type:1] [tank_id:2 LE] [count:1] [positions: count*2]

    This is a tunneled mine placement payload carried inside a 0x2E frame.
    """

    msg_type: Literal[0x4B]
    mine_type: int
    tank_id: int
    positions: list[tuple[int, int]]


# =============================================================================
# Tank Messages
# =============================================================================


class TankRegistryDict(TypedDict):
    """Tank registry entry decoded from 0x2E container.

    Structure (16-20 bytes, verified from captures):
      [subtype:1] [flags:1] [tank_id:2 LE] [info_bytes:12-16]

    For tanks (tank_id < 1000):
      info_bytes structure:
        Standard (flags & 0x2C == 0): [rank_badges:1][zeros:4][unk:2][name]
        Extended (flags & 0x2C != 0): [rank_badges:1][zeros:4][pos:2][unk:3][name]

      rank_badges byte encoding:
        bits 0-2: military rank (0=recruit...7=colonel, overflow for general)
        bits 3-7: badge/award count

    For containers (tank_id >= 1000):
      Equipment/fuel containers on the map are encoded as "tanks".
      info_bytes structure: [y:1][viewport_x:1][type_data:10-14]
      - y is absolute map coordinate
      - viewport_x is relative to player position (player at center ~3)
      - To get absolute x: map_x = player_x + (viewport_x - 3)

    Team encoded in flags lower 2 bits: 0=red, 1=purple, 2=blue, 3=orange.
    """

    msg_type: Literal["tank_registry"]
    flags: int
    tank_id: int
    info_bytes: bytes
    team: str
    tank_name: str
    military_rank: int
    badge_count: int
    is_bot: bool
    is_container: bool
    container_x: int | None  # Absolute x - requires player position to calculate
    container_y: int | None  # Absolute y coordinate
    container_viewport_x: int | None  # Viewport-relative x (player at center ~3)
    tank_y: int | None  # Tank absolute Y coordinate (from info_bytes[5])
    tank_viewport_x: int | None  # Tank viewport-relative X (from info_bytes[6])


class TankStatusShortDict(TypedDict):
    """Enemy tank status with HP and rank from 0x2E container.

    Structure (9 bytes, verified from captures):
      [subtype:1] [flags:1] [tank_id:2 LE] [damage_state:1] [rank:1] [lb_pos:2 LE] [extra:1]

    The damage_state controls how dark the enemy tank name appears (0=full to 3=critical).
    The rank is 0-7 (recruit to general).
    """

    msg_type: Literal["tank_status_short"]
    flags: int
    tank_id: int
    damage_state: int
    rank: int
    leaderboard_position: int


class TankUpdateCompactDict(TypedDict):
    """Compact tank update from 0x2E container (10 bytes).

    Structure (verified from captures):
      [subtype:1] [flags:1] [tank_id:2 LE] [status_data:6]

    Compact status update for tanks, possibly viewport entry notification.
    """

    msg_type: Literal["tank_update_compact"]
    flags: int
    tank_id: int
    status_data: bytes


# TankUpdateExtendedDict deleted 2026-06-19: the 14-byte length
# heuristic never matched a real wire body across 150 production
# capture sessions -- every 14-byte 0x2E body in the corpus is a
# tunneled 0x47 Movement (Lg.h), dispatched before length-based
# container fallback runs. See analysis_scripts/crack_tank_update.py.


class TankUpdateFullDict(TypedDict):
    """Full tank update from 0x2E container (15 bytes).

    Structure (verified from captures):
      [subtype:1] [flags:1] [tank_id:2 LE] [status_data:11]

    Full status update with complete tank information.
    """

    msg_type: Literal["tank_update_full"]
    flags: int
    tank_id: int
    status_data: bytes


# Container TankStatusSync deleted 2026-06-19 (length-only catch-all
# misidentifying short subtypes). Real 0x2E TankStatusSync lives in
# tankpit_bot.protocol (TankStatusSyncDict, 8+ bytes per JS Og.h).


class TankLeaveDict(TypedDict):
    """Tank leave event decoded from 0x2E container.

    Structure (6 bytes, verified from captures):
      [subtype:1] [flags:1] [tank_id:2 LE] [extra:2]

    Sent when a player leaves/disconnects from the game.
    Tank ID at bytes[2:4] as little-endian u16.
    """

    msg_type: Literal["tank_leave"]
    tank_id: int
    flags: int
    extra_data: bytes


# =============================================================================
# Position/Movement Messages
# =============================================================================


class PositionUpdateDict(TypedDict):
    """Position/status update decoded from 0x2E container.

    Structure (verified from captures):
      [subtype:1] [flags:1] [tank_id:2 LE] [x:1] [y:1] [extra:7]

    Periodic position and status updates for tanks.
    x,y are map grid coordinates (0-127 range).
    """

    msg_type: Literal["position_update"]
    flags: int
    tank_id: int
    x: int
    y: int
    extra_data: bytes


# 0x47 Movement lives in tankpit_bot.protocol (MovementDict). The
# container path was deleted 2026-06-19: it had misinterpreted bytes
# 8-11 as a "player_id" but tpclient.js Lg.h reads those bytes as
# lb_score (24-bit BE at 6-8) + rank (a[9]). PlayerIdMapper was also
# deleted -- the protocol decoder has tank_id directly.


# =============================================================================
# Radar Messages
# =============================================================================


# RadarContainerDict, RadarMineDict, and the 0x4F RadarResponse decoder
# all live in tankpit_bot.protocol (single source of truth). Their
# container duplicates were deleted 2026-06-19.


# =============================================================================
# Player List Messages
# =============================================================================


class PlayerListShortDict(TypedDict):
    """Short player list response decoded from 0x2E container.

    Structure (4 bytes, verified from captures):
      [subtype:1] [data:3]

    Sent in response to pressing '/' key (active players query).
    Contains count or summary of active players.
    """

    msg_type: Literal["player_list_short"]
    response_data: bytes


class PlayerListExtendedDict(TypedDict):
    """Extended player list response decoded from 0x2E container.

    Structure (7 bytes, verified from captures):
      [subtype:1] [base_data:3] [extended_data:3]

    Sent when multiple players are active.
    First 4 bytes match short format, with 3 additional bytes.
    """

    msg_type: Literal["player_list_extended"]
    response_data: bytes
    extended_data: bytes


# =============================================================================
# Miscellaneous Messages
# =============================================================================


class TeleportLandedDict(TypedDict):
    """Teleport landed confirmation container message.

    Structure (1 byte):
      [subtype:1] (0x0C = 12)

    Sent by server after teleport completes and tank has landed at new location.
    Arrives 150-2000ms after teleport initiated, just before UI updates position.
    """

    msg_type: Literal["teleport_landed"]
    subtype: int


class ContainerPickupDict(TypedDict):
    """Container pickup event.

    Structure (5 bytes):
      [subtype:1] [x:1] [y:1] [volume:2 LE]

    Sent when a container is picked up.
    - volume = 0: Equipment container
    - volume > 0: Fuel container (volume = fuel received)
    """

    msg_type: Literal["container_pickup"]
    x: int
    y: int
    volume: int
    is_fuel: bool


class TipNotificationDict(TypedDict):
    """Tip/notification container message.

    Structure (29-79 bytes):
      [subtype:1] [notification_data:28-78]

    Contains UI tips and game notifications.
    """

    msg_type: Literal["tip_notification"]
    subtype: int
    length: int
    notification_data: bytes


class ChunkDataDict(TypedDict):
    """Chunk data container message.

    Structure (80-130 bytes):
      [subtype:1] [chunk_data:79-129]

    Contains map/terrain chunk data.
    """

    msg_type: Literal["chunk_data"]
    subtype: int
    length: int
    chunk_data: bytes


class WorldStateDict(TypedDict):
    """World state container message.

    Structure (500+ bytes):
      [subtype:1] [world_data:499+]

    Contains full world/map state data.
    """

    msg_type: Literal["world_state"]
    subtype: int
    length: int
    world_data: bytes


class UnknownContainerDict(TypedDict):
    """Unknown container message that didn't match any known structure.

    Preserved for debugging and future analysis.
    """

    msg_type: Literal["unknown_container"]
    subtype: int
    length: int
    data: bytes


# =============================================================================
# Status Messages (0x3D position+status, 0x2E self-status)
# =============================================================================


# 0x3D TankPositionStatus moved to tankpit_bot.protocol.MovementResponseDict
# 2026-06-19 (single source of truth, with the carrying byte at offset 11
# restored after being dropped from the prior protocol decoder).


# SelfStatusDict was deleted 2026-06-19. The 13-byte 0x2E-nested
# self-status form (with fuel) is decoded by the protocol path's
# TankStatusSync (decode_tank_status_sync), which handles both the short
# 9-byte form and the 13-byte form with fuel at the tail.


# =============================================================================
# Union Types
# =============================================================================

ContainerMessage = (
    MineDetonationDict
    | MinePlacementDict
    | TankRegistryDict
    | PositionUpdateDict
    | TankStatusShortDict
    | TankUpdateCompactDict
    | TankUpdateFullDict
    | TankLeaveDict
    | PlayerListShortDict
    | PlayerListExtendedDict
    | DeactivationDeathDict
    | TeleportLandedDict
    | ContainerPickupDict
    | TipNotificationDict
    | ChunkDataDict
    | WorldStateDict
    | UnknownContainerDict
)


__all__ = [
    "MESSAGE_TYPE_LEVELS",
    "ChunkDataDict",
    "ContainerMessage",
    "ContainerMessageType",
    "ContainerPickupDict",
    "DeactivationDeathDict",
    "DecodeLevel",
    "MineDetonationDict",
    "MinePlacementDict",
    "PlayerListExtendedDict",
    "PlayerListShortDict",
    "PositionUpdateDict",
    "TankLeaveDict",
    "TankRegistryDict",
    "TankStatusShortDict",
    "TankUpdateCompactDict",
    "TankUpdateFullDict",
    "TeleportLandedDict",
    "TipNotificationDict",
    "UnknownContainerDict",
    "WorldStateDict",
    "get_decode_level",
]
