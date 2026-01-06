"""Structure-based decoder for 0x2E container messages.

The 0x2E container wraps various message types. Due to XOR encoding with
session-specific magic keys, the subtype byte (first byte of decoded body)
varies between sessions. This module identifies messages by STRUCTURE
(length, field positions) rather than subtype values.

Message types identified by structure:
- Combat hit: 11-12 bytes with terminator pattern
- Tank registry: 17-20 bytes (name length varies)
- Position update: 14 bytes fixed length
- Tank status sync: 2-3 bytes (heartbeat)

All decoding uses explicit validation without try/except recovery.
"""

from __future__ import annotations

from enum import IntEnum, auto
from typing import Literal, TypedDict


class ContainerMessageType(IntEnum):
    """Types of messages found inside 0x2E containers.

    Identified by structure, not subtype byte value.
    """

    UNKNOWN = 0
    COMBAT_HIT = auto()
    TANK_REGISTRY = auto()
    POSITION_UPDATE = auto()
    TANK_STATUS_SHORT = auto()
    TANK_UPDATE_COMPACT = auto()
    TANK_UPDATE_EXTENDED = auto()
    TANK_UPDATE_FULL = auto()
    TANK_STATUS_SYNC = auto()
    TANK_LEAVE = auto()
    PLAYER_LIST_SHORT = auto()
    PLAYER_LIST_EXTENDED = auto()
    DEACTIVATION_KILL = auto()
    DEACTIVATION_DEATH = auto()
    TELEPORT_LANDED = auto()
    ENTITY_SYNC = auto()
    ENTITY_EXTENDED = auto()
    TIP_NOTIFICATION = auto()
    CHUNK_DATA = auto()
    WORLD_STATE = auto()


class ContainerDecodeError(Exception):
    """Raised when container message decoding fails validation.

    Args:
        message: Description of the validation failure.
    """

    def __init__(self, message: str) -> None:
        """Initialize with error message.

        Args:
            message: Description of the validation failure.
        """
        super().__init__(message)
        self.message = message


# =============================================================================
# Validation Helpers (explicit, no try/except recovery)
# =============================================================================


def require_min_length(data: bytes, min_len: int, context: str) -> None:
    """Validate data meets minimum length requirement.

    Args:
        data: Bytes to validate.
        min_len: Minimum required length.
        context: Context string for error message.

    Raises:
        ContainerDecodeError: If data is too short.
    """
    if len(data) < min_len:
        raise ContainerDecodeError(f"{context}: need at least {min_len} bytes, got {len(data)}")


def require_exact_length(data: bytes, exact_len: int, context: str) -> None:
    """Validate data is exactly the expected length.

    Args:
        data: Bytes to validate.
        exact_len: Expected length.
        context: Context string for error message.

    Raises:
        ContainerDecodeError: If length doesn't match.
    """
    if len(data) != exact_len:
        raise ContainerDecodeError(f"{context}: expected {exact_len} bytes, got {len(data)}")


def require_length_range(data: bytes, min_len: int, max_len: int, context: str) -> None:
    """Validate data length is within expected range.

    Args:
        data: Bytes to validate.
        min_len: Minimum length (inclusive).
        max_len: Maximum length (inclusive).
        context: Context string for error message.

    Raises:
        ContainerDecodeError: If length is outside range.
    """
    if not (min_len <= len(data) <= max_len):
        raise ContainerDecodeError(
            f"{context}: expected {min_len}-{max_len} bytes, got {len(data)}"
        )


def extract_uint16_le(data: bytes, offset: int, context: str) -> int:
    """Extract little-endian uint16 from bytes at offset.

    Args:
        data: Source bytes.
        offset: Byte offset to read from.
        context: Context string for error message.

    Returns:
        Extracted uint16 value.

    Raises:
        ContainerDecodeError: If offset is out of bounds.
    """
    if offset + 2 > len(data):
        raise ContainerDecodeError(
            f"{context}: cannot read uint16 at offset {offset}, data length {len(data)}"
        )
    return data[offset] | (data[offset + 1] << 8)


# =============================================================================
# Combat Hit Message (11-12 bytes)
# =============================================================================


class CombatHitDict(TypedDict):
    """Combat hit event decoded from 0x2E container.

    Structure (verified from captures):
      [subtype:1] [direction:1] [attacker_id:2 LE] [combat_data:6-7] [terminator:1]

    The subtype and terminator bytes are the same value (XOR-dependent).
    Direction 0x09 indicates outgoing hit (you attacked), other values indicate incoming.
    """

    msg_type: Literal["combat_hit"]
    direction: int
    attacker_id: int
    combat_data: bytes
    is_outgoing: bool


def is_combat_hit_structure(data: bytes) -> bool:
    """Check if data matches combat hit message structure.

    Combat hit criteria:
    - Exactly 11 bytes

    Note: Terminator pattern (first==last) varies per session due to XOR encoding.
    We identify solely by length since 11 bytes is unique to combat hits.

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches combat hit pattern.
    """
    return len(data) == 11


def decode_combat_hit(data: bytes) -> CombatHitDict:
    """Decode combat hit message from container body.

    Structure (11 bytes):
      [subtype:1] [direction:1] [attacker_id:2 LE] [combat_data:7]

    Args:
        data: Decoded container body bytes (must be 11 bytes).

    Returns:
        Decoded combat hit data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 11, "CombatHit")

    direction = data[1]
    attacker_id = extract_uint16_le(data, 2, "CombatHit.attacker_id")
    combat_data = bytes(data[4:11])  # Last 7 bytes
    is_outgoing = direction == 0x09

    return CombatHitDict(
        msg_type="combat_hit",
        direction=direction,
        attacker_id=attacker_id,
        combat_data=combat_data,
        is_outgoing=is_outgoing,
    )


# =============================================================================
# Tank Registry Message (16-20 bytes)
# =============================================================================


class TankRegistryDict(TypedDict):
    """Tank registry entry decoded from 0x2E container.

    Structure (16-20 bytes, verified from captures):
      [subtype:1] [flags:1] [tank_id:2 LE] [info_bytes:12-16]

    info_bytes structure:
      Standard (12 bytes): [rank_badges:1][zeros:4][unk1:1][unk2:1][name:5+]
      Extended (15 bytes): [rank_badges:1][zeros:4][pos:2][unk:3][name:5+]

    rank_badges byte encoding:
      bits 0-2: military rank (0=recruit...7=colonel, overflow for general)
      bits 3-7: badge/award count

    Team encoded in flags lower 2 bits: 0=red, 1=purple, 2=blue, 3=orange.
    Extended format indicated by (flags & 0x2C) != 0.
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


def is_tank_registry_structure(data: bytes) -> bool:
    """Check if data matches tank registry message structure.

    Tank registry criteria:
    - Length 16-20 bytes (name length varies)

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches tank registry pattern.
    """
    return 16 <= len(data) <= 20


_TEAM_NAMES: list[str] = ["red", "purple", "blue", "orange"]


def _parse_tank_name(info_bytes: bytes, is_extended: bool) -> str:
    """Extract ASCII tank name from info_bytes.

    Args:
        info_bytes: Raw info bytes from TankRegistry message.
        is_extended: True if extended format (3 extra metadata bytes).

    Returns:
        Tank name as ASCII string, non-printable chars replaced with '?'.
    """
    # Name starts at byte 7 (STD) or byte 10 (EXT)
    name_offset = 10 if is_extended else 7
    if len(info_bytes) <= name_offset:
        return ""
    name_bytes = info_bytes[name_offset:]
    return "".join(chr(b) if 32 <= b < 127 else "?" for b in name_bytes)


def decode_tank_registry(data: bytes) -> TankRegistryDict:
    """Decode tank registry message from container body.

    Structure (16-20 bytes):
      [subtype:1] [flags:1] [tank_id:2 LE] [info_bytes:12-16]

    info_bytes structure:
      [rank_badges:1] [zeros:4] [unk1:1] [unk2:1] [name:variable]
      Extended format adds 3 bytes before name.

    Args:
        data: Decoded container body bytes (must be 16-20 bytes).

    Returns:
        Decoded tank registry data with parsed fields.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_length_range(data, 16, 20, "TankRegistry")

    flags = data[1]
    tank_id = extract_uint16_le(data, 2, "TankRegistry.tank_id")
    info_bytes = bytes(data[4:])

    # Team from lower 2 bits of flags
    team_idx = flags & 0x03
    team = _TEAM_NAMES[team_idx]

    # Parse rank and badges from first info byte
    rank_badges = info_bytes[0] if len(info_bytes) > 0 else 0
    military_rank = rank_badges & 0x07
    badge_count = rank_badges >> 3

    # Bot detection: first 6 bytes of info are all zeros for bots
    # Bots: [zeros:6][bot_num:1][name:variable]
    is_bot = len(info_bytes) >= 6 and all(b == 0 for b in info_bytes[:6])

    # Extended format indicated by flags bits 0x2C being set
    # Standard: flags like 0x02, 0x03 -> name at byte 7
    # Extended: flags like 0x2E, 0x2F (0x2C bits set) -> name at byte 10
    # Bots always use standard format regardless of flags
    is_extended = not is_bot and (flags & 0x2C) != 0

    # Extract tank name
    tank_name = _parse_tank_name(info_bytes, is_extended)

    return TankRegistryDict(
        msg_type="tank_registry",
        flags=flags,
        tank_id=tank_id,
        info_bytes=info_bytes,
        team=team,
        tank_name=tank_name,
        military_rank=military_rank,
        badge_count=badge_count,
        is_bot=is_bot,
    )


# =============================================================================
# Position Update Message (14 bytes)
# =============================================================================


class PositionUpdateDict(TypedDict):
    """Position/status update decoded from 0x2E container.

    Structure (verified from captures):
      [subtype:1] [flags:1] [tank_id:2 LE] [status_bytes:10]

    Periodic position and status updates for tanks.
    """

    msg_type: Literal["position_update"]
    flags: int
    tank_id: int
    status_bytes: bytes


def is_position_update_structure(data: bytes) -> bool:
    """Check if data matches position update message structure.

    Position update criteria:
    - Exactly 13 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches position update pattern.
    """
    return len(data) == 13


def decode_position_update(data: bytes) -> PositionUpdateDict:
    """Decode position update message from container body.

    Structure (13 bytes):
      [subtype:1] [flags:1] [tank_id:2 LE] [status_bytes:9]

    Args:
        data: Decoded container body bytes (must be 13 bytes).

    Returns:
        Decoded position update data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 13, "PositionUpdate")

    flags = data[1]
    tank_id = extract_uint16_le(data, 2, "PositionUpdate.tank_id")
    status_bytes = bytes(data[4:])

    return PositionUpdateDict(
        msg_type="position_update",
        flags=flags,
        tank_id=tank_id,
        status_bytes=status_bytes,
    )


# =============================================================================
# Tank Status Short Message (9 bytes) - Enemy HP/Rank
# =============================================================================


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


def is_tank_status_short_structure(data: bytes) -> bool:
    """Check if data matches tank status short message structure.

    Args:
        data: Decoded container body bytes.

    Returns:
        True if length is exactly 9 bytes.
    """
    return len(data) == 9


def decode_tank_status_short(data: bytes) -> TankStatusShortDict:
    """Decode tank status short message from container body.

    Structure (9 bytes):
      [0]    subtype (ignored, XOR encoded)
      [1]    flags
      [2-3]  tank_id (LE)
      [4]    damage_state (0-3)
      [5]    rank (0-7: recruit to general)
      [6-7]  leaderboard_position (LE)
      [8]    extra byte (ignored)

    Args:
        data: Decoded container body bytes (must be 9 bytes).

    Returns:
        Decoded tank status short data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 9, "TankStatusShort")

    flags = data[1]
    tank_id = extract_uint16_le(data, 2, "TankStatusShort.tank_id")
    damage_state = data[4]
    rank = data[5]
    lb_pos = extract_uint16_le(data, 6, "TankStatusShort.leaderboard_position")

    return TankStatusShortDict(
        msg_type="tank_status_short",
        flags=flags,
        tank_id=tank_id,
        damage_state=damage_state,
        rank=rank,
        leaderboard_position=lb_pos,
    )


# =============================================================================
# Tank Update Compact Message (10 bytes)
# =============================================================================


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


def is_tank_update_compact_structure(data: bytes) -> bool:
    """Check if data matches tank update compact structure.

    Args:
        data: Decoded container body bytes.

    Returns:
        True if length is exactly 10 bytes.
    """
    return len(data) == 10


def decode_tank_update_compact(data: bytes) -> TankUpdateCompactDict:
    """Decode tank update compact message from container body.

    Structure (10 bytes):
      [0]    subtype (ignored, XOR encoded)
      [1]    flags
      [2-3]  tank_id (LE)
      [4-9]  status_data

    Args:
        data: Decoded container body bytes (must be 10 bytes).

    Returns:
        Decoded tank update compact data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 10, "TankUpdateCompact")

    flags = data[1]
    tank_id = extract_uint16_le(data, 2, "TankUpdateCompact.tank_id")
    status_data = bytes(data[4:])

    return TankUpdateCompactDict(
        msg_type="tank_update_compact",
        flags=flags,
        tank_id=tank_id,
        status_data=status_data,
    )


# =============================================================================
# Tank Update Extended Message (14 bytes)
# =============================================================================


class TankUpdateExtendedDict(TypedDict):
    """Extended tank update from 0x2E container (14 bytes).

    Structure (verified from captures):
      [subtype:1] [flags:1] [tank_id:2 LE] [status_data:10]

    Extended status update with additional tank information.
    """

    msg_type: Literal["tank_update_extended"]
    flags: int
    tank_id: int
    status_data: bytes


def is_tank_update_extended_structure(data: bytes) -> bool:
    """Check if data matches tank update extended structure.

    Args:
        data: Decoded container body bytes.

    Returns:
        True if length is exactly 14 bytes.
    """
    return len(data) == 14


def decode_tank_update_extended(data: bytes) -> TankUpdateExtendedDict:
    """Decode tank update extended message from container body.

    Structure (14 bytes):
      [0]    subtype (ignored, XOR encoded)
      [1]    flags
      [2-3]  tank_id (LE)
      [4-13] status_data

    Args:
        data: Decoded container body bytes (must be 14 bytes).

    Returns:
        Decoded tank update extended data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 14, "TankUpdateExtended")

    flags = data[1]
    tank_id = extract_uint16_le(data, 2, "TankUpdateExtended.tank_id")
    status_data = bytes(data[4:])

    return TankUpdateExtendedDict(
        msg_type="tank_update_extended",
        flags=flags,
        tank_id=tank_id,
        status_data=status_data,
    )


# =============================================================================
# Tank Update Full Message (15 bytes)
# =============================================================================


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


def is_tank_update_full_structure(data: bytes) -> bool:
    """Check if data matches tank update full structure.

    Args:
        data: Decoded container body bytes.

    Returns:
        True if length is exactly 15 bytes.
    """
    return len(data) == 15


def decode_tank_update_full(data: bytes) -> TankUpdateFullDict:
    """Decode tank update full message from container body.

    Structure (15 bytes):
      [0]    subtype (ignored, XOR encoded)
      [1]    flags
      [2-3]  tank_id (LE)
      [4-14] status_data

    Args:
        data: Decoded container body bytes (must be 15 bytes).

    Returns:
        Decoded tank update full data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 15, "TankUpdateFull")

    flags = data[1]
    tank_id = extract_uint16_le(data, 2, "TankUpdateFull.tank_id")
    status_data = bytes(data[4:])

    return TankUpdateFullDict(
        msg_type="tank_update_full",
        flags=flags,
        tank_id=tank_id,
        status_data=status_data,
    )


# =============================================================================
# Tank Status Sync Message (2-3 bytes)
# =============================================================================


class TankStatusSyncDict(TypedDict):
    """Tank status sync/heartbeat decoded from 0x2E container.

    Structure:
      [subtype:1] [sync_data:1-2]

    Short sync messages for keepalive or state confirmation.
    """

    msg_type: Literal["tank_status_sync"]
    sync_data: bytes


def is_tank_status_sync_structure(data: bytes) -> bool:
    """Check if data matches tank status sync message structure.

    Tank status sync criteria:
    - Length 2-3 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches tank status sync pattern.
    """
    return 2 <= len(data) <= 3


def decode_tank_status_sync(data: bytes) -> TankStatusSyncDict:
    """Decode tank status sync message from container body.

    Args:
        data: Decoded container body bytes (must pass is_tank_status_sync_structure).

    Returns:
        Decoded tank status sync data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_length_range(data, 2, 3, "TankStatusSync")

    return TankStatusSyncDict(
        msg_type="tank_status_sync",
        sync_data=bytes(data[1:]),
    )


# =============================================================================
# Tank Leave Message (6 bytes) - Player exits game
# =============================================================================


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


def is_tank_leave_structure(data: bytes) -> bool:
    """Check if data matches tank leave message structure.

    Tank leave criteria:
    - Exactly 6 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches tank leave pattern.
    """
    return len(data) == 6


def decode_tank_leave(data: bytes) -> TankLeaveDict:
    """Decode tank leave message from container body.

    Structure (6 bytes):
      [subtype:1] [flags:1] [tank_id:2 LE] [extra:2]

    Args:
        data: Decoded container body bytes (must be 6 bytes).

    Returns:
        Decoded tank leave data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 6, "TankLeave")

    flags = data[1]
    tank_id = extract_uint16_le(data, 2, "TankLeave.tank_id")
    extra_data = bytes(data[4:6])

    return TankLeaveDict(
        msg_type="tank_leave",
        tank_id=tank_id,
        flags=flags,
        extra_data=extra_data,
    )


# =============================================================================
# Player List Response (4 or 7 bytes) - Active players list
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


def is_player_list_short_structure(data: bytes) -> bool:
    """Check if data matches short player list response structure.

    Criteria:
    - Exactly 4 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches player list short pattern.
    """
    return len(data) == 4


def decode_player_list_short(data: bytes) -> PlayerListShortDict:
    """Decode short player list response from container body.

    Args:
        data: Decoded container body bytes (must be 4 bytes).

    Returns:
        Decoded player list response.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 4, "PlayerListShort")

    return PlayerListShortDict(
        msg_type="player_list_short",
        response_data=bytes(data[1:]),
    )


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


def is_player_list_extended_structure(data: bytes) -> bool:
    """Check if data matches extended player list response structure.

    Criteria:
    - Exactly 7 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches player list extended pattern.
    """
    return len(data) == 7


def decode_player_list_extended(data: bytes) -> PlayerListExtendedDict:
    """Decode extended player list response from container body.

    Args:
        data: Decoded container body bytes (must be 7 bytes).

    Returns:
        Decoded player list response.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 7, "PlayerListExtended")

    return PlayerListExtendedDict(
        msg_type="player_list_extended",
        response_data=bytes(data[1:4]),
        extended_data=bytes(data[4:7]),
    )


# =============================================================================
# Deactivation Kill Message (5 bytes) - You killed another tank
# =============================================================================


class DeactivationKillDict(TypedDict):
    """Deactivation event when you killed another tank.

    Structure (5 bytes, verified from captures):
      [subtype:1] [victim_id:2 LE] [killer_id:2 LE]

    Subtype is 0x41 ('A') after XOR decode.
    Sent when another tank is deactivated by you.
    """

    msg_type: Literal["deactivation_kill"]
    victim_id: int
    killer_id: int


def is_deactivation_kill_structure(data: bytes) -> bool:
    """Check if data matches deactivation kill message structure.

    Criteria:
    - Exactly 5 bytes
    - First decoded byte is 0x41 ('A')

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches deactivation kill pattern.
    """
    return len(data) == 5 and data[0] == 0x41


def decode_deactivation_kill(data: bytes) -> DeactivationKillDict:
    """Decode deactivation kill message from container body.

    Structure (5 bytes):
      [subtype:1] [victim_id:2 LE] [killer_id:2 LE]

    Args:
        data: Decoded container body bytes (must be 5 bytes).

    Returns:
        Decoded deactivation kill data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 5, "DeactivationKill")

    victim_id = extract_uint16_le(data, 1, "DeactivationKill.victim_id")
    killer_id = extract_uint16_le(data, 3, "DeactivationKill.killer_id")

    return DeactivationKillDict(
        msg_type="deactivation_kill",
        victim_id=victim_id,
        killer_id=killer_id,
    )


# =============================================================================
# Deactivation Death Message (7 bytes) - You were killed
# =============================================================================


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


def is_deactivation_death_structure(data: bytes) -> bool:
    """Check if data matches deactivation death message structure.

    Criteria:
    - Exactly 7 bytes
    - First decoded byte is 0x43 ('C')

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches deactivation death pattern.
    """
    return len(data) == 7 and data[0] == 0x43


def decode_deactivation_death(data: bytes) -> DeactivationDeathDict:
    """Decode deactivation death message from container body.

    Structure (7 bytes):
      [subtype:1] [flags:1] [killer_id:2 LE] [extra:3]

    Args:
        data: Decoded container body bytes (must be 7 bytes).

    Returns:
        Decoded deactivation death data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 7, "DeactivationDeath")

    flags = data[1]
    killer_id = extract_uint16_le(data, 2, "DeactivationDeath.killer_id")
    extra_data = bytes(data[4:7])

    return DeactivationDeathDict(
        msg_type="deactivation_death",
        flags=flags,
        killer_id=killer_id,
        extra_data=extra_data,
    )


# =============================================================================
# Teleport Landed Container Message (len=1)
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


def is_teleport_landed_structure(data: bytes) -> bool:
    """Check if data matches teleport landed structure.

    Teleport landed criteria:
    - Exactly 1 byte

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches teleport landed pattern.
    """
    return len(data) == 1


def decode_teleport_landed(data: bytes) -> TeleportLandedDict:
    """Decode teleport landed container message.

    Args:
        data: Decoded container body bytes (must be 1 byte).

    Returns:
        Decoded teleport landed data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 1, "TeleportLanded")

    return TeleportLandedDict(
        msg_type="teleport_landed",
        subtype=data[0],
    )


# =============================================================================
# Entity Sync Container Message (len=5)
# =============================================================================


class EntitySyncDict(TypedDict):
    """Entity sync container message.

    Structure (5 bytes):
      [subtype:1] [sync_data:4]

    Broadcasts entity state synchronization updates.
    """

    msg_type: Literal["entity_sync"]
    subtype: int
    sync_data: bytes


def is_entity_sync_structure(data: bytes) -> bool:
    """Check if data matches entity sync structure.

    Entity sync criteria:
    - Exactly 5 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches entity sync pattern.
    """
    return len(data) == 5


def decode_entity_sync(data: bytes) -> EntitySyncDict:
    """Decode entity sync container message.

    Args:
        data: Decoded container body bytes (must be 5 bytes).

    Returns:
        Decoded entity sync data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 5, "EntitySync")

    return EntitySyncDict(
        msg_type="entity_sync",
        subtype=data[0],
        sync_data=bytes(data[1:5]),
    )


# =============================================================================
# Entity Extended Container Message (len=21-28)
# =============================================================================


class EntityExtendedDict(TypedDict):
    """Entity extended information container message.

    Structure (21-28 bytes):
      [subtype:1] [entity_data:20-27]

    Contains extended entity state information (position, status, etc.).
    """

    msg_type: Literal["entity_extended"]
    subtype: int
    length: int
    entity_data: bytes


def is_entity_extended_structure(data: bytes) -> bool:
    """Check if data matches entity extended structure.

    Entity extended criteria:
    - Length 21-28 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches entity extended pattern.
    """
    return 21 <= len(data) <= 28


def decode_entity_extended(data: bytes) -> EntityExtendedDict:
    """Decode entity extended container message.

    Args:
        data: Decoded container body bytes (must be 21-28 bytes).

    Returns:
        Decoded entity extended data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_length_range(data, 21, 28, "EntityExtended")

    return EntityExtendedDict(
        msg_type="entity_extended",
        subtype=data[0],
        length=len(data),
        entity_data=bytes(data[1:]),
    )


# =============================================================================
# Tip Notification Container Message (len=29-79)
# =============================================================================


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


def is_tip_notification_structure(data: bytes) -> bool:
    """Check if data matches tip notification structure.

    Tip notification criteria:
    - Length 29-79 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches tip notification pattern.
    """
    return 29 <= len(data) <= 79


def decode_tip_notification(data: bytes) -> TipNotificationDict:
    """Decode tip notification container message.

    Args:
        data: Decoded container body bytes (must be 29-79 bytes).

    Returns:
        Decoded tip notification data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_length_range(data, 29, 79, "TipNotification")

    return TipNotificationDict(
        msg_type="tip_notification",
        subtype=data[0],
        length=len(data),
        notification_data=bytes(data[1:]),
    )


# =============================================================================
# Chunk Data Container Message (len=80-130)
# =============================================================================


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


def is_chunk_data_structure(data: bytes) -> bool:
    """Check if data matches chunk data structure.

    Chunk data criteria:
    - Length 80-130 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches chunk data pattern.
    """
    return 80 <= len(data) <= 130


def decode_chunk_data(data: bytes) -> ChunkDataDict:
    """Decode chunk data container message.

    Args:
        data: Decoded container body bytes (must be 80-130 bytes).

    Returns:
        Decoded chunk data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_length_range(data, 80, 130, "ChunkData")

    return ChunkDataDict(
        msg_type="chunk_data",
        subtype=data[0],
        length=len(data),
        chunk_data=bytes(data[1:]),
    )


# =============================================================================
# World State Container Message (len=500+)
# =============================================================================


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


def is_world_state_structure(data: bytes) -> bool:
    """Check if data matches world state structure.

    World state criteria:
    - Length >= 500 bytes

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches world state pattern.
    """
    return len(data) >= 500


def decode_world_state(data: bytes) -> WorldStateDict:
    """Decode world state container message.

    Args:
        data: Decoded container body bytes (must be >= 500 bytes).

    Returns:
        Decoded world state data.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_min_length(data, 500, "WorldState")

    return WorldStateDict(
        msg_type="world_state",
        subtype=data[0],
        length=len(data),
        world_data=bytes(data[1:]),
    )


# =============================================================================
# Unknown Container Message
# =============================================================================


class UnknownContainerDict(TypedDict):
    """Unknown container message that didn't match any known structure.

    Preserved for debugging and future analysis.
    """

    msg_type: Literal["unknown_container"]
    subtype: int
    length: int
    data: bytes


def decode_unknown_container(data: bytes) -> UnknownContainerDict:
    """Create unknown container result for unrecognized structures.

    Args:
        data: Decoded container body bytes.

    Returns:
        Unknown container data for debugging.
    """
    require_min_length(data, 1, "UnknownContainer")

    return UnknownContainerDict(
        msg_type="unknown_container",
        subtype=data[0],
        length=len(data),
        data=bytes(data),
    )


# =============================================================================
# Container Message Union and Dispatcher
# =============================================================================


ContainerMessage = (
    CombatHitDict
    | TankRegistryDict
    | PositionUpdateDict
    | TankStatusShortDict
    | TankUpdateCompactDict
    | TankUpdateExtendedDict
    | TankUpdateFullDict
    | TankStatusSyncDict
    | TankLeaveDict
    | PlayerListShortDict
    | PlayerListExtendedDict
    | DeactivationKillDict
    | DeactivationDeathDict
    | TeleportLandedDict
    | EntitySyncDict
    | EntityExtendedDict
    | TipNotificationDict
    | ChunkDataDict
    | WorldStateDict
    | UnknownContainerDict
)


def _identify_tank_update_type(data: bytes) -> ContainerMessageType:
    """Identify tank update message types by structure.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified tank update type, or UNKNOWN if not a tank update.
    """
    # Tank update compact: exactly 10 bytes
    if is_tank_update_compact_structure(data):
        return ContainerMessageType.TANK_UPDATE_COMPACT
    # Tank update extended: exactly 14 bytes
    if is_tank_update_extended_structure(data):
        return ContainerMessageType.TANK_UPDATE_EXTENDED
    # Tank update full: exactly 15 bytes
    if is_tank_update_full_structure(data):
        return ContainerMessageType.TANK_UPDATE_FULL
    return ContainerMessageType.UNKNOWN


def _identify_player_list_type(data: bytes) -> ContainerMessageType:
    """Identify player list message types by structure.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified player list type, or UNKNOWN if not a player list.
    """
    # Player list short: 4 bytes
    if is_player_list_short_structure(data):
        return ContainerMessageType.PLAYER_LIST_SHORT
    # Player list extended: 7 bytes
    # Note: 7 bytes conflicts with deactivation_death, so check deactivation first
    if is_player_list_extended_structure(data):
        return ContainerMessageType.PLAYER_LIST_EXTENDED
    return ContainerMessageType.UNKNOWN


def _identify_deactivation_type(data: bytes) -> ContainerMessageType:
    """Identify deactivation message types by structure.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified deactivation type, or UNKNOWN if not a deactivation.
    """
    # Deactivation kill: 5 bytes
    if is_deactivation_kill_structure(data):
        return ContainerMessageType.DEACTIVATION_KILL
    # Deactivation death: 7 bytes
    if is_deactivation_death_structure(data):
        return ContainerMessageType.DEACTIVATION_DEATH
    return ContainerMessageType.UNKNOWN


def _identify_single_length_type(data: bytes) -> ContainerMessageType:
    """Identify message types that have a single exact length.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified type, or UNKNOWN if not matched.
    """
    # Teleport landed: exactly 1 byte
    if is_teleport_landed_structure(data):
        return ContainerMessageType.TELEPORT_LANDED
    # Tank status sync: 2-3 bytes
    if is_tank_status_sync_structure(data):
        return ContainerMessageType.TANK_STATUS_SYNC
    # Tank leave: 6 bytes
    if is_tank_leave_structure(data):
        return ContainerMessageType.TANK_LEAVE
    # Tank status short: exactly 9 bytes
    if is_tank_status_short_structure(data):
        return ContainerMessageType.TANK_STATUS_SHORT
    # Combat hit: exactly 11 bytes
    if is_combat_hit_structure(data):
        return ContainerMessageType.COMBAT_HIT
    # Position update: exactly 13 bytes
    if is_position_update_structure(data):
        return ContainerMessageType.POSITION_UPDATE
    return ContainerMessageType.UNKNOWN


def identify_container_type(data: bytes) -> ContainerMessageType:
    """Identify the type of container message by structure.

    Order of checks matters - more specific patterns first.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified message type.
    """
    if len(data) < 1:
        return ContainerMessageType.UNKNOWN

    # Single-length types (11, 13, 9, 2-3, 6 bytes)
    single_type = _identify_single_length_type(data)
    if single_type != ContainerMessageType.UNKNOWN:
        return single_type
    # Tank registry: 17-20 bytes
    if is_tank_registry_structure(data):
        return ContainerMessageType.TANK_REGISTRY
    # Tank update types (10, 14, 15 bytes)
    tank_update = _identify_tank_update_type(data)
    if tank_update != ContainerMessageType.UNKNOWN:
        return tank_update
    # Deactivation types (5, 7 bytes) - check before player list due to 7-byte conflict
    deactivation = _identify_deactivation_type(data)
    if deactivation != ContainerMessageType.UNKNOWN:
        return deactivation
    # Entity sync: exactly 5 bytes (after deactivation to avoid conflict with deactivation_kill)
    if is_entity_sync_structure(data):
        return ContainerMessageType.ENTITY_SYNC
    # Player list types (4 bytes only - 7 bytes handled by deactivation_death)
    player_list = _identify_player_list_type(data)
    if player_list != ContainerMessageType.UNKNOWN:
        return player_list
    # Range-based types (21+, 29+, 80+, 500+ bytes)
    range_type = _identify_range_type(data)
    if range_type != ContainerMessageType.UNKNOWN:
        return range_type

    return ContainerMessageType.UNKNOWN


def _identify_range_type(data: bytes) -> ContainerMessageType:
    """Identify message types by length ranges.

    Args:
        data: Decoded container body bytes.

    Returns:
        Identified range type, or UNKNOWN if not matched.
    """
    # Entity extended: 21-28 bytes
    if is_entity_extended_structure(data):
        return ContainerMessageType.ENTITY_EXTENDED
    # Tip notification: 29-79 bytes
    if is_tip_notification_structure(data):
        return ContainerMessageType.TIP_NOTIFICATION
    # Chunk data: 80-130 bytes
    if is_chunk_data_structure(data):
        return ContainerMessageType.CHUNK_DATA
    # World state: 500+ bytes
    if is_world_state_structure(data):
        return ContainerMessageType.WORLD_STATE
    return ContainerMessageType.UNKNOWN


def _decode_tank_update(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode tank update message types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a tank update type.
    """
    if msg_type == ContainerMessageType.TANK_UPDATE_COMPACT:
        return decode_tank_update_compact(data)
    if msg_type == ContainerMessageType.TANK_UPDATE_EXTENDED:
        return decode_tank_update_extended(data)
    if msg_type == ContainerMessageType.TANK_UPDATE_FULL:
        return decode_tank_update_full(data)
    return None


def _decode_player_list(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode player list message types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a player list type.
    """
    if msg_type == ContainerMessageType.PLAYER_LIST_SHORT:
        return decode_player_list_short(data)
    if msg_type == ContainerMessageType.PLAYER_LIST_EXTENDED:
        return decode_player_list_extended(data)
    return None


def _decode_deactivation(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode deactivation message types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a deactivation type.
    """
    if msg_type == ContainerMessageType.DEACTIVATION_KILL:
        return decode_deactivation_kill(data)
    if msg_type == ContainerMessageType.DEACTIVATION_DEATH:
        return decode_deactivation_death(data)
    return None


def _decode_single_type(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode single-length container types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a single-length type.
    """
    if msg_type == ContainerMessageType.COMBAT_HIT:
        return decode_combat_hit(data)
    if msg_type == ContainerMessageType.TANK_REGISTRY:
        return decode_tank_registry(data)
    if msg_type == ContainerMessageType.POSITION_UPDATE:
        return decode_position_update(data)
    if msg_type == ContainerMessageType.TANK_STATUS_SHORT:
        return decode_tank_status_short(data)
    if msg_type == ContainerMessageType.TANK_STATUS_SYNC:
        return decode_tank_status_sync(data)
    if msg_type == ContainerMessageType.TANK_LEAVE:
        return decode_tank_leave(data)
    return None


def decode_container_message(data: bytes) -> ContainerMessage:
    """Decode a 0x2E container message by structure.

    Uses structure-based pattern matching rather than subtype bytes,
    making it robust across sessions with different XOR keys.

    Args:
        data: Decoded container body bytes (after XOR decode, without 0x2E prefix).

    Returns:
        Decoded message as appropriate TypedDict.

    Raises:
        ContainerDecodeError: If known structure fails validation.
    """
    msg_type = identify_container_type(data)

    # Single-length types (1, 2-3, 5, 6, 9, 11, 13, 16-20 bytes)
    single_msg = _decode_single_type(msg_type, data)
    if single_msg is not None:
        return single_msg
    # Tank update types (10, 14, 15 bytes)
    tank_update = _decode_tank_update(msg_type, data)
    if tank_update is not None:
        return tank_update
    # Deactivation types (5, 7 bytes)
    deactivation = _decode_deactivation(msg_type, data)
    if deactivation is not None:
        return deactivation
    # Player list types (4, 7 bytes)
    player_list = _decode_player_list(msg_type, data)
    if player_list is not None:
        return player_list
    # Range-based and misc types (21+, 29+, 80+, 500+ bytes)
    range_msg = _decode_range_type(msg_type, data)
    if range_msg is not None:
        return range_msg

    return decode_unknown_container(data)


def _decode_range_type(msg_type: ContainerMessageType, data: bytes) -> ContainerMessage | None:
    """Decode range-based and miscellaneous container types.

    Args:
        msg_type: Identified message type.
        data: Decoded container body bytes.

    Returns:
        Decoded message, or None if not a range/misc type.
    """
    if msg_type == ContainerMessageType.TELEPORT_LANDED:
        return decode_teleport_landed(data)
    if msg_type == ContainerMessageType.ENTITY_SYNC:
        return decode_entity_sync(data)
    if msg_type == ContainerMessageType.ENTITY_EXTENDED:
        return decode_entity_extended(data)
    if msg_type == ContainerMessageType.TIP_NOTIFICATION:
        return decode_tip_notification(data)
    if msg_type == ContainerMessageType.CHUNK_DATA:
        return decode_chunk_data(data)
    if msg_type == ContainerMessageType.WORLD_STATE:
        return decode_world_state(data)
    return None


__all__ = [
    "ChunkDataDict",
    "CombatHitDict",
    "ContainerDecodeError",
    "ContainerMessage",
    "ContainerMessageType",
    "DeactivationDeathDict",
    "DeactivationKillDict",
    "EntityExtendedDict",
    "EntitySyncDict",
    "PlayerListExtendedDict",
    "PlayerListShortDict",
    "PositionUpdateDict",
    "TankLeaveDict",
    "TankRegistryDict",
    "TankStatusShortDict",
    "TankStatusSyncDict",
    "TankUpdateCompactDict",
    "TankUpdateExtendedDict",
    "TankUpdateFullDict",
    "TeleportLandedDict",
    "TipNotificationDict",
    "UnknownContainerDict",
    "WorldStateDict",
    "decode_chunk_data",
    "decode_combat_hit",
    "decode_container_message",
    "decode_deactivation_death",
    "decode_deactivation_kill",
    "decode_entity_extended",
    "decode_entity_sync",
    "decode_player_list_extended",
    "decode_player_list_short",
    "decode_position_update",
    "decode_tank_leave",
    "decode_tank_registry",
    "decode_tank_status_short",
    "decode_tank_status_sync",
    "decode_tank_update_compact",
    "decode_tank_update_extended",
    "decode_tank_update_full",
    "decode_teleport_landed",
    "decode_tip_notification",
    "decode_unknown_container",
    "decode_world_state",
    "extract_uint16_le",
    "identify_container_type",
    "is_chunk_data_structure",
    "is_combat_hit_structure",
    "is_deactivation_death_structure",
    "is_deactivation_kill_structure",
    "is_entity_extended_structure",
    "is_entity_sync_structure",
    "is_player_list_extended_structure",
    "is_player_list_short_structure",
    "is_position_update_structure",
    "is_tank_leave_structure",
    "is_tank_registry_structure",
    "is_tank_status_short_structure",
    "is_tank_status_sync_structure",
    "is_tank_update_compact_structure",
    "is_tank_update_extended_structure",
    "is_tank_update_full_structure",
    "is_teleport_landed_structure",
    "is_tip_notification_structure",
    "is_world_state_structure",
    "require_exact_length",
    "require_length_range",
    "require_min_length",
]
