"""Tank-related container message decoders.

This module provides decoders for tank registry, status, update, and leave messages.
"""

from __future__ import annotations

from tankpit_bot.container.helpers import (
    extract_uint16_le,
    require_exact_length,
    require_length_range,
)
from tankpit_bot.container.types import (
    TankLeaveDict,
    TankRegistryDict,
    TankStatusShortDict,
    TankStatusSyncDict,
    TankUpdateCompactDict,
    TankUpdateExtendedDict,
    TankUpdateFullDict,
)
from tankpit_bot.protocol.constants import TEAM_NAMES

# Direction byte values for waypoints
DIRECTION_WEST = 0x77  # 'w'
DIRECTION_SOUTH = 0x73  # 's'
DIRECTION_NORTH = 0x6E  # 'n'
DIRECTION_EAST = 0x65  # 'e'
DIRECTION_BYTES = frozenset({DIRECTION_WEST, DIRECTION_SOUTH, DIRECTION_NORTH, DIRECTION_EAST})

# Container subtype byte values (after XOR decoding)
SUBTYPE_MOVEMENT = 0x47  # 'G' - Movement/waypoint messages
SUBTYPE_TANK_REGISTRY = 0x21  # '!' - Tank registry messages


def _parse_tank_name(info_bytes: bytes, is_extended: bool) -> str:
    """Extract ASCII tank name from info_bytes.

    Args:
        info_bytes: Raw info bytes from TankRegistry message.
        is_extended: True if extended format (extra position bytes before name).

    Returns:
        Tank name as ASCII string, non-printable chars replaced with '?'.
    """
    # Standard: [rank_badges:1][zeros:4][unk:2][name] -> offset 7
    # Extended: [rank_badges:1][zeros:4][pos:2][unk:3][name] -> offset 10
    name_offset = 10 if is_extended else 7
    if len(info_bytes) <= name_offset:
        return ""
    name_bytes = info_bytes[name_offset:]
    return "".join(chr(b) if 32 <= b < 127 else "?" for b in name_bytes)


def is_tank_registry_structure(data: bytes) -> bool:
    """Check if data matches tank registry message structure.

    Tank registry criteria:
    - Length 16-20 bytes (name length varies)
    - First byte (subtype) is NOT 0x47 ('G') - Movement uses 0x47
    - Does NOT end with waypoint pattern (4+ direction chars w/s/n/e)

    Movement messages (subtype 0x47 'G') have similar length but use a
    different subtype and end with waypoint directions. Both checks
    prevent misclassification.

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches tank registry pattern.
    """
    if not (16 <= len(data) <= 20):
        return False
    # Reject Movement messages by subtype (0x47 = 'G')
    if data[0] == SUBTYPE_MOVEMENT:
        return False
    # Reject Movement messages - they end with waypoint directions
    # Waypoints are consecutive direction chars (w=0x77, s=0x73, n=0x6e, e=0x65)
    tail4 = data[-4:]
    return not all(b in DIRECTION_BYTES for b in tail4)


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
    team = TEAM_NAMES[team_idx]

    # Parse rank and badges from first info byte
    rank_badges = info_bytes[0] if len(info_bytes) > 0 else 0
    military_rank = rank_badges & 0x07
    badge_count = rank_badges >> 3

    # Bot detection: first 6 bytes of info are all zeros for bots
    # Bots: [zeros:6][bot_num:1][name:variable]
    is_bot = len(info_bytes) >= 6 and all(b == 0 for b in info_bytes[:6])

    # Extended format: flags with 0x2C bits set have extra position bytes
    # Standard: name at offset 7, Extended: name at offset 10
    is_extended = not is_bot and (flags & 0x2C) != 0

    # Extract tank name
    tank_name = _parse_tank_name(info_bytes, is_extended)

    # Container detection: equipment/fuel containers use the same message format
    # but have "names" that are all direction chars (w,s,n,e) or short garbage
    is_container = False
    container_x: int | None = None
    container_y: int | None = None
    container_viewport_x: int | None = None

    if not is_bot:
        # Check if name is all direction/wasd chars
        name_chars = set(tank_name.replace("?", ""))
        is_wasd_name = len(name_chars) > 0 and name_chars <= {"w", "s", "n", "e"}
        # Or short name with non-printable chars
        is_short_garbage = len(tank_name) <= 3 and "?" in tank_name

        if is_wasd_name or is_short_garbage:
            is_container = True
            # Container position encoding:
            # - info[0] = y (absolute map coordinate)
            # - info[1] = viewport-relative x (player at center ~3)
            # Absolute x requires player position: map_x = player_x + (viewport_x - 3)
            container_y = info_bytes[0]
            container_viewport_x = info_bytes[1]
            # container_x left as None - needs player position to calculate
            # Clear the garbage "name" for containers
            tank_name = ""

    # Extract tank position from info_bytes for non-container entries
    # Standard: [rank_badges:1][?:4][pos_y:1][pos_vx:1][name]
    # Extended: [rank_badges:1][?:4][pos_y:1][pos_vx:1][unk:3][name]
    tank_y: int | None = None
    tank_viewport_x: int | None = None
    if not is_container and len(info_bytes) >= 7:
        tank_y = info_bytes[5]
        tank_viewport_x = info_bytes[6]

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
        is_container=is_container,
        container_x=container_x,
        container_y=container_y,
        container_viewport_x=container_viewport_x,
        tank_y=tank_y,
        tank_viewport_x=tank_viewport_x,
    )


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


__all__ = [
    "DIRECTION_BYTES",
    "DIRECTION_EAST",
    "DIRECTION_NORTH",
    "DIRECTION_SOUTH",
    "DIRECTION_WEST",
    "SUBTYPE_MOVEMENT",
    "SUBTYPE_TANK_REGISTRY",
    "decode_tank_leave",
    "decode_tank_registry",
    "decode_tank_status_short",
    "decode_tank_status_sync",
    "decode_tank_update_compact",
    "decode_tank_update_extended",
    "decode_tank_update_full",
    "is_tank_leave_structure",
    "is_tank_registry_structure",
    "is_tank_status_short_structure",
    "is_tank_status_sync_structure",
    "is_tank_update_compact_structure",
    "is_tank_update_extended_structure",
    "is_tank_update_full_structure",
]
