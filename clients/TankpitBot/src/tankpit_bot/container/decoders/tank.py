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


# Container TankStatusShortDict was deleted 2026-06-19. The 9-byte
# length heuristic was wrong on every byte position:
# analysis_scripts/crack_tank_status_short.py proved 74/74 production
# bodies are Og.h TankStatusSync short-form (V['.'] = Og); the
# container layout had tank_id off by one (bytes 2:3 instead of 1:3),
# damage_state at byte 4 (Og.h's rank), rank at byte 5 (Og.h's
# lb_score middle byte), and dropped byte 8 (Og.h's promo_state).
# 9-byte 0x2E bodies now route through ``decode_tank_status_sync``
# from ``decode_0x2e_message``.


# Container TankUpdateCompact (10b), Extended (14b), and Full (15b)
# were deleted 2026-06-19 after analysis_scripts/crack_tank_update.py
# proved them dead. The full audit:
#   * length=14 (Extended): 0 production bodies. Every 14-byte 0x2E
#     body is a tunneled 0x47 Movement (Lg.h) routed before the
#     length-based fallback.
#   * length=15 (Full): 239/239 production bodies are tunneled 0x56
#     Statistics (Wg.h). Tunneled at inner >= 12.
#   * length=10 (Compact): 3 production bodies -- 2 tunneled 0x42
#     BuildPickup (Jg.h, inner >= 9), 1 tunneled 0x28 TankEntry
#     (Uf.h, inner >= 9). All now routed before the length-based
#     fallback runs.
# After the tunneling fix, 0/150 sessions produce any length-based
# TankUpdate* dispatch hit. The types, decoders, structure checks,
# fixtures, and tests are all removed.


# Container TankStatusSync (2-3 byte catch-all) was deleted 2026-06-19:
# it was a length-only catch-all with no subtype guard, misidentifying
# any 2-3 byte container body (0x4F/0x46/0x58/0x3F shorts) as TankStatusSync.
# The real 0x2E TankStatusSync is the 8+ byte protocol path (Og.h).
# Short bodies that no longer match a known subtype fall through to
# UNKNOWN_CONTAINER, which is honest.


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
    "is_tank_leave_structure",
    "is_tank_registry_structure",
]
