"""Position and movement container message decoders.

This module provides decoders for position update and movement messages.
"""

from __future__ import annotations

from tankpit_bot.container.decoders.tank import (
    DIRECTION_BYTES,
    SUBTYPE_MOVEMENT,
)
from tankpit_bot.container.helpers import (
    extract_uint16_le,
    require_exact_length,
    require_min_length,
)
from tankpit_bot.container.types import (
    MovementDict,
    PositionUpdateDict,
)


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
      [subtype:1] [flags:1] [tank_id:2 LE] [x:1] [y:1] [extra:7]

    Args:
        data: Decoded container body bytes (must be 13 bytes).

    Returns:
        Decoded position update data with x,y coordinates.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 13, "PositionUpdate")

    flags = data[1]
    tank_id = extract_uint16_le(data, 2, "PositionUpdate.tank_id")
    x = data[4]
    y = data[5]
    extra_data = bytes(data[6:])

    return PositionUpdateDict(
        msg_type="position_update",
        flags=flags,
        tank_id=tank_id,
        x=x,
        y=y,
        extra_data=extra_data,
    )


def is_movement_structure(data: bytes) -> bool:
    """Check if data matches movement message structure.

    Movement criteria:
    - Length >= 14 bytes (header + at least 2 waypoints)
    - First byte (subtype) is 0x47 ('G'), OR
    - Last 4 bytes are all direction characters (w/s/n/e)

    The subtype check catches short movements where the last 4 bytes may
    include non-direction bytes (e.g., position data padding). The tail
    check catches longer movements and serves as a fallback.

    This distinguishes Movement from TankRegistry which has subtype 0x21 ('!')
    and ends with tank names (alphanumeric, not all directions).

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches movement pattern.
    """
    if len(data) < 14:
        return False
    # Primary check: subtype byte indicates Movement
    if data[0] == SUBTYPE_MOVEMENT:
        return True
    # Fallback check: last 4 bytes are all direction chars
    tail4 = data[-4:]
    return all(b in DIRECTION_BYTES for b in tail4)


def decode_movement(data: bytes) -> MovementDict:
    """Decode movement message from container body.

    Structure:
      [0]     subtype (0x47 'G')
      [1]     flags (0x7E=self, 0x1E=enemy)
      [2-3]   packed: (start_x << 8) | low_byte
      [4]     start_y
      [5-7]   unknown
      [8-11]  player_id (LE uint32)
      [12+]   waypoints

    Args:
        data: Decoded container body bytes (must pass is_movement_structure).

    Returns:
        Decoded movement data with start position and waypoint path.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_min_length(data, 14, "Movement")

    flags = data[1]
    packed = extract_uint16_le(data, 2, "Movement.packed")
    start_x = packed >> 8
    start_y = data[4]
    player_id = int.from_bytes(data[8:12], "little")

    # Extract waypoints (everything after position_data)
    waypoint_bytes = data[12:]
    waypoints = "".join(chr(b) for b in waypoint_bytes if b in DIRECTION_BYTES)

    # Self movements have flags 0x7E (bits 5-6 set), enemy have 0x1E
    is_self = (flags & 0x60) != 0

    return MovementDict(
        msg_type="movement",
        flags=flags,
        start_x=start_x,
        start_y=start_y,
        player_id=player_id,
        tank_id=None,  # Resolved via PlayerIdMapper
        waypoints=waypoints,
        is_self=is_self,
    )


__all__ = [
    "decode_movement",
    "decode_position_update",
    "is_movement_structure",
    "is_position_update_structure",
]
