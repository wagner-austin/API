"""Position container message decoders.

0x47 Movement was moved to the protocol layer 2026-06-19 (single source
of truth -- the container decoder had misinterpreted bytes 8-11 as a
"player_id" field, but tpclient.js Lg.h reads those bytes as lb_score
and rank).
"""

from __future__ import annotations

from tankpit_bot.container.helpers import (
    ContainerDecodeError,
    extract_uint16_le,
    require_exact_length,
)
from tankpit_bot.container.types import PositionUpdateDict

SUBTYPE_POSITION_UPDATE = 0x24


def is_position_update_structure(data: bytes) -> bool:
    """Check if data matches position update message structure.

    Position update criteria:
    - Exactly 13 bytes
    - First byte uses the verified position-update subtype ``0x24``

    Args:
        data: Decoded container body bytes.

    Returns:
        True if structure matches position update pattern.
    """
    return len(data) == 13 and data[0] == SUBTYPE_POSITION_UPDATE


def decode_position_update(data: bytes) -> PositionUpdateDict:
    """Decode position update message from container body.

    Structure (13 bytes):
      [subtype:1=0x24] [flags:1] [tank_id:2 LE] [x:1] [y:1] [extra:7]

    Args:
        data: Decoded container body bytes (must be 13 bytes).

    Returns:
        Decoded position update data with x,y coordinates.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    require_exact_length(data, 13, "PositionUpdate")
    if data[0] != SUBTYPE_POSITION_UPDATE:
        raise ContainerDecodeError(
            f"PositionUpdate: expected subtype 0x{SUBTYPE_POSITION_UPDATE:02X}, got 0x{data[0]:02X}"
        )

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


__all__ = [
    "SUBTYPE_POSITION_UPDATE",
    "decode_position_update",
    "is_position_update_structure",
]
