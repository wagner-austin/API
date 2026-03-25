"""Tank message decoders.

This module handles decoding of tank-related messages:
tank info, entry, exit, status, and status sync.
"""

from __future__ import annotations

from tankpit_bot.container import (
    ContainerMessage,
    decode_container_message,
)
from tankpit_bot.protocol.helpers import require_min_length, x16
from tankpit_bot.protocol.types import (
    TankEntryDict,
    TankExitDict,
    TankInfoDict,
    TankStatusDict,
    TankStatusSyncDict,
)


def decode_tank_info(data: bytes) -> TankInfoDict:
    """Decode tank info from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x21 prefix).

    Returns:
        Decoded tank info.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 10, "TankInfo")
    team = data[0]
    tank_id = x16(data[1], data[2])
    decoration_state = bytes(data[3:7])
    score = 256 * (256 * data[7] + data[8]) + data[9] if len(data) >= 10 else 0
    name = data[10:].decode("utf-8", errors="replace") if len(data) > 10 else ""
    return TankInfoDict(
        msg_type=0x21,
        tank_id=tank_id,
        team=team,
        decoration_state=decoration_state,
        score=score,
        name=name,
    )


def decode_tank_entry(data: bytes) -> TankEntryDict:
    """Decode tank entry from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded tank entry.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 10, "TankEntry")
    tank_id = data[0]
    x = x16(data[1], data[2])
    y = data[3]
    name = data[10:].decode("utf-8", errors="replace") if len(data) > 10 else ""
    return TankEntryDict(msg_type=0x28, tank_id=tank_id, x=x, y=y, name=name)


def decode_tank_exit(data: bytes) -> TankExitDict:
    """Decode tank exit from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x58 prefix).

    Returns:
        Decoded tank exit.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 2, "TankExit")
    return TankExitDict(msg_type=0x58, tank_id=x16(data[0], data[1]))


def decode_tank_status_sync(data: bytes) -> TankStatusSyncDict:
    """Decode tank status sync from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x2E prefix).

    Returns:
        Decoded tank status sync.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 8, "TankStatusSync")
    subtype = data[0]
    tank_id = x16(data[1], data[2])
    damage_state = data[3]
    rank = data[4]
    flags = bytes(data[5:8]) if len(data) > 7 else b""

    if len(data) >= 12:
        lb_pos = x16(data[6], data[7])
        fuel: int | None = x16(data[10], data[11])
    else:
        lb_pos = x16(data[6], data[7]) if len(data) > 7 else 0
        fuel = None

    return TankStatusSyncDict(
        msg_type=0x2E,
        subtype=subtype,
        tank_id=tank_id,
        damage_state=damage_state,
        rank=rank,
        flags=flags,
        leaderboard_position=lb_pos,
        fuel=fuel,
    )


def decode_tank_status(data: bytes) -> TankStatusDict:
    """Decode full tank status from XOR-decoded data.

    Args:
        data: XOR-decoded message body (without 0x3E prefix).

    Returns:
        Decoded tank status.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 13, "TankStatus")
    info_byte = data[0]
    team = info_byte & 0x03
    rank = (info_byte >> 4) & 0x0F
    tank_id = x16(data[1], data[2])
    decoration_state = bytes(data[3:7])
    lb_score = 256 * (256 * data[7] + data[8]) + data[9] if len(data) >= 10 else 0
    lb_pos = 256 * (256 * data[10] + data[11]) + data[12] if len(data) >= 13 else 0
    name = data[13:].decode("utf-8", errors="replace") if len(data) > 13 else ""
    return TankStatusDict(
        msg_type=0x3E,
        team=team,
        rank=rank,
        tank_id=tank_id,
        decoration_state=decoration_state,
        leaderboard_score=lb_score,
        leaderboard_position=lb_pos,
        name=name,
    )


def decode_0x2e_message(data: bytes) -> ContainerMessage:
    """Decode 0x2E container message using structure-based matching.

    Uses container_decoder module which identifies messages by STRUCTURE
    (length, field positions) rather than subtype bytes, since XOR encoding
    with session-specific magic keys causes subtype values to vary.

    Args:
        data: XOR-decoded message body (without 0x2E prefix).

    Returns:
        Decoded container message as appropriate TypedDict.

    Raises:
        ContainerDecodeError: If structure validation fails.
    """
    return decode_container_message(data)


__all__ = [
    "decode_0x2e_message",
    "decode_tank_entry",
    "decode_tank_exit",
    "decode_tank_info",
    "decode_tank_status",
    "decode_tank_status_sync",
]
