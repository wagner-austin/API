"""Text message decoders.

This module handles decoding of text-format messages (no XOR encoding):
join confirmation and world info.
"""

from __future__ import annotations

from tankpit_bot.protocol.constants import MSG_PROMOTION, MSG_TANK_POS
from tankpit_bot.protocol.helpers import (
    DecodeError,
    require_parts,
    require_prefix,
)
from tankpit_bot.protocol.types import JoinConfirmDict, WorldInfoDict


def decode_join_confirm(data: bytes) -> JoinConfirmDict:
    """Decode join confirmation from raw message body.

    Args:
        data: Raw message body (including = prefix).

    Returns:
        Decoded join confirmation.

    Raises:
        DecodeError: If decoding fails.
    """
    text = data.decode("utf-8", errors="replace")
    require_prefix(text, "=", "JoinConfirm")
    parts = text[1:].split("|")
    require_parts(parts, 4, "JoinConfirm")
    return JoinConfirmDict(
        msg_type=0x3D,
        team=int(parts[0]),
        join_date=parts[1],
        name=parts[2],
        rank=int(parts[3]),
        equipment=[int(p) for p in parts[4:8] if p.isdigit()],
    )


def decode_world_info(data: bytes) -> WorldInfoDict:
    """Decode world info from raw message body.

    Args:
        data: Raw message body (including + prefix).

    Returns:
        Decoded world info.

    Raises:
        DecodeError: If decoding fails.
    """
    text = data.decode("utf-8", errors="replace")
    require_prefix(text, "+", "WorldInfo")
    parts = text[1:].split("|")
    require_parts(parts, 8, "WorldInfo")
    flags_str = parts[3].split(",")
    return WorldInfoDict(
        msg_type=0x2B,
        world_id=int(parts[0]),
        name=parts[1],
        field_id=int(parts[2]),
        flags=[int(f) for f in flags_str if f.isdigit()],
        team=int(parts[4]),
        mode=parts[5],
        image=parts[6],
        year=int(parts[7]) if parts[7].isdigit() else 0,
    )


def decode_text_message(raw_body: bytes) -> JoinConfirmDict | WorldInfoDict:
    """Decode a text-format message (no XOR decoding needed).

    Args:
        raw_body: Raw message body including type byte.

    Returns:
        Decoded message object.

    Raises:
        DecodeError: If message type is unknown or decoding fails.
    """
    if len(raw_body) < 1:
        raise DecodeError("decode_text_message: empty body")

    msg_type = raw_body[0]

    if msg_type == MSG_TANK_POS:
        return decode_join_confirm(raw_body)
    if msg_type == MSG_PROMOTION:
        return decode_world_info(raw_body)

    raise DecodeError(f"decode_text_message: unknown type 0x{msg_type:02X}")


__all__ = [
    "decode_join_confirm",
    "decode_text_message",
    "decode_world_info",
]
