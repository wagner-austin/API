"""Text message decoders.

This module handles decoding of text-format messages (no XOR encoding):
join confirmation and world info.
"""

from __future__ import annotations

from tankpit_bot.protocol.constants import (
    MSG_CACHE_UPDATE,
    MSG_DEACTIVATE,
    MSG_PROMOTION,
    MSG_TANK_POS,
)
from tankpit_bot.protocol.types import (
    AutoscrollAckDict,
    ChatAckDict,
    JoinConfirmDict,
    WorldInfoDict,
)
from tankpit_bot.wire.helpers import (
    DecodeError,
    require_parts,
    require_prefix,
)

_ACK_FLAG_DISABLED = ord("0")
_ACK_FLAG_ENABLED = ord("1")


def try_decode_plaintext_ack(raw_body: bytes) -> AutoscrollAckDict | ChatAckDict | None:
    """Decode a plaintext toggle ack, or None when the body is not one.

    The server acknowledges the plaintext client toggles (``A{flag}``
    autoscroll, ``C{flag}`` chat) by echoing the two-byte command back
    un-XORed — raw ``41 30``/``41 31``/``43 30``/``43 31`` on the wire
    (key-probe capture 2026-07-24: autoscroll ack ``4130`` = ``"A0"``,
    chat ack ``4331`` = ``"C1"``). Both letters are overloaded with
    XOR-encoded binary frames (0x41 Deactivation, 0x43 CacheUpdate),
    so the discrimination must happen here, BEFORE any XOR decode:
    exactly two raw bytes with an ASCII ``0``/``1`` flag.

    Args:
        raw_body: Raw message body including type byte, NOT XOR-decoded.

    Returns:
        The decoded ack, or None when the body is not a plaintext ack
        (and should take the XOR-decoded binary route instead).
    """
    if len(raw_body) != 2 or raw_body[1] not in (_ACK_FLAG_DISABLED, _ACK_FLAG_ENABLED):
        return None
    enabled = raw_body[1] == _ACK_FLAG_ENABLED
    if raw_body[0] == MSG_DEACTIVATE:
        return AutoscrollAckDict(msg_type="autoscroll_ack", enabled=enabled)
    if raw_body[0] == MSG_CACHE_UPDATE:
        return ChatAckDict(msg_type="chat_ack", enabled=enabled)
    return None


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
        game_start=parts[1],
        name=parts[2],
        rank=int(parts[3]),
        active_forces=[int(p) for p in parts[4:8] if p.isdigit()],
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
    "try_decode_plaintext_ack",
]
