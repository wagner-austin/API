"""Miscellaneous message decoders.

This module handles decoding of miscellaneous messages:
action done, chat, statistics, active forces.
"""

from __future__ import annotations

from tankpit_bot.protocol.helpers import require_min_length, x16
from tankpit_bot.protocol.types import (
    ActionDoneDict,
    ActiveForcesDict,
    BuildPickupDict,
    ChatMessageDict,
    DecorationDict,
    PromotionDict,
    StatisticsDict,
)


def decode_action_done(data: bytes) -> ActionDoneDict:
    """Decode action done message.

    Args:
        data: XOR-decoded message body.

    Returns:
        Empty action done dict.
    """
    return ActionDoneDict(msg_type=0x54)


def decode_chat_message(data: bytes) -> ChatMessageDict:
    """Decode chat message from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded chat message.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 3, "ChatMessage")
    return ChatMessageDict(
        msg_type=0x4D,
        sender_id=x16(data[0], data[1]),
        message_type=data[2],
        x=data[3] if len(data) > 3 else None,
        y=data[4] if len(data) > 4 else None,
    )


def decode_statistics(data: bytes) -> StatisticsDict:
    """Decode statistics from XOR-decoded data.

    Trace-verified from tpclient.js Wg.h (line 4617-4621):
      Long format (len > 12):
        a[0:2]  = hours (LE u16)
        a[2]    = minutes
        a[3]    = seconds
        a[4:8]  = destroyed (32-bit BE)
        a[8:10] = deactivated (LE u16)
        a[10:14] = promo_points (32-bit BE)
      Short format (len <= 12):
        a[0:2]  = hours (LE u16)
        a[2]    = minutes
        a[3]    = seconds
        a[4:6]  = destroyed (LE u16)
        a[6:8]  = deactivated (LE u16)
        a[8:12] = promo_points (32-bit BE)

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded statistics.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 12, "Statistics")
    if len(data) > 12:
        destroyed = int.from_bytes(data[4:8], "big")
        deactivated = x16(data[8], data[9])
        score = int.from_bytes(data[10:14], "big")
    else:
        destroyed = x16(data[4], data[5])
        deactivated = x16(data[6], data[7])
        score = int.from_bytes(data[8:12], "big")
    return StatisticsDict(
        msg_type=0x56,
        playtime_hours=x16(data[0], data[1]),
        playtime_minutes=data[2],
        playtime_seconds=data[3],
        destroyed=destroyed,
        deactivated=deactivated,
        score=score,
    )


def decode_promotion(data: bytes) -> PromotionDict:
    """Decode the binary 0x2B '+' Promotion message.

    Trace-verified from tpclient.js Rf.h (V["+"]):
      a[0] = new_rank
      a[1] = was_promoted (1 = banner shown; 0 = silent rank set)

    Args:
        data: XOR-decoded message body (without the 0x2B prefix).

    Returns:
        Decoded promotion event.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 2, "Promotion")
    return PromotionDict(
        msg_type=0x2B,
        new_rank=data[0],
        was_promoted=data[1] == 1,
    )


def decode_build_pickup(data: bytes) -> BuildPickupDict:
    """Decode the 0x42 'B' BuildPickup message.

    Trace-verified from tpclient.js Jg.h (V.B):
      X(a[0], a[1]) = tank_id
      a[2:4]        = source_x, source_y
      a[4:6]        = drop_x, drop_y
      a[6]          = direction
      a[7]          = obstacle_type (1 = bridge module; other non-zero
                                     values = obstacle subtypes; 0 = cleared)
      a[8]          = flag (pickup-visibility branch in JS)

    Args:
        data: XOR-decoded message body (without the 0x42 prefix).

    Returns:
        Decoded build / pickup event.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 9, "BuildPickup")
    return BuildPickupDict(
        msg_type=0x42,
        tank_id=x16(data[0], data[1]),
        source_x=data[2],
        source_y=data[3],
        drop_x=data[4],
        drop_y=data[5],
        direction=data[6],
        obstacle_type=data[7],
        flag=data[8],
    )


def decode_decoration(data: bytes) -> DecorationDict:
    """Decode the 0x4E 'N' Decoration / award message.

    Trace-verified from tpclient.js Sf.h (V.N):
      X(a[0], a[1]) = tank_id (LE u16)
      a[2]          = slot
      a[3]          = level

    Args:
        data: XOR-decoded message body (without the 0x4E prefix).

    Returns:
        Decoded decoration event.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 4, "Decoration")
    return DecorationDict(
        msg_type=0x4E,
        tank_id=x16(data[0], data[1]),
        slot=data[2],
        level=data[3],
    )


def decode_active_forces(data: bytes) -> ActiveForcesDict:
    """Decode active forces from XOR-decoded data.

    Args:
        data: XOR-decoded message body.

    Returns:
        Decoded active forces.

    Raises:
        DecodeError: If decoding fails.
    """
    require_min_length(data, 4, "ActiveForces")
    return ActiveForcesDict(msg_type=0x2A, team_counts=[data[i] for i in range(4)])


__all__ = [
    "decode_action_done",
    "decode_active_forces",
    "decode_build_pickup",
    "decode_chat_message",
    "decode_decoration",
    "decode_promotion",
    "decode_statistics",
]
