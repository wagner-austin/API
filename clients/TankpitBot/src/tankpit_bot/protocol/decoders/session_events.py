"""Decoders for non-geometric server messages.

This module handles the server-emitted message kinds that do NOT
update the world-state geometry (positions, tiles, mines, etc.). They
fall into two themes:

* **Session / lobby**: chat (0x4D), active_players (0x2F), top10
  (0x31), ping_response (0x60), connection_lost (0x7E).
* **Scoring / progress**: statistics (0x56), promotion (0x2B),
  build_pickup (0x42), decoration (0x4E), active_forces (0x2A),
  action_done (0x54).

Previously named ``misc.py`` -- the rename reflects that these are a
coherent group (non-state-mutating session/scoring events) rather
than a catch-all dumping ground.
"""

from __future__ import annotations

from tankpit_bot.protocol.types import (
    ActionDoneDict,
    ActiveForcesDict,
    ActivePlayerEntry,
    ActivePlayersDict,
    BuildPickupDict,
    ChatMessageDict,
    ConnectionLostDict,
    DecorationDict,
    PingResponseDict,
    PromotionDict,
    StatisticsDict,
    Top10Dict,
    Top10EntryDict,
)
from tankpit_bot.wire.helpers import DecodeError, require_min_length, x16


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


def decode_active_players(data: bytes) -> ActivePlayersDict:
    """Decode 0x2F ``/`` ActivePlayers from XOR-decoded body bytes.

    JS Yg.h (``tpclient.pretty.js:4653``) reads repeating 3-byte
    records: ``(tank_id_lo, tank_id_hi, rank)``. The body length must
    be a multiple of 3.

    Args:
        data: XOR-decoded message body (after the outer 0x2F byte).

    Returns:
        Decoded :class:`ActivePlayersDict` with one entry per record.

    Raises:
        DecodeError: When the body length isn't a multiple of 3.
    """
    if len(data) % 3 != 0:
        raise DecodeError(f"ActivePlayers: body must be a multiple of 3 bytes, got {len(data)}")
    players = [
        ActivePlayerEntry(tank_id=x16(data[i], data[i + 1]), rank=data[i + 2])
        for i in range(0, len(data), 3)
    ]
    return ActivePlayersDict(msg_type=0x2F, players=players)


def decode_top10(data: bytes) -> Top10Dict:
    """Decode 0x31 ``1`` Top10 from XOR-decoded body bytes.

    JS Zg.h (``tpclient.pretty.js:4679``) wire shape:

      ``a[0]      = team_filter (255 = all-team)``
      ``a[1..3]   = viewer_score (24-bit BE)``
      ``a[4]      = viewer_position``
      ``a[5..]    = rows of [position, score(3 BE), team, rank,
                              name_len, name(name_len bytes)]``

    Args:
        data: XOR-decoded message body (after the outer 0x31 byte).

    Returns:
        Decoded :class:`Top10Dict`.

    Raises:
        DecodeError: When the body is shorter than the 5-byte header
            or a row is truncated.
    """
    require_min_length(data, 5, "Top10")
    team_filter = data[0]
    viewer_score = 256 * (256 * data[1] + data[2]) + data[3]
    viewer_position = data[4]
    entries: list[Top10EntryDict] = []
    offset = 5
    while offset < len(data):
        if offset + 7 > len(data):
            raise DecodeError(
                f"Top10: truncated row header at offset {offset}, body len {len(data)}"
            )
        position = data[offset]
        score = 256 * (256 * data[offset + 1] + data[offset + 2]) + data[offset + 3]
        team = data[offset + 4]
        rank = data[offset + 5]
        name_len = data[offset + 6]
        name_start = offset + 7
        name_end = name_start + name_len
        if name_end > len(data):
            raise DecodeError(
                f"Top10: row name spills past body end "
                f"(name_start={name_start}, name_len={name_len}, body_len={len(data)})"
            )
        name = data[name_start:name_end].decode("utf-8", errors="replace")
        entries.append(
            Top10EntryDict(
                position=position,
                score=score,
                team=team,
                rank=rank,
                name=name,
                tank_id=-1,
            )
        )
        offset = name_end
    return Top10Dict(
        msg_type=0x31,
        team_filter=team_filter,
        viewer_score=viewer_score,
        viewer_position=viewer_position,
        entries=entries,
    )


def decode_ping_response(data: bytes) -> PingResponseDict:
    """Decode 0x60 PingResponse.

    JS V[``\\``] = we (``tpclient.pretty.js:3839``) handler is a no-op
    on the body; we mirror that by returning a bare typed message
    suitable for emission into the events stream as a structured
    heartbeat record.

    Args:
        data: XOR-decoded message body (unused; the handler is bare).

    Returns:
        :class:`PingResponseDict` carrying just the msg_type tag.
    """
    del data
    return PingResponseDict(msg_type=0x60)


def decode_connection_lost(data: bytes) -> ConnectionLostDict:
    """Decode 0x7E ConnectionLost.

    JS V[``~``] = xe (``tpclient.pretty.js:3829``) triggers a
    disconnect with no body; we mirror it as a typed event so the
    bot's session log captures the server-side disconnect signal.

    Args:
        data: XOR-decoded message body (unused).

    Returns:
        :class:`ConnectionLostDict` carrying just the msg_type tag.
    """
    del data
    return ConnectionLostDict(msg_type=0x7E)


__all__ = [
    "decode_action_done",
    "decode_active_forces",
    "decode_active_players",
    "decode_build_pickup",
    "decode_chat_message",
    "decode_connection_lost",
    "decode_decoration",
    "decode_ping_response",
    "decode_promotion",
    "decode_statistics",
    "decode_top10",
]
