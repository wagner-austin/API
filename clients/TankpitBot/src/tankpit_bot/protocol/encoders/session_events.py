"""Session/scoring message encoders — inverses of ``decoders.session_events``."""

from __future__ import annotations

from tankpit_bot.protocol.types import (
    ActionDoneDict,
    ActiveForcesDict,
    ActivePlayersDict,
    BuildPickupDict,
    ChatMessageDict,
    ConnectionLostDict,
    DecorationDict,
    PingResponseDict,
    PromotionDict,
    StatisticsDict,
    Top10Dict,
)
from tankpit_bot.wire.helpers import pack16, pack24


def encode_action_done(message: ActionDoneDict) -> bytes:
    """Encode a 0x54 ActionDone payload (inverse of ``decode_action_done``).

    The decoder ignores the body; a single zero byte satisfies the
    envelope's 1-byte minimum. No corpus samples exist to pin the wire
    byte.

    Args:
        message: Decoded action-done heartbeat.

    Returns:
        A 1-byte body.
    """
    del message
    return bytes([0])


def encode_chat_message(message: ChatMessageDict) -> bytes:
    """Encode a 0x4D ChatMessage payload (inverse of ``decode_chat_message``).

    Args:
        message: Decoded chat message.

    Returns:
        Payload bytes without the 0x4D prefix; the optional x/y tail
        bytes are emitted only when present.
    """
    out = pack16(message["sender_id"]) + bytes([message["message_type"]])
    if message["x"] is not None:
        out += bytes([message["x"]])
        if message["y"] is not None:
            out += bytes([message["y"]])
    return out


def encode_statistics(message: StatisticsDict) -> bytes:
    """Encode a 0x56 Statistics payload (inverse of ``decode_statistics``).

    Emits the 14-byte LONG format (32-bit destroyed) — the only form
    in the corpus (317/317 bodies, 2026-07-21).

    Args:
        message: Decoded statistics.

    Returns:
        Payload bytes without the 0x56 prefix.
    """
    return (
        pack16(message["playtime_hours"])
        + bytes([message["playtime_minutes"], message["playtime_seconds"]])
        + message["destroyed"].to_bytes(4, "big")
        + pack16(message["deactivated"])
        + message["score"].to_bytes(4, "big")
    )


def encode_promotion(message: PromotionDict) -> bytes:
    """Encode a 0x2B Promotion payload (inverse of ``decode_promotion``).

    Args:
        message: Decoded promotion event.

    Returns:
        Payload bytes without the 0x2B prefix.
    """
    return bytes([message["new_rank"], 1 if message["was_promoted"] else 0])


def encode_build_pickup(message: BuildPickupDict) -> bytes:
    """Encode a 0x42 BuildPickup payload (inverse of ``decode_build_pickup``).

    Args:
        message: Decoded build/pickup event.

    Returns:
        Payload bytes without the 0x42 prefix.
    """
    return pack16(message["tank_id"]) + bytes(
        [
            message["source_x"],
            message["source_y"],
            message["drop_x"],
            message["drop_y"],
            message["direction"],
            message["obstacle_type"],
            message["flag"],
        ]
    )


def encode_decoration(message: DecorationDict) -> bytes:
    """Encode a 0x4E Decoration payload (inverse of ``decode_decoration``).

    Args:
        message: Decoded decoration/award event.

    Returns:
        Payload bytes without the 0x4E prefix.
    """
    return pack16(message["tank_id"]) + bytes([message["slot"], message["level"]])


def encode_active_forces(message: ActiveForcesDict) -> bytes:
    """Encode a 0x2A ActiveForces payload (inverse of ``decode_active_forces``).

    Args:
        message: Decoded per-team player counts.

    Returns:
        Payload bytes without the 0x2A prefix.
    """
    return bytes(message["team_counts"])


def encode_active_players(message: ActivePlayersDict) -> bytes:
    """Encode a 0x2F ActivePlayers payload (inverse of ``decode_active_players``).

    Args:
        message: Decoded player roster.

    Returns:
        Payload bytes without the 0x2F prefix (3-byte records).
    """
    out = bytearray()
    for player in message["players"]:
        out += pack16(player["tank_id"]) + bytes([player["rank"]])
    return bytes(out)


def encode_top10(message: Top10Dict) -> bytes:
    """Encode a 0x31 Top10 payload (inverse of ``decode_top10``).

    Args:
        message: Decoded leaderboard snapshot.

    Returns:
        Payload bytes without the 0x31 prefix: the 5-byte header, then
        one variable-length row per entry.
    """
    out = bytearray(
        bytes([message["team_filter"]])
        + pack24(message["viewer_score"])
        + bytes([message["viewer_position"]])
    )
    for entry in message["entries"]:
        name = entry["name"].encode("utf-8")
        out += (
            bytes([entry["position"]])
            + pack24(entry["score"])
            + bytes([entry["team"], entry["rank"], len(name)])
            + name
        )
    return bytes(out)


def encode_ping_response(message: PingResponseDict) -> bytes:
    """Encode a 0x60 PingResponse payload (inverse of ``decode_ping_response``).

    Args:
        message: Decoded ping heartbeat (carries nothing).

    Returns:
        An empty body — the JS handler is a no-op on the body.
    """
    del message
    return b""


def encode_connection_lost(message: ConnectionLostDict) -> bytes:
    """Encode a 0x7E ConnectionLost payload (inverse of ``decode_connection_lost``).

    Args:
        message: Decoded disconnect signal (carries nothing).

    Returns:
        An empty body — the JS handler triggers a disconnect with no body.
    """
    del message
    return b""


__all__ = [
    "encode_action_done",
    "encode_active_forces",
    "encode_active_players",
    "encode_build_pickup",
    "encode_chat_message",
    "encode_connection_lost",
    "encode_decoration",
    "encode_ping_response",
    "encode_promotion",
    "encode_statistics",
    "encode_top10",
]
