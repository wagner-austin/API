"""Session-broadcast dispatch: chat, promotions, rosters, and top-10.

The announcement channels that carry no per-tank state. Called by the
tank dispatcher in :mod:`tankpit_bot.sniffer.world_state_dispatch`.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot import browser, protocol
from tankpit_bot.runtime_logging import (
    emit_diagnostic,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_tanks import (
    _update_tank_position,
)
from tankpit_bot.state import (
    set_self_rank,
)

log = get_logger(__name__)


def _dispatch_self_promotion(ws: WorldService, new_rank: int, was_promoted: bool) -> None:
    """Apply a 0x2B Promotion (Rf) to self_state and emit a diagnostic.

    JS Rf.prototype.h: ``a.i.l = this.j`` -- the server-authoritative
    rank assignment to the player's own tank. ``was_promoted`` is the
    UI banner flag; ``new_rank`` is the absolute new rank index.

    Args:
        ws: World service instance.
        new_rank: New rank index (0-8).
        was_promoted: True when the server intends a "promoted" banner;
            False on silent rank resets (e.g. join-time initialization).
    """
    ws.world_state = set_self_rank(ws.world_state, new_rank, browser.get_current_time_ms())
    emit_diagnostic(
        diagnostic_kind="self_promotion",
        new_rank=new_rank,
        was_promoted=was_promoted,
    )


def _dispatch_chat_message(
    ws: WorldService,
    sender_id: int,
    message_id: int,
    x: int | None,
    y: int | None,
) -> None:
    """Record an inbound 0x4D chat broadcast.

    Two consumers: the events stream (a ``chat_received`` diagnostic
    with the preset text resolved from the E[] table) and the world
    service's send-receipt latch -- the server echoes the bot's own
    chats back as 0x4D, and that echo is the ONLY confirmation a chat
    survived the server-side flood mute (sniff-20260729-214411: after
    8 rapid sends every later chat was silently swallowed; wiki
    [[chat-messages]]). The carried ``x, y`` is whatever the sender's
    client put in the send frame -- self-reported, not
    server-verified -- so it never mutates tank positions.

    Args:
        ws: World service instance.
        sender_id: Chatting tank's id from the wire.
        message_id: Preset chat message ID (E[] table index).
        x: Sender-reported X tile, or None on a coordinate-less frame.
        y: Sender-reported Y tile, or None on a coordinate-less frame.
    """
    from tankpit_bot.protocol.chat import chat_message_text

    text = chat_message_text(message_id)
    self_state = ws.world_state["self_state"]
    is_self_echo = self_state is not None and sender_id == self_state["tank_id"]
    if is_self_echo:
        ws.last_chat_echo_message_id = message_id
    else:
        # Any chat from another tank marks them as a responsive,
        # combat-consenting player (human-consent contract 2026-07-30).
        ws.chat_seen_tank_ids.add(sender_id)
    sender = ws.world_state["tanks"].get(str(sender_id))
    sender_name = sender["name"] if sender is not None else ""
    log.info(
        "CHAT: %s (id=%d) says %r%s",
        sender_name if sender_name else f"tank-{sender_id}",
        sender_id,
        text,
        " [self echo]" if is_self_echo else "",
    )
    emit_diagnostic(
        diagnostic_kind="chat_received",
        sender_id=sender_id,
        sender_name=sender_name,
        message_id=message_id,
        text=text,
        x=x if x is not None else -1,
        y=y if y is not None else -1,
        is_self_echo=is_self_echo,
    )


def _emit_active_players(
    ws: WorldService,
    players: list[protocol.ActivePlayerEntry],
) -> None:
    """Persist an 0x2F ActivePlayers roster and emit a structured diagnostic.

    Args:
        ws: World service instance.
        players: Decoded roster entries in server-sent order.
    """
    ws.active_players = [(player["tank_id"], player["rank"]) for player in players]
    emit_diagnostic(
        diagnostic_kind="active_players",
        count=len(players),
        tank_ids=",".join(str(player["tank_id"]) for player in players),
    )


def _emit_top10(
    ws: WorldService,
    team_filter: int,
    viewer_score: int,
    viewer_position: int,
    entries: list[protocol.Top10EntryDict],
) -> None:
    """Persist a 0x31 Top10 snapshot on the world service + emit a diagnostic.

    The Top10 broadcast can come with zero rows (very fresh sessions
    or empty leaderboards); guard the ``entries[0]`` peek so we still
    emit a structured event with row_count=0.

    Args:
        ws: World service instance.
        team_filter: Wire's team_filter byte (255 = all teams).
        viewer_score: 24-bit BE score for the viewing player.
        viewer_position: 1-based leaderboard rank for the viewer.
        entries: Decoded Top10 rows in server-sent order.
    """
    ws.top10_viewer_score = viewer_score
    ws.top10_viewer_position = viewer_position
    ws.top10_team_filter = team_filter
    top_name: str = entries[0]["name"] if entries else ""
    top_score: int = entries[0]["score"] if entries else 0
    emit_diagnostic(
        diagnostic_kind="top10",
        team_filter=team_filter,
        viewer_score=viewer_score,
        viewer_position=viewer_position,
        row_count=len(entries),
        top_name=str(top_name),
        top_score=int(top_score),
    )


def _dispatch_session_broadcasts(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch session-level server broadcasts.

    Covers 0x2F ActivePlayers, 0x31 Top10, 0x60 PingResponse, and 0x7E
    ConnectionLost -- all of which carry no tank-state geometry but
    DO carry session information the bot's events stream should
    capture. Split out of :func:`_dispatch_tank_announcements` to keep
    the latter under the C901 complexity ceiling.

    Args:
        ws: World service instance.
        decoded: Decoded binary protocol message.

    Returns:
        True when the message matched one of the broadcast shapes,
        False otherwise (so the caller can fall through to other
        dispatchers).
    """
    match decoded:
        case {"msg_type": 0x2F, "players": list(players)}:
            _emit_active_players(ws, players)
            return True
        case {
            "msg_type": 0x31,
            "team_filter": int(team_filter),
            "viewer_score": int(viewer_score),
            "viewer_position": int(viewer_position),
            "entries": list(entries),
        }:
            _emit_top10(ws, team_filter, viewer_score, viewer_position, entries)
            return True
        case {"msg_type": 0x60}:
            ws.last_ping_response_ms = browser.get_current_time_ms()
            emit_diagnostic(diagnostic_kind="ping_response")
            return True
        case {"msg_type": 0x7E}:
            emit_diagnostic(diagnostic_kind="connection_lost")
            return True
    return False


def _dispatch_tank_announcements(ws: WorldService, decoded: protocol.BinaryMessage) -> bool:
    """Dispatch announcement-style messages with no positional effect.

    Covers 0x29 TankExit, 0x2B Promotion, 0x4E Decoration, 0x56
    Statistics. The 0x42 BuildPickup is handled here too because it
    behaves like an event observation -- it does mutate the actor's
    position via :func:`_update_tank_position` but contributes no
    structural world-state change beyond that.
    """
    match decoded:
        case {
            "msg_type": 0x29,
            "team": int(team),
            "tank_id": int(tid),
            "was_silent": bool(was_silent),
            "was_eliminated": bool(was_eliminated),
        }:
            emit_diagnostic(
                diagnostic_kind="tank_exit_announcement",
                team=team,
                tank_id=tid,
                was_silent=was_silent,
                was_eliminated=was_eliminated,
            )
            return True
        case {
            "msg_type": 0x2B,
            "new_rank": int(new_rank),
            "was_promoted": bool(was_promoted),
        }:
            _dispatch_self_promotion(ws, new_rank, was_promoted)
            return True
        case {
            "msg_type": 0x4E,
            "tank_id": int(tid),
            "slot": int(slot),
            "level": int(level),
        }:
            emit_diagnostic(
                diagnostic_kind="tank_decoration",
                tank_id=tid,
                slot=slot,
                level=level,
            )
            return True
        case {
            "msg_type": 0x42,
            "tank_id": int(tid),
            "source_x": int(sx),
            "source_y": int(sy),
            "drop_x": int(dx),
            "drop_y": int(dy),
            "obstacle_type": int(obstacle_type),
        }:
            _update_tank_position(ws, tid, sx, sy, "wire_0x42_build_pickup")
            emit_diagnostic(
                diagnostic_kind="build_pickup",
                tank_id=tid,
                source_x=sx,
                source_y=sy,
                drop_x=dx,
                drop_y=dy,
                obstacle_type=obstacle_type,
            )
            return True
        case {
            "msg_type": 0x56,
            "playtime_hours": int(hours),
            "playtime_minutes": int(minutes),
            "playtime_seconds": int(seconds),
            "destroyed": int(destroyed),
            "deactivated": int(deactivated),
            "score": int(score),
        }:
            playtime_total = hours * 3600 + minutes * 60 + seconds
            ws.career_destroyed = destroyed
            ws.career_deactivated = deactivated
            ws.career_score = score
            ws.career_playtime_seconds_total = playtime_total
            ws.career_stats_last_update_ms = browser.get_current_time_ms()
            emit_diagnostic(
                diagnostic_kind="self_statistics",
                playtime_hours=hours,
                playtime_minutes=minutes,
                playtime_seconds=seconds,
                playtime_seconds_total=playtime_total,
                destroyed=destroyed,
                deactivated=deactivated,
                score=score,
            )
            return True
        case {"msg_type": 0x3C, "message": str(message)}:
            # ``message`` is reserved by the runtime logger as the
            # human-readable channel line; use ``text`` for the payload.
            emit_diagnostic(diagnostic_kind="supervisor_text", text=message)
            return True
        case {
            "msg_type": 0x4D,
            "sender_id": int(sender_id),
            "message_type": int(chat_message_id),
            "x": chat_x,
            "y": chat_y,
        }:
            _dispatch_chat_message(
                ws,
                sender_id,
                chat_message_id,
                chat_x if isinstance(chat_x, int) else None,
                chat_y if isinstance(chat_y, int) else None,
            )
            return True
    return _dispatch_session_broadcasts(ws, decoded)


__all__ = [
    "log",
]
