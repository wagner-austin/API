"""Session-event payloads: promotions, stats, rosters, and lifecycle.

One of the nine payload families under
:mod:`tankpit_bot.protocol.types`, split from the former single
959-line module. Membership mirrors
:mod:`tankpit_bot.protocol.decoders.session_events` -- the decoder that
produces these payloads owns their definitions.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class BuildPickupDict(TypedDict):
    """Obstacle / bridge build / pickup event (0x42 'B' message).

    Trace-verified from tpclient.js Jg.h (V.B):
      X(a[0], a[1]) = tank_id (LE u16)
      a[2]          = source_x  -- where the tank was when the action fired
      a[3]          = source_y
      a[4]          = drop_x    -- target tile receiving the obstacle / bridge
      a[5]          = drop_y
      a[6]          = direction -- new facing for the acting tank (passed to We())
      a[7]          = obstacle_type -- assigned to tile.j; ``1`` means bridge
                       module, other non-zero values are obstacle subtypes.
                       Production captures (2026-06-19) show ``2`` for a
                       regular obstacle drop; ``0`` is the cleared state.
      a[8]          = flag      -- influences pickup-visibility branch (this.s in JS)

    JS Jg.prototype.h:
      * Updates the tank's facing at (source_x, source_y).
      * Stamps ``drop_x, drop_y`` tile's ``j`` field with ``obstacle_type``.
      * For the player's own tank, prints "Bridge module built"
        (when ``obstacle_type == 1``), "Obstacle dropped", or
        "Obstacle picked up" depending on ``a.la`` (carry state).
    """

    msg_type: Literal[0x42]
    tank_id: int
    source_x: int
    source_y: int
    drop_x: int
    drop_y: int
    direction: int
    obstacle_type: int
    flag: int


class DecorationDict(TypedDict):
    """Decoration / award notification (0x4E 'N' message).

    Trace-verified from tpclient.js Sf.h (V.N):
      X(a[0], a[1]) = tank_id (LE u16)
      a[2]          = slot (decoration slot index into the tank's ``v[]`` table)
      a[3]          = level (new decoration level for that slot)

    JS Sf.prototype.h prints a banner only when the new ``level`` raises
    the tank's current ``v[slot]`` -- the new value is assigned
    unconditionally. The decoration label is
    ``nb[3 * slot + level - 1]`` (3 medals per slot).
    """

    msg_type: Literal[0x4E]
    tank_id: int
    slot: int
    level: int


class PromotionDict(TypedDict):
    """Binary promotion notification (0x2B '+' message, gameplay).

    Trace-verified from tpclient.js Rf.h (V["+"]):
      a[0] = new_rank (target rank, indexes into rank-name table)
      a[1] = was_promoted (1 = "You have been promoted!" banner;
                           0 = silent rank set, e.g. on join)

    Distinct from the text-format ``WorldInfoDict`` (also 0x2B) emitted
    by the server at lobby/ROOM_LIST time. The two are disambiguated by
    wire body length: Rf carries exactly 2 XOR-decoded payload bytes.
    """

    msg_type: Literal[0x2B]
    new_rank: int
    was_promoted: bool


class ActionDoneDict(TypedDict):
    """Action completion marker (0x54 'T' message)."""

    msg_type: Literal[0x54]


class ChatMessageDict(TypedDict):
    """Chat message (M message)."""

    msg_type: Literal[0x4D]
    sender_id: int
    message_type: int
    x: int | None
    y: int | None


class StatisticsDict(TypedDict):
    """Statistics display (V message)."""

    msg_type: Literal[0x56]
    playtime_hours: int
    playtime_minutes: int
    playtime_seconds: int
    destroyed: int
    deactivated: int
    score: int


class ActiveForcesDict(TypedDict):
    """Active forces count (* message)."""

    msg_type: Literal[0x2A]
    team_counts: list[int]


class ActivePlayerEntry(TypedDict):
    """One row in an 0x2F ActivePlayers list.

    Attributes:
        tank_id: 16-bit LE tank id from the wire.
        rank: 1-byte rank index used by the JS client to render the
            ``ec[rank]`` label in the active-players banner. Same
            domain as ``TankStateDict.rank``.
    """

    tank_id: int
    rank: int


class ActivePlayersDict(TypedDict):
    """0x2F ``/`` ActivePlayers list.

    Server-broadcast roster of every active player in the room,
    decoded from the JS Yg.h handler at ``tpclient.pretty.js:4653``.
    Wire shape is repeating 3-byte records: ``(tank_id_lo,
    tank_id_hi, rank)``. The bot consumes this to know who's actually
    in the room without needing to spam ``/`` queries.
    """

    msg_type: Literal[0x2F]
    players: list[ActivePlayerEntry]


class Top10EntryDict(TypedDict):
    """One row of the 0x31 ``1`` Top10 leaderboard.

    Attributes:
        position: 1-based leaderboard rank in this Top10 list.
        score: 24-bit BE score.
        team: Team id (0-3).
        rank: Military rank (0-8).
        name: UTF-8 player name.
        tank_id: ``-1`` when the server does not echo the persistent
            tank id on this row; otherwise the value carried by the
            wire. The JS client hyperlinks ``tank_id >= 500`` rows to
            ``/tanks/profile?tank_id=...`` so the value is at least
            sometimes a persistent identifier.
    """

    position: int
    score: int
    team: int
    rank: int
    name: str
    tank_id: int


class Top10Dict(TypedDict):
    """0x31 ``1`` Top10 leaderboard broadcast.

    Wire shape (JS Zg.h at ``tpclient.pretty.js:4679``):
      a[0]      = team_filter (255 = all-team Top10, else team id)
      a[1..3]   = viewer's score (24-bit BE)
      a[4]      = viewer's leaderboard position
      a[5..]    = repeating rows: position(1), score(3 BE), team(1),
                  rank(1), name_len(1), name(name_len bytes)

    Attributes:
        team_filter: ``255`` for the all-team list, else the team id
            this Top10 row applies to.
        viewer_score: 24-bit BE score of the player viewing the list.
        viewer_position: 1-based leaderboard position of the viewer.
        entries: Decoded rows in the order the server sent them
            (top to bottom).
    """

    msg_type: Literal[0x31]
    team_filter: int
    viewer_score: int
    viewer_position: int
    entries: list[Top10EntryDict]


class PingResponseDict(TypedDict):
    """0x60 `` ` `` PingResponse from the server.

    JS V[``\\``] = we (``tpclient.pretty.js:3839``) handler is a no-op:
    the server just acknowledges the bot is still considered
    connected. Decoded for telemetry so the bot's events stream can
    timestamp every heartbeat.
    """

    msg_type: Literal[0x60]


class ConnectionLostDict(TypedDict):
    """0x7E ``~`` ConnectionLost from the server.

    JS V[``~``] = xe (``tpclient.pretty.js:3829``) triggers a
    disconnect. Decoded so the bot's events stream records WHY a
    session ended even when the transport layer doesn't surface a
    structured reason.
    """

    msg_type: Literal[0x7E]


__all__ = [
    "ActionDoneDict",
    "ActiveForcesDict",
    "ActivePlayerEntry",
    "ActivePlayersDict",
    "BuildPickupDict",
    "ChatMessageDict",
    "ConnectionLostDict",
    "DecorationDict",
    "PingResponseDict",
    "PromotionDict",
    "StatisticsDict",
    "Top10Dict",
    "Top10EntryDict",
]
