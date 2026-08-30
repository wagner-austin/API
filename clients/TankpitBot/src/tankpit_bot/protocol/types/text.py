"""Text-format message payloads (no XOR encoding) and the two acks.

One of the nine payload families under
:mod:`tankpit_bot.protocol.types`, split from the former single
959-line module. Membership mirrors
:mod:`tankpit_bot.protocol.decoders.text` -- the decoder that
produces these payloads owns their definitions.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class JoinConfirmDict(TypedDict):
    """Join confirmation (= message) - TEXT format.

    Format: =<team>|<game_start>|<name>|<rank>|<orange>|<purple>|
    <blue>|<red>

    The four trailing counts are ACTIVE FORCES -- how many tanks are
    playing each color in that room -- not equipment, which is what
    they were named here until 2026-08-28. See
    :attr:`active_forces`.
    """

    msg_type: Literal[0x3D]
    team: int
    game_start: str
    """The date the client's lobby panel prints as "Game start".

    Named for the label the game itself renders (``tpclient.js`` builds
    the panel as ``Game start: `` + this field), NOT for a meaning we
    have pinned -- it was called ``join_date`` until 2026-08-28 on no
    evidence at all, which is the mistake this name avoids repeating.

    It is the ROOM's date, not the account's or the tank's. It varies
    per room within one account (Arterial: ``Sep. 25, 2012`` on World,
    ``Jan. 08, 2013`` on Practice) and is IDENTICAL across accounts on
    the same room -- Arterial read ``Sep. 25, 2012`` on World in the
    2026-08-13 capture and Artax read the same value there on
    2026-08-28, fifteen days and a room-id rotation apart (World was
    room 5, then 6). Practice pairs the same way -- ``Jan. 08, 2013``
    for both Arterial and Artax -- so both rooms are confirmed by two
    accounts each, and the rooms disagree with each other. A
    per-account or per-tank date could not behave like that.

    Stable across MAP rotation too: World has run Desert and other
    fields while keeping this date, so it dates the room itself, not
    the map currently loaded in it."""
    name: str
    rank: int
    active_forces: list[int]
    """Tanks playing each color in the room, the order the client's
    lobby panel prints them: orange, purple, blue, red.

    Operator ground truth 2026-08-28: "there's always 9 bots of each
    color", which is why a world empty of humans reads ``9,9,9,9``
    (measured live, Artax/World) while ``api/active_games`` reports
    ``playing=0`` -- the API counts humans, this counts every tank.
    The archived ``9|10|10|9`` sample is the same 9 bots plus one
    human on two of the colors."""


class WorldInfoDict(TypedDict):
    """World/map info (+ message) - TEXT format.

    Format: +<id>|<name>|<field>|<flags>|<team>|<mode>|<image>|<year>
    """

    msg_type: Literal[0x2B]
    world_id: int
    name: str
    field_id: int
    flags: list[int]
    team: int
    mode: str
    image: str
    year: int


class ChatAckDict(TypedDict):
    """Chat-toggle acknowledgment (1-byte 0x43 'C' message).

    The 0x43 type byte is overloaded: cache patches are 4-byte
    entries, while the server answers a client chat toggle (Ka,
    "C{enabled}") with a single flag byte. Discovered live
    2026-07-24 when the key probe's Z press crashed the decode
    pipeline; the official client's $g handler reads 4-byte entries
    without length validation and silently mis-parses this frame.
    """

    msg_type: Literal["chat_ack"]
    enabled: bool


class AutoscrollAckDict(TypedDict):
    """Autoscroll-toggle acknowledgment (short 0x41 'A' message).

    The 0x41 type byte is overloaded like 0x43: deactivations carry
    six bytes, while the server echoes a client autoscroll toggle
    (Ia, "A{enabled}") with a short flag frame. Discovered live by
    the 2026-07-24 key probe ('a' press).
    """

    msg_type: Literal["autoscroll_ack"]
    enabled: bool


__all__ = [
    "AutoscrollAckDict",
    "ChatAckDict",
    "JoinConfirmDict",
    "WorldInfoDict",
]
