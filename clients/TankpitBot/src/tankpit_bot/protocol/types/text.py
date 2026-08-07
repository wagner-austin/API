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

    Format: =<team>|<join_date>|<name>|<rank>|<eq1>|<eq2>|<eq3>|<eq4>
    """

    msg_type: Literal[0x3D]
    team: int
    join_date: str
    name: str
    rank: int
    equipment: list[int]


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
