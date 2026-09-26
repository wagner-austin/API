"""Typed decode of client command payloads for the sim server.

The client sends ``[!][type][cmd][args...]`` frames (XOR applied after
the ``!`` prefix — see ``protocol.commands``). The transport strips
the prefix and XOR; this module turns the plaintext payload into a
typed command the tick processor can queue.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

from tankpit_bot.protocol.chat import CMD_CHAT
from tankpit_bot.protocol.commands import (
    CMD_BLOCK,
    CMD_DEPOSIT_FUEL,
    CMD_ENTER_GAME,
    CMD_INVENTORY,
    CMD_KEEPALIVE,
    CMD_MAP_OPEN,
    CMD_MAP_TELEPORT,
    CMD_MINE,
    CMD_MOVE,
    CMD_PICKUP_EQUIPMENT,
    CMD_PICKUP_FUEL,
    CMD_RADAR,
    CMD_SCOPE,
    CMD_SHOOT,
    CMD_STATISTICS,
    CMD_TOGGLE_EQUIPMENT,
)
from tankpit_bot.wire.helpers import require_min_length, x16


class ClientCommandKind(StrEnum):
    """Which client command a decoded payload is; ``OTHER`` is every unmapped byte."""

    MOVE = "move"
    SHOOT = "shoot"
    TELEPORT = "teleport"
    RADAR = "radar"
    MINE = "mine"
    MAP_OPEN = "map_open"
    PICKUP_FUEL = "pickup_fuel"
    PICKUP_EQUIPMENT = "pickup_equipment"
    DEPOSIT_FUEL = "deposit_fuel"
    TOGGLE_EQUIPMENT = "toggle_equipment"
    BLOCK = "block"
    CHAT = "chat"
    SCOPE = "scope"
    STATISTICS = "statistics"
    KEEPALIVE = "keepalive"
    ENTER_GAME = "enter_game"
    INVENTORY = "inventory"
    OTHER = "other"


_COORD_KINDS: dict[int, ClientCommandKind] = {
    CMD_MOVE: ClientCommandKind.MOVE,
    CMD_MAP_TELEPORT: ClientCommandKind.TELEPORT,
    CMD_PICKUP_FUEL: ClientCommandKind.PICKUP_FUEL,
    CMD_PICKUP_EQUIPMENT: ClientCommandKind.PICKUP_EQUIPMENT,
    CMD_BLOCK: ClientCommandKind.BLOCK,
}
_BARE_KINDS: dict[int, ClientCommandKind] = {
    CMD_RADAR: ClientCommandKind.RADAR,
    CMD_MINE: ClientCommandKind.MINE,
    CMD_MAP_OPEN: ClientCommandKind.MAP_OPEN,
    # The statistics key. 279 of 386 archived 0x56 frames follow it as
    # the client's most recent sent command, which is what makes the
    # server's answer a RESPONSE rather than a broadcast
    # ([[session-state-deglobalisation]]).
    CMD_STATISTICS: ClientCommandKind.STATISTICS,
    # The client keep-alive (JS class ``dc``, [[client-commands]]). A
    # BARE kind because it carries no arguments -- the whole payload is
    # ``02 21`` in all 11,871 archived sends -- and it belongs in this
    # table rather than in ``other`` because the server has a LAW for
    # it (silence), and a sim that cannot name it cannot obey that law.
    CMD_KEEPALIVE: ClientCommandKind.KEEPALIVE,
    # The two commands a REAL client sends that ours does not. Both
    # were unmapped until 2026-09-03 and both therefore decoded to
    # ``other``, the one kind the server refuses -- the keep-alive's
    # crash was not the only one waiting ([[client-commands]]).
    CMD_ENTER_GAME: ClientCommandKind.ENTER_GAME,
    CMD_INVENTORY: ClientCommandKind.INVENTORY,
}


class ClientCommandDict(TypedDict):
    """One decoded client command.

    ``x``/``y`` are 0 for commands without coordinates; ``target_id``
    is the shoot command's optional entity id (0 = positional shot);
    ``slot`` is the equipment-toggle slot (1-5, 0 for every other
    kind); ``message_id`` is the chat command's preset message id
    (0 for every other kind); ``direction`` is the scope command's
    compass byte (0=N clockwise through 7=NW, 0 for every other
    kind); ``amount`` is the fuel-deposit command's requested units
    (0 for every other kind). ``command`` preserves the raw command
    byte for ``other`` kinds.
    """

    kind: ClientCommandKind
    command: int
    x: int
    y: int
    target_id: int
    slot: int
    message_id: int
    direction: int
    amount: int


def decode_client_command(payload: bytes) -> ClientCommandDict:
    """Decode one plaintext client command payload.

    Args:
        payload: XOR-decoded bytes after the ``!`` prefix:
            ``[type][cmd][args...]``.

    Returns:
        The typed command.

    Raises:
        DecodeError: If the payload is shorter than ``[type][cmd]`` or
            a coordinate command is missing its coordinates.
    """
    require_min_length(payload, 2, "ClientCommand")
    command = payload[1]
    if command in _COORD_KINDS:
        require_min_length(payload, 4, "ClientCommand.coords")
        return ClientCommandDict(
            kind=_COORD_KINDS[command],
            command=command,
            x=payload[2],
            y=payload[3],
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
            amount=0,
        )
    if command == CMD_SHOOT:
        require_min_length(payload, 4, "ClientCommand.shoot")
        target_id = x16(payload[4], payload[5]) if len(payload) >= 6 else 0
        return ClientCommandDict(
            kind=ClientCommandKind.SHOOT,
            command=command,
            x=payload[2],
            y=payload[3],
            target_id=target_id,
            slot=0,
            message_id=0,
            direction=0,
            amount=0,
        )
    if command == CMD_TOGGLE_EQUIPMENT:
        require_min_length(payload, 3, "ClientCommand.toggle")
        return ClientCommandDict(
            kind=ClientCommandKind.TOGGLE_EQUIPMENT,
            command=command,
            x=0,
            y=0,
            target_id=0,
            slot=payload[2] - ord("0"),
            message_id=0,
            direction=0,
            amount=0,
        )
    if command == CMD_CHAT:
        # [type][0x6D][message_id][x][y][flag] — the 6-byte Hb frame
        # (wiki [[chat-messages]], wire-verified sniff-20260729-214411).
        require_min_length(payload, 5, "ClientCommand.chat")
        return ClientCommandDict(
            kind=ClientCommandKind.CHAT,
            command=command,
            x=payload[3],
            y=payload[4],
            target_id=0,
            slot=0,
            message_id=payload[2],
            direction=0,
            amount=0,
        )
    if command == CMD_SCOPE:
        # [type]['Z'][direction] — the 3-byte Rb scope-extend frame
        # (wire-measured 2026-08-01, [[viewport-shift-protocol]]).
        require_min_length(payload, 3, "ClientCommand.scope")
        return ClientCommandDict(
            kind=ClientCommandKind.SCOPE,
            command=command,
            x=0,
            y=0,
            target_id=0,
            slot=0,
            message_id=0,
            direction=payload[2],
            amount=0,
        )
    if command == CMD_DEPOSIT_FUEL:
        # [type][0x44][amount_lo][amount_hi][x][y] -- the AMOUNT leads
        # and the tile follows, which is why the byte read as
        # shoot-shaped for months. JS ``Wb.prototype.h`` writes it in
        # that order and six live deposits confirm it
        # ([[fuel-system]], [[client-commands]]).
        require_min_length(payload, 6, "ClientCommand.deposit_fuel")
        return ClientCommandDict(
            kind=ClientCommandKind.DEPOSIT_FUEL,
            command=command,
            x=payload[4],
            y=payload[5],
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
            amount=x16(payload[2], payload[3]),
        )
    if command in _BARE_KINDS:
        return ClientCommandDict(
            kind=_BARE_KINDS[command],
            command=command,
            x=0,
            y=0,
            target_id=0,
            slot=0,
            message_id=0,
            direction=0,
            amount=0,
        )
    return ClientCommandDict(
        kind=ClientCommandKind.OTHER,
        command=command,
        x=0,
        y=0,
        target_id=0,
        slot=0,
        message_id=0,
        direction=0,
        amount=0,
    )


class SimError(Exception):
    """Raised when the sim is asked for behavior outside its build stage."""


__all__ = [
    "ClientCommandDict",
    "ClientCommandKind",
    "SimError",
    "decode_client_command",
]
