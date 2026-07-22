"""Typed decode of client command payloads for the sim server.

The client sends ``[!][type][cmd][args...]`` frames (XOR applied after
the ``!`` prefix — see ``protocol.commands``). The transport strips
the prefix and XOR; this module turns the plaintext payload into a
typed command the tick processor can queue.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot.protocol.commands import (
    CMD_MAP_OPEN,
    CMD_MAP_TELEPORT,
    CMD_MINE,
    CMD_MOVE,
    CMD_PICKUP_EQUIPMENT,
    CMD_PICKUP_FUEL,
    CMD_RADAR,
    CMD_SHOOT,
    CMD_TOGGLE_EQUIPMENT,
)
from tankpit_bot.protocol.helpers import require_min_length, x16

ClientCommandKind = Literal[
    "move",
    "shoot",
    "teleport",
    "radar",
    "mine",
    "map_open",
    "pickup_fuel",
    "pickup_equipment",
    "toggle_equipment",
    "other",
]

_COORD_KINDS: dict[int, ClientCommandKind] = {
    CMD_MOVE: "move",
    CMD_MAP_TELEPORT: "teleport",
    CMD_PICKUP_FUEL: "pickup_fuel",
    CMD_PICKUP_EQUIPMENT: "pickup_equipment",
}
_BARE_KINDS: dict[int, ClientCommandKind] = {
    CMD_RADAR: "radar",
    CMD_MINE: "mine",
    CMD_MAP_OPEN: "map_open",
}


class ClientCommandDict(TypedDict):
    """One decoded client command.

    ``x``/``y`` are 0 for commands without coordinates; ``target_id``
    is the shoot command's optional entity id (0 = positional shot);
    ``slot`` is the equipment-toggle slot (1-5, 0 for every other
    kind). ``command`` preserves the raw command byte for ``other``
    kinds.
    """

    kind: ClientCommandKind
    command: int
    x: int
    y: int
    target_id: int
    slot: int


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
        )
    if command == CMD_SHOOT:
        require_min_length(payload, 4, "ClientCommand.shoot")
        target_id = x16(payload[4], payload[5]) if len(payload) >= 6 else 0
        return ClientCommandDict(
            kind="shoot",
            command=command,
            x=payload[2],
            y=payload[3],
            target_id=target_id,
            slot=0,
        )
    if command == CMD_TOGGLE_EQUIPMENT:
        require_min_length(payload, 3, "ClientCommand.toggle")
        return ClientCommandDict(
            kind="toggle_equipment",
            command=command,
            x=0,
            y=0,
            target_id=0,
            slot=payload[2] - ord("0"),
        )
    if command in _BARE_KINDS:
        return ClientCommandDict(
            kind=_BARE_KINDS[command],
            command=command,
            x=0,
            y=0,
            target_id=0,
            slot=0,
        )
    return ClientCommandDict(kind="other", command=command, x=0, y=0, target_id=0, slot=0)


class SimError(Exception):
    """Raised when the sim is asked for behavior outside its build stage."""


__all__ = [
    "ClientCommandDict",
    "ClientCommandKind",
    "SimError",
    "decode_client_command",
]
