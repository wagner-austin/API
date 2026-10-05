"""The networked sim server: its rooms, served over WebSockets.

``tankpit-sim-serve`` hosts one or more sim rooms
(:mod:`tankpit_bot.sim.net_room`) behind a WebSocket port. Each binary
message a client sends is one wire payload: length-prefixed frames,
plaintext in the lobby and ``!``-led XOR'd commands in play, exactly the
bytes the page client puts on the real socket. The server answers lobby
frames at once, and once a tick sends each seated player its batch,
enveloped and ciphered as the real server does.

The socket layer holds no game state. :class:`NetServer` maps sockets to
:class:`~tankpit_bot.sim.net_host.NetHost` connection ids and moves
bytes; the host decides everything. A text message, a frame the host
refuses, or anything else a client gets wrong raises in that client's
handler, which the WebSocket library ends with a 1011 close while every
other connection plays on; the handler's ``finally`` takes the tank off
the field either way.

The accounts are this server's own (:mod:`tankpit_bot.sim.net_accounts`),
never tankpit.com's.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from platform_core.logging import LogLevel, get_logger
from platform_core.rich_logging import setup_rich_logging
from websockets.asyncio.server import ServerConnection, broadcast, serve

from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.sim.net_accounts import load_account_book
from tankpit_bot.sim.net_host import NetHost
from tankpit_bot.sim.net_room import NetError, field_room_info, open_field_room

log = get_logger(__name__)

DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_ROOM = "1:field01:p"

_SERVE_FLAGS = frozenset(
    {
        "--bind",
        "--port",
        "--accounts",
        "--room",
        "--ticks",
        "--tick-ms",
        "--layout",
        "--population-seed",
    }
)
"""Every flag ``tankpit-sim-serve`` reads; each takes one value."""


class NetServer:
    """The sockets of one networked server, and its tick."""

    def __init__(self, host: NetHost) -> None:
        """Serve a host's rooms.

        Args:
            host: The rooms and connections this server's sockets feed.
        """
        self.host = host
        self._sockets: dict[int, ServerConnection] = {}

    async def handle(self, socket: ServerConnection) -> None:
        """Serve one client socket until it closes.

        Args:
            socket: The client's WebSocket.

        Raises:
            NetError: If the client sends a text message (``SIM_NET_TEXT``);
                the wire is binary.
            LobbyError: For an AUTH the account book refuses.
            DecodeError: For a torn payload or a command before AUTH.
            SimError: For a command the sim cannot decode.
        """
        connection_id = self.host.open()
        self._sockets[connection_id] = socket
        try:
            async for message in socket:
                if isinstance(message, str):
                    raise NetError(
                        "SIM_NET_TEXT: the wire is binary; a text message ended this connection"
                    )
                payload = base64.b64encode(message).decode("ascii")
                for reply in self.host.receive(connection_id, payload):
                    await socket.send(base64.b64decode(reply))
        finally:
            del self._sockets[connection_id]
            self.host.close(connection_id)

    def tick(self) -> int:
        """Advance every room one tick and send each player its batch.

        A socket that closed after the tick began is skipped rather than
        written to: its handler is about to take its tank off the field,
        and its last batch has no reader.

        Returns:
            How many batches were sent.
        """
        payloads = self.host.tick()
        for connection_id, payload in payloads.items():
            broadcast([self._sockets[connection_id]], base64.b64decode(payload))
        return len(payloads)

    async def tick_for(self, ticks: int | None, interval_seconds: float) -> int:
        """Tick at the wire's cadence, for a number of ticks or forever.

        Args:
            ticks: How many ticks to play; None never stops.
            interval_seconds: The pause after each tick.

        Returns:
            How many ticks were played.
        """
        played = 0
        while ticks is None or played < ticks:
            self.tick()
            played += 1
            await asyncio.sleep(interval_seconds)
        return played


class ServeArgs(NamedTuple):
    """``tankpit-sim-serve``'s flags.

    Attributes:
        bind: The address to listen on.
        port: The port to listen on; 0 picks a free one.
        accounts: The account file.
        rooms: Each room as ``ID:FIELD:MODE`` (mode ``p`` practice, ``n`` open).
        ticks: How many ticks to serve; None serves until interrupted.
        tick_ms: The pause after each tick, the wire's 2 s by default; a
            smoke run may fast-forward.
        layout: The practice layout's provenance; None derives it per room.
        population_seed: The container seed; None derives it per room.
    """

    bind: str
    port: int
    accounts: Path
    rooms: tuple[str, ...]
    ticks: int | None
    tick_ms: int
    layout: str | None
    population_seed: int | None


def parse_serve_args(argv: Sequence[str]) -> ServeArgs:
    """Read ``tankpit-sim-serve``'s flags.

    Args:
        argv: The arguments, program name excluded.

    Returns:
        The flags.

    Raises:
        NetError: If a flag is unknown, lacks its value, or the account
            file is not named (``SIM_SERVE_USAGE``).
        ValueError: If a numeric flag is not a number.
    """
    values: dict[str, str] = {}
    rooms: list[str] = []
    for index in range(0, len(argv), 2):
        flag = argv[index]
        if flag not in _SERVE_FLAGS or index + 1 >= len(argv):
            raise NetError(f"SIM_SERVE_USAGE: unknown flag or missing value at {flag!r}")
        if flag == "--room":
            rooms.append(argv[index + 1])
        else:
            values[flag] = argv[index + 1]
    if "--accounts" not in values:
        raise NetError("SIM_SERVE_USAGE: --accounts PATH names this server's account file")
    ticks, seed = values.get("--ticks"), values.get("--population-seed")
    return ServeArgs(
        bind=values.get("--bind", DEFAULT_BIND),
        port=int(values.get("--port", str(DEFAULT_PORT))),
        accounts=Path(values["--accounts"]),
        rooms=tuple(rooms) if rooms else (DEFAULT_ROOM,),
        ticks=None if ticks is None else int(ticks),
        tick_ms=int(values.get("--tick-ms", str(TICK_RATE_MS))),
        layout=values.get("--layout"),
        population_seed=None if seed is None else int(seed),
    )


def build_host(args: ServeArgs) -> NetHost:
    """Open every room the flags name, over the account file.

    Args:
        args: The flags.

    Returns:
        The host, no connections open.

    Raises:
        NetError: If a room is not ``ID:FIELD:MODE`` with mode ``p`` or
            ``n`` (``SIM_SERVE_ROOM``), or two rooms share an id.
        FieldChoiceError: If a room names no shipped field.
        LobbyError: If the account file holds an invalid record.
    """
    rooms = []
    for spec in args.rooms:
        parts = spec.split(":")
        if len(parts) != 3 or parts[2] not in ("p", "n") or not parts[0]:
            raise NetError(f"SIM_SERVE_ROOM: {spec!r} is not ID:FIELD:MODE with mode p or n")
        room_id, field, mode = parts
        name = "Practice" if mode == "p" else f"World ({field})"
        info = field_room_info(room_id, name, f"{field}.gif", practice=mode == "p")
        rooms.append(
            open_field_room(info, layout=args.layout, population_seed=args.population_seed)
        )
    return NetHost(tuple(rooms), load_account_book(args.accounts))


async def serve_rooms(server: NetServer, args: ServeArgs) -> int:
    """Listen on the flags' port and tick until the tick count is played.

    Args:
        server: The server to put behind the port.
        args: The flags.

    Returns:
        How many ticks were played.
    """
    async with serve(server.handle, args.bind, args.port) as listening:
        # getsockname() is Any in typeshed, the address shape varying by
        # family; this listener is TCP, so it is a (host, port) pair.
        bound: tuple[str, int] = next(iter(listening.sockets)).getsockname()
        log.info(
            "sim server on ws://%s:%d, rooms %s",
            args.bind,
            bound[1],
            ", ".join(room.info["room_id"] for room in server.host.rooms),
        )
        return await server.tick_for(args.ticks, args.tick_ms / 1000)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for ``tankpit-sim-serve``.

    Args:
        argv: ``--accounts PATH`` and optionally ``--bind ADDR``,
            ``--port N``, ``--room ID:FIELD:MODE`` (repeatable),
            ``--ticks N``, ``--tick-ms N``, ``--layout NAME``,
            ``--population-seed N``.
            Uses ``sys.argv[1:]`` when None.

    Returns:
        0 once the tick count is served.
    """
    args = parse_serve_args(list(argv) if argv is not None else sys.argv[1:])
    setup_rich_logging(level=LogLevel.INFO)
    played = asyncio.run(serve_rooms(NetServer(build_host(args)), args))
    sys.stdout.write(f"sim server: {played} ticks served\n")
    return 0


__all__ = [
    "DEFAULT_BIND",
    "DEFAULT_PORT",
    "DEFAULT_ROOM",
    "NetServer",
    "ServeArgs",
    "build_host",
    "main",
    "parse_serve_args",
    "serve_rooms",
]
