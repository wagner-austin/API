"""The networked server's connections, independent of any socket library.

A :class:`NetHost` holds the rooms and every open connection, and turns
what a client sends into what the server answers. It is synchronous and
knows nothing of sockets: :mod:`tankpit_bot.sim.net_server` reads bytes
off a WebSocket, hands them here as a payload, and writes back what this
returns. The payload is the wire's base64 currency, the same one the
capture archive and the in-process link (:mod:`tankpit_bot.sim.session`)
speak, so one split (:func:`~tankpit_bot.sim.transport.route_client_frames`)
serves both.

A connection's life, in the archive's own order
(:mod:`tankpit_bot.sim.lobby`):

1. Its first frame is AUTH. The account must be one this server issued
   (:mod:`tankpit_bot.sim.net_accounts`); the frame's magic builds the
   connection's cipher table. The lobby answers with the room list.
2. Select and enter go to the connection's lobby. Entry seats a tank in
   the room on the troop the client chose (:meth:`NetRoom.seat`).
3. From then its ``!`` frames are commands for that tank, and each tick
   it is sent its batch, enveloped and XOR'd as the real wire is.
4. A quit frame, or the socket closing, takes the tank off the field.

Anything else is a protocol error, raised with a ``SIM_NET_*``,
``SIM_LOBBY_*`` or decode code for the socket layer to end that one
connection with.
"""

from __future__ import annotations

from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.sim.lobby import SimLobby, parse_auth_frame
from tankpit_bot.sim.net_accounts import AccountBookProtocol
from tankpit_bot.sim.net_room import NetError, NetRoom
from tankpit_bot.sim.transport import (
    encode_plaintext_payload,
    encode_tick_payload,
    route_client_frames,
)


class _Connection:
    """One client's place in the host: its lobby, cipher and seat."""

    def __init__(self) -> None:
        """Start before AUTH: no account, no table, no seat."""
        self.account_id: str | None = None
        self.lobby: SimLobby | None = None
        self.table: bytes | None = None
        self.room: NetRoom | None = None
        self.tank_id: int | None = None


class NetHost:
    """Every room and every open connection of one networked server."""

    def __init__(self, rooms: tuple[NetRoom, ...], accounts: AccountBookProtocol) -> None:
        """Hold the rooms, keyed by id, and the account book.

        Args:
            rooms: The rooms, in the order the lobby lists them.
            accounts: The accounts a client may join as.

        Raises:
            NetError: If there are no rooms, or two share an id
                (``SIM_NET_ROOMS``).
        """
        ids = [room.info["room_id"] for room in rooms]
        if not ids or len(set(ids)) != len(ids):
            raise NetError(f"SIM_NET_ROOMS: a host needs rooms with distinct ids, not {ids}")
        self._rooms = rooms
        self._accounts = accounts
        self._connections: dict[int, _Connection] = {}
        self._opened = 0

    def open(self) -> int:
        """Admit a new connection, before its AUTH.

        Returns:
            The connection's id, for :meth:`receive` and :meth:`close`.
        """
        self._opened += 1
        self._connections[self._opened] = _Connection()
        return self._opened

    def _require(self, connection_id: int) -> _Connection:
        """Find an open connection.

        Args:
            connection_id: The id :meth:`open` gave.

        Returns:
            The connection.

        Raises:
            NetError: If no open connection has the id (``SIM_NET_UNKNOWN``).
        """
        connection = self._connections.get(connection_id)
        if connection is None:
            raise NetError(f"SIM_NET_UNKNOWN: no open connection {connection_id}")
        return connection

    def receive(self, connection_id: int, payload: str) -> list[str]:
        """Take one payload a client sent and answer what needs an answer now.

        Lobby frames are answered at once. Commands are queued for the
        next tick, whose batch carries their consequences.

        Args:
            connection_id: The sending connection.
            payload: The base64 of the WebSocket message's bytes.

        Returns:
            The lobby's replies, one payload per frame, as the real
            server sends each row separately.

        Raises:
            NetError: For an unknown connection, a command from a
                connection with no seat (``SIM_NET_NOT_SEATED``), or a
                seat the room refuses.
            LobbyError: For a first frame that is not AUTH, or an
                account the book does not admit.
            DecodeError: For a torn payload, or a command before AUTH.
            SimError: For a command the sim cannot decode.
        """
        connection = self._require(connection_id)
        routed = route_client_frames(payload, connection.table)
        replies = [reply for body in routed.lobby for reply in self._lobby_frame(connection, body)]
        for command in routed.commands:
            if connection.room is None or connection.tank_id is None:
                raise NetError(f"SIM_NET_NOT_SEATED: connection {connection_id} has no tank")
            connection.room.server.queue_command(connection.tank_id, command)
        return [encode_plaintext_payload([reply]) for reply in replies]

    def _lobby_frame(self, connection: _Connection, body: bytes) -> list[bytes]:
        """Answer one plaintext frame, seating or unseating as it says.

        Args:
            connection: The sending connection.
            body: The frame body, lead byte included.

        Returns:
            The lobby's reply frames.

        Raises:
            LobbyError: If the connection's first frame is not an AUTH
                the book admits.
            NetError: If the account is connected already
                (``SIM_NET_IN_USE``), or the room refuses the seat.
        """
        if connection.lobby is None:
            auth = parse_auth_frame(body)
            account = self._accounts.verify(auth["account_id"], auth["token"])
            if any(other.account_id == auth["account_id"] for other in self._connections.values()):
                raise NetError(f"SIM_NET_IN_USE: account {auth['account_id']!r} is connected")
            connection.account_id = auth["account_id"]
            connection.lobby = SimLobby(account, tuple(room.info for room in self._rooms))
            connection.table = build_session_xor_table(auth["magic"])
        lobby = connection.lobby
        replies = lobby.handle_frame(body)
        if lobby.quit:
            self._unseat(connection)
        elif connection.tank_id is None and lobby.entry is not None:
            room = next(r for r in self._rooms if r.info["room_id"] == lobby.entry.room_id)
            connection.tank_id = room.seat(lobby.account, lobby.entry.troop)
            connection.room = room
        return replies

    def _unseat(self, connection: _Connection) -> None:
        """Take a connection's tank off the field, if it has one.

        Args:
            connection: The connection leaving play.
        """
        if connection.room is not None and connection.tank_id is not None:
            connection.room.leave(connection.tank_id)
        connection.room = None
        connection.tank_id = None

    def tick(self) -> dict[int, str]:
        """Advance every room one tick and encode each player's batch.

        Returns:
            One payload per seated connection, keyed by connection id.
            None is empty: every tick carries a status sync for each
            living tank, the seated player's own among them.

        Raises:
            SimError: If a message would exceed what the cipher covers.
        """
        batches = {room.info["room_id"]: room.advance() for room in self._rooms}
        payloads: dict[int, str] = {}
        for connection_id, connection in self._connections.items():
            room, tank_id, table = connection.room, connection.tank_id, connection.table
            if room is None or tank_id is None or table is None:
                continue
            payloads[connection_id] = encode_tick_payload(
                batches[room.info["room_id"]][tank_id], table
            )
        return payloads

    def close(self, connection_id: int) -> None:
        """Forget a connection whose socket closed, unseating it first.

        Args:
            connection_id: The closing connection.

        Raises:
            NetError: If no open connection has the id.
        """
        self._unseat(self._require(connection_id))
        del self._connections[connection_id]

    @property
    def rooms(self) -> tuple[NetRoom, ...]:
        """The rooms, in lobby order."""
        return self._rooms

    @property
    def connections(self) -> int:
        """How many connections are open."""
        return len(self._connections)


__all__ = ["NetHost"]
