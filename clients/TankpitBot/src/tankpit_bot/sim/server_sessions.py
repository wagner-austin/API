"""Which connections a field has, and what one can prove about a click.

The registry of :class:`ClientSession` objects, one per connected tank,
and the two questions every router asks of it. The move family and the
shoot family both refuse a click that leaves the acting tank's own
viewport, and both ask it HERE: "this tank's own window" is a lookup
by tank id, never a comparison against the one window the server
happens to have.

A field starts with no connections. :meth:`SimServerSessionsMixin.connect`
admits one into a field that may already be running — joining a room
mid-play is the normal case on the real server, not the exception — and
the connections already there are told with the same 0x28 TankEntry a
churn visitor's arrival draws ([[session-state-deglobalisation]]).
"""

from __future__ import annotations

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.sim.client_session import ClientSession
from tankpit_bot.sim.commands import SimError
from tankpit_bot.sim.outbox import TickOutbox
from tankpit_bot.sim.wire_statements import entry_statement
from tankpit_bot.sim.world import SimWorldDict


class SimServerSessionsMixin:
    """The connection registry for the simulator's command routers.

    The attributes below are DECLARATIONS, not assignments: the
    server's ``__init__`` remains their single owner.
    """

    world: SimWorldDict
    terrain: TerrainMapProtocol
    _sessions: dict[int, ClientSession]
    _outbox: TickOutbox

    def connect(self, tank_id: int) -> ClientSession:
        """Admit a connection for a tank already on the field.

        The connections already present learn of the arrival through
        this tick's outbox; the joiner learns the room from its own
        join burst (:meth:`handshake`), which a real client asks for
        with CMD_ENTER_GAME.

        Args:
            tank_id: The tank the connection speaks for.

        Returns:
            The new connection's session.

        Raises:
            SimError: If no living tank has this id, or a connection
                already speaks for it — one tank, one connection.
        """
        tank = self.world["tanks"].get(tank_id)
        if tank is None or not tank["alive"]:
            raise SimError(f"no living tank {tank_id} to connect")
        if tank_id in self._sessions:
            raise SimError(f"tank {tank_id} is already connected")
        self._outbox.broadcast([entry_statement(self.world, tank_id)])
        session = ClientSession(self.world, self.terrain, tank_id)
        self._sessions[tank_id] = session
        self._outbox.admit(tank_id)
        return session

    @property
    def sessions(self) -> tuple[ClientSession, ...]:
        """Every connection on the field, in the order they joined.

        Returns:
            The connected sessions.
        """
        return tuple(self._sessions.values())

    def session_for(self, tank_id: int) -> ClientSession | None:
        """The connection speaking for a tank, if any.

        A tank the sim drives itself — a practice-roster bot, the
        scripted opponent, a ghost — has no connection and therefore
        no stored viewport, which is why the answer is nullable rather
        than a session with an empty window: "no connection" and "a
        connection seeing nothing" are different facts and the
        refusal laws must not confuse them.

        Args:
            tank_id: The tank whose connection is wanted.

        Returns:
            That tank's session, or None when nothing is connected
            for it.
        """
        return self._sessions.get(tank_id)

    def require_session(self, tank_id: int) -> ClientSession:
        """The connection speaking for a tank, which must exist.

        Args:
            tank_id: A connected tank.

        Returns:
            Its session.

        Raises:
            SimError: If nothing is connected for the tank — a caller
                asking for a connection's own answer (its join burst)
                about a tank with none is a harness bug.
        """
        session = self._sessions.get(tank_id)
        if session is None:
            raise SimError(f"tank {tank_id} has no connection")
        return session

    def click_leaves_own_window(self, tank_id: int, x: int, y: int) -> bool:
        """Can the server PROVE this click left the tank's own window?

        Proof requires a window to check against. An unconnected tank
        has none, so its clicks are never refused on these grounds —
        the server is not entitled to invent a viewport for a tank
        nobody is watching through.

        Args:
            tank_id: The clicking tank.
            x: Clicked tile X.
            y: Clicked tile Y.

        Returns:
            True only when a connection exists for this tank AND the
            tile lies outside its stored 0x5A window.
        """
        session = self.session_for(tank_id)
        return session is not None and not session.viewport.in_window(x, y)


__all__ = ["SimServerSessionsMixin"]
