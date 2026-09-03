"""Which connection speaks for a tank, and what it can prove about a click.

Two routers need the same question answered — the move family and the
shoot family both refuse a click that leaves the acting tank's own
viewport — and both used to answer it by reaching into the server's
single ``session`` and comparing ids inline. That is correct with one
connection and silently wrong with several: the comparison does not
mean "this tank's own window", it means "the one window we happen to
have".

Asking it HERE, once, is what makes the difference invisible at the
call sites. When a session registry replaces the single session, only
:meth:`SimServerSessionsMixin.session_for` changes.
"""

from __future__ import annotations

from tankpit_bot.sim.client_session import ClientSession


class SimServerSessionsMixin:
    """Connection lookup for the simulator's command routers.

    The attribute below is a DECLARATION, not an assignment: the
    server's ``__init__`` remains its single owner.
    """

    session: ClientSession

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
        return self.session if tank_id == self.session.client_id else None

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
