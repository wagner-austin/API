"""Movement message payloads: walk commands and their responses.

One of the nine payload families under
:mod:`tankpit_bot.protocol.types`, split from the former single
959-line module. Membership mirrors
:mod:`tankpit_bot.protocol.decoders.movement` -- the decoder that
produces these payloads owns their definitions.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class MovementDict(TypedDict):
    """Movement path (0x47 'G' message).

    Layout from tpclient.js Lg.h (V.G), verified 2026-06-19:
      a[0:2]  = tank_id (LE u16)
      a[2]    = start_x
      a[3]    = start_y
      a[4]    = direction
      a[5]    = damage_state (assigned to b.u in Lg.prototype.h; NOT damage_state)
      a[6:9]  = lb_score (24-bit BE)
      a[9]    = rank (assigned to b.l in Lg.prototype.h)
      a[10]   = animation flag (passed to Re constructor, not tank state)
      a[11]   = is_carrying (1=true)
      a[12:]  = waypoints (direction chars)

    ``waypoints`` collapses the nsew path to its final position (one
    entry, or empty when stationary); ``path_tiles`` preserves the
    wire's true step count — one fuel per step, exact even on
    non-minimal paths around obstacles ([[game-economy]] walk row) —
    and ``path`` keeps the raw nsew route the SERVER chose, since the
    client only sends a destination click and the server pathfinds.
    """

    msg_type: Literal[0x47]
    tank_id: int
    start_x: int
    start_y: int
    direction: int
    damage_state: int
    lb_score: int
    rank: int
    flag: int
    is_carrying: bool
    waypoints: list[tuple[int, int]]
    path_tiles: int
    path: str


class MovementResponseDict(TypedDict):
    """Movement response (0x3D '=' binary message).

    Layout from tpclient.js Mg.h (V["="]):
      a[0]    = team
      a[1:3]  = tank_id (LE u16)
      a[3]    = x
      a[4]    = y
      a[5]    = direction
      a[6]    = damage_state (assigned to b.u; NOT damage_state)
      a[7]    = rank
      a[8:11] = lb_score (24-bit BE)
      a[11]   = carrying flag
    """

    msg_type: Literal[0x3D]
    team: int
    tank_id: int
    x: int
    y: int
    direction: int
    damage_state: int
    rank: int
    lb_score: int
    carrying: int


__all__ = [
    "MovementDict",
    "MovementResponseDict",
]
