"""Map-data message payloads: the full-map snapshot and its rows.

One of the nine payload families under
:mod:`tankpit_bot.protocol.types`, split from the former single
959-line module. Membership mirrors
:mod:`tankpit_bot.protocol.decoders.map_data` -- the decoder that
produces these payloads owns their definitions.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class MapTankEntry(TypedDict):
    """One tank slot parsed from the 0x4C MapData blob.

    Trace-verified from the per-entry loop in JS Ig.h (V.L):
      a[c+0]      = x
      a[c+1]      = y
      X(a[c+2:4]) = tank_id (LE u16)
      a[c+4]      = packed:
                    rank   = (byte >> 4) & 0xF
                    damage = (byte >> 2) & 0x3
                    team   =  byte       & 0x3
    """

    x: int
    y: int
    tank_id: int
    rank: int
    damage: int
    team: int


class MapDataDict(TypedDict):
    """Whole-map snapshot (0x4C 'L' message).

    Trace-verified from tpclient.js Ig.h (V.L). The body has two
    sections:

      1. Fuel-dot run-length list -- the map's yellow-pixel fuel
         atlas. Total RLE byte count is ``X(a[0], a[1])`` (LE u16)
         and the cells live in ``a[2 : 2+count]``. A 2-D cursor
         ``(d, e)`` starts at ``(1, 1)``; each byte ``h`` advances
         ``d`` by ``h``, wrapping to ``e += 1, d %= 256`` whenever
         ``d`` exceeds 255. Cells valued 255 are pure continuation --
         they advance the cursor but emit no dot. Every other cell
         emits the cursor as a ``(x, y)`` fuel-dot position. The
         atlas is server-cached per session (byte-identical across
         map opens); ~40% of dots still hold fuel when visited, and
         every verified dot held high-volume fuel. Restored
         2026-07-03 (decoded for length only 2026-06-22 to then).

      2. Tank entries -- 5 bytes each, packed to the end of the body.
         See :class:`MapTankEntry` for the per-entry layout.

    JS Ig.prototype.h stores each tank entry into the map slot at
    ``(x << 8) + y`` and assigns ``team`` / ``damage`` / ``rank``
    (verified against Mg.prototype.h's identical field assignment
    order: ``c.h = team``, ``c.u = damage``, ``c.l = rank``).
    """

    msg_type: Literal[0x4C]
    fuel_dots: list[tuple[int, int]]
    tanks: list[MapTankEntry]


__all__ = [
    "MapDataDict",
    "MapTankEntry",
]
