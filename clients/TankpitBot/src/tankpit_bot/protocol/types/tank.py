"""Tank message payloads: info, entry, exit, removal, and status.

One of the nine payload families under
:mod:`tankpit_bot.protocol.types`, split from the former single
959-line module. Membership mirrors
:mod:`tankpit_bot.protocol.decoders.tank` -- the decoder that
produces these payloads owns their definitions.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class TankInfoDict(TypedDict):
    """Tank info (0x21 '!' message).

    Trace-verified from tpclient.js Tf.h (line 3896-3901):
      a[0]    = team (a[0] & 255)
      a[1:3]  = tank_id (LE u16)
      a[3:7]  = decoration_state (4 bytes, decoded by yg() into 9 x 2-bit slots)
      a[7:10] = persistent_tank_id (24-bit BE, sets a.aa for profile links)
      a[10:]  = name (UTF-8 string)

    NOTE: This message does NOT contain the tank's current rank.
    """

    msg_type: Literal[0x21]
    tank_id: int
    team: int
    decoration_state: bytes
    persistent_tank_id: int
    name: str


class TankEntryDict(TypedDict):
    """Tank entry (( message).

    Layout from tpclient.js Uf.h (V["("]), verified 2026-06-19:
      a[0]   = flags (255=known tank)
      a[1:3] = tank_id (LE u16)
      a[3]   = packed byte: team(bits 0-1), damage_state(bits 2-3), rank(bits 4-7)
      a[4:7] = score (24-bit BE)
      a[7]   = x position
      a[8]   = y position
    """

    msg_type: Literal[0x28]
    team: int
    tank_id: int
    rank: int
    damage_state: int
    score: int
    x: int
    y: int


class TankRemoveDict(TypedDict):
    """Tank removal from world (0x58 'X' message).

    Trace-verified from tpclient.js Ug.h (V.X):
      a[0:2] = tank_id (LE u16)

    Server-driven removal — clears the tile entry, releases the tank slot,
    and drops the rendered tank. No accompanying display text.
    """

    msg_type: Literal[0x58]
    tank_id: int


class TankExitDict(TypedDict):
    """Tank exit/elimination announcement (0x29 ')' message).

    Trace-verified from tpclient.js Vf.h (V[")"]):
      a[0]   = team
      a[1:3] = tank_id (LE u16)
      a[3]   = was_silent (1 = no display text emitted)
      a[4]   = was_eliminated (1 = "eliminated from the game",
                               0 = "left the game")

    Pure announcement — the renderer prints a log line unless
    ``was_silent``. Separate from 0x58 TankRemove, which physically
    removes the tank from the world.
    """

    msg_type: Literal[0x29]
    team: int
    tank_id: int
    was_silent: bool
    was_eliminated: bool


class TankStatusSyncDict(TypedDict):
    """Tank status sync (0x2E message).

    Layout from tpclient.js Og.h (V["."]), verified 2026-06-19:
      a[0]    = team (subtype)
      a[1:3]  = tank_id (LE u16)
      a[3]    = damage_state (b.u; dual-purpose: rank_category on init, damage during gameplay)
      a[4]    = rank (b.l)
      a[5:8]  = lb_score (24-bit BE)
      a[8]    = promo_state -- present when the body is at least 9 bytes
      a[9]    = promo_bar_lit (if long form)
      a[10:12] = fuel (LE u16, if long form)

    The 9-byte short form carries ``promo_state``; the 13-byte long
    form carries ``fuel`` as well. Production corpus
    (analysis_scripts/crack_tank_status_short.py) confirms 74/74
    9-byte 0x2E bodies have promo_state in ``[0, 5]``.

    ``a[9]`` was documented as ``has_fuel_bar`` and then DROPPED by the
    decoder, leaving the encoder to hardcode it to 1 ("21,278/21,278
    corpus bodies"). Both halves were wrong. The JS reads fuel
    unconditionally at long form (``Cc(a.v, this.o)`` runs whatever
    ``a[9]`` says), so the byte gates nothing about fuel; it rides with
    ``promo_state`` into ``Dc``, which stores it as the promotion
    bar's colour — ``Fc`` fills the bar to ``2 * promo_state`` pixels
    and paints it green when the flag is set, dark red when it is
    clear. Its meaning beyond that colour is unproven. A 2026-08-06
    archive sweep found 219 of 70,532 long-form bodies carrying 0, all
    on the client's own tank across four sessions and every fuel
    level, so the hardcoded 1 was a genuine round-trip defect that the
    archive walk had been crashing before it could report
    ([[session-state-deglobalisation]]).

    ``promo_bar_lit`` and ``fuel`` are present together or not at all:
    the only observed body lengths are 8, 9, and 12, and byte 9 exists
    only in the 12-byte form.
    """

    msg_type: Literal[0x2E]
    subtype: int
    tank_id: int
    damage_state: int
    rank: int
    lb_score: int
    promo_state: int | None
    promo_bar_lit: bool | None
    fuel: int | None


class TankStatusDict(TypedDict):
    """Full tank status (0x3E '>' message).

    ``damage_state`` is bits 2-3 of the info byte (the packed-byte
    convention shared with TankEntry/MapTankEntry: team 0-1, damage
    2-3, rank 4-7). Corpus 2026-07-21: 223 of 244 bodies carry a
    nonzero value there — dropping it broke byte-identical
    round-trips.
    """

    msg_type: Literal[0x3E]
    team: int
    rank: int
    damage_state: int
    tank_id: int
    decoration_state: bytes
    leaderboard_score: int
    leaderboard_position: int
    name: str


__all__ = [
    "TankEntryDict",
    "TankExitDict",
    "TankInfoDict",
    "TankRemoveDict",
    "TankStatusDict",
    "TankStatusSyncDict",
]
