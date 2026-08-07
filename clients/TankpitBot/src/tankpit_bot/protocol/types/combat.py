"""Combat message payloads: shot events and deactivations.

One of the nine payload families under
:mod:`tankpit_bot.protocol.types`, split from the former single
959-line module. Membership mirrors
:mod:`tankpit_bot.protocol.decoders.combat` -- the decoder that
produces these payloads owns their definitions.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class ShootEventDict(TypedDict):
    """Shooting / hit event (0x53 'S' message).

    Layout from tpclient.js Gg.h (V.S), re-verified 2026-06-19 against
    real wire bytes from runs/bot/bot-20260619-050303 msg t+25.47s
    `53 02 15 05 b2 7d b2 7e b2 7e 01`:
      a[0]    = team byte (red=0, purple=1, blue=2, orange=3)
      a[1:3]  = shooter_id (LE u16)  -- the tank that fired
      a[3]    = source_x  -- shooter's tile X (live position)
      a[4]    = source_y  -- shooter's tile Y
      a[5]    = target_x  -- shot's landing tile X (homing's final tile)
      a[6]    = target_y  -- shot's landing tile Y
      a[7]    = aim_x     -- aim tile X (the tile the gun is pointed at)
      a[8]    = aim_y     -- aim tile Y
      a[9]    = weapon (0=single, 1=dual, 2=missile, 3=homing)

    Prior decoder named a[3,4] as target and a[5,6] as projectile_start
    -- reversed. Prior names for a[7..9] as fuel/weapon/ammo were also
    wrong. Three-way validation (enemy src tracking, homing tgt tile,
    wire damage events) confirmed the corrected layout.

    a[7]/a[8] semantics promoted from ``unk1``/``unk2`` to
    ``aim_x``/``aim_y`` 2026-06-20. JS evidence: ``Gg.h`` passes them to
    the projectile-animation constructor ``yf`` as ``z`` and ``O``;
    inside ``yf``, ``this.qa = 24 * z + 12`` and ``this.ta = 16 * O + 8``
    are PIXEL CENTRES of the tile the tank's gun is aimed at, and
    ``yf.start()`` uses ``atan2(this.h - this.qa, this.ta - this.i)`` to
    set the tank's facing direction. For straight shots aim == target;
    for guided weapons (missile/homing) aim is the initial barrel
    direction and target is the homing impact tile.

    Hit detection per JS Gg.prototype.h case 18: shot landed on a named
    tank tile -> hit. That tile lookup uses (target_x, target_y).
    """

    msg_type: Literal[0x53]
    team: int
    shooter_id: int
    source_x: int
    source_y: int
    target_x: int
    target_y: int
    aim_x: int
    aim_y: int
    weapon: int


class DeactivationDict(TypedDict):
    """Kill/deactivation event (0x41 'A' message).

    Layout from tpclient.js Pg.h (V.A), verified 2026-06-19:
      a[0]  = status byte
      a[1:3] = victim_id (LE u16)
      a[3]  = promo_eligible (1=earned extra points)
      a[4:6] = killer_id (LE u16)
      If killer_id >= 65530: mine kill (team = killer_id - 65530)
    """

    msg_type: Literal[0x41]
    status: int
    victim_id: int
    promo_eligible: bool
    killer_id: int
    is_mine_kill: bool


__all__ = [
    "DeactivationDict",
    "ShootEventDict",
]
