"""Per-tile tank-body occupancy derived from the tank registry.

A tank body occupies its tile: the server's route planner will not
walk another tank through it, and a walk aimed past one stops at the
body and draws ``0x52 error_code=1`` ("You can't go there!"). The user
contract, 2026-08-04, states the server's behaviour verbatim -- *"you
walk until you hit the block then stop and you get the error message"*
-- and names the full blocker set: terrain, another tank, a movable
block, or a visible mine ([[walk-mechanics]], [[flag-triage-20260729]]
F6).

Three of those four already compose into the one passability answer
(:class:`tankpit_bot.bot.ai.ferry.FerryAwareTerrain`): static terrain
from the minimap, movable blocks from the wire terrain vocabulary, and
hostile mines from the mine registry. This module supplies the fourth.

Occupancy is a *projection* of world state, not a mutation of it --
same shape as :mod:`tankpit_bot.state.scan_coverage`. It is deliberately
NOT a passability answer on its own: consumers get it through the
composed terrain view so no call site can forget to ask.
"""

from __future__ import annotations

from tankpit_bot.state.types import (
    VIEWPORT_PRESENCE_TTL_MS,
    TankStateDict,
    WorldStateDict,
    coord_key,
    has_known_position,
)


def is_tank_body_present(tank: TankStateDict, now_ms: int) -> bool:
    """Return whether a tank's body currently occupies its recorded tile.

    Three conditions, each load-bearing:

    * **Not self.** The bot cannot be blocked by its own body, and
      pathfinding starts on that tile. ``is_self`` is set from the
      ``self_state`` tank-id comparison in
      :mod:`tankpit_bot.state.mutations`, so it is the single source
      of that identity -- this module does not re-derive it.
    * **Position ever observed.** Viewport-freshness does NOT imply a
      position: 0x21 TankInfo and 0x3E TankStatus route as
      ``storage_source == "viewport"`` with no coordinates, and the
      login choreography sends the full-roster 0x21 dump FIRST -- so
      every tank spends its opening 9-46 s at the ``(0, 0)``
      construction default (measured 2026-08-04; see
      :func:`~tankpit_bot.state.types.tank.has_known_position`).
      Without this gate the session's first 5 s wall off the map
      corner with the whole roster's phantom bodies.
    * **Viewport-fresh.** Only a ``storage_source == "viewport"``
      observation inside :data:`VIEWPORT_PRESENCE_TTL_MS` proves the
      tank was in the bot's local sensing window. Without this gate the
      registry's global roster -- refreshed for every tank on the map
      by 0x4C MapData and 0x2E TankStatusSync -- would wall off tiles
      all over the board. The position can still lag the presence gate
      by up to the TTL (a status message refreshes presence without
      moving the recorded tile); that imprecision is inherent to 2 s
      observation granularity and recorded as F6 open work.

    Liveness is deliberately NOT a condition: **a corpse stands where
    it died and blocks like a body.** The 2026-08-04 archive measure
    ("six clean walks onto corpse tiles, zero blocked crossings")
    briefly gated this on ``liveness == "alive"``; run
    bot-20260813-204615 falsified it with exact-window wire receipts
    (HUD flag 3): the bot killed orange-6 at (33,189) from (34,189)
    -- static rock on every other neighbor -- and SEVEN consecutive
    pickups through that tile closed ``0x52 code=1`` as PURE refusals
    (no 0x47 echo: the first step was already blocked) while 0x2E
    kept restating tank 532 at (33,189) with the dead-corpse sprite
    (direction=32) and no mine stood anywhere in the pocket. The old
    miner could not see this class: its blocking-proof pattern
    required a partial-walk ECHO stopping adjacent to the corpse, and
    a bot already standing adjacent gets a bare receipt -- its six
    "clean crossings" (corpse tiles fixed from the victim's last wire
    position, which displaced and ranged kills falsify) are the
    suspect measurement, recorded in [[flag-triage-20260729]]. The
    wire itself keeps the belief honest: the corpse's position is
    restated for as long as it stands, so the same viewport-freshness
    gate that bounds live bodies bounds corpses, and reactivation
    flips ``liveness`` back with a fresh position either way.

    Args:
        tank: Tank registry entry to test.
        now_ms: Current wall-clock time in milliseconds.

    Returns:
        True when the tank's body should be treated as occupying
        ``(tank["x"], tank["y"])`` for movement planning.
    """
    if tank["is_self"]:
        return False
    if not has_known_position(tank):
        return False
    return now_ms - tank["last_viewport_observation_ms"] <= VIEWPORT_PRESENCE_TTL_MS


def occupied_tank_keys(world: WorldStateDict, now_ms: int) -> frozenset[str]:
    """Return ``"x,y"`` keys of tiles occupied by other tanks' bodies.

    Args:
        world: Current world state carrying the tank registry.
        now_ms: Current wall-clock time in milliseconds.

    Returns:
        Frozen set of ``"x,y"`` keys, one per tank body that passes
        :func:`is_tank_body_present`. Empty when no other tank is
        viewport-fresh.
    """
    return frozenset(
        coord_key(tank["x"], tank["y"])
        for tank in world["tanks"].values()
        if is_tank_body_present(tank, now_ms)
    )


__all__ = [
    "is_tank_body_present",
    "occupied_tank_keys",
]
