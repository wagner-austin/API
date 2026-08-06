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

    * **Alive.** Corpses do NOT block walking -- archive-disproven
      2026-08-04 (`analysis_scripts/mine_corpse_blocking.py`): six
      clean 0x47 echoes of the bot walking ONTO a fresh corpse tile
      2-10 s after its own kill, inside the 22 s corpse window, zero
      blocked crossings. Kills drop NO loot (user contract
      2026-08-04); the crossings are ordinary post-kill restock
      collection routes through the current viewport that happen to
      pass the corpse tile -- incidental, which makes them unbiased
      evidence. The first cut of this module counted deactivated
      tanks ("a corpse stands where it died"), which would have
      walled off tiles on those restock routes for up to the
      presence TTL after every kill. The ~22 s corpse window governs
      the tank's RESPAWN choreography ([[deactivation-format]]), not
      tile passability; the sim server's ``_blocked_by_world``
      always gated on ``alive`` and now the client matches it.

    Args:
        tank: Tank registry entry to test.
        now_ms: Current wall-clock time in milliseconds.

    Returns:
        True when the tank's body should be treated as occupying
        ``(tank["x"], tank["y"])`` for movement planning.
    """
    if tank["is_self"]:
        return False
    if tank["liveness"] != "alive":
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
