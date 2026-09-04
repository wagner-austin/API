"""How many extractors the map itself says must stay protected.

Split out of the expander when the rebuild gate pushed it past the module
ceiling: the floor is a pure derivation from the survey -- what the map
offers minus what the rival takes -- and shares nothing with the ordering
of spenders that IS the expander ([[policy-economy]]).
"""

from __future__ import annotations

#: Pools the rival lands before the expansion race is decided.
#:
#: **Measured, not chosen.** Every winning duel_lake solo trace ends the race
#: with the survey reading 8-9 of the map's 9 pools occupied and the bot
#: holding 6-7 of them: the Very Hard opponent claims two, sometimes three,
#: and the race is won by taking everything else
#: (`runs/traces/vh-solo24`, log 2026-08-03; census strings in
#: `runs/sweeps/vh-solo24`).
RIVAL_POOL_SHARE = 2

#: The floor no map can lower: the first extractor is always protected.
#:
#: Across 46 duels, matches without an economy at all lost outright -- final
#: income at or below 38/s failed 6 of 7 -- so however few pools a map offers,
#: claiming the first is never outranked by replacing a loss
#: ([[policy-holding-ground]]).
FLOOR_MINIMUM = 1


def economy_floor(visible: int, unreachable: int) -> int:
    """Derive the extractor count below which expansion outranks a loss.

    **The number used to be a literal seven, and seven was one map's answer.**
    The definitive solo run's traces split cleanly on it and nothing else in
    the first 1,500 samples: the duel_lake opening is a bloodless expansion
    race, every win reached 6-7 extractors by s1500 and every loss stalled at
    4-5, so the floor was raised from four to seven and Very Hard went from
    0/24 to 14/24 (`runs/traces/vh-solo24`, log 2026-08-03).

    Carried to four other maps, the same seven lost every match it did not
    stalemate -- 0W/5L/3S -- because their extractor peaks were 2-4: a floor
    the map cannot fund is never crossed, so expansion stayed protected
    forever and the army channels starved on maps where the race was already
    over (`runs/sweeps/xmap-*`, log 2026-08-05). The number was never the
    policy; the map's own pool count was.

    So the floor is what duel_lake's seven actually measured: **everything
    the builder can reach, minus the share the rival takes.** On duel_lake's
    9 reachable pools that is exactly the seven the traces demanded; on a
    map offering four it is two, and protection ends where the map's race
    does instead of where duel_lake's did.

    Args:
        visible: Pools the sample carries in total.
        unreachable: Pools the builder cannot walk to at all.

    Returns:
        The derived floor, never below :data:`FLOOR_MINIMUM`.
    """
    return max(FLOOR_MINIMUM, visible - unreachable - RIVAL_POOL_SHARE)


__all__ = ["FLOOR_MINIMUM", "RIVAL_POOL_SHARE", "economy_floor"]
