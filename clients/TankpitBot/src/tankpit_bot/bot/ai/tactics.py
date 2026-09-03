"""Tactical decision functions for the AI system.

Pure functions that determine when to proactively radar, when to open
the map for enemy discovery, and what equipment should be active.
"""

from __future__ import annotations

from tankpit_bot.physics.capacity import inventory_capacity


def combat_radar_min(rank: int) -> int:
    """Return the minimum extra-radar count for HUNT-entry readiness.

    This is bot POLICY, not game physics — it lives here (the lowest
    equipment-policy module) because both the HUNT-entry gate and the
    radar-hoard toggle read it. User contract (2026-07-06): weapons
    must be at cap for HUNT entry, but extra radars are permitted up
    to 5 below cap because scan coverage during the fight consumes
    them faster than the between-kill restock can top them up. The
    floor is ``inventory_capacity(rank) - 5``.

    Args:
        rank: Wire rank field, ``0`` (recruit) through ``8`` (general).

    Returns:
        Minimum extra-radar count below which HUNT entry is refused:
        15 at recruit, 20 at private, 25 at corporal, ..., 55 at
        general.
    """
    return inventory_capacity(rank) - 5


SWARM_MUSTER_QUORUM = 2
"""War-ready fleet members (self included) a swarm bot needs standing
before it will OPEN a human fight nobody is in yet. Joining a fight a
sibling already holds needs no quorum — reinforcement beats
book-keeping. Two is the smallest number that is not fighting alone,
which is the serial trickle the doctrine exists to end (operator
order 2026-09-01). Policy, so it lives here beside the wartime
arithmetic both the acquisition gate and the readiness floor read."""


def wartime_inventory_ready(dual: int, homing: int, radar: int, rank: int) -> bool:
    """Return True when an inventory clears the wartime readiness floor.

    The 80%/50% wartime bar (operator ruling 2026-09-01, verbatim:
    "like 80% equipment and 50% radar?"), extracted here as pure
    arithmetic because two layers consult it: the HUNT-entry gate
    while a war is live, and the fleetshare report's ``war_ready``
    row that feeds the swarm doctrine's muster quorum.

    Args:
        dual: Dual-shot count.
        homing: Homing-shot count.
        radar: Extra-radar count.
        rank: Wire rank field, ``0`` (recruit) through ``8`` (general).

    Returns:
        True when both weapons are at 80% of the rank cap and radars
        at half of it.
    """
    cap = inventory_capacity(rank)
    war_weapon_floor = (cap * 4) // 5
    return dual >= war_weapon_floor and homing >= war_weapon_floor and radar >= cap // 2


def compute_desired_equipment(
    mode: str,
    fuel: int,
    dual_shots_count: int = 99,
    homing_shots_count: int = 99,
) -> set[int]:
    """Compute which equipment slots should be enabled.

    Returns the set of slot numbers (1-5) that should be active.
    Dual shots (2), homing shots (4), and extra radar (5) stay enabled
    while stocked. Shields and missiles stay off. (The 2026-08-28
    radar-hoard band was reverted the same day by operator order --
    radar spending follows the reveal-floor economics, not a stock
    band.)

    Args:
        mode: Current AI behavior mode name.
        fuel: Current fuel level.
        dual_shots_count: Number of dual shots remaining.
        homing_shots_count: Number of homing shots remaining.

    Returns:
        Set of equipment slot numbers that should be enabled.
    """
    desired: set[int] = {5}  # Extra radar always on

    # Dual shots when we have stock (avoids "You can't do this" when depleted)
    if dual_shots_count > 0:
        desired.add(2)
    if homing_shots_count > 0:
        desired.add(4)

    return desired


__all__ = [
    "combat_radar_min",
    "compute_desired_equipment",
]
