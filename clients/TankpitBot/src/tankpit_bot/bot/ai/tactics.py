"""Tactical decision functions for the AI system.

Pure functions that determine when to proactively radar, when to open
the map for enemy discovery, and what equipment should be active.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIConfigDict
from tankpit_bot.physics.capacity import inventory_capacity
from tankpit_bot.state.types import (
    ContainerStateDict,
    WorldStateDict,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


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


def _is_within_observable_viewport(world: WorldStateDict, x: int, y: int) -> bool:
    """Return True when a coordinate lies inside the current visible viewport.

    Args:
        world: Current world state.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.

    Returns:
        True if the coordinate lies inside the observable viewport bounds.
    """
    left, top, right, bottom = viewport_visible_bounds(world["viewport"])
    return left <= x <= right and top <= y <= bottom


def _is_visible_fuel_container(world: WorldStateDict, container: ContainerStateDict) -> bool:
    """Return True when a fuel container is currently visible in the viewport.

    Args:
        world: Current world state.
        container: Container candidate.

    Returns:
        True if the container is fuel, non-empty, and inside the observable frame.
    """
    return (
        container["is_fuel"]
        and container["volume"] > 0
        and _is_within_observable_viewport(world, container["x"], container["y"])
    )


def should_proactive_radar(
    fuel: int,
    world: WorldStateDict,
    last_scan_ms: int,
    now: int,
    config: AIConfigDict,
) -> bool:
    """Check if a proactive radar scan is needed for fuel discovery.

    Triggers when fuel is low (<=fuel_low_threshold), no containers are
    visible, and the scan cooldown has elapsed. One radar per viewport
    is enough — collect everything before scanning again.

    Args:
        fuel: Current fuel level.
        world: Current world state.
        last_scan_ms: Timestamp of last radar scan.
        now: Current timestamp in milliseconds.
        config: AI configuration.

    Returns:
        True if a proactive radar scan should be performed.
    """
    # Respect scan cooldown first (cheap check)
    if now - last_scan_ms < config["scan_cooldown_ms"]:
        return False

    # Don't radar when FUEL containers are already visible — collect them first.
    # Equipment containers don't count — we need fuel, not equipment.
    for container in world["containers"].values():
        if _is_visible_fuel_container(world, container):
            return False

    # Only radar when fuel is low
    return fuel <= config["fuel_low_threshold"]


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
    "should_proactive_radar",
]
