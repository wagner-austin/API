"""Tactical decision functions for the AI system.

Pure functions that determine when to proactively radar, when to open
the map for enemy discovery, and what equipment should be active.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIConfigDict
from tankpit_bot.state.types import ContainerStateDict, SelfStateDict, TankStateDict, WorldStateDict
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds


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


def _is_visible_enemy_tank(
    world: WorldStateDict,
    self_state: SelfStateDict,
    tank: TankStateDict,
) -> bool:
    """Return True when an enemy tank is currently visible in the viewport.

    Args:
        world: Current world state.
        self_state: Player state for team filtering.
        tank: Tank candidate.

    Returns:
        True if the tank is a live enemy inside the observable viewport.
    """
    return (
        not tank["is_self"]
        and tank["team"] != self_state["team"]
        and not (tank["x"] == 0 and tank["y"] == 0)
        and _is_within_observable_viewport(world, tank["x"], tank["y"])
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


def should_map_open_for_enemies(
    world: WorldStateDict,
    self_state: SelfStateDict,
    last_map_open_ms: int,
    now: int,
    config: AIConfigDict,
) -> bool:
    """Check if the bot should open the map to discover enemy positions.

    Map open (CMD_MAP_OPEN, 'f' key) reveals global enemy tank positions.
    This is the only way to find enemies outside the visible viewport.

    Triggers when no enemy tanks are visible in the world state (excluding
    dead tanks at 0,0) and the map open cooldown has elapsed.

    Args:
        world: Current world state.
        self_state: Player's own state for team filtering.
        last_map_open_ms: Timestamp of last map open command.
        now: Current timestamp in milliseconds.
        config: AI configuration with cooldown settings.

    Returns:
        True if a map open should be performed.
    """
    # Respect cooldown
    if now - last_map_open_ms < config["map_open_cooldown_ms"]:
        return False

    for tank in world["tanks"].values():
        if _is_visible_enemy_tank(world, self_state, tank):
            return False

    return True


def compute_desired_equipment(
    mode: str,
    fuel: int,
    dual_shots_count: int = 99,
    homing_shots_count: int = 99,
) -> set[int]:
    """Compute which equipment slots should be enabled.

    Returns the set of slot numbers (1-5) that should be active.
    Dual shots (2), homing shots (4), and extra radar (5) stay enabled
    while stocked. Shields and missiles stay off.

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
    "compute_desired_equipment",
    "should_map_open_for_enemies",
    "should_proactive_radar",
]
