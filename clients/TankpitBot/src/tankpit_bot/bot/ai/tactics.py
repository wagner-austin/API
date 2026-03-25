"""Tactical decision functions for the AI system.

Pure functions that determine when to proactively radar, when to teleport
to search for resources, and what equipment should be active based on
the current behavior mode and game state.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.types import AIConfigDict, BehaviorScoreDict
from tankpit_bot.state.types import SelfStateDict, WorldStateDict

# Fuel buffer above low threshold for triggering proactive radar.
# At fuel_low_threshold + this value, radar is used to discover fuel early.
_RADAR_FUEL_BUFFER = 200


def should_proactive_radar(
    fuel: int,
    world: WorldStateDict,
    last_scan_ms: int,
    now: int,
    config: AIConfigDict,
) -> bool:
    """Check if a proactive radar scan is needed.

    Triggers when fuel is approaching the low threshold and no fuel
    containers are currently visible, so the bot discovers nearby
    fuel before running critically low.

    Args:
        fuel: Current fuel level.
        world: Current world state.
        last_scan_ms: Timestamp of last radar scan.
        now: Current timestamp in milliseconds.
        config: AI configuration.

    Returns:
        True if a proactive radar scan should be performed.
    """
    # Only when fuel is approaching low threshold
    if fuel >= config["fuel_low_threshold"] + _RADAR_FUEL_BUFFER:
        return False
    # Only if no fuel containers are visible
    has_fuel = any(c["is_fuel"] for c in world["containers"].values())
    if has_fuel:
        return False
    # Respect scan cooldown
    return now - last_scan_ms >= config["scan_cooldown_ms"]


def should_teleport_search(
    behavior: BehaviorScoreDict,
    fuel: int,
    world: WorldStateDict,
    last_scan_ms: int,
    now: int,
    config: AIConfigDict,
) -> bool:
    """Check if the bot should teleport to find resources.

    Triggers when fuel is low, no containers are visible anywhere,
    and we have already scanned recently (so the area is confirmed
    empty). The bot teleports to the farthest waypoint to search
    a new area.

    Args:
        behavior: Chosen behavior from evaluators.
        fuel: Current fuel level.
        world: Current world state.
        last_scan_ms: Timestamp of last radar scan.
        now: Current timestamp in milliseconds.
        config: AI configuration.

    Returns:
        True if a teleport search should be performed.
    """
    # Only when fuel is below low threshold
    if fuel >= config["fuel_low_threshold"]:
        return False
    # Only if area is completely empty (no containers at all)
    if len(world["containers"]) > 0:
        return False
    # Only if we scanned recently (area is confirmed empty)
    scan_age = now - last_scan_ms
    if scan_age >= config["scan_cooldown_ms"]:
        return False
    # Only override low-priority behaviors (PATROL, zero-score)
    return behavior["score"] <= 100


def find_teleport_target(
    config: AIConfigDict,
    self_state: SelfStateDict,
) -> tuple[int, int]:
    """Find the best teleport target for resource searching.

    Picks the farthest patrol waypoint from the current position
    to maximize the chance of finding new resources.

    Args:
        config: AI configuration with patrol waypoints.
        self_state: Player's own state for position.

    Returns:
        Tuple of (x, y) coordinates for the teleport target.
    """
    waypoints = config["patrol_waypoints"]
    best = waypoints[0]
    best_dist = 0
    for wx, wy in waypoints:
        dist = abs(wx - self_state["x"]) + abs(wy - self_state["y"])
        if dist > best_dist:
            best_dist = dist
            best = (wx, wy)
    return best


def compute_desired_equipment(
    mode: str,
    fuel: int,
    target_damage: int,
    fuel_critical_threshold: int,
    is_teleport: bool = False,
) -> set[int]:
    """Compute which equipment slots should be enabled.

    Returns the set of slot numbers (1-5) that should be active
    for the current behavior mode, fuel level, and combat state.
    All other combat slots (1-4) should be disabled.

    Args:
        mode: Current AI behavior mode name.
        fuel: Current fuel level.
        target_damage: Damage state of the hunt target (0-3).
        fuel_critical_threshold: Fuel level below which shields activate.
        is_teleport: Whether the current command is a teleport.

    Returns:
        Set of equipment slot numbers that should be enabled.
    """
    desired: set[int] = {5}  # Extra radar always on

    shields_on = (
        mode == "DEFEND"
        or (mode == "COLLECT_FUEL" and fuel < fuel_critical_threshold)
        or is_teleport
    )
    if shields_on:
        desired.add(1)

    if mode == "HUNT":
        desired.add(2)  # Dual shots always during HUNT
        if target_damage >= 3:
            desired.add(4)  # Homing only when enemy is critical

    return desired


__all__ = [
    "compute_desired_equipment",
    "find_teleport_target",
    "should_proactive_radar",
    "should_teleport_search",
]
