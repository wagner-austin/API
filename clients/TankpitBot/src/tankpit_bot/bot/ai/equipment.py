"""Container predicates for the AI system.

Pure functions that evaluate whether containers are pursuable and
whether locked targets should be released. Search functions that find
specific containers live in ``equipment_search.py``. Per-tile scan
coverage lives in :mod:`tankpit_bot.state.scan_coverage`.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.state.types import (
    ContainerStateDict,
    MineStateDict,
    SelfStateDict,
    WorldStateDict,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

log = get_logger(__name__)

_CONTAINER_FRESHNESS_TTL_MS = 30000

_LOCK_RELEASE_MIN_GAP = 10


def is_lock_release_warranted(
    self_state: SelfStateDict,
    locked_x: int,
    locked_y: int,
    candidate_x: int,
    candidate_y: int,
) -> bool:
    """Return True when a candidate is enough closer to drop a locked target.

    Args:
        self_state: Player state for the distance origin.
        locked_x: Locked target X coordinate.
        locked_y: Locked target Y coordinate.
        candidate_x: Fresh candidate X coordinate.
        candidate_y: Fresh candidate Y coordinate.

    Returns:
        True when the candidate is at most half the locked distance and
        at least ``_LOCK_RELEASE_MIN_GAP`` tiles closer.
    """
    sx, sy = self_state["x"], self_state["y"]
    locked_dist = manhattan_distance(sx, sy, locked_x, locked_y)
    candidate_dist = manhattan_distance(sx, sy, candidate_x, candidate_y)
    if candidate_dist * 2 > locked_dist:
        return False
    return locked_dist - candidate_dist >= _LOCK_RELEASE_MIN_GAP


def is_container_pursuable(
    container: ContainerStateDict,
    *,
    want_fuel: bool,
    now_ms: int,
) -> bool:
    """Return True when a tracked container is worth pursuing at all.

    This is the SINGLE definition of pursuability: candidate selection,
    opportunistic pickups, and lock continuation must all apply it.

    Args:
        container: Tracked container to check.
        want_fuel: True to require fuel, False to require equipment.
        now_ms: Current timestamp for freshness filtering. ``0`` disables
            the TTL.

    Returns:
        True when the container matches the kind, has no failed pickup,
        and is within the freshness TTL.
    """
    if container["is_fuel"] != want_fuel:
        return False
    if container["failed_pickups"] > 0:
        return False
    return not (now_ms > 0 and _is_stale(container, now_ms))


def _viewport_bounds(world: WorldStateDict) -> tuple[int, int, int, int]:
    """Return inclusive observable viewport bounds from world state."""
    return viewport_visible_bounds(world["viewport"])


def _is_stale(container: ContainerStateDict, now_ms: int) -> bool:
    """Return True when a container's timestamp is older than the freshness TTL.

    Args:
        container: Container to check.
        now_ms: Current timestamp.

    Returns:
        True when the container's last-seen time exceeds the TTL.
    """
    age = now_ms - container["timestamp_ms"]
    return age > _CONTAINER_FRESHNESS_TTL_MS


def hostile_mines(world: WorldStateDict) -> dict[str, MineStateDict]:
    """Return tracked mines that would damage the bot if detonated.

    Tankpit mines do not damage tanks on the placer's team -- friendly
    mines are passable. Every blocking / pathing check should query
    this filtered view instead of ``world["mines"]`` directly so the
    bot doesn't treat its own team's defensive layout as obstacles.

    If ``self_state`` is not yet known, every mine is treated as
    hostile (defensive default).

    Args:
        world: Current world state.

    Returns:
        Mines indexed by ``"x,y"`` key whose team is different from the
        bot's team, or every tracked mine when ``self_state`` is None.
    """
    self_state = world["self_state"]
    if self_state is None:
        return world["mines"]
    self_team = self_state["team"]
    return {key: mine for key, mine in world["mines"].items() if mine["team"] != self_team}


__all__ = [
    "_viewport_bounds",
    "hostile_mines",
    "is_container_pursuable",
    "is_lock_release_warranted",
]
