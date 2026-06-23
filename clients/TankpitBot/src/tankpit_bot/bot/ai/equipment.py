"""Container predicates and scan coverage for the AI system.

Pure functions that evaluate whether containers are pursuable, whether
viewport areas have been scanned, and whether locked targets should be
released. Search functions that find specific containers live in
``equipment_search.py``.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.bot.ai.threats import manhattan_distance
from tankpit_bot.state.types import (
    ContainerStateDict,
    SelfStateDict,
    WorldStateDict,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

log = get_logger(__name__)

_CONTAINER_FRESHNESS_TTL_MS = 30000

_LOCK_RELEASE_MIN_GAP = 10

_SCAN_COVERAGE_OVERLAP_TILES = 4

_SCAN_COVERAGE_TTL_MS = 45000

#: Public alias for cross-module consumers (recover_fuel_mode).
SCAN_COVERAGE_TTL_MS = _SCAN_COVERAGE_TTL_MS


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


def is_area_scanned(world: WorldStateDict, left: int, top: int, now_ms: int) -> bool:
    """Return True when a viewport origin has fresh overlapping scan coverage.

    Args:
        world: Current world state with scan coverage records.
        left: Queried viewport left X coordinate.
        top: Queried viewport top Y coordinate.
        now_ms: Current timestamp for coverage freshness.

    Returns:
        True when a fresh, mostly-overlapping scan covers the origin.
    """
    for key, scanned_ms in world["scanned_viewports"].items():
        if now_ms - scanned_ms > _SCAN_COVERAGE_TTL_MS:
            continue
        key_left_text, _, key_top_text = key.partition(",")
        if (
            abs(int(key_left_text) - left) <= _SCAN_COVERAGE_OVERLAP_TILES
            and abs(int(key_top_text) - top) <= _SCAN_COVERAGE_OVERLAP_TILES
        ):
            return True
    return False


def is_tile_scanned(world: WorldStateDict, x: int, y: int, now_ms: int) -> bool:
    """Return True when a world tile sits inside fresh scan coverage.

    Args:
        world: Current world state with scan coverage records.
        x: World tile X coordinate.
        y: World tile Y coordinate.
        now_ms: Current timestamp for coverage freshness.

    Returns:
        True when a fresh scan's viewport contained the tile.
    """
    width = world["viewport"]["width"]
    height = world["viewport"]["height"]
    for key, scanned_ms in world["scanned_viewports"].items():
        if now_ms - scanned_ms > _SCAN_COVERAGE_TTL_MS:
            continue
        key_left_text, _, key_top_text = key.partition(",")
        scan_left = int(key_left_text)
        scan_top = int(key_top_text)
        if scan_left <= x < scan_left + width and scan_top <= y < scan_top + height:
            return True
    return False


def is_current_viewport_scanned(world: WorldStateDict) -> bool:
    """Return True when the current viewport has authoritative local coverage.

    Args:
        world: Current world state.

    Returns:
        True if the current viewport area is covered by a fresh radar scan.
    """
    viewport = world["viewport"]
    return is_area_scanned(world, viewport["left"], viewport["top"], world["timestamp_ms"])


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


__all__ = [
    "SCAN_COVERAGE_TTL_MS",
    "_viewport_bounds",
    "is_area_scanned",
    "is_container_pursuable",
    "is_current_viewport_scanned",
    "is_lock_release_warranted",
    "is_tile_scanned",
]
