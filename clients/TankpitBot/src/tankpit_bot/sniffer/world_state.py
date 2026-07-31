"""World state tracking — singleton bridge to WorldService.

This module maintains module-level functions that delegate to a singleton
``WorldService`` instance. Consumer modules are being migrated to receive
``WorldService`` directly via dependency injection. Once all consumers are
migrated, this module will be removed.

The singleton instance is accessible via ``get_world_service()`` for
modules that need the full service during migration.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import WorldStateDict

_service = WorldService()


def get_world_service() -> WorldService:
    """Return the singleton WorldService instance.

    Returns:
        The module-level WorldService instance.
    """
    return _service


def reset_world_state() -> None:
    """Reset world state for new session (used by tests)."""
    global _service
    _service = WorldService()


def get_world_state() -> WorldStateDict:
    """Get the current world state.

    Returns:
        Current WorldStateDict with containers, mines, self_state, etc.
    """
    return _service.get_world_state()


def get_terrain_map() -> _test_hooks.TerrainMapProtocol | None:
    """Get the current terrain map, loading if needed.

    Returns:
        TerrainMap instance, or None if terrain GIF not found.
    """
    return _service.get_terrain_map()


def check_and_clear_radar_scan_complete() -> bool:
    """Check if a radar scan completed since last check, then clear.

    Returns:
        True if radar completion was observed.
    """
    return _service.check_and_clear_radar_scan_complete()


def mark_radar_scan_complete() -> None:
    """Record that the server completed a radar scan."""
    _service.mark_radar_scan_complete()


def register_room_image(room_id: str, image: str) -> None:
    """Register a room's field image from a ROOM_LIST message.

    Args:
        room_id: Room ID (e.g. "2").
        image: Field image filename (e.g. "field42.gif").
    """
    _service.register_room_image(room_id, image)


def set_selected_room(room_id: str) -> None:
    """Track which room was selected from a SELECT message.

    Args:
        room_id: Room ID that was selected.
    """
    _service.set_selected_room(room_id)


def update_world_state_from_position(x: int, y: int) -> None:
    """Update world state with new self position.

    Args:
        x: Self X coordinate.
        y: Self Y coordinate.
    """
    _service.update_world_state_from_position(x, y)


def mark_move_target_failed(x: int, y: int, timestamp_ms: int) -> None:
    """Record a move destination that stalled and timed out.

    Args:
        x: Failed destination X coordinate.
        y: Failed destination Y coordinate.
        timestamp_ms: When the failure was detected.
    """
    _service.mark_move_target_failed(x, y, timestamp_ms)


def is_move_target_failed(x: int, y: int, now_ms: int) -> bool:
    """Check if a move target was recently marked as failed.

    Args:
        x: Destination X coordinate.
        y: Destination Y coordinate.
        now_ms: Current timestamp for TTL check.

    Returns:
        True if the target failed recently and should be avoided.
    """
    return _service.is_move_target_failed(x, y, now_ms)


def record_movement_rejection(timestamp_ms: int) -> None:
    """Record a server cant_go refusal of any movement leg.

    Args:
        timestamp_ms: When the rejection arrived.
    """
    _service.record_movement_rejection(timestamp_ms)


def recent_movement_rejections(now_ms: int, window_ms: int) -> int:
    """Count movement rejections inside the trailing window.

    Args:
        now_ms: Current wall-clock ms.
        window_ms: Trailing window length.

    Returns:
        Number of rejections newer than ``now_ms - window_ms``.
    """
    return _service.recent_movement_rejections(now_ms, window_ms)


def mark_container_desync(timestamp_ms: int) -> None:
    """Record a disproven remembered-container belief (code=4 pickup).

    Args:
        timestamp_ms: When the empty-container rejection arrived.
    """
    _service.container_desync_ms = timestamp_ms


def clear_container_desync() -> None:
    """Answer a container desync without a scan.

    Used when live coverage already tells the whole story (radar-spend
    economics, s9-4): rescanning ground scanned seconds earlier buys
    nothing, so the disproof is considered answered by the existing
    coverage.
    """
    _service.container_desync_ms = 0


def container_desync_pending() -> bool:
    """Check whether a container desync awaits its radar resync.

    Returns:
        True while a code=4 disproof has not yet been answered by a
        radar response (which reconciles the viewport and clears it).
    """
    return _service.container_desync_ms > 0


def recent_own_mine_hit(now_ms: int) -> bool:
    """Check whether a walk-over mine hit landed within the flip window.

    User doctrine 2026-07-30: "walk to targets or containers in
    viewport but if we hit a mine teleport to target or container.
    then resume walking within viewport." One window is enough for
    the flipped approach to dispatch; afterwards walking resumes.

    Args:
        now_ms: Current timestamp.

    Returns:
        True while the last own-tile detonation is fresh.
    """
    return now_ms - _service.last_own_mine_hit_ms < _OWN_MINE_HIT_FLIP_MS


# The reactive walk->teleport flip stays live long enough for the next
# decision to dispatch the teleport approach (a few server windows),
# then expires so walking resumes per the doctrine.
_OWN_MINE_HIT_FLIP_MS = 6_000


def get_incoming_damage_window(now_ms: int, window_ms: int) -> tuple[int, int]:
    """Return fuel-confirmed incoming (hits, fuel) in the trailing window.

    The damage-aware engagement break's rate instrument -- reads the
    session damage book ([[bot-behavior-contract]] §3.3).

    Args:
        now_ms: Current wall-clock ms.
        window_ms: Trailing window length in ms.

    Returns:
        ``(hits, fuel)`` confirmed within the window.
    """
    from tankpit_bot.ledger.damage_book import incoming_damage_window

    return incoming_damage_window(_service.damage_book, now_ms, window_ms)


def mark_scan_viewport_failed(viewport_left: int, viewport_top: int, timestamp_ms: int) -> None:
    """Record a viewport whose radar scan stalled and timed out.

    Args:
        viewport_left: Failed viewport left X coordinate.
        viewport_top: Failed viewport top Y coordinate.
        timestamp_ms: When the failure was detected.
    """
    _service.mark_scan_viewport_failed(viewport_left, viewport_top, timestamp_ms)


def is_scan_viewport_failed(viewport_left: int, viewport_top: int, now_ms: int) -> bool:
    """Check whether a viewport recently had a stalled radar scan.

    Args:
        viewport_left: Viewport left X coordinate.
        viewport_top: Viewport top Y coordinate.
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        True if radar recently stalled for that viewport.
    """
    return _service.is_scan_viewport_failed(viewport_left, viewport_top, now_ms)


__all__ = [
    "check_and_clear_radar_scan_complete",
    "clear_container_desync",
    "container_desync_pending",
    "get_incoming_damage_window",
    "get_terrain_map",
    "get_world_service",
    "get_world_state",
    "is_move_target_failed",
    "is_scan_viewport_failed",
    "mark_container_desync",
    "mark_move_target_failed",
    "mark_radar_scan_complete",
    "mark_scan_viewport_failed",
    "recent_movement_rejections",
    "record_movement_rejection",
    "register_room_image",
    "reset_world_state",
    "set_selected_room",
    "update_world_state_from_position",
]
