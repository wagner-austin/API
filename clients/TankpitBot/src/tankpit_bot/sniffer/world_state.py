"""World state tracking from radar, movement, and inventory messages.

This module maintains the current world state (containers, mines, player position,
inventory) and renders ASCII visualizations of the game world.

Inventory is tracked from binary protocol messages (0x49, 0x67, 0x74) instead
of DOM scraping, providing reliable absolute counts without false transitions.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.inventory import (
    InventoryItem,
    InventoryState,
    ItemType,
)
from tankpit_bot.state import (
    WorldStateDict,
    make_empty_world_state,
    update_self_position,
    viewport_scan_key,
)
from tankpit_bot.state.viewport_geometry import (
    viewport_radar_bounds,
    viewport_visible_bounds,
)

log = get_logger(__name__)

# Canonical item order matching protocol array indices [0..4]
_ITEM_TYPES: list[ItemType] = [
    "armor_shields",
    "dual_shots",
    "missile_shots",
    "homing_shots",
    "extra_radars",
]

# Module-level world state - updated as messages are processed
_world_state: WorldStateDict = make_empty_world_state()

# Module-level terrain map - loaded on first radar response
_terrain_map: _test_hooks.TerrainMapProtocol | None = None

# Room tracking: room_id -> field image name (e.g. "2" -> "field42.gif")
_room_images: dict[str, str] = {}
_selected_room: str | None = None

# Combat hit tracking.
#
# CombatHit (0x2E) fires for EVERY shot — hits, misses, corpses, mines.
# The last byte of combat_data is the WEAPON TYPE:
#   00=single, 01=dual, 02=missile, 03=homing
#
# Hit detection: the game only uses special ammo (dual/missile/homing) when
# shooting at an actual enemy. If the target position is empty, it falls back
# to single shot (00) even with dual enabled. So:
#   weapon_byte > 0 = hit (special ammo used = enemy at target)
#   weapon_byte == 0 with dual enabled = miss (no enemy at target)
#
# The bot always keeps dual shots enabled, so 00 = miss, 01+ = hit.
_got_confirmed_hit: bool = False

# Tracks whether ANY CombatHit response arrived for our shot (weapon_byte >= 0).
# Separate from _got_confirmed_hit which only tracks weapon_byte > 0.
# When weapon_byte=0 arrives, the server DID process our shot — it just used
# single ammo (no dual available or target was empty).
_got_our_shot_response: bool = False

# Weapon byte → inventory item key. Used to decrement ammo on confirmed hits.
# weapon_byte: 1=dual, 2=missile, 3=homing. 0=single (no ammo consumed).
_WEAPON_BYTE_TO_ITEM: dict[int, ItemType] = {
    1: "dual_shots",
    2: "missile_shots",
    3: "homing_shots",
}

# Kill tracking — tank IDs killed via Deactivation protocol message.
# Corpses stay at their death position, so the AI must filter by ID not position.
_killed_tank_ids: set[int] = set()

# Teleport tracking — set to True when TeleportLanded is received from server.
# Drained by the tick loop so it knows the teleport completed.
_teleport_landed: bool = False

# Radar scan tracking — set when a radar result/ack arrives from the server.
# Drained by the bot state machine so SCANNING can complete even when the scan
# finds zero containers.
_radar_scan_complete: bool = False

# Some radar scans arrive as a differential cache refresh (0x4F CombinedTileUpdate)
# followed by a RadarAck instead of an explicit RadarResponse. Track the most
# recent combined-tile refresh so RadarAck can promote it to authoritative
# container truth for the current viewport.
_pending_radar_cache_refresh_ms: int = 0
_RADAR_CACHE_REFRESH_WINDOW_MS = 2000

# Some tunneled 0x2E -> 0x4F radar packets are also differential snapshots with
# zero container/mine deltas. In that form, RadarAck(found=True) means "there
# are still radar resources in this viewport, but nothing changed since last
# known state". Track the empty differential so RadarAck can preserve existing
# authoritative resource state instead of clearing it.
_pending_radar_empty_delta_ms: int = 0

# Failed move targets — coordinates where a move stalled and timed out.
# Maps "x,y" key to timestamp_ms of the failure. Cleared on radar refresh
# and session reset. The planner rejects these coordinates until they expire
# or are re-confirmed by fresh world data.
_failed_move_targets: dict[str, int] = {}

# TTL for failed move targets (30 seconds). After this, the target is
# eligible again in case the obstacle was transient.
_FAILED_MOVE_TTL_MS = 30000

# Failed scan viewports — viewport origins where radar stalled and timed out.
# Maps "left,top" key to timestamp_ms of the failure. Cleared when the viewport
# later receives authoritative local confirmation.
_failed_scan_viewports: dict[str, int] = {}

# TTL for failed scan viewports (30 seconds). During this window the planner
# should avoid reissuing radar for the same viewport.
_FAILED_SCAN_VIEWPORT_TTL_MS = 30000


def mark_radar_scan_complete() -> None:
    """Record that the server completed a radar scan."""
    global _radar_scan_complete
    _radar_scan_complete = True


def _mark_pending_radar_cache_refresh() -> None:
    """Record that a recent combined-tile update may belong to a radar scan."""
    global _pending_radar_cache_refresh_ms
    _pending_radar_cache_refresh_ms = get_current_time_ms()


def _consume_pending_radar_cache_refresh() -> bool:
    """Return True if a recent combined-tile update should count as radar."""
    global _pending_radar_cache_refresh_ms
    if _pending_radar_cache_refresh_ms <= 0:
        return False
    now = get_current_time_ms()
    recent = now - _pending_radar_cache_refresh_ms <= _RADAR_CACHE_REFRESH_WINDOW_MS
    _pending_radar_cache_refresh_ms = 0
    return recent


def _mark_pending_radar_empty_delta() -> None:
    """Record that a zero-delta tunneled radar result was observed."""
    global _pending_radar_empty_delta_ms
    _pending_radar_empty_delta_ms = get_current_time_ms()


def _consume_pending_radar_empty_delta() -> bool:
    """Return True if a recent zero-delta tunneled radar result is pending."""
    global _pending_radar_empty_delta_ms
    if _pending_radar_empty_delta_ms <= 0:
        return False
    now = get_current_time_ms()
    recent = now - _pending_radar_empty_delta_ms <= _RADAR_CACHE_REFRESH_WINDOW_MS
    _pending_radar_empty_delta_ms = 0
    return recent


def check_and_clear_radar_scan_complete() -> bool:
    """Check if a radar scan completed since last check, then clear.

    Returns:
        True if radar completion was observed.
    """
    global _radar_scan_complete
    result = _radar_scan_complete
    _radar_scan_complete = False
    return result


# Inventory tracking from binary protocol (0x49, 0x67, 0x74)
# Default: all disabled, zero counts. The game starts with most items disabled
# (armor, missile, homing disabled; dual and radar enabled). The protocol
# messages (0x49 InventorySync) will set the correct state on first sync.
_inventory_state: InventoryState = InventoryState(
    armor_shields=InventoryItem(count=0, enabled=False),
    dual_shots=InventoryItem(count=0, enabled=False),
    missile_shots=InventoryItem(count=0, enabled=False),
    homing_shots=InventoryItem(count=0, enabled=False),
    extra_radars=InventoryItem(count=0, enabled=False),
)


def _make_empty_inventory() -> InventoryState:
    """Create an empty inventory state with all items at zero.

    Returns:
        InventoryState with all counts at 0 and enabled False.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=0, enabled=False),
        dual_shots=InventoryItem(count=0, enabled=False),
        missile_shots=InventoryItem(count=0, enabled=False),
        homing_shots=InventoryItem(count=0, enabled=False),
        extra_radars=InventoryItem(count=0, enabled=False),
    )


def reset_world_state() -> None:
    """Reset world state for new session (used by tests)."""
    global _world_state, _terrain_map, _room_images, _selected_room, _inventory_state
    global _got_confirmed_hit, _got_our_shot_response
    global _killed_tank_ids, _teleport_landed
    global _radar_scan_complete, _pending_radar_cache_refresh_ms, _pending_radar_empty_delta_ms
    _world_state = make_empty_world_state()
    _terrain_map = None
    _room_images = {}
    _selected_room = None
    _inventory_state = _make_empty_inventory()
    _got_confirmed_hit = False
    _got_our_shot_response = False
    _killed_tank_ids = set()
    _teleport_landed = False
    _radar_scan_complete = False
    _pending_radar_cache_refresh_ms = 0
    _pending_radar_empty_delta_ms = 0
    _failed_move_targets.clear()
    _failed_scan_viewports.clear()


def get_world_state() -> WorldStateDict:
    """Get the current world state.

    Returns:
        Current WorldStateDict with containers, mines, self_state, etc.
    """
    return _world_state


def get_terrain_map() -> _test_hooks.TerrainMapProtocol | None:
    """Get the current terrain map, loading if needed.

    Returns:
        TerrainMap instance, or None if terrain GIF not found.
    """
    return _load_terrain_map_if_needed()


def register_room_image(room_id: str, image: str) -> None:
    """Register a room's field image from a ROOM_LIST message.

    Args:
        room_id: Room ID (e.g. "2").
        image: Field image filename (e.g. "field42.gif").
    """
    global _room_images
    _room_images[room_id] = image


def set_selected_room(room_id: str) -> None:
    """Track which room was selected from a SELECT message.

    Resets the terrain map so the correct one loads on next render.

    Args:
        room_id: Room ID that was selected.
    """
    global _selected_room, _terrain_map
    _selected_room = room_id
    _terrain_map = None
    image = _room_images.get(room_id)
    log.info("Selected room %s (field image: %s)", room_id, image or "unknown")


def _find_field_gif(image: str) -> Path | None:
    """Find the local GIF file for a field image name.

    The game sends names like "field42.gif" but local files use a rendered
    suffix: "field42-r.gif" or "field42_r.gif".

    Args:
        image: Field image filename from server (e.g. "field42.gif").

    Returns:
        Path to the local GIF file, or None if not found.
    """
    stem = image.removesuffix(".gif")
    candidates = [
        Path(f"{stem}_r.gif"),
        Path(f"{stem}-r.gif"),
    ]
    for path in candidates:
        if _test_hooks.path_exists(path):
            return path
    return None


def _load_terrain_map_if_needed() -> _test_hooks.TerrainMapProtocol | None:
    """Load terrain map for the selected room.

    Uses the field image from the room list to find the correct GIF.

    Returns:
        TerrainMap instance, or None if file not found.
    """
    global _terrain_map
    if _terrain_map is not None:
        return _terrain_map

    # Try to load from selected room's field image
    if _selected_room is not None:
        image = _room_images.get(_selected_room)
        if image is not None:
            gif_path = _find_field_gif(image)
            if gif_path is not None:
                _terrain_map = _test_hooks.load_terrain_map(gif_path)
                log.info("Loaded terrain map from %s (room %s)", gif_path, _selected_room)
                return _terrain_map
            log.warning("No local GIF found for %s", image)

    # Fallback: try known GIF paths
    gif_paths = [
        Path("field01_r.gif"),
        Path("field42-r.gif"),
    ]
    for gif_path in gif_paths:
        if _test_hooks.path_exists(gif_path):
            _terrain_map = _test_hooks.load_terrain_map(gif_path)
            log.info("Loaded terrain map from %s (fallback)", gif_path)
            return _terrain_map

    return None


def update_world_state_from_position(x: int, y: int) -> None:
    """Update world state with new self position.

    Args:
        x: Self X coordinate.
        y: Self Y coordinate.
    """
    global _world_state
    _world_state = update_self_position(_world_state, x, y, get_current_time_ms())


def mark_move_target_failed(x: int, y: int, timestamp_ms: int) -> None:
    """Record a move destination that stalled and timed out.

    The planner should avoid re-selecting this coordinate until
    it is cleared by a radar refresh or the TTL expires.

    Args:
        x: Failed destination X coordinate.
        y: Failed destination Y coordinate.
        timestamp_ms: When the failure was detected.
    """
    key = f"{x},{y}"
    _failed_move_targets[key] = timestamp_ms
    log.info("MOVE: marked (%d,%d) as failed target", x, y)


def mark_scan_viewport_failed(viewport_left: int, viewport_top: int, timestamp_ms: int) -> None:
    """Record a viewport whose radar scan stalled and timed out.

    Args:
        viewport_left: Failed viewport left X coordinate.
        viewport_top: Failed viewport top Y coordinate.
        timestamp_ms: When the failure was detected.
    """
    key = viewport_scan_key(viewport_left, viewport_top)
    _failed_scan_viewports[key] = timestamp_ms
    log.info(
        "SCAN: marked viewport (%d,%d) as failed target",
        viewport_left,
        viewport_top,
    )


def is_scan_viewport_failed(viewport_left: int, viewport_top: int, now_ms: int) -> bool:
    """Check whether a viewport recently had a stalled radar scan.

    Args:
        viewport_left: Viewport left X coordinate.
        viewport_top: Viewport top Y coordinate.
        now_ms: Current timestamp for TTL evaluation.

    Returns:
        True if radar recently stalled for that viewport.
    """
    key = viewport_scan_key(viewport_left, viewport_top)
    failed_ms = _failed_scan_viewports.get(key)
    if failed_ms is None:
        return False
    return (now_ms - failed_ms) < _FAILED_SCAN_VIEWPORT_TTL_MS


def clear_failed_scan_viewport(viewport_left: int, viewport_top: int) -> None:
    """Clear a failed-scan mark for a specific viewport origin.

    Args:
        viewport_left: Viewport left X coordinate.
        viewport_top: Viewport top Y coordinate.
    """
    key = viewport_scan_key(viewport_left, viewport_top)
    _failed_scan_viewports.pop(key, None)


def is_move_target_failed(x: int, y: int, now_ms: int) -> bool:
    """Check if a move target was recently marked as failed.

    Args:
        x: Destination X coordinate.
        y: Destination Y coordinate.
        now_ms: Current timestamp for TTL check.

    Returns:
        True if the target failed recently and should be avoided.
    """
    key = f"{x},{y}"
    failed_ms = _failed_move_targets.get(key)
    if failed_ms is None:
        return False
    return (now_ms - failed_ms) < _FAILED_MOVE_TTL_MS


def clear_failed_move_targets() -> None:
    """Clear all failed move targets. Called on fresh radar data."""
    _failed_move_targets.clear()


def _viewport_bounds(world: WorldStateDict) -> tuple[int, int, int, int]:
    """Return inclusive visible viewport bounds.

    Args:
        world: Current world state.

    Returns:
        Inclusive ``(left, top, right, bottom)`` viewport bounds.
    """
    return viewport_visible_bounds(world["viewport"])


def _radar_bounds(world: WorldStateDict) -> tuple[int, int, int, int]:
    """Return inclusive current radar coverage bounds.

    Args:
        world: Current world state.

    Returns:
        Inclusive ``(left, top, right, bottom)`` radar bounds.
    """
    return viewport_radar_bounds(world["viewport"])


__all__ = [
    "check_and_clear_radar_scan_complete",
    "clear_failed_move_targets",
    "clear_failed_scan_viewport",
    "get_terrain_map",
    "get_world_state",
    "is_move_target_failed",
    "is_scan_viewport_failed",
    "mark_move_target_failed",
    "mark_radar_scan_complete",
    "mark_scan_viewport_failed",
    "register_room_image",
    "reset_world_state",
    "set_selected_room",
    "update_world_state_from_position",
]
