"""World state tracking from radar, movement, and inventory messages.

This module maintains the current world state (containers, mines, player position,
inventory) and renders ASCII visualizations of the game world.

Inventory is tracked from binary protocol messages (0x49, 0x67, 0x74) instead
of DOM scraping, providing reliable absolute counts without false transitions.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks, protocol
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.container import RadarContainerDict, RadarMineDict
from tankpit_bot.inventory import (
    InventoryChange,
    InventoryItem,
    InventoryState,
    ItemType,
    diff_inventory,
)
from tankpit_bot.runtime_logging import emit_world
from tankpit_bot.sniffer.viewport import get_viewport_left, update_viewport_origin
from tankpit_bot.state import (
    ContainerStateDict,
    MineStateDict,
    WorldStateDict,
    add_mine,
    add_mine_from_radar,
    coord_key,
    make_container_state,
    make_empty_world_state,
    make_terrain_tile,
    mark_viewport_scanned,
    pickup_container,
    remove_mine,
    remove_tank,
    render_world_ascii,
    set_self_fuel,
    update_container_from_radar,
    update_self_position,
    update_tank_damage,
    update_tank_from_registry,
    viewport_scan_key,
)
from tankpit_bot.state.viewport_geometry import (
    VIEWPORT_PATCH_WIDTH,
    make_visible_viewport_state,
    viewport_patch_world_coords,
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


def mark_combat_hit(weapon_byte: int) -> None:
    """Called when we receive a CombatHit where we are the attacker.

    Records that the server processed our shot. If weapon_byte > 0,
    special ammo was consumed (hit confirmed) and the corresponding
    inventory count is decremented. If weapon_byte == 0, the server
    used single shot — either because dual is depleted or the target
    position was empty.

    Args:
        weapon_byte: Last byte of combat_data (0=single, 1=dual,
            2=missile, 3=homing).
    """
    global _got_confirmed_hit, _got_our_shot_response
    _got_our_shot_response = True
    if weapon_byte > 0:
        _got_confirmed_hit = True
        _decrement_ammo_for_weapon(weapon_byte)


def check_and_clear_combat_hit() -> bool:
    """Check if our shot hit (special ammo was used), then clear.

    Returns:
        True if shot connected (weapon_byte > 0), False if miss.
    """
    global _got_confirmed_hit
    result = _got_confirmed_hit
    _got_confirmed_hit = False
    return result


def peek_combat_hit() -> bool:
    """Return whether a confirmed outgoing hit is currently buffered.

    Returns:
        True if an outgoing hit has been observed and not yet consumed.
    """
    return _got_confirmed_hit


def peek_our_shot_response() -> bool:
    """Return whether any CombatHit response for our shot is buffered.

    This is True for both weapon_byte > 0 (hit) and weapon_byte == 0
    (single shot). Use this to distinguish "server responded with
    single shot" from "server hasn't responded yet".

    Returns:
        True if any shot response has been observed and not yet consumed.
    """
    return _got_our_shot_response


def check_and_clear_our_shot_response() -> bool:
    """Check if any CombatHit for our shot arrived, then clear.

    Returns:
        True if the server sent a CombatHit response for our shot
        (any weapon_byte, including 0).
    """
    global _got_our_shot_response
    result = _got_our_shot_response
    _got_our_shot_response = False
    return result


def _decrement_ammo_for_weapon(weapon_byte: int) -> None:
    """Decrement inventory count for the ammo type consumed by a hit.

    When the server confirms a hit with special ammo (weapon_byte > 0),
    one unit of that ammo type was consumed. This keeps the local
    inventory in sync with the server between full InventorySync (0x49)
    messages.

    Args:
        weapon_byte: Weapon type from CombatHit (1=dual, 2=missile,
            3=homing).
    """
    global _inventory_state
    item_key = _WEAPON_BYTE_TO_ITEM.get(weapon_byte)
    if item_key is None:
        return
    current = _inventory_state[item_key]
    if current["count"] <= 0:
        return
    new_count = current["count"] - 1
    updated_item = InventoryItem(count=new_count, enabled=current["enabled"])
    old = _inventory_state
    _inventory_state = InventoryState(
        armor_shields=updated_item if item_key == "armor_shields" else old["armor_shields"],
        dual_shots=updated_item if item_key == "dual_shots" else old["dual_shots"],
        missile_shots=updated_item if item_key == "missile_shots" else old["missile_shots"],
        homing_shots=updated_item if item_key == "homing_shots" else old["homing_shots"],
        extra_radars=updated_item if item_key == "extra_radars" else old["extra_radars"],
    )
    log.info("AMMO: %s consumed by hit (%d -> %d)", item_key, current["count"], new_count)


def mark_tank_killed(tank_id: int) -> None:
    """Record a tank as killed via Deactivation protocol message.

    Args:
        tank_id: The killed tank's ID.
    """
    _killed_tank_ids.add(tank_id)


def drain_killed_tank_ids() -> set[int]:
    """Get and clear all killed tank IDs since last drain.

    Returns:
        Set of tank IDs that were killed.
    """
    global _killed_tank_ids
    result = _killed_tank_ids
    _killed_tank_ids = set()
    return result


def mark_teleport_landed() -> None:
    """Record that the server confirmed a teleport landing."""
    global _teleport_landed
    _teleport_landed = True


def check_and_clear_teleport_landed() -> bool:
    """Check if a teleport landed since last check, then clear.

    Returns:
        True if teleport landed confirmation was received.
    """
    global _teleport_landed
    result = _teleport_landed
    _teleport_landed = False
    return result


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


def get_inventory_state() -> InventoryState:
    """Get the current inventory state from binary protocol tracking.

    Returns:
        Current InventoryState with counts and enabled flags.
    """
    return _inventory_state


def update_inventory_from_protocol(
    counts: list[int],
    enabled: list[bool],
) -> list[InventoryChange]:
    """Set absolute inventory state from a 0x49 (Inventory) message.

    The 0x49 message with alternate=True provides absolute counts on room join.
    Computes diff against previous state and logs changes.

    Args:
        counts: List of 5 item counts [armor, dual, missile, homing, radar].
        enabled: List of 5 enabled flags matching the same order.

    Returns:
        List of inventory changes detected.
    """
    global _inventory_state
    old = _inventory_state
    _inventory_state = InventoryState(
        armor_shields=InventoryItem(count=counts[0], enabled=enabled[0]),
        dual_shots=InventoryItem(count=counts[1], enabled=enabled[1]),
        missile_shots=InventoryItem(count=counts[2], enabled=enabled[2]),
        homing_shots=InventoryItem(count=counts[3], enabled=enabled[3]),
        extra_radars=InventoryItem(count=counts[4], enabled=enabled[4]),
    )
    changes = diff_inventory(old, _inventory_state)
    _log_inventory_changes(changes)
    return changes


def update_inventory_from_gain(gained: list[int]) -> list[InventoryChange]:
    """Apply equipment gain deltas from a 0x67 (EquipmentGain) message.

    Adds the gained amounts to the current counts.

    Args:
        gained: List of 5 gain amounts [armor, dual, missile, homing, radar].

    Returns:
        List of inventory changes detected.
    """
    global _inventory_state
    old = _inventory_state
    _inventory_state = InventoryState(
        armor_shields=InventoryItem(
            count=old["armor_shields"]["count"] + gained[0],
            enabled=old["armor_shields"]["enabled"],
        ),
        dual_shots=InventoryItem(
            count=old["dual_shots"]["count"] + gained[1],
            enabled=old["dual_shots"]["enabled"],
        ),
        missile_shots=InventoryItem(
            count=old["missile_shots"]["count"] + gained[2],
            enabled=old["missile_shots"]["enabled"],
        ),
        homing_shots=InventoryItem(
            count=old["homing_shots"]["count"] + gained[3],
            enabled=old["homing_shots"]["enabled"],
        ),
        extra_radars=InventoryItem(
            count=old["extra_radars"]["count"] + gained[4],
            enabled=old["extra_radars"]["enabled"],
        ),
    )
    changes = diff_inventory(old, _inventory_state)
    _log_inventory_changes(changes)
    return changes


def update_inventory_from_toggle(enabled: list[bool]) -> list[InventoryChange]:
    """Update enabled flags from a 0x74 (EquipmentToggle) message.

    Preserves counts, only updates the enabled/disabled state.

    Args:
        enabled: List of 5 enabled flags [armor, dual, missile, homing, radar].

    Returns:
        List of inventory changes detected.
    """
    global _inventory_state
    old = _inventory_state
    _inventory_state = InventoryState(
        armor_shields=InventoryItem(
            count=old["armor_shields"]["count"],
            enabled=enabled[0],
        ),
        dual_shots=InventoryItem(
            count=old["dual_shots"]["count"],
            enabled=enabled[1],
        ),
        missile_shots=InventoryItem(
            count=old["missile_shots"]["count"],
            enabled=enabled[2],
        ),
        homing_shots=InventoryItem(
            count=old["homing_shots"]["count"],
            enabled=enabled[3],
        ),
        extra_radars=InventoryItem(
            count=old["extra_radars"]["count"],
            enabled=enabled[4],
        ),
    )
    changes = diff_inventory(old, _inventory_state)
    _log_inventory_changes(changes)
    return changes


def _log_inventory_changes(changes: list[InventoryChange]) -> None:
    """Log inventory changes with human-readable messages.

    Args:
        changes: List of inventory changes to log.
    """
    for change in changes:
        item_display = change["item"].replace("_", " ")
        if change["delta"] != 0:
            if change["delta"] > 0:
                log.info(
                    "[INV:GAINED] %s: +%d (%d->%d)",
                    item_display,
                    change["delta"],
                    change["old_count"],
                    change["new_count"],
                )
            else:
                log.info(
                    "[INV:USED] %s: %d (%d->%d)",
                    item_display,
                    change["delta"],
                    change["old_count"],
                    change["new_count"],
                )
        if change["enabled_changed"]:
            state_str = "enabled" if change["now_enabled"] else "disabled"
            log.info("[INV:TOGGLE] %s: %s", item_display, state_str)


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


def update_world_state_from_radar(
    containers: list[RadarContainerDict],
    mines: list[RadarMineDict],
) -> None:
    """Update world state with radar scan results.

    Args:
        containers: List of containers from radar.
        mines: List of mines from radar.
    """
    global _world_state, _pending_radar_cache_refresh_ms, _pending_radar_empty_delta_ms
    ts = get_current_time_ms()
    _pending_radar_cache_refresh_ms = 0
    _pending_radar_empty_delta_ms = 0
    mark_radar_scan_complete()
    clear_failed_move_targets()
    _reconcile_radar_viewport_resources(containers, mines)
    viewport = _world_state["viewport"]
    clear_failed_scan_viewport(viewport["left"], viewport["top"])
    _world_state = mark_viewport_scanned(_world_state, viewport["left"], viewport["top"], ts)

    # Add containers
    for c in containers:
        _world_state = update_container_from_radar(_world_state, c["x"], c["y"], c["volume"], ts)

    # Add mines
    for m in mines:
        _world_state = add_mine_from_radar(_world_state, m["x"], m["y"], m["team"], ts)


def _containers_from_current_radar_cache() -> list[RadarContainerDict]:
    """Synthesize authoritative radar containers from current terrain cache."""
    left, top, right, bottom = _radar_bounds(_world_state)
    containers: list[RadarContainerDict] = []
    for tile in _world_state["terrain"].values():
        x = tile["x"]
        y = tile["y"]
        if not (left <= x <= right and top <= y <= bottom):
            continue
        cache_value = tile["cache_value"]
        if cache_value == -1:
            containers.append(RadarContainerDict(x=x, y=y, volume=-1))
        elif cache_value > 0:
            containers.append(RadarContainerDict(x=x, y=y, volume=cache_value))
    return containers


def update_world_state_from_radar_cache() -> None:
    """Promote a differential radar cache refresh to authoritative containers.

    Some radar scans refresh the current viewport through a combined tile/cache
    update plus ``RadarAck`` instead of an explicit ``radar_response`` packet.
    In that case, the terrain cache already contains the latest fuel/equipment
    truth for the radar envelope; use it to rebuild authoritative containers for
    the current viewport.
    """
    global _world_state
    ts = get_current_time_ms()
    containers = _containers_from_current_radar_cache()
    mark_radar_scan_complete()
    clear_failed_move_targets()
    _reconcile_radar_viewport_resources(containers, None)
    viewport = _world_state["viewport"]
    clear_failed_scan_viewport(viewport["left"], viewport["top"])
    _world_state = mark_viewport_scanned(_world_state, viewport["left"], viewport["top"], ts)
    for container in containers:
        _world_state = update_container_from_radar(
            _world_state,
            container["x"],
            container["y"],
            container["volume"],
            ts,
        )
    emit_world(
        "Radar cache refresh: promoted %d containers from combined tile update",
        len(containers),
    )


def update_world_state_from_radar_known_resources() -> None:
    """Confirm a zero-delta differential radar without discarding known resources.

    Some radar scans only confirm that already-known resources still exist in
    the current viewport. Those scans arrive as an empty tunneled 0x4F result
    followed by RadarAck(found=True). Preserve existing authoritative
    containers inside the radar bounds instead of treating the scan as empty.
    Do not re-promote terrain cache here: stale cache entries can linger for
    already-picked containers until a later tile update clears them.
    """
    global _world_state
    ts = get_current_time_ms()
    left, top, right, bottom = _radar_bounds(_world_state)
    containers_by_key: dict[str, RadarContainerDict] = {}

    for container in _world_state["containers"].values():
        x = container["x"]
        y = container["y"]
        if left <= x <= right and top <= y <= bottom:
            containers_by_key[coord_key(x, y)] = RadarContainerDict(
                x=x,
                y=y,
                volume=container["volume"],
            )

    containers = list(containers_by_key.values())
    mark_radar_scan_complete()
    clear_failed_move_targets()
    viewport = _world_state["viewport"]
    clear_failed_scan_viewport(viewport["left"], viewport["top"])
    _world_state = mark_viewport_scanned(_world_state, viewport["left"], viewport["top"], ts)
    for container in containers:
        _world_state = update_container_from_radar(
            _world_state,
            container["x"],
            container["y"],
            container["volume"],
            ts,
        )
    emit_world(
        "Radar differential refresh: preserved %d known containers in viewport",
        len(containers),
    )


def _clear_container_tile_cache(x: int, y: int) -> None:
    """Clear cached resource data for a tile without changing terrain."""
    global _world_state
    key = coord_key(x, y)
    existing = _world_state["terrain"].get(key)
    if existing is None:
        return
    new_terrain = dict(_world_state["terrain"])
    new_terrain[key] = make_terrain_tile(
        x=x,
        y=y,
        terrain_type=existing["terrain_type"],
        cache_value=0,
        overlay_value=existing["overlay_value"],
    )
    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=_world_state["containers"],
        mines=_world_state["mines"],
        terrain=new_terrain,
        viewport=_world_state["viewport"],
        scanned_viewports=_world_state["scanned_viewports"],
        timestamp_ms=_world_state["timestamp_ms"],
    )


def update_world_state_from_tank_registry_container(
    container_y: int,
    container_viewport_x: int,
) -> None:
    """Ignore non-radar container registry hints for planning state.

    Tank registry container entries expose coarse location hints but do not
    provide trustworthy resource truth. Container planning is radar-driven, so
    these messages must not populate ``world["containers"]``.

    Args:
        container_y: Absolute Y coordinate.
        container_viewport_x: Viewport-relative X coordinate.
    """
    viewport_left = get_viewport_left()
    if viewport_left is None:
        log.info(
            "Ignoring tank_registry container: viewport_left not yet known (y=%d, vx=%d)",
            container_y,
            container_viewport_x,
        )
        return
    container_x = viewport_left + container_viewport_x
    log.debug(
        "Ignoring tank_registry container hint at (%d, %d); radar is authoritative",
        container_x,
        container_y,
    )


def update_world_state_from_tank_entry(tank_id: int, x: int, y: int, name: str) -> None:
    """Add or update tank from TankEntry (0x28) — has position but no team."""
    global _world_state
    ts = get_current_time_ms()
    key = str(tank_id)
    existing = _world_state["tanks"].get(key)
    team = existing["team"] if existing else 0
    rank = existing["rank"] if existing else 0
    _world_state = update_tank_from_registry(
        _world_state,
        tank_id,
        team,
        name,
        rank,
        False,
        x,
        y,
        ts,
    )


def update_world_state_from_tank_info(tank_id: int, team: int, name: str) -> None:
    """Store/update tank from TankInfo (0x21)."""
    global _world_state
    ts = get_current_time_ms()
    key = str(tank_id)
    existing = _world_state["tanks"].get(key)
    _world_state = update_tank_from_registry(
        _world_state,
        tank_id,
        team,
        name,
        existing["rank"] if existing else 0,
        existing["is_bot"] if existing else False,
        existing["x"] if existing else 0,
        existing["y"] if existing else 0,
        ts,
    )


def update_world_state_from_tank_status(
    tank_id: int,
    team: int,
    rank: int,
    name: str,
) -> None:
    """Store/update tank from TankStatus (0x3E)."""
    global _world_state
    ts = get_current_time_ms()
    key = str(tank_id)
    existing = _world_state["tanks"].get(key)
    _world_state = update_tank_from_registry(
        _world_state,
        tank_id,
        team,
        name,
        rank,
        existing["is_bot"] if existing else False,
        existing["x"] if existing else 0,
        existing["y"] if existing else 0,
        ts,
    )


def update_world_state_from_tank_registry(
    tank_id: int,
    name: str,
    team_str: str,
    rank: int,
    is_bot: bool,
    tank_y: int,
    tank_viewport_x: int,
) -> None:
    """Store tank with position from tank_registry message.

    Computes absolute X from viewport_left + viewport_x.

    Args:
        tank_id: Tank ID.
        name: Tank name.
        team_str: Team name string ("red", "purple", "blue", "orange").
        rank: Military rank (0-7).
        is_bot: Whether tank is a bot.
        tank_y: Absolute Y coordinate.
        tank_viewport_x: Viewport-relative X coordinate.
    """
    global _world_state
    from tankpit_bot.protocol.constants import TEAM_NAMES

    # Convert team string to int
    team = TEAM_NAMES.index(team_str) if team_str in TEAM_NAMES else 0

    viewport_left = get_viewport_left()
    if viewport_left is None:
        log.info(
            "Cannot add tank_registry tank: viewport_left not yet known (tank=%d, y=%d, vx=%d)",
            tank_id,
            tank_y,
            tank_viewport_x,
        )
        return
    tank_x = viewport_left + tank_viewport_x

    ts = get_current_time_ms()
    _world_state = update_tank_from_registry(
        _world_state,
        tank_id,
        team,
        name,
        rank,
        is_bot,
        tank_x,
        tank_y,
        ts,
    )


def update_world_state_from_move_response_full(
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
) -> None:
    """Update self_state and tank position from MovementResponse (0x3D).

    The first 0x3D received establishes the bot's identity (tank_id, team, rank).
    All 0x3D messages update the corresponding tank's position.

    Args:
        tank_id: Tank ID.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Team ID (0-3).
        rank: Military rank.
    """
    global _world_state
    ts = get_current_time_ms()

    # Update self_state with real identity data
    self_state = _world_state["self_state"]
    if self_state is None or self_state["tank_id"] == 0:
        from tankpit_bot.state.types import SelfStateDict

        _world_state = WorldStateDict(
            self_state=SelfStateDict(
                tank_id=tank_id,
                x=x,
                y=y,
                team=team,
                rank=rank,
                fuel=self_state["fuel"] if self_state else 0,
                leaderboard_position=0,
            ),
            tanks=_world_state["tanks"],
            containers=_world_state["containers"],
            mines=_world_state["mines"],
            terrain=_world_state["terrain"],
            viewport=_world_state["viewport"],
            scanned_viewports=_world_state["scanned_viewports"],
            timestamp_ms=ts,
        )
    elif self_state["tank_id"] == tank_id:
        # Update self position
        update_world_state_from_position(x, y)

    # Update the tank in the tank list
    key = str(tank_id)
    existing = _world_state["tanks"].get(key)
    name = existing["name"] if existing else ""
    is_bot = existing["is_bot"] if existing else False
    _world_state = update_tank_from_registry(
        _world_state,
        tank_id,
        team,
        name,
        rank,
        is_bot,
        x,
        y,
        ts,
    )


def update_world_state_from_tank_damage(tank_id: int, damage_state: int) -> None:
    """Update tank damage from TankStatusSync (0x2E)."""
    global _world_state
    ts = get_current_time_ms()
    _world_state = update_tank_damage(_world_state, tank_id, damage_state, ts)


def update_world_state_from_tank_exit(tank_id: int) -> None:
    """Remove tank from world state on TankExit (0x58)."""
    global _world_state
    _world_state = remove_tank(_world_state, tank_id, get_current_time_ms())


def _update_tank_position(tank_id: int, x: int, y: int) -> None:
    """Update any tank's position from a position-carrying message.

    Creates the tank if it doesn't exist yet, preserving existing metadata.

    Args:
        tank_id: Tank ID.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
    """
    global _world_state
    ts = get_current_time_ms()
    key = str(tank_id)
    existing = _world_state["tanks"].get(key)
    _world_state = update_tank_from_registry(
        _world_state,
        tank_id,
        existing["team"] if existing else 0,
        existing["name"] if existing else "",
        existing["rank"] if existing else 0,
        existing["is_bot"] if existing else False,
        x,
        y,
        ts,
    )


def _update_enemy_from_detection(tank_id: int, x: int, y: int, team: int, rank: int) -> None:
    """Update enemy tank position from EnemyDetection (0x48) response.

    Sent by server in response to CMD_NEAREST_ENEMY ('e' key).
    Contains absolute x,y for the nearest enemy.

    Args:
        tank_id: Enemy tank ID.
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        team: Enemy team ID.
        rank: Enemy military rank.
    """
    global _world_state
    ts = get_current_time_ms()
    key = str(tank_id)
    existing = _world_state["tanks"].get(key)
    name = existing["name"] if existing else ""
    is_bot = existing["is_bot"] if existing else False
    _world_state = update_tank_from_registry(
        _world_state,
        tank_id,
        team,
        name,
        rank,
        is_bot,
        x,
        y,
        ts,
    )
    log.info(
        "ENEMY_DETECT: tank=%d at (%d,%d) team=%d rank=%d name=%s",
        tank_id,
        x,
        y,
        team,
        rank,
        name,
    )


def update_world_state_from_fuel_total(fuel_total: int) -> None:
    """Update world state with new absolute fuel level.

    The server sends the new fuel total (not a delta) in FuelGain/FuelDeposit.

    Args:
        fuel_total: New absolute fuel level.
    """
    global _world_state
    ts = get_current_time_ms()
    old_fuel = _world_state["self_state"]["fuel"] if _world_state["self_state"] is not None else 0
    _world_state = set_self_fuel(_world_state, fuel_total, ts)
    delta = fuel_total - old_fuel
    emit_world("Fuel: %d -> %d (%+d)", old_fuel, fuel_total, delta)


def update_world_state_from_container_pickup(x: int, y: int) -> None:
    """Update world state when container is picked up.

    Removes container and adds its fuel to self_state.

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    global _world_state
    ts = get_current_time_ms()
    _world_state = pickup_container(_world_state, x, y, ts)
    _clear_container_tile_cache(x, y)
    emit_world("Picked up container at (%d, %d)", x, y)


def remove_container_at(x: int, y: int) -> None:
    """Remove a container from world state at the given position.

    Used when the bot detects a container is unreachable (stuck timeout).

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    global _world_state
    key = f"{x},{y}"
    if key in _world_state["containers"]:
        new_containers = dict(_world_state["containers"])
        del new_containers[key]
        _world_state = WorldStateDict(
            self_state=_world_state["self_state"],
            tanks=_world_state["tanks"],
            containers=new_containers,
            mines=_world_state["mines"],
            terrain=_world_state["terrain"],
            viewport=_world_state["viewport"],
            scanned_viewports=_world_state["scanned_viewports"],
            timestamp_ms=_world_state["timestamp_ms"],
        )
        _clear_container_tile_cache(x, y)
        log.info("Removed unreachable container at (%d, %d)", x, y)


def increment_container_failed_pickups(x: int, y: int) -> None:
    """Increment the failed_pickups counter on a container.

    Called when a pickup attempt stalls. The container stays in world
    state but is deprioritized by the planner. If the container is
    later re-confirmed by a fresh radar/viewport, failed_pickups
    resets to 0.

    Args:
        x: Container X coordinate.
        y: Container Y coordinate.
    """
    global _world_state
    key = f"{x},{y}"
    container = _world_state["containers"].get(key)
    if container is None:
        return
    new_container = make_container_state(
        x=container["x"],
        y=container["y"],
        is_fuel=container["is_fuel"],
        volume=container["volume"],
        timestamp_ms=container["timestamp_ms"],
        failed_pickups=container["failed_pickups"] + 1,
    )
    new_containers = dict(_world_state["containers"])
    new_containers[key] = new_container
    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=new_containers,
        mines=_world_state["mines"],
        terrain=_world_state["terrain"],
        viewport=_world_state["viewport"],
        scanned_viewports=_world_state["scanned_viewports"],
        timestamp_ms=_world_state["timestamp_ms"],
    )
    log.info(
        "Container (%d,%d) failed_pickups: %d -> %d",
        x,
        y,
        container["failed_pickups"],
        new_container["failed_pickups"],
    )


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


def _reconcile_radar_viewport_resources(
    containers: list[RadarContainerDict],
    mines: list[RadarMineDict] | None,
) -> None:
    """Make current viewport resources match an authoritative radar scan.

    Radar covers the full visible viewport. Any tracked container or mine
    inside the current viewport that is absent from the radar response is
    stale and must be removed before adding the freshly scanned results.

    Args:
        containers: Containers returned by radar.
        mines: Mines returned by radar. ``None`` skips mine reconciliation for
            differential radar cache refreshes that only encode containers.
    """
    global _world_state
    left, top, right, bottom = _radar_bounds(_world_state)
    radar_container_keys = {coord_key(item["x"], item["y"]) for item in containers}
    radar_mine_keys = (
        {coord_key(item["x"], item["y"]) for item in mines}
        if mines is not None
        else None
    )

    new_containers: dict[str, ContainerStateDict] | None = None
    for key, container in _world_state["containers"].items():
        x = container["x"]
        y = container["y"]
        if left <= x <= right and top <= y <= bottom and key not in radar_container_keys:
            if new_containers is None:
                new_containers = dict(_world_state["containers"])
            del new_containers[key]

    new_mines: dict[str, MineStateDict] | None = None
    if radar_mine_keys is not None:
        for key, mine in _world_state["mines"].items():
            x = mine["x"]
            y = mine["y"]
            if left <= x <= right and top <= y <= bottom and key not in radar_mine_keys:
                if new_mines is None:
                    new_mines = dict(_world_state["mines"])
                del new_mines[key]

    if new_containers is None and new_mines is None:
        return

    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=_world_state["containers"] if new_containers is None else new_containers,
        mines=_world_state["mines"] if new_mines is None else new_mines,
        terrain=_world_state["terrain"],
        viewport=_world_state["viewport"],
        scanned_viewports=_world_state["scanned_viewports"],
        timestamp_ms=get_current_time_ms(),
    )


def render_world_state_ascii() -> str | None:
    """Render current world state as ASCII.

    Returns:
        ASCII representation, or None if terrain map not loaded.
    """
    terrain = _load_terrain_map_if_needed()
    if terrain is None:
        return None
    return render_world_ascii(_world_state, terrain)


def _update_viewport_entities(
    viewport_left: int,
    viewport_top: int,
    entities: list[dict[str, int]],
) -> None:
    """Apply a visible viewport update using explicit viewport origin from 0x5A.

    Args:
        viewport_left: Absolute left edge of the visible 16x16 viewport.
        viewport_top: Absolute top edge of the visible 16x16 viewport.
        entities: Viewport entity dicts with col, row, entity_id, value, terrain_type.
    """
    global _world_state

    update_viewport_origin(viewport_left, viewport_top)

    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=_world_state["containers"],
        mines=_world_state["mines"],
        terrain=_world_state["terrain"],
        viewport=make_visible_viewport_state(viewport_left, viewport_top),
        scanned_viewports=_world_state["scanned_viewports"],
        timestamp_ms=_world_state["timestamp_ms"],
    )

    _update_viewport_tiles(entities, viewport_left, viewport_top)
    clear_failed_scan_viewport(viewport_left, viewport_top)


def _update_viewport_tiles(
    entities: list[dict[str, int]],
    vp_left: int,
    vp_top: int,
) -> None:
    """Apply ``0x5A`` tile patches to viewport terrain and visual cache only.

    Client JS applies ``0x5A`` rows into per-tile cache, overlay, and terrain
    fields. For bot planning, however, resource truth is radar-driven: these
    passive viewport patches must not add/remove actionable fuel or equipment
    targets in ``world["containers"]``.

    Args:
        entities: Viewport entity list.
        vp_left: Viewport left offset.
        vp_top: Viewport top offset.
    """
    global _world_state
    ts = get_current_time_ms()
    new_terrain = dict(_world_state["terrain"])

    for ent in entities:
        abs_x, abs_y = viewport_patch_world_coords(
            vp_left,
            vp_top,
            ent["col"],
            ent["row"],
        )
        cache_value = ent["cache_value"]
        overlay_value = ent["overlay_value"]
        terrain_type = ent["terrain_type"]
        key = coord_key(abs_x, abs_y)
        new_terrain[key] = make_terrain_tile(
            x=abs_x,
            y=abs_y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=_world_state["containers"],
        mines=_world_state["mines"],
        terrain=new_terrain,
        viewport=_world_state["viewport"],
        scanned_viewports=_world_state["scanned_viewports"],
        timestamp_ms=ts,
    )


def _update_cache_tiles(updates: list[tuple[int, int, int]]) -> None:
    """Apply absolute cache-only tile updates to terrain/visual cache only.

    Args:
        updates: Absolute `(x, y, cache_value)` triples.
    """
    global _world_state

    new_terrain = dict(_world_state["terrain"])
    timestamp_ms = get_current_time_ms()
    for x, y, cache_value in updates:
        key = coord_key(x, y)
        existing = new_terrain.get(key)
        terrain_type = existing["terrain_type"] if existing is not None else 0
        overlay_value = existing["overlay_value"] if existing is not None else 255
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=_world_state["containers"],
        mines=_world_state["mines"],
        terrain=new_terrain,
        viewport=_world_state["viewport"],
        scanned_viewports=_world_state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def _update_overlay_tiles(updates: list[tuple[int, int, int]]) -> None:
    """Apply absolute overlay-only tile updates to world state.

    Args:
        updates: Absolute `(x, y, overlay_value)` triples.
    """
    global _world_state

    new_terrain = dict(_world_state["terrain"])
    timestamp_ms = get_current_time_ms()
    for x, y, overlay_value in updates:
        key = coord_key(x, y)
        existing = new_terrain.get(key)
        terrain_type = existing["terrain_type"] if existing is not None else 0
        cache_value = existing["cache_value"] if existing is not None else 0
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=_world_state["containers"],
        mines=_world_state["mines"],
        terrain=new_terrain,
        viewport=_world_state["viewport"],
        scanned_viewports=_world_state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def _update_terrain_tiles(updates: list[tuple[int, int, int]]) -> None:
    """Apply absolute terrain/structure tile updates to world state.

    Args:
        updates: Absolute `(x, y, terrain_type)` triples from protocol 0x4A.
    """
    global _world_state

    new_terrain = dict(_world_state["terrain"])
    timestamp_ms = get_current_time_ms()

    for x, y, terrain_type in updates:
        key = coord_key(x, y)
        existing = new_terrain.get(key)
        cache_value = existing["cache_value"] if existing is not None else 0
        overlay_value = existing["overlay_value"] if existing is not None else 255
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            cache_value=cache_value,
            overlay_value=overlay_value,
        )

    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=_world_state["containers"],
        mines=_world_state["mines"],
        terrain=new_terrain,
        viewport=_world_state["viewport"],
        scanned_viewports=_world_state["scanned_viewports"],
        timestamp_ms=timestamp_ms,
    )


def _apply_waypoints(start_x: int, start_y: int, waypoints: str) -> tuple[int, int]:
    """Apply waypoints to calculate final position.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        waypoints: Path string (e.g., "wsss" = west, south, south, south).

    Returns:
        Tuple of (final_x, final_y) after applying all waypoints.
    """
    x, y = start_x, start_y
    for wp in waypoints:
        if wp == "n":
            y -= 1
        elif wp == "s":
            y += 1
        elif wp == "e":
            x += 1
        elif wp == "w":
            x -= 1
    return x, y


def _is_absolute_position(x: int, y: int) -> bool:
    """Check if position_update coordinates are absolute world coordinates.

    Position updates contain either:
    - Absolute world coordinates (after join/teleport): x,y >= 18
    - Viewport-relative coordinates (during movement): x,y < 18

    Position updates use the same 18x18 patch envelope as ``0x5A`` viewport
    rows, so coordinates within that range are viewport-relative and do not
    represent absolute world position.

    Args:
        x: X coordinate from position_update.
        y: Y coordinate from position_update.

    Returns:
        True if coordinates are absolute world coordinates.
    """
    return x >= VIEWPORT_PATCH_WIDTH or y >= VIEWPORT_PATCH_WIDTH


def _render_ascii_if_available(event: str) -> None:
    """Render ASCII viewport if terrain map is available.

    Args:
        event: Event name for logging (e.g., "Enter", "Teleport", "Move").
    """
    ascii_view = render_world_state_ascii()
    if ascii_view is not None:
        emit_world("[WorldState %s]\n%s", event, ascii_view)


def _dispatch_resource_update(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch resource and inventory messages.

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {"msg_type": 0x2E, "fuel": int(fuel)} if fuel is not None:
            update_world_state_from_fuel_total(fuel)
            return True
        case {"msg_type": 0x44, "fuel_total": int(fuel_total)}:
            update_world_state_from_fuel_total(fuel_total)
            return True
        case {"msg_type": 0x64, "fuel_total": int(fuel_total)}:
            update_world_state_from_fuel_total(fuel_total)
            return True
        case {"msg_type": 0x49, "counts": list(counts), "enabled": list(enabled)}:
            update_inventory_from_protocol(counts, enabled)
            return True
        case {"msg_type": 0x67, "gained": list(gained)}:
            update_inventory_from_gain(gained)
            return True
        case {"msg_type": 0x74, "enabled": list(enabled)}:
            update_inventory_from_toggle(enabled)
            return True
        case {"msg_type": 0x46, "found": bool(found)}:
            if _consume_pending_radar_cache_refresh():
                update_world_state_from_radar_cache()
            elif _consume_pending_radar_empty_delta():
                if found:
                    update_world_state_from_radar_known_resources()
                else:
                    update_world_state_from_radar([], [])
            else:
                mark_radar_scan_complete()
            return True
    return False


def _dispatch_tank_update(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tank-related messages to update world state.

    Handles tank entry (0x28), info (0x21), status (0x3E),
    damage (0x2E), exit (0x58), and enemy detection (0x48) messages.

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {"msg_type": 0x28, "tank_id": int(tid), "x": int(tx), "y": int(ty), "name": str(name)}:
            update_world_state_from_tank_entry(tid, tx, ty, name)
            return True
        case {"msg_type": 0x21, "tank_id": int(tid), "team": int(team), "name": str(name)}:
            update_world_state_from_tank_info(tid, team, name)
            return True
        case {
            "msg_type": 0x3E,
            "tank_id": int(tid),
            "team": int(team),
            "rank": int(rank),
            "name": str(name),
        }:
            update_world_state_from_tank_status(tid, team, rank, name)
            return True
        case {"msg_type": 0x2E, "tank_id": int(tid), "damage_state": int(dmg)}:
            update_world_state_from_tank_damage(tid, dmg)
            return True
        case {"msg_type": 0x58, "tank_id": int(tid)}:
            update_world_state_from_tank_exit(tid)
            return True
        case {
            "msg_type": 0x48,
            "tank_id": int(tid),
            "x": int(x),
            "y": int(y),
            "team": int(team),
            "rank": int(rank),
        }:
            _update_enemy_from_detection(tid, x, y, team, rank)
            return True
        case {
            "msg_type": 0x41,
            "victim_id": int(vid),
        }:
            # Tank killed — invalidate position so we stop targeting
            _update_tank_position(vid, 0, 0)
            mark_tank_killed(vid)
            log.info("DEACTIVATED: tank=%d killed, position invalidated", vid)
            return True
    return False


def _resolve_waypoint_destination(
    start_x: int,
    start_y: int,
    waypoints: list[tuple[int, int]],
) -> tuple[int, int]:
    """Resolve the final destination from protocol waypoint tuples.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        waypoints: Waypoints list from the protocol movement decoder.

    Returns:
        Final destination after applying the waypoint tuple list.
    """
    final_x: int = start_x
    final_y: int = start_y
    if waypoints:
        final_x, final_y = waypoints[0]
    return (final_x, final_y)


def _handle_waypoint_movement(sx: int, sy: int, wps: list[tuple[int, int]]) -> None:
    """Handle 0x47 waypoint movement for non-self tanks.

    Resolves the final destination from waypoints and updates the tank
    that matches the start position.

    Args:
        sx: Start X coordinate.
        sy: Start Y coordinate.
        wps: Waypoints list of (x, y) tuples from protocol decoder.
    """
    final_x, final_y = _resolve_waypoint_destination(sx, sy, wps)
    for tank in _world_state["tanks"].values():
        if tank["x"] == sx and tank["y"] == sy and not tank["is_self"]:
            _update_tank_position(tank["tank_id"], final_x, final_y)
            break


def _dispatch_container_movement(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch container-decoded movement messages (msg_type="movement").

    Container MovementDict has string waypoints and player_id-based
    tank identification, unlike protocol MovementDict (0x47) which uses
    positional tuple waypoints.

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": "movement",
            "start_x": int(sx),
            "start_y": int(sy),
            "waypoints": str(wps),
            "is_self": True,
        }:
            fx, fy = _apply_waypoints(sx, sy, wps)
            update_world_state_from_position(fx, fy)
            _render_ascii_if_available("SelfMovement")
            return True
        case {
            "msg_type": "movement",
            "start_x": int(sx),
            "start_y": int(sy),
            "player_id": int(pid),
            "waypoints": str(wps),
            "is_self": False,
        }:
            from tankpit_bot.sniffer.player_tracking import _player_id_mapper

            resolved_tid = _player_id_mapper.get_tank_id(pid)
            if resolved_tid is not None:
                fx, fy = _apply_waypoints(sx, sy, wps)
                _update_tank_position(resolved_tid, fx, fy)
            return True
    return False


def _dispatch_binary_position_update(
    flags: int,
    tank_id: int,
    x: int,
    y: int,
) -> bool:
    """Dispatch one decoded ``position_update`` message.

    Args:
        flags: Position-update flags.
        tank_id: Tank identifier.
        x: Reported x coordinate.
        y: Reported y coordinate.

    Returns:
        True after handling the position update.
    """
    is_self = (flags & 0x02) != 0
    if is_self and _is_absolute_position(x, y):
        update_world_state_from_position(x, y)
        _render_ascii_if_available("Enter/Teleport")
    elif not is_self and _is_absolute_position(x, y):
        _update_tank_position(tank_id, x, y)
    return True


def _dispatch_protocol_movement_update(
    tank_id: int,
    start_x: int,
    start_y: int,
    waypoints: list[tuple[int, int]],
) -> bool:
    """Dispatch one decoded protocol ``0x47`` movement message.

    Args:
        tank_id: Moving tank id.
        start_x: Absolute movement start x.
        start_y: Absolute movement start y.
        waypoints: Absolute waypoint tuples from the protocol decoder.

    Returns:
        True after handling the movement.
    """
    self_state = _world_state["self_state"]
    is_self = self_state is not None and tank_id == self_state["tank_id"]
    if is_self:
        final_x, final_y = _resolve_waypoint_destination(start_x, start_y, waypoints)
        update_world_state_from_position(final_x, final_y)
        _render_ascii_if_available("SelfMovement")
    else:
        _handle_waypoint_movement(start_x, start_y, waypoints)
    return True


def _dispatch_tile_patch_update(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tile patch updates for cache, overlay, terrain, and viewport.

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {"msg_type": 0x4A, "updates": list(updates)}:
            _update_terrain_tiles(updates)
            return True
        case {"msg_type": 0x40, "updates": list(updates)}:
            _update_overlay_tiles(updates)
            return True
        case {"msg_type": 0x43, "updates": list(updates)}:
            _update_cache_tiles(updates)
            return True
        case {
            "msg_type": 0x4F,
            "cache_updates": list(cache_updates),
            "overlay_updates": list(overlay_updates),
        }:
            _update_cache_tiles(cache_updates)
            _update_overlay_tiles(overlay_updates)
            _mark_pending_radar_cache_refresh()
            return True
        case {
            "msg_type": 0x5A,
            "viewport_left": int(viewport_left),
            "viewport_top": int(viewport_top),
            "entities": list(entities),
        }:
            _update_viewport_entities(viewport_left, viewport_top, entities)
            return True
    return False


def _dispatch_position_update(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch position and movement messages to update world state.

    Handles position_update, protocol movement (0x47), MovementResponse (0x3D),
    container movement, and viewport updates (0x5A).

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": "position_update",
            "flags": int(flags),
            "tank_id": int(tid),
            "x": int(x),
            "y": int(y),
        }:
            return _dispatch_binary_position_update(flags, tid, x, y)
        case {
            "msg_type": 0x47,
            "tank_id": int(tid),
            "start_x": int(sx),
            "start_y": int(sy),
            "waypoints": list(wps),
        }:
            return _dispatch_protocol_movement_update(tid, sx, sy, wps)
        case {
            "msg_type": 0x3D,
            "tank_id": int(tid),
            "x": int(x),
            "y": int(y),
            "team": int(team),
            "rank": int(rank),
        }:
            update_world_state_from_move_response_full(tid, x, y, team, rank)
            _render_ascii_if_available("MovementResponse")
            return True
        case _:
            if _dispatch_tile_patch_update(decoded):
                return True
    return _dispatch_container_movement(decoded)


def _dispatch_tank_event(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch tank lifecycle events (leave, deactivation, damage, update).

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": "tank_update_compact" | "tank_update_extended" | "tank_update_full",
            "flags": int(flags),
            "tank_id": int(tid),
            "status_data": bytes(sd),
        }:
            # Flag 0xCD appears during obstacle pickup/drop interactions. The
            # payload bytes correlate with structure/object coordinates rather
            # than tank position, so treating the first two bytes as x/y
            # pollutes tank state.
            if flags == 0xCD:
                return True
            if len(sd) >= 2:
                _update_tank_position(tid, sd[0], sd[1])
            return True
        case {
            "msg_type": "tank_status_short",
            "tank_id": int(tid),
            "damage_state": int(dmg),
        }:
            update_world_state_from_tank_damage(tid, dmg)
            return True
        case {"msg_type": "tank_leave", "tank_id": int(tid)}:
            update_world_state_from_tank_exit(tid)
            return True
        case {"msg_type": "deactivation_kill", "victim_id": int(vid)}:
            # Log the raw victim_id and check if it matches any known tank
            known_tanks = list(_world_state["tanks"].keys())
            log.info(
                "DEACTIVATION_KILL: victim_id=%d (0x%04X) known_tanks=%s",
                vid,
                vid,
                known_tanks[:10],
            )
            _update_tank_position(vid, 0, 0)
            mark_tank_killed(vid)
            return True
        case {"msg_type": "deactivation_death", "killer_id": int(kid)}:
            log.info("DEACTIVATION_DEATH: killed by tank=%d", kid)
            return True
    return False


def _dispatch_container_message(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch container-level messages (tank_registry, tank_update, etc.).

    Args:
        decoded: Decoded binary protocol message.

    Returns:
        True if the message was handled, False otherwise.
    """
    match decoded:
        case {
            "msg_type": 0x4B,
            "mine_type": int(mine_type),
            "tank_id": int(tank_id),
            "positions": list(positions),
        }:
            return _dispatch_mine_placement(mine_type, tank_id, positions)
        case {"msg_type": 0x45, "positions": list(positions)}:
            return _dispatch_mine_detonation(positions)
        case {
            "msg_type": "tank_registry",
            "is_container": True,
            "container_y": int(cy),
            "container_viewport_x": int(cvx),
        }:
            update_world_state_from_tank_registry_container(cy, cvx)
            log.info("Container from tank_registry: y=%d vx=%d", cy, cvx)
            return True
        case {"msg_type": "container_pickup", "x": int(x), "y": int(y)}:
            update_world_state_from_container_pickup(x, y)
            return True
        case {"msg_type": "teleport_landed"}:
            emit_world("TELEPORT_LANDED: server confirmed teleport")
            mark_teleport_landed()
            return True
        case {
            "msg_type": "combat_hit",
            "attacker_id": int(aid),
            "direction": int(),
            "is_outgoing": bool(),
            "combat_data": bytes(cdata),
        }:
            self_state = _world_state["self_state"]
            # Our shot: check weapon byte for hit/miss
            if self_state is not None and aid == self_state["tank_id"]:
                weapon_byte = cdata[-1] if len(cdata) > 0 else 0
                log.info("OUR_SHOT: weapon_byte=%d data=%s", weapon_byte, cdata.hex())
                mark_combat_hit(weapon_byte)
            return True
        case {
            "msg_type": "tank_registry",
            "is_container": False,
            "tank_id": int(tid),
            "tank_name": str(name),
            "team": str(team_str),
            "military_rank": int(rank),
            "is_bot": bool(is_bot),
            "tank_y": int(ty),
            "tank_viewport_x": int(tvx),
        }:
            update_world_state_from_tank_registry(tid, name, team_str, rank, is_bot, ty, tvx)
            return True
    return _dispatch_tank_event(decoded)


def _dispatch_mine_placement(
    mine_type: int,
    tank_id: int,
    positions: list[tuple[int, int]],
) -> bool:
    """Dispatch tunneled mine placement into world state.

    Args:
        mine_type: Mine type from protocol payload.
        tank_id: ID of the placing tank.
        positions: Absolute mine coordinates.

    Returns:
        True after attempting to apply the placement.
    """
    global _world_state
    self_state = _world_state["self_state"]
    team: int | None = None
    if self_state is not None and self_state["tank_id"] == tank_id:
        team = self_state["team"]
    else:
        tank_state = _world_state["tanks"].get(str(tank_id))
        if tank_state is not None:
            team = tank_state["team"]
    if team is None:
        return True
    timestamp_ms = get_current_time_ms()
    for x, y in positions:
        _world_state = add_mine(
            _world_state,
            x,
            y,
            mine_type,
            tank_id,
            team,
            timestamp_ms,
        )
    return True


def _dispatch_mine_detonation(
    positions: list[tuple[int, int]],
) -> bool:
    """Dispatch tunneled mine detonation into world state.

    Args:
        positions: Absolute mine coordinates removed by the detonation.

    Returns:
        True after applying the removals.
    """
    global _world_state
    timestamp_ms = get_current_time_ms()
    for x, y in positions:
        _world_state = remove_mine(_world_state, x, y, timestamp_ms)
    return True


def _parse_world_state_blob(wd: bytes) -> None:
    """Parse world_state blob from map response to extract all tank positions.

    Format (verified from world_state_dump.bin):
    - [terrain_count:2 LE] — number of terrain delta bytes
    - [terrain_count terrain delta bytes]
    - Repeated 5-byte tank entries until end:
      [x:1][y:1][id_lo:1][id_hi:1][packed:1]
      where tank_id = id_lo + id_hi*256 (LE), team = packed & 3, rank = (packed>>4) & 15
    """
    if len(wd) < 2:
        return

    terrain_count = wd[0] | (wd[1] << 8)
    tank_data_start = 2 + terrain_count

    if tank_data_start > len(wd):
        log.warning("WorldState blob too short: %d bytes, terrain_count=%d", len(wd), terrain_count)
        return

    remaining = wd[tank_data_start:]
    num_tanks = len(remaining) // 5
    if num_tanks == 0:
        return

    updated = 0
    for i in range(num_tanks):
        entry = remaining[i * 5 : i * 5 + 5]
        x = entry[0]
        y = entry[1]
        tank_id = entry[2] | (entry[3] << 8)
        packed = entry[4]
        team = packed & 0x03
        rank = (packed >> 4) & 0x0F

        # Update or create this tank in world state
        _update_map_tank(tank_id, x, y, team, rank)
        updated += 1

    log.info(
        "MAP_POSITIONS: parsed %d tanks from world_state blob (%d bytes, %d terrain)",
        updated,
        len(wd),
        terrain_count,
    )


def _update_map_tank(tank_id: int, x: int, y: int, team: int, rank: int) -> None:
    """Update a tank's position/team/rank from map data.

    Preserves existing name and is_bot fields if the tank is already known.
    """
    global _world_state
    ts = get_current_time_ms()
    key = str(tank_id)
    existing = _world_state["tanks"].get(key)
    _world_state = update_tank_from_registry(
        _world_state,
        tank_id,
        team,
        existing["name"] if existing else "",
        rank,
        existing["is_bot"] if existing else False,
        x,
        y,
        ts,
    )


def dispatch_world_state_update(decoded: protocol.BinaryMessage) -> None:
    """Dispatch decoded message to update world state, inventory, and render ASCII.

    Delegates to specialized dispatchers for resources, tanks, positions,
    and container messages.

    Args:
        decoded: Decoded binary protocol message.
    """
    if _dispatch_resource_update(decoded):
        return
    if _dispatch_tank_update(decoded):
        return
    if _dispatch_position_update(decoded):
        return
    if _dispatch_container_message(decoded):
        return

    match decoded:
        case {"msg_type": "world_state", "world_data": bytes(wd)}:
            _parse_world_state_blob(wd)
            return
        case {"msg_type": 0x4F, "containers": list(containers), "mines": list(mines)}:
            # Tunneled 0x2E -> 0x4F radar scan results can be differential.
            # Non-empty lists are authoritative immediately. Empty lists need
            # the following RadarAck(found=...) to distinguish "no resources"
            # from "no deltas, keep existing viewport resources".
            if not containers and not mines:
                _mark_pending_radar_empty_delta()
            else:
                update_world_state_from_radar(containers, mines)
                _render_ascii_if_available("Radar")
            return
        case {"msg_type": "radar_response", "containers": list(containers), "mines": list(mines)}:
            update_world_state_from_radar(containers, mines)
            _render_ascii_if_available("Radar")


__all__ = [
    "check_and_clear_combat_hit",
    "check_and_clear_our_shot_response",
    "check_and_clear_radar_scan_complete",
    "check_and_clear_teleport_landed",
    "clear_failed_move_targets",
    "dispatch_world_state_update",
    "drain_killed_tank_ids",
    "get_inventory_state",
    "get_terrain_map",
    "get_world_state",
    "increment_container_failed_pickups",
    "is_move_target_failed",
    "is_scan_viewport_failed",
    "mark_combat_hit",
    "mark_move_target_failed",
    "mark_radar_scan_complete",
    "mark_scan_viewport_failed",
    "mark_tank_killed",
    "mark_teleport_landed",
    "peek_combat_hit",
    "peek_our_shot_response",
    "register_room_image",
    "remove_container_at",
    "render_world_state_ascii",
    "reset_world_state",
    "set_selected_room",
    "update_inventory_from_gain",
    "update_inventory_from_protocol",
    "update_inventory_from_toggle",
    "update_world_state_from_container_pickup",
    "update_world_state_from_fuel_total",
    "update_world_state_from_position",
    "update_world_state_from_radar",
    "update_world_state_from_tank_registry_container",
]
