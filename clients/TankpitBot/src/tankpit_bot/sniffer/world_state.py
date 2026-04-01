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
from tankpit_bot.sniffer.viewport import get_viewport_left
from tankpit_bot.state import (
    WorldStateDict,
    add_mine,
    add_mine_from_radar,
    coord_key,
    make_container_state,
    make_empty_world_state,
    make_terrain_tile,
    pickup_container,
    remove_tank,
    render_world_ascii,
    set_self_fuel,
    update_container_from_radar,
    update_self_position,
    update_tank_damage,
    update_tank_from_registry,
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

# Failed move targets — coordinates where a move stalled and timed out.
# Maps "x,y" key to timestamp_ms of the failure. Cleared on radar refresh
# and session reset. The planner rejects these coordinates until they expire
# or are re-confirmed by fresh world data.
_failed_move_targets: dict[str, int] = {}

# TTL for failed move targets (30 seconds). After this, the target is
# eligible again in case the obstacle was transient.
_FAILED_MOVE_TTL_MS = 30000


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
    global _radar_scan_complete
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
    _failed_move_targets.clear()


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
    global _world_state
    ts = get_current_time_ms()
    mark_radar_scan_complete()
    clear_failed_move_targets()

    # Add containers
    for c in containers:
        _world_state = update_container_from_radar(_world_state, c["x"], c["y"], c["volume"], ts)

    # Add mines
    for m in mines:
        _world_state = add_mine_from_radar(_world_state, m["x"], m["y"], m["team"], ts)


def update_world_state_from_tank_registry_container(
    container_y: int,
    container_viewport_x: int,
) -> None:
    """Update world state with container from tank_registry message.

    Tank registry containers have viewport-relative x coordinate.
    Absolute x = viewport_left + container_viewport_x.

    Args:
        container_y: Absolute Y coordinate.
        container_viewport_x: Viewport-relative X coordinate.
    """
    global _world_state
    # Use sniffer's viewport tracking which is updated from position_update messages
    viewport_left = get_viewport_left()
    if viewport_left is None:
        log.info(
            "Cannot add container: viewport_left not yet known (y=%d, vx=%d)",
            container_y,
            container_viewport_x,
        )
        return
    container_x = viewport_left + container_viewport_x
    ts = get_current_time_ms()
    # Volume unknown from tank_registry, use 1 as placeholder (is_fuel=True)
    _world_state = update_container_from_radar(_world_state, container_x, container_y, 1, ts)
    log.debug("Added container from tank_registry: (%d, %d)", container_x, container_y)


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

    # Compute absolute x from viewport
    viewport_left = get_viewport_left()
    tank_x = viewport_left + tank_viewport_x if viewport_left is not None else tank_viewport_x

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
    log.info("Fuel: %d -> %d (%+d)", old_fuel, fuel_total, delta)


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
    log.info("Picked up container at (%d, %d)", x, y)


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
            timestamp_ms=_world_state["timestamp_ms"],
        )
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
    """Apply a viewport update using explicit viewport origin from 0x5A.

    Args:
        viewport_left: Absolute left edge of the observable 18x18 frame.
        viewport_top: Absolute top edge of the observable 18x18 frame.
        entities: Viewport entity dicts with col, row, entity_id, value, terrain_type.
    """
    global _world_state

    from tankpit_bot.state.types import ViewportStateDict

    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=_world_state["containers"],
        mines=_world_state["mines"],
        terrain=_world_state["terrain"],
        viewport=ViewportStateDict(left=viewport_left, top=viewport_top, width=18, height=18),
        timestamp_ms=_world_state["timestamp_ms"],
    )

    _add_containers_from_entities(entities, viewport_left, viewport_top)


def _add_containers_from_entities(
    entities: list[dict[str, int]],
    vp_left: int,
    vp_top: int,
) -> None:
    """Add containers from ``0x5A`` tile patches to world state.

    Client JS applies ``0x5A`` rows into tile cache fields, not tank presence.
    ``entity_id > 0`` marks fuel on the tile, ``entity_id == -1`` marks
    equipment, and ``entity_id == 0`` means no container cache update.

    Args:
        entities: Viewport entity list.
        vp_left: Viewport left offset.
        vp_top: Viewport top offset.
    """
    global _world_state
    ts = get_current_time_ms()

    for ent in entities:
        eid = ent.get("entity_id", -1)
        abs_x = vp_left + ent["col"]
        abs_y = vp_top + ent["row"]

        if eid == -1:
            _world_state = update_container_from_radar(
                _world_state,
                abs_x,
                abs_y,
                -1,
                ts,
            )
        elif eid > 0:
            _world_state = update_container_from_radar(
                _world_state,
                abs_x,
                abs_y,
                eid,
                ts,
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
        entity_id = existing["entity_id"] if existing is not None else 0
        new_terrain[key] = make_terrain_tile(
            x=x,
            y=y,
            terrain_type=terrain_type,
            entity_id=entity_id,
        )

    _world_state = WorldStateDict(
        self_state=_world_state["self_state"],
        tanks=_world_state["tanks"],
        containers=_world_state["containers"],
        mines=_world_state["mines"],
        terrain=new_terrain,
        viewport=_world_state["viewport"],
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

    The viewport is 18x18, so coordinates within that range are viewport-relative
    and don't represent actual world position.

    Args:
        x: X coordinate from position_update.
        y: Y coordinate from position_update.

    Returns:
        True if coordinates are absolute world coordinates.
    """
    viewport_size = 18
    return x >= viewport_size or y >= viewport_size


def _render_ascii_if_available(event: str) -> None:
    """Render ASCII viewport if terrain map is available.

    Args:
        event: Event name for logging (e.g., "Enter", "Teleport", "Move").
    """
    ascii_view = render_world_state_ascii()
    if ascii_view is not None:
        log.info("[WorldState %s]\n%s", event, ascii_view)


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
        case {"msg_type": 0x46}:
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
            is_self = (flags & 0x02) != 0
            if is_self and _is_absolute_position(x, y):
                update_world_state_from_position(x, y)
                _render_ascii_if_available("Enter/Teleport")
            elif not is_self and _is_absolute_position(x, y):
                _update_tank_position(tid, x, y)
            return True
        case {
            "msg_type": 0x47,
            "tank_id": int(tid),
            "start_x": int(sx),
            "start_y": int(sy),
            "waypoints": list(wps),
        }:
            self_state = _world_state["self_state"]
            is_self = self_state is not None and tid == self_state["tank_id"]
            if is_self:
                final_x, final_y = _resolve_waypoint_destination(sx, sy, wps)
                update_world_state_from_position(final_x, final_y)
                _render_ascii_if_available("SelfMovement")
            else:
                _handle_waypoint_movement(sx, sy, wps)
            return True
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
        case {"msg_type": 0x4A, "updates": list(updates)}:
            _update_terrain_tiles(updates)
            return True
        case {
            "msg_type": 0x5A,
            "viewport_left": int(viewport_left),
            "viewport_top": int(viewport_top),
            "entities": list(entities),
        }:
            _update_viewport_entities(viewport_left, viewport_top, entities)
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
            log.info("TELEPORT_LANDED: server confirmed teleport")
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
    "mark_combat_hit",
    "mark_move_target_failed",
    "mark_radar_scan_complete",
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
