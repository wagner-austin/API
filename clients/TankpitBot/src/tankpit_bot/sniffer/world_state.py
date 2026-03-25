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
    add_mine_from_radar,
    make_empty_world_state,
    pickup_container,
    remove_tank,
    render_world_ascii,
    set_self_fuel,
    update_container_from_radar,
    update_self_position_and_viewport,
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

# Combat hit tracking — True when ANY CombatHit from us arrives (= shot connected)
_got_combat_hit: bool = False


def mark_combat_hit() -> None:
    """Called when we receive a CombatHit where we are the attacker."""
    global _got_combat_hit
    _got_combat_hit = True


def check_and_clear_combat_hit() -> bool:
    """Check if we got a CombatHit since last check, then clear."""
    global _got_combat_hit
    result = _got_combat_hit
    _got_combat_hit = False
    return result


# Inventory tracking from binary protocol (0x49, 0x67, 0x74)
_inventory_state: InventoryState = InventoryState(
    armor_shields=InventoryItem(count=0, enabled=True),
    dual_shots=InventoryItem(count=0, enabled=True),
    missile_shots=InventoryItem(count=0, enabled=True),
    homing_shots=InventoryItem(count=0, enabled=True),
    extra_radars=InventoryItem(count=0, enabled=True),
)


def _make_empty_inventory() -> InventoryState:
    """Create an empty inventory state with all items at zero.

    Returns:
        InventoryState with all counts at 0 and enabled True.
    """
    return InventoryState(
        armor_shields=InventoryItem(count=0, enabled=True),
        dual_shots=InventoryItem(count=0, enabled=True),
        missile_shots=InventoryItem(count=0, enabled=True),
        homing_shots=InventoryItem(count=0, enabled=True),
        extra_radars=InventoryItem(count=0, enabled=True),
    )


def reset_world_state() -> None:
    """Reset world state for new session (used by tests)."""
    global _world_state, _terrain_map, _room_images, _selected_room, _inventory_state
    _world_state = make_empty_world_state()
    _terrain_map = None
    _room_images = {}
    _selected_room = None
    _inventory_state = _make_empty_inventory()


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
        Path("field42-r.gif"),
        Path("field01_r.gif"),
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
    _world_state = update_self_position_and_viewport(_world_state, x, y, get_current_time_ms())


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


def render_world_state_ascii() -> str | None:
    """Render current world state as ASCII.

    Returns:
        ASCII representation, or None if terrain map not loaded.
    """
    terrain = _load_terrain_map_if_needed()
    if terrain is None:
        return None
    return render_world_ascii(_world_state, terrain)


def _update_viewport_entities(entities: list[dict]) -> None:  # type: ignore[type-arg]
    """Invalidate tanks that should be in viewport but aren't in entity list.

    The ViewportUpdate (0x5A) contains all entities currently visible.
    Any tank we think is in the viewport but isn't in this list has left.
    """
    vp = _world_state["viewport"]
    if vp["left"] == 0 and vp["top"] == 0 and vp["width"] == 18:
        # Viewport not initialized yet
        return

    # Collect entity_ids from viewport update
    visible_ids: set[int] = set()
    for ent in entities:
        eid = ent.get("entity_id", -1)
        if eid > 0:
            visible_ids.add(eid)

    # Check all tanks — if they claim to be in viewport but aren't visible, invalidate
    for tank in _world_state["tanks"].values():
        if tank["is_self"] or tank["x"] == 0 and tank["y"] == 0:
            continue
        in_vp = (
            vp["left"] <= tank["x"] < vp["left"] + vp["width"]
            and vp["top"] <= tank["y"] < vp["top"] + vp["height"]
        )
        if in_vp and tank["tank_id"] not in visible_ids:
            log.info(
                "VIEWPORT: %s (tank=%d) no longer visible, invalidating",
                tank["name"],
                tank["tank_id"],
            )
            _update_tank_position(tank["tank_id"], 0, 0)


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
            log.info("DEACTIVATED: tank=%d killed, position invalidated", vid)
            return True
    return False


def _dispatch_position_update(decoded: protocol.BinaryMessage) -> bool:
    """Dispatch position and movement messages to update world state.

    Handles position_update, movement waypoints, and MovementResponse (0x3D).

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
            "start_x": int(sx),
            "start_y": int(sy),
            "waypoints": list(wps),
        }:
            # 0x47 bytes 0,1 are player_id (not tank_id). Match by start position.
            self_state = _world_state["self_state"]
            is_self = self_state is not None and self_state["x"] == sx and self_state["y"] == sy
            if not is_self:
                # waypoints[0] is (final_x, final_y) if path was parsed
                final_x, final_y = wps[0] if wps else (sx, sy)
                for tank in _world_state["tanks"].values():
                    if tank["x"] == sx and tank["y"] == sy and not tank["is_self"]:
                        _update_tank_position(tank["tank_id"], final_x, final_y)
                        break
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
        case {
            "msg_type": 0x5A,
            "entities": list(entities),
        }:
            _update_viewport_entities(entities)
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
        case {"msg_type": "combat_hit", "attacker_id": int(aid)}:
            self_state = _world_state["self_state"]
            if self_state is not None and aid == self_state["tank_id"]:
                mark_combat_hit()
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
        case {
            "msg_type": "tank_update_compact" | "tank_update_extended" | "tank_update_full",
            "tank_id": int(tid),
            "status_data": bytes(sd),
        }:
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
            _update_tank_position(vid, 0, 0)
            log.info("DEACTIVATION_KILL: tank=%d killed, position invalidated", vid)
            return True
        case {"msg_type": "deactivation_death", "killer_id": int(kid)}:
            log.info("DEACTIVATION_DEATH: killed by tank=%d", kid)
            return True
    return False


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
    "dispatch_world_state_update",
    "get_inventory_state",
    "get_terrain_map",
    "get_world_state",
    "register_room_image",
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
