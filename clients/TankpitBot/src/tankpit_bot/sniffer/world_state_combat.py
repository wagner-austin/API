"""Combat and lifecycle event tracking for world state.

Handles combat hit confirmation, shot response tracking, ammo decrement,
kill tracking, and teleport landing confirmation.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_service import WEAPON_BYTE_TO_ITEM, WorldService

log = get_logger(__name__)


def mark_combat_hit(ws: WorldService, weapon_byte: int) -> None:
    """Called when we receive a CombatHit where we are the attacker.

    Records that the server processed our shot. If weapon_byte > 0,
    special ammo was consumed (hit confirmed) and the corresponding
    inventory count is decremented.

    Args:
        ws: World service instance.
        weapon_byte: Last byte of combat_data (0=single, 1=dual,
            2=missile, 3=homing).
    """
    ws.got_our_shot_response = True
    if weapon_byte > 0:
        ws.got_confirmed_hit = True
        _decrement_ammo_for_weapon(ws, weapon_byte)


def check_and_clear_combat_hit(ws: WorldService) -> bool:
    """Check if our shot hit (special ammo was used), then clear.

    Args:
        ws: World service instance.

    Returns:
        True if shot connected (weapon_byte > 0), False if miss.
    """
    result = ws.got_confirmed_hit
    ws.got_confirmed_hit = False
    return result


def peek_combat_hit(ws: WorldService) -> bool:
    """Return whether a confirmed outgoing hit is currently buffered.

    Args:
        ws: World service instance.

    Returns:
        True if an outgoing hit has been observed and not yet consumed.
    """
    return ws.got_confirmed_hit


def peek_our_shot_response(ws: WorldService) -> bool:
    """Return whether any CombatHit response for our shot is buffered.

    Args:
        ws: World service instance.

    Returns:
        True if any shot response has been observed and not yet consumed.
    """
    return ws.got_our_shot_response


def check_and_clear_our_shot_response(ws: WorldService) -> bool:
    """Check if any CombatHit for our shot arrived, then clear.

    Args:
        ws: World service instance.

    Returns:
        True if the server sent a CombatHit response for our shot.
    """
    result = ws.got_our_shot_response
    ws.got_our_shot_response = False
    return result


def _decrement_ammo_for_weapon(ws: WorldService, weapon_byte: int) -> None:
    """Decrement inventory count for the ammo type consumed by a hit.

    Args:
        ws: World service instance.
        weapon_byte: Weapon type from CombatHit (1=dual, 2=missile,
            3=homing).
    """
    item_key = WEAPON_BYTE_TO_ITEM.get(weapon_byte)
    if item_key is None:
        return
    current = ws.inventory_state[item_key]
    if current["count"] <= 0:
        return
    new_count = current["count"] - 1
    updated_item = InventoryItem(count=new_count, enabled=current["enabled"])
    old = ws.inventory_state
    ws.inventory_state = InventoryState(
        armor_shields=updated_item if item_key == "armor_shields" else old["armor_shields"],
        dual_shots=updated_item if item_key == "dual_shots" else old["dual_shots"],
        missile_shots=updated_item if item_key == "missile_shots" else old["missile_shots"],
        homing_shots=updated_item if item_key == "homing_shots" else old["homing_shots"],
        extra_radars=updated_item if item_key == "extra_radars" else old["extra_radars"],
    )
    log.info("AMMO: %s consumed by hit (%d -> %d)", item_key, current["count"], new_count)


def mark_tank_killed(ws: WorldService, tank_id: int) -> None:
    """Record a tank as killed via Deactivation protocol message.

    Also anchors the tank's current world-state position as its death
    tile so the registry-truth module can suppress corpse re-ingestion.

    Args:
        ws: World service instance.
        tank_id: The killed tank's ID.
    """
    ws.killed_tank_ids.add(tank_id)
    existing = ws.world_state["tanks"].get(str(tank_id))
    if existing is not None:
        ws.tank_death_anchors[tank_id] = (existing["x"], existing["y"])


def drain_killed_tank_ids(ws: WorldService) -> set[int]:
    """Get and clear all killed tank IDs since last drain.

    Args:
        ws: World service instance.

    Returns:
        Set of tank IDs that were killed.
    """
    result = ws.killed_tank_ids
    ws.killed_tank_ids = set()
    return result


def get_death_anchor(ws: WorldService, tank_id: int) -> tuple[int, int] | None:
    """Return the death-tile anchor for a killed tank.

    Args:
        ws: World service instance.
        tank_id: Tank ID to look up.

    Returns:
        ``(x, y)`` tuple of the tile where the tank was last killed,
        or ``None`` if the tank has no death anchor.
    """
    return ws.tank_death_anchors.get(tank_id)


def clear_death_anchor(ws: WorldService, tank_id: int) -> None:
    """Clear a tank's death-tile anchor after respawn evidence.

    Args:
        ws: World service instance.
        tank_id: Tank whose death anchor to clear.
    """
    ws.tank_death_anchors.pop(tank_id, None)


def mark_teleport_landed(ws: WorldService) -> None:
    """Record that the server confirmed a teleport landing.

    Args:
        ws: World service instance.
    """
    ws.teleport_landed = True


def check_and_clear_teleport_landed(ws: WorldService) -> bool:
    """Check if a teleport landed since last check, then clear.

    Args:
        ws: World service instance.

    Returns:
        True if teleport landed confirmation was received.
    """
    result = ws.teleport_landed
    ws.teleport_landed = False
    return result


__all__ = [
    "check_and_clear_combat_hit",
    "check_and_clear_our_shot_response",
    "check_and_clear_teleport_landed",
    "clear_death_anchor",
    "drain_killed_tank_ids",
    "get_death_anchor",
    "mark_combat_hit",
    "mark_tank_killed",
    "mark_teleport_landed",
    "peek_combat_hit",
    "peek_our_shot_response",
]
