"""Combat and lifecycle event tracking for world state.

Handles combat hit confirmation, shot response tracking, ammo decrement,
kill tracking, and teleport landing confirmation.
"""

from __future__ import annotations

from platform_core.logging import get_logger

import tankpit_bot.sniffer.world_state as _ws
from tankpit_bot.inventory import InventoryItem, InventoryState

log = get_logger(__name__)


def mark_combat_hit(weapon_byte: int) -> None:
    """Called when we receive a CombatHit where we are the attacker.

    Records that the server processed our shot. If weapon_byte > 0,
    special ammo was consumed (hit confirmed) and the corresponding
    inventory count is decremented.

    Args:
        weapon_byte: Last byte of combat_data (0=single, 1=dual,
            2=missile, 3=homing).
    """
    _ws._got_our_shot_response = True
    if weapon_byte > 0:
        _ws._got_confirmed_hit = True
        _decrement_ammo_for_weapon(weapon_byte)


def check_and_clear_combat_hit() -> bool:
    """Check if our shot hit (special ammo was used), then clear.

    Returns:
        True if shot connected (weapon_byte > 0), False if miss.
    """
    result = _ws._got_confirmed_hit
    _ws._got_confirmed_hit = False
    return result


def peek_combat_hit() -> bool:
    """Return whether a confirmed outgoing hit is currently buffered.

    Returns:
        True if an outgoing hit has been observed and not yet consumed.
    """
    return _ws._got_confirmed_hit


def peek_our_shot_response() -> bool:
    """Return whether any CombatHit response for our shot is buffered.

    Returns:
        True if any shot response has been observed and not yet consumed.
    """
    return _ws._got_our_shot_response


def check_and_clear_our_shot_response() -> bool:
    """Check if any CombatHit for our shot arrived, then clear.

    Returns:
        True if the server sent a CombatHit response for our shot.
    """
    result = _ws._got_our_shot_response
    _ws._got_our_shot_response = False
    return result


def _decrement_ammo_for_weapon(weapon_byte: int) -> None:
    """Decrement inventory count for the ammo type consumed by a hit.

    Args:
        weapon_byte: Weapon type from CombatHit (1=dual, 2=missile,
            3=homing).
    """
    item_key = _ws._WEAPON_BYTE_TO_ITEM.get(weapon_byte)
    if item_key is None:
        return
    current = _ws._inventory_state[item_key]
    if current["count"] <= 0:
        return
    new_count = current["count"] - 1
    updated_item = InventoryItem(count=new_count, enabled=current["enabled"])
    old = _ws._inventory_state
    _ws._inventory_state = InventoryState(
        armor_shields=updated_item if item_key == "armor_shields" else old["armor_shields"],
        dual_shots=updated_item if item_key == "dual_shots" else old["dual_shots"],
        missile_shots=updated_item if item_key == "missile_shots" else old["missile_shots"],
        homing_shots=updated_item if item_key == "homing_shots" else old["homing_shots"],
        extra_radars=updated_item if item_key == "extra_radars" else old["extra_radars"],
    )
    log.info("AMMO: %s consumed by hit (%d -> %d)", item_key, current["count"], new_count)


def mark_tank_killed(tank_id: int) -> None:
    """Record a tank as killed via Deactivation protocol message.

    Also anchors the tank's current world-state position as its death
    tile so the registry-truth module can suppress corpse re-ingestion
    (the client registry keeps rendering deactivated sprites for minutes).

    Args:
        tank_id: The killed tank's ID.
    """
    _ws._killed_tank_ids.add(tank_id)
    existing = _ws._world_state["tanks"].get(str(tank_id))
    if existing is not None:
        _ws._tank_death_anchors[tank_id] = (existing["x"], existing["y"])


def drain_killed_tank_ids() -> set[int]:
    """Get and clear all killed tank IDs since last drain.

    Returns:
        Set of tank IDs that were killed.
    """
    result = _ws._killed_tank_ids
    _ws._killed_tank_ids = set()
    return result


def get_death_anchor(tank_id: int) -> tuple[int, int] | None:
    """Return the death-tile anchor for a killed tank.

    Args:
        tank_id: Tank ID to look up.

    Returns:
        ``(x, y)`` tuple of the tile where the tank was last killed,
        or ``None`` if the tank has no death anchor.
    """
    return _ws._tank_death_anchors.get(tank_id)


def clear_death_anchor(tank_id: int) -> None:
    """Clear a tank's death-tile anchor after respawn evidence.

    Called when a registry observation places the tank away from its
    death tile -- proof that the tank respawned and its old corpse
    sprite is gone.

    Args:
        tank_id: Tank whose death anchor to clear.
    """
    _ws._tank_death_anchors.pop(tank_id, None)


def mark_teleport_landed() -> None:
    """Record that the server confirmed a teleport landing."""
    _ws._teleport_landed = True


def check_and_clear_teleport_landed() -> bool:
    """Check if a teleport landed since last check, then clear.

    Returns:
        True if teleport landed confirmation was received.
    """
    result = _ws._teleport_landed
    _ws._teleport_landed = False
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
