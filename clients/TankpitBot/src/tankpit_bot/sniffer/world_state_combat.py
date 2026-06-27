"""Combat and lifecycle event tracking for world state.

Handles combat hit confirmation, shot response tracking, ammo decrement,
kill tracking, and teleport landing confirmation.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.inventory import InventoryItem, replace_inventory_slot
from tankpit_bot.sniffer.world_service import WEAPON_BYTE_TO_ITEM, WorldService

log = get_logger(__name__)


def mark_combat_hit(ws: WorldService, weapon_byte: int, victim_id: int) -> None:
    """Called when a 0x53 ShootEvent arrives where we are the attacker.

    The legacy function name predates the 2026-06-19 decoder
    unification, which deleted the container ``CombatHit`` decoder and
    consolidated shot resolution on the protocol-path ``ShootEvent``
    (0x53). The semantics are unchanged: the authoritative hit signal
    is tile-occupancy. JS ``Gg.prototype.h`` switch case 18 prints
    "You hit X" exactly when the shot's target tile contains a named
    tank; we translate that to ``victim_id > 0``.

    Server-side ammo: the server only decrements dual / missile /
    homing counts on a confirmed hit (user-confirmed 2026-06-24 from
    inventory deltas in run 2026-06-24 11:32). The local decrement
    must mirror that rule -- otherwise the local shadow count drifts
    below the authoritative server count between every miss and the
    next 0x49/0x67/0x74 correction.

    Args:
        ws: World service instance.
        weapon_byte: ShootEvent ``weapon`` field (0=single, 1=dual,
            2=missile, 3=homing). Decrements that ammo type on a
            confirmed hit only.
        victim_id: Tank id present at the shot's target tile, or -1 if
            the tile was empty (miss).
    """
    ws.got_our_shot_response = True
    ws.last_shot_victim_id = victim_id
    if victim_id > 0:
        ws.got_confirmed_hit = True
        if weapon_byte > 0:
            _decrement_ammo_for_weapon(ws, weapon_byte)


def check_and_clear_combat_hit(ws: WorldService) -> bool:
    """Check if our most recent shot hit a tank, then clear.

    Hit = any tank existed at the wire-reported target tile of our shot.
    Authoritative per the JS shoot handler (case 18: "You hit X").

    Args:
        ws: World service instance.

    Returns:
        True if shot landed on an occupied tile, False if tile was empty.
    """
    result = ws.got_confirmed_hit
    ws.got_confirmed_hit = False
    return result


def check_and_clear_last_shot_victim_id(ws: WorldService) -> int:
    """Return the tank id our last shot landed on, then clear.

    Used by combat_feedback to tell intended-target hits apart from
    incidental hits (homing seeker landing on a closer enemy than the
    bot commanded).

    Args:
        ws: World service instance.

    Returns:
        Tank id of victim on target tile, or -1 if tile was empty.
    """
    result = ws.last_shot_victim_id
    ws.last_shot_victim_id = -1
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
    """Return whether any 0x53 ShootEvent response for our shot is buffered.

    Args:
        ws: World service instance.

    Returns:
        True if any shot response has been observed and not yet consumed.
    """
    return ws.got_our_shot_response


def check_and_clear_ammo_delta_hit(ws: WorldService) -> bool:
    """Return True if a tracking weapon's ammo dropped since the last shoot.

    The server only debits dual / missile / homing ammo on a confirmed
    hit, so a negative delta between the pending-shoot snapshot and
    the current count is authoritative proof that the shot landed --
    including in the off-viewport pursuit case where the wire's
    ``victim_id`` lookup misses because the target isn't in the local
    registry at the impact tile (live run 2026-06-24 12:43).

    Args:
        ws: World service instance.

    Returns:
        True when any of ``dual_shots`` / ``missile_shots`` /
        ``homing_shots`` shows a strictly negative delta vs the
        snapshot. Clears the snapshot after reading. False when no
        snapshot is pending or no decrement is visible.
    """
    snap = ws.pending_shot_inventory_snapshot
    if snap is None:
        return False
    current = ws.inventory_state
    decreased = (
        current["dual_shots"]["count"] < snap["dual_shots"]["count"]
        or current["missile_shots"]["count"] < snap["missile_shots"]["count"]
        or current["homing_shots"]["count"] < snap["homing_shots"]["count"]
    )
    ws.pending_shot_inventory_snapshot = None
    return decreased


def check_and_clear_our_shot_response(ws: WorldService) -> bool:
    """Check if any 0x53 ShootEvent for our shot arrived, then clear.

    Args:
        ws: World service instance.

    Returns:
        True if the server sent a ShootEvent response for our shot.
    """
    result = ws.got_our_shot_response
    ws.got_our_shot_response = False
    return result


def _decrement_ammo_for_weapon(ws: WorldService, weapon_byte: int) -> None:
    """Decrement inventory count for the ammo type consumed by a hit.

    Args:
        ws: World service instance.
        weapon_byte: 0x53 ShootEvent ``weapon`` field (1=dual,
            2=missile, 3=homing).
    """
    item_key = WEAPON_BYTE_TO_ITEM.get(weapon_byte)
    if item_key is None:
        return
    current = ws.inventory_state[item_key]
    if current["count"] <= 0:
        return
    new_count = current["count"] - 1
    ws.inventory_state = replace_inventory_slot(
        ws.inventory_state,
        item_key,
        InventoryItem(count=new_count, enabled=current["enabled"]),
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


def check_and_clear_command_error(ws: WorldService) -> int:
    """Return and clear the last command error code from a Supervisor message.

    Error codes from tpclient.js Gb[] array:
      0 = "You can't do this"
      1 = "You can't go there!"
      4 = "Empty container"
      5 = "Tank full"
      8 = "Insufficient fuel"
     -1 = no error pending

    Args:
        ws: World service instance.

    Returns:
        Error code (0-10), or -1 if no error pending.
    """
    result = ws.last_command_error
    ws.last_command_error = -1
    return result


__all__ = [
    "check_and_clear_ammo_delta_hit",
    "check_and_clear_combat_hit",
    "check_and_clear_command_error",
    "check_and_clear_last_shot_victim_id",
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
