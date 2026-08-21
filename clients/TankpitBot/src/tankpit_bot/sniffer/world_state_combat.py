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

    **The per-shot ammo consumption IS the hit signal** (user contract
    2026-07-02). The server only spends dual / missile / homing ammo
    on a shot that lands, and it encodes the spend in the ShootEvent
    ``weapon`` field: ``weapon > 0`` means one consumable was debited
    (hit), ``weapon == 0`` means a free single resolved against empty
    ground (miss). This is the same per-shot inventory delta the page
    client renders. Live proof, run 2026-07-02 01:21: five pursuit
    homings each carried ``weapon=3`` and orange-3 died to the fifth,
    while the wire's ``victim_id`` was ``-1`` on every one of them
    (the tile-occupancy lookup cannot see an off-viewport target).
    The pre-2026-07-02 classifier keyed the hit on ``victim_id > 0``
    and derived the ammo decrement FROM it -- calling the winning
    pursuit shots misses and leaving the ammo-delta signal circularly
    dependent on the guess it existed to correct.

    ``victim_id`` remains recorded for kill attribution and
    intended-target diagnostics only.

    Args:
        ws: World service instance.
        weapon_byte: ShootEvent ``weapon`` field (0=single, 1=dual,
            2=missile, 3=homing). Any value above 0 is a server-side
            ammo debit and therefore a confirmed hit.
        victim_id: Tank id present at the shot's target tile, or -1
            when the tile is empty or the target is off-viewport.
    """
    ws.got_our_shot_response = True
    ws.last_shot_victim_id = victim_id
    if weapon_byte > 0:
        ws.got_confirmed_hit = True
        _decrement_ammo_for_weapon(ws, weapon_byte)


def mark_pending_ground_shot(ws: WorldService, aim_x: int, aim_y: int, dispatch_ms: int) -> None:
    """Record a dispatched ground-aimed shot awaiting its 0x53 receipt.

    Called by the executor when a shoot command with ``target_id == 0``
    (clearance fire at a tile) goes to the wire. The tick loop's
    ground-shot resolver consumes the mark when the echo arrives and
    emits the shot's ``fired`` resolution. At most one shot is in
    flight per tick, so overwriting a stale mark is the correct
    supersede semantics — the replaced shot's decision already closed
    through the ledger's dispatched-supersede path.

    Args:
        ws: World service instance.
        aim_x: Commanded aim tile X.
        aim_y: Commanded aim tile Y.
        dispatch_ms: Wall-clock dispatch timestamp.
    """
    ws.pending_ground_shot_aim_x = aim_x
    ws.pending_ground_shot_aim_y = aim_y
    ws.pending_ground_shot_dispatch_ms = dispatch_ms


def clear_pending_ground_shot(ws: WorldService) -> None:
    """Drop any pending ground-shot mark.

    Called when a tank-targeted shot dispatches (the combat feedback
    classifier owns the next echo) and by the resolver once the mark
    is consumed.

    Args:
        ws: World service instance.
    """
    ws.pending_ground_shot_aim_x = 0
    ws.pending_ground_shot_aim_y = 0
    ws.pending_ground_shot_dispatch_ms = 0


def check_and_clear_combat_hit(ws: WorldService) -> bool:
    """Check if our most recent shot consumed ammo (= hit), then clear.

    Hit = the ShootEvent ``weapon`` field recorded an ammo debit
    (weapon > 0). Consumption is the authoritative per-shot hit
    ledger (user contract 2026-07-02); ``weapon=0`` singles are free
    and resolve against empty ground.

    Args:
        ws: World service instance.

    Returns:
        True if the shot debited ammo, False if it was a free single.
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


def mark_tank_killed(ws: WorldService, tank_id: int, killer_id: int) -> None:
    """Record a tank as killed via Deactivation protocol message.

    Args:
        ws: World service instance.
        tank_id: The killed tank's ID.
        killer_id: The 0x41 killer's tank ID — kept with the victim so
            the drain consumers can tell our kills from the room's.
    """
    ws.killed_tank_ids[tank_id] = killer_id


def drain_killed_tank_ids(ws: WorldService) -> dict[int, int]:
    """Get and clear all killed tank IDs since last drain.

    Args:
        ws: World service instance.

    Returns:
        Mapping of victim tank ID to killer tank ID.
    """
    result = ws.killed_tank_ids
    ws.killed_tank_ids = {}
    return result


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


def peek_command_error(ws: WorldService) -> int:
    """Return the pending command error code without consuming it.

    Used by the shot-feedback wait to end the wait immediately when a
    0x52 rejection arrives for the in-flight shot -- the actual
    consumption happens in the feedback classifier so nothing races
    the in-flight-action machinery.

    Args:
        ws: World service instance.

    Returns:
        Pending error code (0-10), or -1 if no error pending.
    """
    return ws.last_command_error


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
    "clear_pending_ground_shot",
    "drain_killed_tank_ids",
    "mark_combat_hit",
    "mark_pending_ground_shot",
    "mark_tank_killed",
    "mark_teleport_landed",
    "peek_combat_hit",
    "peek_command_error",
    "peek_our_shot_response",
]
