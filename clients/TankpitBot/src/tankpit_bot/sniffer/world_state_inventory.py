"""Inventory tracking from binary protocol messages.

Handles absolute inventory sync (0x49), equipment gains (0x67), and
equipment toggles (0x74). All functions update the module-level
inventory state in ``world_state``.
"""

from __future__ import annotations

from platform_core.logging import get_logger

import tankpit_bot.sniffer.world_state as _ws
from tankpit_bot.inventory import (
    InventoryChange,
    InventoryItem,
    InventoryState,
    diff_inventory,
)

log = get_logger(__name__)


def get_inventory_state() -> InventoryState:
    """Get the current inventory state from binary protocol tracking.

    Returns:
        Current InventoryState with counts and enabled flags.
    """
    return _ws._inventory_state


def update_inventory_from_protocol(
    counts: list[int],
    enabled: list[bool],
) -> list[InventoryChange]:
    """Set absolute inventory state from a 0x49 (Inventory) message.

    Args:
        counts: List of 5 item counts [armor, dual, missile, homing, radar].
        enabled: List of 5 enabled flags matching the same order.

    Returns:
        List of inventory changes detected.
    """
    old = _ws._inventory_state
    _ws._inventory_state = InventoryState(
        armor_shields=InventoryItem(count=counts[0], enabled=enabled[0]),
        dual_shots=InventoryItem(count=counts[1], enabled=enabled[1]),
        missile_shots=InventoryItem(count=counts[2], enabled=enabled[2]),
        homing_shots=InventoryItem(count=counts[3], enabled=enabled[3]),
        extra_radars=InventoryItem(count=counts[4], enabled=enabled[4]),
    )
    changes = diff_inventory(old, _ws._inventory_state)
    _log_inventory_changes(changes)
    return changes


def update_inventory_from_gain(gained: list[int]) -> list[InventoryChange]:
    """Apply equipment gain deltas from a 0x67 (EquipmentGain) message.

    Args:
        gained: List of 5 gain amounts [armor, dual, missile, homing, radar].

    Returns:
        List of inventory changes detected.
    """
    old = _ws._inventory_state
    _ws._inventory_state = InventoryState(
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
    changes = diff_inventory(old, _ws._inventory_state)
    _log_inventory_changes(changes)
    return changes


def update_inventory_from_toggle(enabled: list[bool]) -> list[InventoryChange]:
    """Update enabled flags from a 0x74 (EquipmentToggle) message.

    Args:
        enabled: List of 5 enabled flags [armor, dual, missile, homing, radar].

    Returns:
        List of inventory changes detected.
    """
    old = _ws._inventory_state
    _ws._inventory_state = InventoryState(
        armor_shields=InventoryItem(count=old["armor_shields"]["count"], enabled=enabled[0]),
        dual_shots=InventoryItem(count=old["dual_shots"]["count"], enabled=enabled[1]),
        missile_shots=InventoryItem(count=old["missile_shots"]["count"], enabled=enabled[2]),
        homing_shots=InventoryItem(count=old["homing_shots"]["count"], enabled=enabled[3]),
        extra_radars=InventoryItem(count=old["extra_radars"]["count"], enabled=enabled[4]),
    )
    changes = diff_inventory(old, _ws._inventory_state)
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


__all__ = [
    "get_inventory_state",
    "update_inventory_from_gain",
    "update_inventory_from_protocol",
    "update_inventory_from_toggle",
]
