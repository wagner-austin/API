"""Equipment-specific pickup-completion helpers for live action-lab probes.

Mirrors the fuel-side defaults exposed by ``pickup_phase`` (which watch the
fuel counter on ``self_state``), but watches the total inventory item count
managed by the binary-protocol decoders for ``0x49`` / ``0x67`` / ``0x74``.
"""

from __future__ import annotations

from typing import Literal

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.inventory import InventoryState
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state

_EQUIPMENT_PICKUP_POLL_INTERVAL_MS = 100.0


class EquipmentPickupError(Exception):
    """Raised when an equipment pickup phase cannot proceed."""


def total_inventory_count(state: InventoryState) -> int:
    """Return the sum of all five inventory slot counts.

    Args:
        state: Current inventory state.

    Returns:
        Total number of items currently held across every slot.
    """
    return (
        state["armor_shields"]["count"]
        + state["dual_shots"]["count"]
        + state["missile_shots"]["count"]
        + state["homing_shots"]["count"]
        + state["extra_radars"]["count"]
    )


def get_completed_equipment_pickup_outcome(
    probe: action_session.WorldStateProviderProtocol,
    *,
    target_x: int,
    target_y: int,
    inventory_count_before: int,
) -> tuple[Literal["picked_up_equipment"], int, int] | None:
    """Return a completed pickup outcome once the inventory total grows.

    Args:
        probe: Probe exposing the latest world state. The ``probe`` argument
            mirrors the fuel-side helper signature even though the equipment
            inventory is read from the global inventory tracker rather than
            from ``WorldStateDict``.
        target_x: Pickup target X tile (for diagnostic parity).
        target_y: Pickup target Y tile (for diagnostic parity).
        inventory_count_before: Inventory total before pickup tracking began.

    Returns:
        Completed pickup tuple when the inventory total has grown, otherwise
        ``None``.
    """
    _ = (probe, target_x, target_y)
    current_total = total_inventory_count(get_inventory_state())
    if current_total > inventory_count_before:
        return (
            "picked_up_equipment",
            action_hooks.get_current_time_ms(),
            current_total,
        )
    return None


def wait_for_equipment_pickup_outcome(
    page: action_session.WaitPageProtocol,
    probe: action_session.BufferedWorldStateProviderProtocol,
    *,
    target_x: int,
    target_y: int,
    pickup_started_ms: int,
    inventory_count_before: int,
    timeout_ms: int,
) -> tuple[Literal["picked_up_equipment", "pickup_timeout"], int, int]:
    """Wait for an equipment pickup to complete or time out.

    Args:
        page: Page used for polling waits.
        probe: Probe exposing buffered world state.
        target_x: Pickup target X tile (for diagnostic parity).
        target_y: Pickup target Y tile (for diagnostic parity).
        pickup_started_ms: Timestamp when pickup tracking started.
        inventory_count_before: Inventory total before pickup started.
        timeout_ms: Maximum wait time in milliseconds.

    Returns:
        Terminal pickup status, completion timestamp, and inventory total
        observed at completion.
    """
    while action_hooks.get_current_time_ms() - pickup_started_ms < timeout_ms:
        action_hooks.drain_buffered_messages(probe)
        outcome = get_completed_equipment_pickup_outcome(
            probe,
            target_x=target_x,
            target_y=target_y,
            inventory_count_before=inventory_count_before,
        )
        if outcome is not None:
            return outcome
        page.wait_for_timeout(_EQUIPMENT_PICKUP_POLL_INTERVAL_MS)
    return (
        "pickup_timeout",
        action_hooks.get_current_time_ms(),
        total_inventory_count(get_inventory_state()),
    )


__all__ = [
    "EquipmentPickupError",
    "get_completed_equipment_pickup_outcome",
    "total_inventory_count",
    "wait_for_equipment_pickup_outcome",
]
