"""Narration for the radar, mine and equipment commands.

Pure functions from a resolved outcome to the messages ONE observer
receives; see :mod:`tankpit_bot.sim.narrate.movement` for the shape and
the meaning of ``observer_id``.
"""

from __future__ import annotations

from tankpit_bot.container.types import MineDetonationDict, MinePlacementDict
from tankpit_bot.protocol.constants import SUPERVISOR_ERROR_INVENTORY_FULL
from tankpit_bot.protocol.types import (
    BinaryMessage,
    EquipmentGainDict,
    EquipmentToggleDict,
    InventoryDict,
    RadarResultDict,
    RadarScanResultDict,
    SupervisorDict,
)
from tankpit_bot.sim.actions import MinePressOutcomeDict, RadarOutcomeDict
from tankpit_bot.sim.equipment import EquipmentGrantDict
from tankpit_bot.sim.movement import PickupRecordDict
from tankpit_bot.sim.narrate.movement import pickup_message
from tankpit_bot.sim.world import SimWorldDict


def _inventory_snapshot(world: SimWorldDict, tank_id: int) -> InventoryDict:
    """Build the 0x49 snapshot of one tank's current counts.

    Args:
        world: Simulated world, post-resolution. Read only.
        tank_id: The tank whose inventory is reported.

    Returns:
        The 0x49 message carrying that tank's counts and flags.
    """
    tank = world["tanks"][tank_id]
    return InventoryDict(
        msg_type=0x49,
        show=False,
        alternate=False,
        counts=list(tank["counts"]),
        enabled=list(tank["enabled"]),
    )


def narrate_radar(
    world: SimWorldDict,
    outcome: RadarOutcomeDict,
    observer_id: int,
) -> list[BinaryMessage]:
    """Narrate one resolved radar scan to a single observer.

    A scan's results are PER-RECIPIENT: measured across 341 archived
    sessions, 7,014 scan results arrived against 7,053 own radar
    commands and ZERO arrived in a session that sent none — other
    players' scans never reach this connection ([[recipient-policy]]).

    The extra-consumption snapshot LEADS the scan results — live radar
    windows are 84% ``49+4F+46`` (response-shape differ 2026-08-01);
    the sim's end-of-tick snapshot had it trailing.

    Args:
        world: Simulated world, post-scan. Read only.
        outcome: The scan's resolved outcome.
        observer_id: The connection being narrated for.

    Returns:
        The messages this observer receives, in emission order.
    """
    if outcome["tank_id"] != observer_id:
        return []
    messages: list[BinaryMessage] = []
    if outcome["consumed_extra"]:
        messages.append(_inventory_snapshot(world, outcome["tank_id"]))
    messages.append(
        RadarScanResultDict(
            msg_type=0x4F,
            containers=outcome["containers"],
            mines=outcome["mines"],
            mine_clears=[],
        )
    )
    messages.append(RadarResultDict(msg_type=0x46, detection_type=0, found=outcome["enemy_found"]))
    return messages


def narrate_mine_press(
    outcome: MinePressOutcomeDict,
    observer_id: int,
) -> list[BinaryMessage]:
    """Narrate one resolved mine press to a single observer.

    The 0x4B placement is PER-RECIPIENT — it is the placer's own
    receipt, the same discipline the 0x52 rejections and the fuel sync
    follow. Every one of the 23 placements in the whole archive names
    the capturing client; not one reports another player's press,
    however heavily they mined. Other players' mines reach a client as
    overlay and radar reveals instead ([[mine-mechanics]]).

    The 0x45 detonation is NOT per-recipient: it broadcasts, which is
    why the archive carries 296 of them against 23 placements.

    Args:
        outcome: The press's resolved outcome.
        observer_id: The connection being narrated for.

    Returns:
        The messages this observer receives, in emission order.
    """
    messages: list[BinaryMessage] = []
    if outcome["placed"] and outcome["tank_id"] == observer_id:
        messages.append(
            MinePlacementDict(
                msg_type=0x4B,
                mine_type=outcome["mine_type"],
                tank_id=outcome["tank_id"],
                positions=outcome["placed"],
            )
        )
    if outcome["detonated"]:
        messages.append(MineDetonationDict(msg_type=0x45, positions=outcome["detonated"]))
    return messages


def narrate_equipment_pickup(
    world: SimWorldDict,
    grant: EquipmentGrantDict,
    tank_id: int,
    kind: str,
    observer_id: int,
) -> list[BinaryMessage]:
    """Narrate one resolved equipment pickup to a single observer.

    A grant emits the 0x67 gained array followed by its 0x49 snapshot
    — the archive shows every 0x67 immediately followed by one — and
    then the container-pickup record closing the drained container
    (live order over 2,170 archive windows; the sim had no pickup
    record at all until the 2026-08-01 response-shape differ).

    A full-inventory attempt on an explicit ``pickup_equipment`` click
    answers with the measured 0x52 error 7 and leaves the container;
    incidental arrivals at full inventory are silent. Both 0x67 and
    the 0x52 are PER-RECIPIENT: production treats any 0x67 as a SELF
    gain, so another tank's grant resolves silently
    ([[recipient-policy]]).

    Args:
        world: Simulated world, post-grant. Read only.
        grant: The pickup's resolved outcome.
        tank_id: The arriving tank.
        kind: The command kind that caused the arrival.
        observer_id: The connection being narrated for.

    Returns:
        The messages this observer receives, in emission order.
    """
    if tank_id != observer_id:
        return []
    if grant["kind"] == "granted":
        tank = world["tanks"][tank_id]
        return [
            EquipmentGainDict(msg_type=0x67, show_message=True, gained=grant["gained"]),
            _inventory_snapshot(world, tank_id),
            pickup_message([PickupRecordDict(x=tank["x"], y=tank["y"], remaining_volume=0)]),
        ]
    if kind != "pickup_equipment":
        return []
    return [
        SupervisorDict(
            msg_type=0x52,
            reset_action=1,
            close_map=0,
            error_code=SUPERVISOR_ERROR_INVENTORY_FULL,
        )
    ]


def narrate_equipment_toggle(
    world: SimWorldDict,
    tank_id: int,
    observer_id: int,
) -> list[BinaryMessage]:
    """Narrate one resolved equipment toggle to a single observer.

    The 0x74 carries all five enabled flags and names no tank, so it
    can only describe the recipient's own loadout — it is
    per-recipient by construction ([[recipient-policy]]).

    Args:
        world: Simulated world, post-toggle. Read only.
        tank_id: The toggling tank.
        observer_id: The connection being narrated for.

    Returns:
        The messages this observer receives, in emission order.
    """
    if tank_id != observer_id:
        return []
    return [EquipmentToggleDict(msg_type=0x74, enabled=list(world["tanks"][tank_id]["enabled"]))]


__all__ = [
    "narrate_equipment_pickup",
    "narrate_equipment_toggle",
    "narrate_mine_press",
    "narrate_radar",
]
