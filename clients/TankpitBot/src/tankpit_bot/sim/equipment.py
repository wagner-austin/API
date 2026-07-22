"""Law 7 addendum — equipment containers and the 0x67 grant.

Archive-mined law (2026-07-22, 1,149 exact-pre ``0x67 -> next 0x49``
pairs across 246 sessions): every successful pickup grants EXACTLY
ONE slot; counts hard-cap at 25 (zero past-25 in the corpus); the
uncapped stack rolls are 5-9 for the weapon slots and 2-4 for radar
(radar really is the smallest stack); slot choice is RANDOM among
deficient slots (128 homing-over-needier-dual, 37 the reverse, 89
radar-while-a-weapon-short — the wiki's "deterministic most-behind"
claim is falsified). All slots at cap -> the server rejects the
pickup with 0x52 error 7 (``SUPERVISOR_ERROR_INVENTORY_FULL``) and
the container stays.

Sim assumption (documented, [[physics-module-roadmap]]): the sim is
deterministic, so it grants the MOST-DEFICIENT slot (lowest slot
index on ties) with the measured midpoint stack — 7 for weapons, 3
for radar — clipped to the cap. Distribution-faithful randomness is
deliberately traded for reproducibility.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot.sim.world import EQUIPMENT_SLOTS, SimTankDict, SimWorldDict

EQUIPMENT_CAP = 25
"""Hard per-slot cap — 0/1,149 corpus grants pushed a count past it."""

WEAPON_STACK = 7
"""Deterministic weapon-slot stack: midpoint of the measured 5-9 rolls."""

RADAR_STACK = 3
"""Deterministic radar stack: midpoint of the measured 2-4 rolls."""

_SLOT_STACKS = (WEAPON_STACK, WEAPON_STACK, WEAPON_STACK, WEAPON_STACK, RADAR_STACK)


class EquipmentGrantDict(TypedDict):
    """One resolved equipment pickup attempt.

    ``gained`` is the five-slot 0x67 payload (a single nonzero entry
    on a grant); ``inventory_full`` grants carry all zeros and leave
    the container in the world.
    """

    kind: Literal["granted", "inventory_full"]
    gained: list[int]


def resolve_equipment_pickup(world: SimWorldDict, tank_id: int) -> EquipmentGrantDict | None:
    """Resolve an equipment container under the arriving tank, if any.

    Args:
        world: Simulated world (mutated: grant applied, container
            consumed on success).
        tank_id: The arriving tank.

    Returns:
        The grant outcome, or None when the tank's tile holds no
        equipment container.
    """
    tank = world["tanks"][tank_id]
    containers = [e for e in world["equipment"] if (e["x"], e["y"]) == (tank["x"], tank["y"])]
    if not containers:
        return None
    gained = _grant(tank)
    if not any(gained):
        return EquipmentGrantDict(kind="inventory_full", gained=gained)
    world["equipment"].remove(containers[0])
    return EquipmentGrantDict(kind="granted", gained=gained)


def _grant(tank: SimTankDict) -> list[int]:
    """Apply the deterministic grant to the neediest slot.

    Args:
        tank: The collecting tank (mutated on a grant).

    Returns:
        The five-slot gained array (all zeros at full inventory).
    """
    gained = [0] * EQUIPMENT_SLOTS
    deficits = [EQUIPMENT_CAP - count for count in tank["counts"]]
    best = 0
    for slot in range(1, EQUIPMENT_SLOTS):
        if deficits[slot] > deficits[best]:
            best = slot
    if deficits[best] <= 0:
        return gained
    amount = min(_SLOT_STACKS[best], deficits[best])
    tank["counts"][best] += amount
    gained[best] = amount
    return gained


__all__ = [
    "EQUIPMENT_CAP",
    "RADAR_STACK",
    "WEAPON_STACK",
    "EquipmentGrantDict",
    "resolve_equipment_pickup",
]
