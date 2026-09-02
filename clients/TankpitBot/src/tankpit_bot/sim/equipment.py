"""Law 7 addendum — equipment containers and the 0x67 grant.

Archive-mined law (2026-07-22, 1,149 exact-pre ``0x67 -> next 0x49``
pairs across 246 sessions): every successful pickup grants EXACTLY
ONE slot; counts hard-cap at the rank inventory capacity — the
all-private corpus capped at 25, i.e. ``inventory_capacity(1)``, and
the rank-derived law (``20 + 5 * rank``) is the better-attested form
(official rules table; recruit cap 20 confirmed by the
bot-20260725-211120 promotion crossing). The uncapped stack rolls are
5-9 for the weapon slots and 2-4 for radar (radar really is the
smallest stack); slot choice is RANDOM among deficient slots (128
homing-over-needier-dual, 37 the reverse, 89 radar-while-a-weapon-
short — the wiki's "deterministic most-behind" claim is falsified).
All slots at cap -> the server rejects the pickup with 0x52 error 7,
the ``equipment_pickup_refusal`` law in ``physics/supervisor.py``,
and the container stays.

Sim assumption (documented, [[physics-module-roadmap]]): the sim is
deterministic, so it grants the MOST-DEFICIENT slot (lowest slot
index on ties) with the measured midpoint stack — 7 for weapons, 3
for radar — clipped to the cap. Distribution-faithful randomness is
deliberately traded for reproducibility.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot.physics.capacity import inventory_capacity
from tankpit_bot.physics.supervisor import equipment_pickup_refusal
from tankpit_bot.sim.world import EQUIPMENT_SLOTS, SimTankDict, SimWorldDict

RADAR_SLOT = 4
"""Index of the extra-radar slot in the five-slot counts array."""

WEAPON_STACK_ROLL = (5, 9)
"""Measured uncapped weapon-slot stack range (dual/homing, 1,149 grants)."""

RADAR_STACK_ROLL = (2, 4)
"""Measured uncapped radar stack range — radar really is the smallest."""

WEAPON_STACK = (WEAPON_STACK_ROLL[0] + WEAPON_STACK_ROLL[1]) // 2
"""Deterministic weapon-slot stack: midpoint of the measured 5-9 rolls."""

RADAR_STACK = (RADAR_STACK_ROLL[0] + RADAR_STACK_ROLL[1]) // 2
"""Deterministic radar stack: midpoint of the measured 2-4 rolls."""

MERCY_BUNDLE_ROLLS = ((0, 0), (1, 4), (0, 0), (1, 1), (1, 2))
"""Measured radar-zero kill-reward ranges per slot (armor, dual,
missile, homing, radar) — the archive's 5 silent bundles rolled dual
+1..4, homing exactly +1, radar +1..2, and may overfill past the cap."""

MERCY_BUNDLE = tuple((low + high) // 2 for low, high in MERCY_BUNDLE_ROLLS)
"""Deterministic sim mercy bundle: per-slot midpoints of the rolls."""


def kill_grants_mercy(radar_count: int) -> bool:
    """The radar-zero kill-reward trigger (archive-cracked 2026-07-22).

    Deterministic in the corpus: 5/5 kills at radar zero granted the
    silent bundle, 0/254 kills at radar > 0 granted, no exceptions.

    Args:
        radar_count: The killer's extra-radar count at the kill.

    Returns:
        True when the kill earns the silent mercy bundle.
    """
    return radar_count == 0


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
    if equipment_pickup_refusal(list(tank["counts"]), tank["rank"]) is not None:
        return EquipmentGrantDict(kind="inventory_full", gained=[0] * EQUIPMENT_SLOTS)
    gained = _grant(tank)
    world["equipment"].remove(containers[0])
    return EquipmentGrantDict(kind="granted", gained=gained)


def _grant(tank: SimTankDict) -> list[int]:
    """Apply the deterministic grant to the neediest slot.

    The caller has already ruled out the all-at-cap refusal via
    ``equipment_pickup_refusal``, so at least one slot is deficient.

    Args:
        tank: The collecting tank (mutated).

    Returns:
        The five-slot gained array (exactly one nonzero entry).
    """
    gained = [0] * EQUIPMENT_SLOTS
    cap = inventory_capacity(tank["rank"])
    deficits = [cap - count for count in tank["counts"]]
    best = 0
    for slot in range(1, EQUIPMENT_SLOTS):
        if deficits[slot] > deficits[best]:
            best = slot
    amount = min(_SLOT_STACKS[best], deficits[best])
    tank["counts"][best] += amount
    gained[best] = amount
    return gained


def toggle_equipment_slot(world: SimWorldDict, tank_id: int, slot: int) -> None:
    """Flip one equipment slot for a tank.

    The toggle is free and server-authoritative. An out-of-range press
    changes nothing — the real UI has no such button, so a slot outside
    1-5 is the client's problem and is ignored exactly as live rather
    than refused.

    This is the resolve half of the toggle: it owns the world effect so
    the 0x74 answer can be narrated purely
    (:func:`tankpit_bot.sim.narrate.resources.narrate_equipment_toggle`).

    Args:
        world: Simulated world (mutated when the slot is in range).
        tank_id: The toggling tank.
        slot: Equipment slot, 1-5.
    """
    tank = world["tanks"][tank_id]
    if 1 <= slot <= len(tank["enabled"]):
        tank["enabled"][slot - 1] = not tank["enabled"][slot - 1]


__all__ = [
    "MERCY_BUNDLE",
    "MERCY_BUNDLE_ROLLS",
    "RADAR_SLOT",
    "RADAR_STACK",
    "RADAR_STACK_ROLL",
    "WEAPON_STACK",
    "WEAPON_STACK_ROLL",
    "EquipmentGrantDict",
    "kill_grants_mercy",
    "resolve_equipment_pickup",
    "toggle_equipment_slot",
]
