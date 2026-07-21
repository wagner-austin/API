"""Fuel cost of every player action.

The complete action economy, closed 2026-07-20 (``wiki/pages/
game-economy.md``, "What's still open: Nothing"): everything costs 10
except walking (1/tile), the free-ammo single shot (6), block
operations (free), and teleports (distance-priced). Each symbol below
is bound to a claim in the game-economy claim block and verified by
``scripts.physics_claims`` on every ``make check``.
"""

from __future__ import annotations

from math import isqrt

WALK_COST_PER_TILE = 1
"""Fuel per tile walked. Wiki: [[game-economy]]#walk-cost."""

SINGLE_SHOT_COST = 6
"""Fuel per single shot (weapon=0, consumes no ammo).
Wiki: [[game-economy]]#single-shot-cost."""

DUAL_SHOT_COST = 10
"""Fuel per dual shot (weapon=1, 1 dual per landed shot).
Wiki: [[game-economy]]#dual-shot-cost."""

MISSILE_SHOT_COST = 10
"""Fuel per missile (weapon=2, 1 missile per landed shot).
Wiki: [[game-economy]]#missile-shot-cost."""

HOMING_SHOT_COST = 10
"""Fuel per homing shot (weapon=3, 1 homing per landed shot; the
debit sometimes lands as two -5 steps across sync boundaries).
Wiki: [[game-economy]]#homing-shot-cost."""

RADAR_COST = 10
"""Fuel per extra-radar scan. Wiki: [[game-economy]]#radar-cost."""

MINE_PRESS_COST = 10
"""Fuel per mine press, flat — independent of how many of the 3x3
field's mines actually land. Wiki: [[game-economy]]#mine-press-cost."""

BLOCK_OP_COST = 0
"""Fuel per movable-block pickup or drop (free; towing movement pays
the normal walk cost). Wiki: [[game-economy]]#block-op-cost."""


def teleport_cost(
    start_x: int,
    start_y: int,
    target_x: int,
    target_y: int,
) -> int:
    """Compute the exact fuel cost for a teleport.

    Tankpit teleport fuel cost scales with Euclidean distance:

    ``floor(6 * sqrt(dx^2 + dy^2))``

    This implementation uses integer square root over ``36 *
    distance_sq`` so the returned value is exact without
    floating-point drift.

    The server charges distance to the ACTUAL landing tile, not the
    requested target — when the landing drifts (mines, terrain), a
    target-based estimate can be off by a few fuel.
    Wiki: [[game-economy]]#teleport-cost.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        target_x: Destination X coordinate.
        target_y: Destination Y coordinate.

    Returns:
        Exact integer fuel cost for the teleport.
    """
    delta_x = target_x - start_x
    delta_y = target_y - start_y
    distance_sq = delta_x * delta_x + delta_y * delta_y
    return isqrt(36 * distance_sq)


__all__ = [
    "BLOCK_OP_COST",
    "DUAL_SHOT_COST",
    "HOMING_SHOT_COST",
    "MINE_PRESS_COST",
    "MISSILE_SHOT_COST",
    "RADAR_COST",
    "SINGLE_SHOT_COST",
    "WALK_COST_PER_TILE",
    "teleport_cost",
]
