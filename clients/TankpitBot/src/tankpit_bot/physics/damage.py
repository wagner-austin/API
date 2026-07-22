"""Fuel loss to the victim per damage source.

The game's "health" is the fuel reserve: damage decrements fuel, and
a tank at zero fuel is deactivated. Values wire-verified 2026-06-20
against 0x2E fuel deltas in the multi-tank PvP capture
(``wiki/pages/game-economy.md``, Damage taken).
"""

from __future__ import annotations

SINGLE_HIT_VICTIM_COST = 45
"""Fuel lost when hit by an enemy single shot (weapon=0).
Wiki: [[game-economy]]#single-hit-victim-cost."""

DUAL_HIT_VICTIM_COST = 90
"""Fuel lost when hit by an enemy dual shot (weapon=1).
Wiki: [[game-economy]]#dual-hit-victim-cost."""

MISSILE_HIT_VICTIM_COST = 45
"""Fuel lost when hit by an enemy missile (weapon=2). Measured
2026-07-21: five isolated hits, each exactly -45 at the echo instant.
Wiki: [[game-economy]]#missile-hit-victim-cost."""

HOMING_HIT_VICTIM_COST = 45
"""Fuel lost when hit by an enemy homing shot (weapon=3). Measured
2026-07-21: five isolated hits, each exactly -45 at the echo instant.
Wiki: [[game-economy]]#homing-hit-victim-cost."""

ARMOR_ABSORB_PER_SHIELD = 45
"""Damage one armor shield absorbs. With shields on, damage is fully
absorbed and shields are consumed at damage/45 per hit: singles,
missiles, and homings eat 1 shield; duals eat 2 (measured 2026-07-21,
sixteen incoming hits, fuel untouched throughout).
Wiki: [[game-economy]]#armor-absorb-per-shield."""

MINE_DETONATION_COST = 45
"""Fuel lost by walking onto a hostile mine. This is why hostile
mine tiles are impassable in the composed decision terrain
([[terrain-composition]]). Wiki: [[game-economy]]#mine-detonation-cost."""

__all__ = [
    "ARMOR_ABSORB_PER_SHIELD",
    "DUAL_HIT_VICTIM_COST",
    "HOMING_HIT_VICTIM_COST",
    "MINE_DETONATION_COST",
    "MISSILE_HIT_VICTIM_COST",
    "SINGLE_HIT_VICTIM_COST",
]
