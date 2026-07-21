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

MINE_DETONATION_COST = 45
"""Fuel lost by walking onto a hostile mine. This is why hostile
mine tiles are impassable in the composed decision terrain
([[terrain-composition]]). Wiki: [[game-economy]]#mine-detonation-cost."""

__all__ = [
    "DUAL_HIT_VICTIM_COST",
    "MINE_DETONATION_COST",
    "SINGLE_HIT_VICTIM_COST",
]
