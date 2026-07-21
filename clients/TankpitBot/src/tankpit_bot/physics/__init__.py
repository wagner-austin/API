"""The game's physics, one symbol per wiki claim.

This package is the SINGLE code home for every reverse-engineered
game rule: action fuel costs, damage taken, rank-derived capacities,
and combat timing. Each public symbol is bound to exactly one wiki
claim (a ``json claims`` fenced block in ``wiki/pages/``) and the
binding is machine-checked by ``scripts.physics_claims`` inside
``make check`` — a constant that drifts from its wiki claim is a red
gate, in either direction.

Rules for this package (see ``wiki/pages/physics-module-roadmap.md``):

* Pure functions and constants only — no I/O, no world state, no
  imports from ``bot``/``state``/``sniffer``.
* Every symbol's docstring names its wiki page and claim id.
* New game facts land here first; planners and selectors import from
  here and never restate the number locally.
"""

from __future__ import annotations

from tankpit_bot.physics.capacity import (
    DEPOSIT_FLOOR,
    free_radar_radius,
    fuel_capacity,
    inventory_capacity,
)
from tankpit_bot.physics.combat import REROUTE_TTL_MS
from tankpit_bot.physics.costs import (
    BLOCK_OP_COST,
    DUAL_SHOT_COST,
    HOMING_SHOT_COST,
    MINE_PRESS_COST,
    MISSILE_SHOT_COST,
    RADAR_COST,
    SINGLE_SHOT_COST,
    WALK_COST_PER_TILE,
    teleport_cost,
)
from tankpit_bot.physics.damage import (
    DUAL_HIT_VICTIM_COST,
    MINE_DETONATION_COST,
    SINGLE_HIT_VICTIM_COST,
)

__all__ = [
    "BLOCK_OP_COST",
    "DEPOSIT_FLOOR",
    "DUAL_HIT_VICTIM_COST",
    "DUAL_SHOT_COST",
    "HOMING_SHOT_COST",
    "MINE_DETONATION_COST",
    "MINE_PRESS_COST",
    "MISSILE_SHOT_COST",
    "RADAR_COST",
    "REROUTE_TTL_MS",
    "SINGLE_HIT_VICTIM_COST",
    "SINGLE_SHOT_COST",
    "WALK_COST_PER_TILE",
    "free_radar_radius",
    "fuel_capacity",
    "inventory_capacity",
    "teleport_cost",
]
