"""Law 3 — queue-model shot resolution (wiki [[shoot-event-format]],
[[weapon-selection]], [[game-economy]]).

A shot resolves against the TILE state at processing time — there is
no range mechanic. The server picks the weapon (dual by default
against an enemy, homing when the enemy moved this same tick, missile
only against an obstructed enemy), clips non-missile shots at the
first blocking tile, applies the measured damage table with armor
absorption, bills the victim instantly, and defers the shooter's
firing cost to the next tick (charge latency).
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.physics.costs import (
    DUAL_SHOT_COST,
    HOMING_SHOT_COST,
    MISSILE_SHOT_COST,
    SINGLE_SHOT_COST,
)
from tankpit_bot.physics.damage import (
    ARMOR_ABSORB_PER_SHIELD,
    DUAL_HIT_VICTIM_COST,
    HOMING_HIT_VICTIM_COST,
    MISSILE_HIT_VICTIM_COST,
    SINGLE_HIT_VICTIM_COST,
)
from tankpit_bot.sim.world import SimTankDict, SimWorldDict

WEAPON_SINGLE = 0
WEAPON_DUAL = 1
WEAPON_MISSILE = 2
WEAPON_HOMING = 3

# Equipment slot indexes on the 0x49 wire ([armor, dual, missile,
# homing, radar]).
SLOT_ARMOR = 0
SLOT_DUAL = 1
SLOT_MISSILE = 2
SLOT_HOMING = 3
SLOT_RADAR = 4

_FIRING_COSTS: dict[int, int] = {
    WEAPON_SINGLE: SINGLE_SHOT_COST,
    WEAPON_DUAL: DUAL_SHOT_COST,
    WEAPON_MISSILE: MISSILE_SHOT_COST,
    WEAPON_HOMING: HOMING_SHOT_COST,
}
_VICTIM_COSTS: dict[int, int] = {
    WEAPON_SINGLE: SINGLE_HIT_VICTIM_COST,
    WEAPON_DUAL: DUAL_HIT_VICTIM_COST,
    WEAPON_MISSILE: MISSILE_HIT_VICTIM_COST,
    WEAPON_HOMING: HOMING_HIT_VICTIM_COST,
}
_AMMO_SLOT: dict[int, int] = {
    WEAPON_DUAL: SLOT_DUAL,
    WEAPON_MISSILE: SLOT_MISSILE,
    WEAPON_HOMING: SLOT_HOMING,
}

# Wire damage_state counts DOWN toward deactivation: 0 (full) -> 3 ->
# 2 -> 1 (critical). Every observed kill died from tier 1.
_DAMAGE_PROGRESSION: dict[int, int] = {0: 3, 3: 2, 2: 1, 1: 1}


class ShotOutcomeDict(TypedDict):
    """Everything one processed shot changed.

    ``impact_x/y`` is the resolved tile (obstruction-clipped for
    non-missile shots); ``victim_id`` is the living enemy hit there,
    or None. ``mine_cascade`` carries up to two detonation packets:
    the directly-shot mine, then its adjacent chain. ``shooter_debit``
    is the firing cost the server bills at the NEXT tick.
    """

    shooter_id: int
    weapon: int
    source_x: int
    source_y: int
    aim_x: int
    aim_y: int
    impact_x: int
    impact_y: int
    victim_id: int | None
    victim_deactivated: bool
    shields_consumed: int
    mine_cascade: list[list[tuple[int, int]]]
    shooter_debit: int
    ammo_slot: int | None
    kind: Literal["shot"]


def _living_enemy_at(
    world: SimWorldDict, shooter: SimTankDict, x: int, y: int
) -> SimTankDict | None:
    """Return the living enemy tank on a tile, if any.

    Args:
        world: Simulated world.
        shooter: The firing tank.
        x: Tile X.
        y: Tile Y.

    Returns:
        The enemy tank at (x, y), or None.
    """
    for tank in world["tanks"].values():
        if tank["alive"] and tank["team"] != shooter["team"] and (tank["x"], tank["y"]) == (x, y):
            return tank
    return None


def _ray_tiles(sx: int, sy: int, tx: int, ty: int) -> list[tuple[int, int]]:
    """Tiles crossed from source (exclusive) to target (inclusive).

    Integer line walk (Bresenham): the projectile visits each tile on
    the straight ray toward the click.

    Args:
        sx: Source X.
        sy: Source Y.
        tx: Target X.
        ty: Target Y.

    Returns:
        Ray tiles in flight order.
    """
    tiles: list[tuple[int, int]] = []
    dx = abs(tx - sx)
    dy = abs(ty - sy)
    step_x = 1 if tx > sx else -1
    step_y = 1 if ty > sy else -1
    error = dx - dy
    x, y = sx, sy
    while (x, y) != (tx, ty):
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x += step_x
        if doubled < dx:
            error += dx
            y += step_y
        tiles.append((x, y))
    return tiles


def _clip_impact(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    shooter: SimTankDict,
    target_x: int,
    target_y: int,
) -> tuple[int, int, bool]:
    """Resolve a non-missile shot's impact tile along the ray.

    Args:
        world: Simulated world.
        terrain: Static terrain.
        shooter: The firing tank.
        target_x: Clicked tile X.
        target_y: Clicked tile Y.

    Returns:
        ``(impact_x, impact_y, obstructed)`` — the first blocking tile
        (rock terrain or any tank) or the click tile when the line of
        sight is clear. Water is NOT an obstruction.
    """
    for x, y in _ray_tiles(shooter["x"], shooter["y"], target_x, target_y):
        if (x, y) == (target_x, target_y):
            break
        if terrain.get_terrain(x, y) == terrain.ROCK:
            return x, y, True
        for tank in world["tanks"].values():
            if (
                tank["alive"]
                and tank["tank_id"] != shooter["tank_id"]
                and (tank["x"], tank["y"]) == (x, y)
            ):
                return x, y, True
    return target_x, target_y, False


def _slot_ready(tank: SimTankDict, slot: int) -> bool:
    """Report whether an equipment slot is enabled with rounds left.

    Args:
        tank: The tank to inspect.
        slot: Equipment slot index.

    Returns:
        True when the slot is enabled and has at least one round.
    """
    return tank["enabled"][slot] and tank["counts"][slot] > 0


def _select_weapon(
    shooter: SimTankDict,
    enemy_at_click: SimTankDict | None,
    obstructed: bool,
    enemy_moved_this_tick: bool,
) -> int:
    """Server-side weapon selection (wiki [[weapon-selection]]).

    Args:
        shooter: The firing tank.
        enemy_at_click: Living enemy on the clicked tile, if any.
        obstructed: Whether the line of sight to the click is blocked.
        enemy_moved_this_tick: Whether that enemy moved this tick.

    Returns:
        The weapon byte the server fires.
    """
    if enemy_at_click is None:
        return WEAPON_SINGLE
    if obstructed:
        if _slot_ready(shooter, SLOT_MISSILE):
            return WEAPON_MISSILE
        return WEAPON_SINGLE
    if enemy_moved_this_tick and _slot_ready(shooter, SLOT_HOMING):
        return WEAPON_HOMING
    if _slot_ready(shooter, SLOT_DUAL):
        return WEAPON_DUAL
    return WEAPON_SINGLE


def _apply_hit(victim: SimTankDict, weapon: int, outcome: ShotOutcomeDict) -> None:
    """Apply the measured damage table to a hit victim.

    Armor fully absorbs damage at one shield per 45 while shields are
    enabled and available; otherwise the victim's fuel drops by the
    weapon's victim cost, the damage tier advances, and the tank
    deactivates at zero fuel.

    Args:
        victim: The tank on the impact tile (mutated).
        weapon: Firing weapon byte.
        outcome: Outcome being built (mutated).
    """
    damage = _VICTIM_COSTS[weapon]
    shields_needed = damage // ARMOR_ABSORB_PER_SHIELD
    if victim["enabled"][SLOT_ARMOR] and victim["counts"][SLOT_ARMOR] >= shields_needed:
        victim["counts"][SLOT_ARMOR] -= shields_needed
        outcome["shields_consumed"] = shields_needed
        return
    victim["fuel"] = max(0, victim["fuel"] - damage)
    victim["damage_state"] = _DAMAGE_PROGRESSION[victim["damage_state"]]
    if victim["fuel"] == 0:
        victim["alive"] = False
        outcome["victim_deactivated"] = True


def _detonate_mines(world: SimWorldDict, x: int, y: int, outcome: ShotOutcomeDict) -> None:
    """Detonate a shot mine and its adjacent chain (two 0x45 packets).

    Args:
        world: Simulated world (mutated).
        x: Impact tile X.
        y: Impact tile Y.
        outcome: Outcome being built (mutated).
    """
    direct = [mine for mine in world["mines"] if (mine["x"], mine["y"]) == (x, y)]
    if not direct:
        return
    for mine in direct:
        world["mines"].remove(mine)
    chain = [
        mine for mine in list(world["mines"]) if abs(mine["x"] - x) <= 1 and abs(mine["y"] - y) <= 1
    ]
    for mine in chain:
        world["mines"].remove(mine)
    outcome["mine_cascade"].append([(x, y)])
    if chain:
        outcome["mine_cascade"].append([(mine["x"], mine["y"]) for mine in chain])


def process_shot(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    shooter_id: int,
    target_x: int,
    target_y: int,
    moved_this_tick: frozenset[int],
) -> ShotOutcomeDict:
    """Process one shoot command at the current tick.

    Args:
        world: Simulated world (mutated).
        terrain: Static terrain of the world's field.
        shooter_id: The firing tank (must exist and be alive).
        target_x: Clicked tile X.
        target_y: Clicked tile Y.
        moved_this_tick: Ids of tanks whose move commands processed
            earlier in this same tick (drives homing selection).

    Returns:
        The typed outcome; the world reflects it. The firing cost is
        NOT yet billed — the server applies ``shooter_debit`` at the
        next tick (measured charge latency).
    """
    shooter = world["tanks"][shooter_id]
    enemy_at_click = _living_enemy_at(world, shooter, target_x, target_y)
    impact_x, impact_y, obstructed = _clip_impact(world, terrain, shooter, target_x, target_y)
    weapon = _select_weapon(
        shooter,
        enemy_at_click,
        obstructed,
        enemy_at_click is not None and enemy_at_click["tank_id"] in moved_this_tick,
    )
    if weapon in (WEAPON_MISSILE, WEAPON_HOMING):
        impact_x, impact_y = target_x, target_y
    outcome = ShotOutcomeDict(
        shooter_id=shooter_id,
        weapon=weapon,
        source_x=shooter["x"],
        source_y=shooter["y"],
        aim_x=target_x,
        aim_y=target_y,
        impact_x=impact_x,
        impact_y=impact_y,
        victim_id=None,
        victim_deactivated=False,
        shields_consumed=0,
        mine_cascade=[],
        shooter_debit=_FIRING_COSTS[weapon],
        ammo_slot=None,
        kind="shot",
    )
    victim = _living_enemy_at(world, shooter, impact_x, impact_y)
    if victim is not None:
        outcome["victim_id"] = victim["tank_id"]
        _apply_hit(victim, weapon, outcome)
        ammo_slot = _AMMO_SLOT.get(weapon)
        if ammo_slot is not None:
            shooter["counts"][ammo_slot] = max(0, shooter["counts"][ammo_slot] - 1)
            outcome["ammo_slot"] = ammo_slot
    else:
        _detonate_mines(world, impact_x, impact_y, outcome)
    return outcome


__all__ = [
    "SLOT_ARMOR",
    "SLOT_DUAL",
    "SLOT_HOMING",
    "SLOT_MISSILE",
    "SLOT_RADAR",
    "WEAPON_DUAL",
    "WEAPON_HOMING",
    "WEAPON_MISSILE",
    "WEAPON_SINGLE",
    "ShotOutcomeDict",
    "process_shot",
]
