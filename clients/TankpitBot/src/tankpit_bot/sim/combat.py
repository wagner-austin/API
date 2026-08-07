"""Laws 3 and 4 — queue-model shot resolution and homing reroute
(wiki [[shoot-event-format]], [[weapon-selection]], [[game-economy]]).

A shot resolves against the TILE state at processing time — there is
no range mechanic. The server picks the weapon (dual by default
against an enemy, homing when the enemy moved this same tick, missile
only against an obstructed enemy), clips non-missile shots at the
first blocking tile, applies the measured damage table with armor
absorption, bills the victim instantly, and defers the shooter's
firing cost to the next tick (charge latency).

Law 4 adds id-targeted resolution: an id-carrying shot follows the
tank, not the clicked tile. A visible target reroutes the click to
its current position (the queue-race conversion — a same-tick mover
draws homing instead of a miss); a DEPARTED target (0x58 emitted)
keeps drawing guaranteed homing hits until the measured reroute TTL
(``physics.combat.REROUTE_TTL_MS`` = 12,920 ms; corpus boundary
[12.91, 12.93] s fire-time, 2026-07-22 sweep, confirmed live by run
bot-20260726-194658: hits to +12.0 s, miss at +14.0 s), after which
the id no longer resolves and the shot is a free single miss.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.physics.combat import REROUTE_TTL_MS
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
from tankpit_bot.sim.blocks import BLOCK_LAND, BLOCK_STACKED, block_tile_value
from tankpit_bot.sim.world import SimTankDict, SimWorldDict
from tankpit_bot.state.scan_coverage import tile_key

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
        (rock terrain, a land/stacked movable block, or any tank) or
        the click tile when the line of sight is clear. Water is NOT
        an obstruction, and neither is a flat water-block bridge
        ([[weapon-selection]]: blocks obstruct non-missile shots;
        the bridge-at-water-level exemption is a sim assumption).
    """
    for x, y in _ray_tiles(shooter["x"], shooter["y"], target_x, target_y):
        if (x, y) == (target_x, target_y):
            break
        if terrain.get_terrain(x, y) == terrain.ROCK:
            return x, y, True
        if block_tile_value(world, terrain, x, y) in (BLOCK_LAND, BLOCK_STACKED):
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
    weapon's victim cost and the tank deactivates at zero fuel. The
    damage tier is NOT stored state — it is the fuel quartile
    (``physics.damage_tier``, corpus-fitted 2026-07-23), derived at
    every emission point.

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
    if world["mines"].pop(tile_key(x, y), None) is None:
        return
    # The cascade is the eight neighbours, read by key rather than by
    # sweeping the field — the minefield is dense enough that a sweep
    # per impact dominated the tick ([[session-state-deglobalisation]]).
    chain: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbour = world["mines"].pop(tile_key(x + dx, y + dy), None)
            if neighbour is not None:
                chain.append((neighbour["x"], neighbour["y"]))
    outcome["mine_cascade"].append([(x, y)])
    if chain:
        outcome["mine_cascade"].append(chain)


def _reroute_departed(
    shooter: SimTankDict,
    target: SimTankDict,
    target_x: int,
    target_y: int,
    departed_age_ms: int,
) -> ShotOutcomeDict:
    """Resolve an id-shot at a departed tank (law 4).

    Within the measured TTL the server keeps rerouting: the shot fires
    as a guaranteed homing hit (ammo debited = hit, per the
    consumption-equals-hit contract) even though the target's position
    is dark. Past the TTL the id no longer resolves — a free single
    with nothing debited, the measured genuine miss. A shooter without
    a ready homing slot cannot reroute either (the human analogue
    needs homing enabled).

    Args:
        shooter: The firing tank (mutated on ammo debit).
        target: The departed tank (mutated on hit).
        target_x: Clicked tile X (the stale aim — position is dark).
        target_y: Clicked tile Y.
        departed_age_ms: Milliseconds since the target's 0x58.

    Returns:
        The typed outcome.
    """
    rerouted = departed_age_ms <= REROUTE_TTL_MS and _slot_ready(shooter, SLOT_HOMING)
    weapon = WEAPON_HOMING if rerouted else WEAPON_SINGLE
    outcome = ShotOutcomeDict(
        shooter_id=shooter["tank_id"],
        weapon=weapon,
        source_x=shooter["x"],
        source_y=shooter["y"],
        aim_x=target_x,
        aim_y=target_y,
        impact_x=target_x,
        impact_y=target_y,
        victim_id=None,
        victim_deactivated=False,
        shields_consumed=0,
        mine_cascade=[],
        shooter_debit=_FIRING_COSTS[weapon],
        ammo_slot=None,
        kind="shot",
    )
    if rerouted:
        outcome["victim_id"] = target["tank_id"]
        _apply_hit(target, WEAPON_HOMING, outcome)
        shooter["counts"][SLOT_HOMING] = max(0, shooter["counts"][SLOT_HOMING] - 1)
        outcome["ammo_slot"] = SLOT_HOMING
    return outcome


def process_shot(
    world: SimWorldDict,
    terrain: TerrainMapProtocol,
    shooter_id: int,
    target_x: int,
    target_y: int,
    moved_this_tick: frozenset[int],
    target_id: int,
    departed_age_ms: int | None,
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
        target_id: The shot's entity id (0 = positional shot). An
            id-shot at a living VISIBLE tank reroutes the click to the
            tank's current tile before positional resolution — the
            law-4 queue-race conversion.
        departed_age_ms: Milliseconds since the target's 0x58
            TankRemove, or None when the target has not departed —
            drives the reroute-TTL path for id-shots.

    Returns:
        The typed outcome; the world reflects it. The firing cost is
        NOT yet billed — the server applies ``shooter_debit`` at the
        next tick (measured charge latency).
    """
    shooter = world["tanks"][shooter_id]
    target = world["tanks"].get(target_id) if target_id != 0 else None
    if target is not None and target["alive"]:
        if departed_age_ms is not None:
            return _reroute_departed(shooter, target, target_x, target_y, departed_age_ms)
        target_x, target_y = target["x"], target["y"]
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
