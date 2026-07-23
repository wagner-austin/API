"""Law 3 — queue-model shots: selection, clipping, damage, mines."""

from __future__ import annotations

from tankpit_bot.sim.combat import (
    SLOT_ARMOR,
    SLOT_DUAL,
    SLOT_HOMING,
    SLOT_MISSILE,
    WEAPON_DUAL,
    WEAPON_HOMING,
    WEAPON_MISSILE,
    WEAPON_SINGLE,
    process_shot,
)
from tankpit_bot.sim.world import SimMineDict, SimWorldDict, make_sim_tank, make_sim_world
from tests.in_memory_terrain_map import InMemoryTerrainMap

_NOBODY: frozenset[int] = frozenset()


def _arena() -> SimWorldDict:
    """Shooter 9 (team 0) at (10, 10), enemy 11 (team 1) at (15, 10)."""
    world = make_sim_world("field01_r.gif")
    world["tanks"][9] = make_sim_tank(9, 0, 1, 10, 10, 1000)
    world["tanks"][11] = make_sim_tank(11, 1, 1, 15, 10, 500)
    return world


def test_ground_shot_is_a_single_with_full_deferred_cost() -> None:
    """No enemy at the click: weapon 0, impact at the click, debit 6."""
    world = _arena()
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 12, 12, _NOBODY, 0, None)
    assert outcome["weapon"] == WEAPON_SINGLE
    assert (outcome["impact_x"], outcome["impact_y"]) == (12, 12)
    assert outcome["shooter_debit"] == 6
    assert world["tanks"][9]["fuel"] == 1000


def test_terrain_clips_the_impact_tile_and_still_bills() -> None:
    """A mountain on the ray stops the shot at the mountain tile."""
    world = _arena()
    terrain = InMemoryTerrainMap(terrain_data={(12, 10): "#"})
    outcome = process_shot(world, terrain, 9, 15, 10, _NOBODY, 0, None)
    assert outcome["weapon"] == WEAPON_SINGLE
    assert (outcome["impact_x"], outcome["impact_y"]) == (12, 10)
    assert outcome["shooter_debit"] == 6
    assert world["tanks"][11]["fuel"] == 500


def test_obstructed_enemy_with_missiles_ready_fires_missile() -> None:
    """Missiles fly over terrain and hit the clicked enemy for 45."""
    world = _arena()
    world["tanks"][9]["counts"][SLOT_MISSILE] = 3
    terrain = InMemoryTerrainMap(terrain_data={(12, 10): "#"})
    outcome = process_shot(world, terrain, 9, 15, 10, _NOBODY, 0, None)
    assert outcome["weapon"] == WEAPON_MISSILE
    assert (outcome["impact_x"], outcome["impact_y"]) == (15, 10)
    assert outcome["victim_id"] == 11
    assert world["tanks"][11]["fuel"] == 455
    assert world["tanks"][9]["counts"][SLOT_MISSILE] == 2
    assert outcome["ammo_slot"] == SLOT_MISSILE


def test_obstructed_enemy_without_missiles_clips_to_terrain() -> None:
    """Missiles disabled: the shot degrades to a clipped single."""
    world = _arena()
    terrain = InMemoryTerrainMap(terrain_data={(12, 10): "#"})
    outcome = process_shot(world, terrain, 9, 15, 10, _NOBODY, 0, None)
    assert outcome["weapon"] == WEAPON_SINGLE
    assert (outcome["impact_x"], outcome["impact_y"]) == (12, 10)
    assert outcome["victim_id"] is None


def test_tank_in_the_line_of_sight_obstructs() -> None:
    """Any tank on the ray blocks a non-missile shot."""
    world = _arena()
    world["tanks"][12] = make_sim_tank(12, 0, 1, 12, 10, 500)
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 0, None)
    assert outcome["weapon"] == WEAPON_SINGLE
    assert (outcome["impact_x"], outcome["impact_y"]) == (12, 10)


def test_stationary_enemy_takes_a_dual_when_duals_ready() -> None:
    """Clear line, stationary enemy, duals loaded: weapon 1 for 90."""
    world = _arena()
    world["tanks"][9]["counts"][SLOT_DUAL] = 5
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 0, None)
    assert outcome["weapon"] == WEAPON_DUAL
    assert world["tanks"][11]["fuel"] == 500 - 90
    assert world["tanks"][9]["counts"][SLOT_DUAL] == 4


def test_enemy_without_duals_takes_a_single() -> None:
    """No dual rounds: the default degrades to a 45 single."""
    world = _arena()
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 0, None)
    assert outcome["weapon"] == WEAPON_SINGLE
    assert world["tanks"][11]["fuel"] == 455


def test_same_tick_mover_draws_a_homing() -> None:
    """An enemy that moved this tick is hit by homing when loaded."""
    world = _arena()
    world["tanks"][9]["counts"][SLOT_HOMING] = 2
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, frozenset({11}), 0, None)
    assert outcome["weapon"] == WEAPON_HOMING
    assert world["tanks"][11]["fuel"] == 455
    without = _arena()
    fallback = process_shot(without, InMemoryTerrainMap(), 9, 15, 10, frozenset({11}), 0, None)
    assert fallback["weapon"] == WEAPON_SINGLE


def test_armor_fully_absorbs_at_one_shield_per_45() -> None:
    """Shields eat the hit — 2 for a dual — and fuel stays put."""
    world = _arena()
    world["tanks"][9]["counts"][SLOT_DUAL] = 1
    world["tanks"][11]["counts"][SLOT_ARMOR] = 5
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 0, None)
    assert outcome["shields_consumed"] == 2
    assert world["tanks"][11]["counts"][SLOT_ARMOR] == 3
    assert world["tanks"][11]["fuel"] == 500


def test_deactivation_at_zero_fuel() -> None:
    """Fuel is the health pool: the tank dies when it reaches zero."""
    world = _arena()
    world["tanks"][11]["fuel"] = 90
    first = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 0, None)
    assert first["victim_deactivated"] is False
    assert world["tanks"][11]["fuel"] == 45
    second = process_shot(world, InMemoryTerrainMap(), 9, 15, 10, _NOBODY, 0, None)
    assert second["victim_deactivated"] is True
    assert world["tanks"][11]["alive"] is False
    assert world["tanks"][11]["fuel"] == 0


def test_shooting_a_mine_cascades_two_packets() -> None:
    """The shot mine and its adjacent chain arrive as two packets."""
    world = _arena()
    world["mines"] = [
        SimMineDict(x=13, y=12, team=1),
        SimMineDict(x=14, y=12, team=1),
        SimMineDict(x=13, y=13, team=1),
        SimMineDict(x=20, y=20, team=1),
    ]
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 13, 12, _NOBODY, 0, None)
    assert outcome["mine_cascade"] == [[(13, 12)], [(14, 12), (13, 13)]]
    assert world["mines"] == [SimMineDict(x=20, y=20, team=1)]


def test_lone_mine_is_one_packet() -> None:
    """A solo mine detonates without a chain packet."""
    world = _arena()
    world["mines"] = [SimMineDict(x=13, y=12, team=1)]
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 13, 12, _NOBODY, 0, None)
    assert outcome["mine_cascade"] == [[(13, 12)]]


def test_pure_vertical_ray_resolves() -> None:
    """A straight-down shot walks its ray without horizontal steps."""
    world = _arena()
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 10, 14, _NOBODY, 0, None)
    assert (outcome["impact_x"], outcome["impact_y"]) == (10, 14)
    assert outcome["weapon"] == WEAPON_SINGLE


def test_shot_at_own_tile_has_an_empty_ray() -> None:
    """Clicking the shooter's own tile resolves there with no clipping."""
    world = _arena()
    outcome = process_shot(world, InMemoryTerrainMap(), 9, 10, 10, _NOBODY, 0, None)
    assert (outcome["impact_x"], outcome["impact_y"]) == (10, 10)
    assert outcome["victim_id"] is None
    assert outcome["mine_cascade"] == []
