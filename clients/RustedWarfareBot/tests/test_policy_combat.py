"""Choosing what to attack, exercised as the pure function it is.

No socket and no game: a world state goes in and a set of engagements comes
out, which is what keeps the fighting logic arguable without a match running.
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats, Weapon
from rw_bot.policy.combat import (
    choose_target,
    engagements,
    find_army,
    find_targets,
    is_armed,
    is_mobile,
)
from rw_bot.wire.state import Entity, Sample


def _weapon(damage: float = 17.0, reach: float = 110.0) -> Weapon:
    return Weapon(
        shoot_delay=50.0,
        attack_range=reach,
        direct_damage=damage,
        direct_damage_volley=damage,
        area_damage=0.0,
        area_damage_volley=0.0,
    )


def _unit(
    type_name: str,
    *,
    speed: float = 1.0,
    armed: bool = True,
) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=350,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=_weapon() if armed else None,
    )


_CATALOGUE = {
    "c_tank": _unit("c_tank"),
    "builder": _unit("builder", speed=0.6, armed=False),
    "commandCenter": _unit("commandCenter", speed=0.0, armed=False),
    "c_turret_t1": _unit("c_turret_t1", speed=0.0),
    "editorOrBuilder": _unit("editorOrBuilder", speed=0.0, armed=False),
}


def _entity(
    unit_id: int,
    type_name: str,
    x: float = 0.0,
    y: float = 0.0,
    *,
    mine: bool = True,
    hostile: bool = False,
    complete: bool = True,
) -> Entity:
    return Entity(
        index=0,
        unit_id=unit_id,
        type_name=type_name,
        class_name="units.x",
        x=x,
        y=y,
        team=0 if mine else 1,
        mine=mine,
        hostile=hostile,
        movement="LAND",
        group=1,
        hp=100.0,
        max_hp=100.0,
        complete=complete,
        queued=0,
    )


def _sample(*entities: Entity) -> Sample:
    return Sample(
        frame=1,
        clock_ms=10,
        credits=4000,
        entities=tuple(entities),
        pools=(),
        options=(),
    )


def test_an_armed_mobile_unit_is_army() -> None:
    tank = _entity(1, "c_tank")
    assert find_army(_sample(tank), _CATALOGUE) == (tank,)


def test_a_builder_is_not_army() -> None:
    """Unarmed. Sending it at a tank is a Builder thrown away."""
    assert find_army(_sample(_entity(1, "builder")), _CATALOGUE) == ()
    assert is_armed(_entity(1, "builder"), _CATALOGUE) is False


def test_a_turret_is_armed_but_not_army() -> None:
    """It cannot travel, so an order to attack anything distant is undeliverable."""
    turret = _entity(1, "c_turret_t1")
    assert is_armed(turret, _CATALOGUE) is True
    assert is_mobile(turret, _CATALOGUE) is False
    assert find_army(_sample(turret), _CATALOGUE) == ()


def test_the_editor_placeholder_is_never_army() -> None:
    """Owned, off-map, and not a unit -- the same exclusion producer selection needs."""
    assert find_army(_sample(_entity(217, "editorOrBuilder")), _CATALOGUE) == ()


def test_an_unfinished_tank_is_not_army() -> None:
    """It does not exist yet, whatever the roster says."""
    assert find_army(_sample(_entity(1, "c_tank", complete=False)), _CATALOGUE) == ()


def test_an_enemy_tank_is_not_our_army() -> None:
    enemy = _entity(9, "c_tank", mine=False, hostile=True)
    assert find_army(_sample(enemy), _CATALOGUE) == ()


def test_a_type_the_catalogue_does_not_know_is_not_army() -> None:
    """Unknown weapon means unknown fight; the safe answer is to stay home."""
    assert find_army(_sample(_entity(1, "mysteryTank")), _CATALOGUE) == ()
    assert is_mobile(_entity(1, "mysteryTank"), _CATALOGUE) is False


def test_targets_are_the_engines_hostiles_not_merely_the_unowned() -> None:
    """An ally is not mine and is not a target, which "not mine" gets wrong."""
    ally = _entity(5, "c_tank", mine=False, hostile=False)
    enemy = _entity(9, "c_tank", mine=False, hostile=True)
    assert find_targets(_sample(ally, enemy)) == (enemy,)


def test_the_whole_army_commits_to_one_target() -> None:
    """Concentrating fire is the one tactic that matters at this scale."""
    near = _entity(9, "c_tank", 100.0, 0.0, mine=False, hostile=True)
    far = _entity(10, "c_tank", 900.0, 0.0, mine=False, hostile=True)
    world = _sample(
        _entity(1, "c_tank", 0.0, 0.0),
        _entity(2, "c_tank", 10.0, 0.0),
        near,
        far,
    )
    orders = engagements(world, _CATALOGUE)
    assert [e["attacker_id"] for e in orders] == [1, 2]
    assert {e["target_id"] for e in orders} == {9}


def test_the_target_is_nearest_to_the_army_not_to_one_unit() -> None:
    """A split force converges instead of each unit picking its own enemy."""
    army = (_entity(1, "c_tank", 0.0, 0.0), _entity(2, "c_tank", 1000.0, 0.0))
    near_to_one = _entity(9, "c_tank", 0.0, 400.0, mine=False, hostile=True)
    near_to_centre = _entity(10, "c_tank", 500.0, 0.0, mine=False, hostile=True)
    assert choose_target(army, (near_to_one, near_to_centre)) == near_to_centre


def test_the_current_target_is_kept_while_it_lives() -> None:
    """Commitment, and the whole reason the churn happened.

    Nearest is measured from the army centre, and that centre shifts whenever a
    unit dies or a new one rolls out. Re-choosing every sample re-tasked the
    whole army on a flip that could be a few world units wide.
    """
    held = _entity(9, "c_tank", 900.0, 0.0, mine=False, hostile=True)
    nearer = _entity(10, "c_tank", 10.0, 0.0, mine=False, hostile=True)
    army = (_entity(1, "c_tank", 0.0, 0.0),)
    assert choose_target(army, (held, nearer), holding=9) == held
    assert choose_target(army, (held, nearer)) == nearer


def test_a_target_that_is_gone_is_replaced() -> None:
    """Holding is not clinging: a dead target frees the army to re-commit."""
    nearer = _entity(10, "c_tank", 10.0, 0.0, mine=False, hostile=True)
    army = (_entity(1, "c_tank", 0.0, 0.0),)
    assert choose_target(army, (nearer,), holding=9) == nearer


def test_the_held_target_carries_through_engagements() -> None:
    world = _sample(
        _entity(1, "c_tank", 0.0, 0.0),
        _entity(9, "c_tank", 900.0, 0.0, mine=False, hostile=True),
        _entity(10, "c_tank", 10.0, 0.0, mine=False, hostile=True),
    )
    assert engagements(world, _CATALOGUE, holding=9)[0]["target_id"] == 9
    assert engagements(world, _CATALOGUE)[0]["target_id"] == 10


def test_no_army_produces_no_orders() -> None:
    enemy = _entity(9, "c_tank", mine=False, hostile=True)
    assert engagements(_sample(enemy), _CATALOGUE) == ()
    assert choose_target((), (enemy,)) is None


def test_no_enemy_produces_no_orders() -> None:
    tank = _entity(1, "c_tank")
    assert engagements(_sample(tank), _CATALOGUE) == ()
    assert choose_target((tank,), ()) is None


def test_the_engagement_reason_names_both_sides() -> None:
    """The run log has to be readable back without re-deriving the choice."""
    world = _sample(
        _entity(1, "c_tank", 0.0, 0.0),
        _entity(9, "commandCenter", 50.0, 0.0, mine=False, hostile=True),
    )
    assert engagements(world, _CATALOGUE)[0]["reason"] == "c_tank -> commandCenter 9"
