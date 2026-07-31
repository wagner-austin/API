"""The raid, exercised without a match running.

What is tested: the objective is the frontier extractor and only income
qualifies, a party is drafted whole from the gathered or not at all, survivors
below strength disband and fight their way home, a confirmed-dead objective is
reported to the memory, and no order is ever re-sent while its pairing holds
([[issuing-orders]]).

The v2 rules carry the v1 refutation's weight: one-at-a-time replacement was
an attrition conveyor that fed lone units across the map forever, and the arm
lost every seat of its batch for it (log: 2026-07-29).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.intel import Intel
from rw_bot.policy.raid import INCOME_TYPES, Raider, income_objectives
from rw_bot.wire.state import Entity, Sample
from tests.wire_fixtures import enemy, entity, sample


def _stats(type_name: str, speed: float) -> UnitStats:
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
        weapon=None,
    )


_CATALOGUE: dict[str, UnitStats] = {
    "commandCenter": _stats("commandCenter", 0.0),
    "c_tank": _stats("c_tank", 1.1),
}

_CENTRE = entity(1, "commandCenter", x=0.0, y=0.0)


def _tank(unit_id: int, x: float = 50.0, y: float = 0.0) -> Entity:
    return entity(unit_id, "c_tank", x=x, y=y)


def _seen(intel: Intel, *hostiles: Entity, frame: int = 100) -> Intel:
    intel.observe(sample(*hostiles, frame=frame))
    return intel


def _world(*extra: Entity) -> Sample:
    return sample(_CENTRE, *extra)


def test_only_income_is_an_objective() -> None:
    """Raiding the army is the waves' job; raiding defences is what waves die to."""
    intel = _seen(
        Intel(),
        enemy(9, "extractorT1", x=900.0),
        enemy(10, "c_turret_t1", x=300.0),
        enemy(11, "c_tank", x=200.0),
    )
    assert [s["unit_id"] for s in income_objectives(intel)] == [9]
    assert all(t.startswith("extractor") for t in INCOME_TYPES)


def test_the_frontier_extractor_is_assaulted_first() -> None:
    """Nearest to our anchor: reachable before the deep ones."""
    intel = _seen(Intel(), enemy(9, "extractorT1", x=2000.0), enemy(8, "extractorT2", x=900.0))
    raider = Raider(size=2)
    army = (_tank(20), _tank(21), _tank(22))
    orders = raider.strike(_world(*army), intel, army, _CATALOGUE, True)
    assert [(o["unit_id"], o["x"]) for o in orders] == [(20, 900.0), (21, 900.0)]
    assert raider.party() == frozenset({20, 21})
    assert raider.raids == 1
    assert raider.marches == 2


def test_orders_are_not_resent_while_the_objective_holds() -> None:
    intel = _seen(Intel(), enemy(9, "extractorT1", x=900.0))
    raider = Raider(size=2)
    army = (_tank(20), _tank(21))
    world = _world(*army)
    raider.strike(world, intel, army, _CATALOGUE, True)
    assert raider.strike(world, intel, army, _CATALOGUE, True) == ()
    assert raider.marches == 2


def test_survivors_below_strength_disband_and_fight_home() -> None:
    """A party reduced below the size that makes one is not one any more --
    the waves' own rule. V1 replaced the fallen one recruit at a time instead,
    and each crossed the map alone: the conveyor the refutation convicted.
    The road home crosses the same ground the road out did, so the survivors
    attack-move rather than walk."""
    intel = _seen(Intel(), enemy(9, "extractorT1", x=900.0))
    raider = Raider(size=2)
    army = (_tank(20), _tank(21), _tank(22))
    raider.strike(_world(*army), intel, army, _CATALOGUE, True)
    survivors = (_tank(21, x=600.0), _tank(22))
    orders = raider.strike(_world(*survivors), intel, survivors, _CATALOGUE, True)
    assert [(o["unit_id"], o["x"], o["y"]) for o in orders] == [(21, 0.0, 0.0)]
    assert raider.party() == frozenset()
    # Homeward orders are not marches; the party took two on the way out.
    assert raider.marches == 2


def test_a_fresh_party_is_drafted_whole_from_the_gathered() -> None:
    """A party starts together the way a wave does: only units already at the
    gathering ground qualify, so nobody forms up en route."""
    intel = _seen(Intel(), enemy(9, "extractorT1", x=900.0))
    raider = Raider(size=2)
    home = (_tank(21), _tank(22))
    orders = raider.strike(_world(*home), intel, home, _CATALOGUE, True)
    assert [o["unit_id"] for o in orders] == [21, 22]
    assert raider.party() == frozenset({21, 22})


def test_the_draft_takes_only_gathered_units() -> None:
    """A unit past the rally radius is somewhere on the map, not in formation;
    drafting it would rebuild the conveyor with extra steps."""
    intel = _seen(Intel(), enemy(9, "extractorT1", x=900.0))
    raider = Raider(size=2)
    spread = (_tank(20, x=500.0), _tank(21))
    assert raider.strike(_world(*spread), intel, spread, _CATALOGUE, True) == ()
    assert raider.party() == frozenset()
    gathered = (_tank(20, x=500.0), _tank(21), _tank(22, y=30.0))
    orders = raider.strike(_world(*gathered), intel, gathered, _CATALOGUE, True)
    assert [o["unit_id"] for o in orders] == [21, 22]


def test_the_draft_waits_for_the_campaigns_leave() -> None:
    """Whether the army can spare a party is the campaign's call, made against
    the wave gate's own figure; v1 never asked and drafted from the gate
    itself. A party already out is managed regardless -- the gate arbitrates
    drafting, not the raid in progress."""
    intel = _seen(Intel(), enemy(9, "extractorT1", x=900.0))
    raider = Raider(size=2)
    army = (_tank(20), _tank(21))
    assert raider.strike(_world(*army), intel, army, _CATALOGUE, False) == ()
    assert raider.party() == frozenset()
    raider.strike(_world(*army), intel, army, _CATALOGUE, True)
    assert raider.party() == frozenset({20, 21})
    # Leave withdrawn mid-raid: the party out is still the party.
    assert raider.strike(_world(*army), intel, army, _CATALOGUE, False) == ()
    assert raider.party() == frozenset({20, 21})


def test_a_raider_standing_on_a_ghost_reports_the_death() -> None:
    """The memory cannot see a kill; the raider standing where the sighting
    said, seeing nothing, is the confirmation.
    """
    intel = _seen(Intel(), enemy(9, "extractorT1", x=900.0))
    raider = Raider(size=1)
    afar = (_tank(20, x=50.0),)
    raider.strike(_world(*afar), intel, afar, _CATALOGUE, True)
    arrived = (_tank(20, x=890.0),)
    assert raider.strike(_world(*arrived), intel, arrived, _CATALOGUE, True) == ()
    assert income_objectives(intel) == ()
    # With nothing left remembered, the party disbands back to the waves.
    assert raider.strike(_world(*arrived), intel, arrived, _CATALOGUE, True) == ()
    assert raider.party() == frozenset()


def test_a_visible_objective_is_not_forgotten_on_arrival() -> None:
    """Standing next to a live extractor is the assault, not a confirmation."""
    target = enemy(9, "extractorT1", x=900.0)
    intel = _seen(Intel(), target)
    raider = Raider(size=1)
    home = (_tank(20),)
    raider.strike(_world(*home), intel, home, _CATALOGUE, True)
    arrived = (_tank(20, x=890.0),)
    raider.strike(_world(*arrived, target), intel, arrived, _CATALOGUE, True)
    assert [s["unit_id"] for s in income_objectives(intel)] == [9]


def test_no_memory_or_no_anchor_means_no_raid() -> None:
    raider = Raider(size=1)
    army = (_tank(20),)
    assert raider.strike(_world(*army), Intel(), army, _CATALOGUE, True) == ()
    intel = _seen(Intel(), enemy(9, "extractorT1", x=900.0))
    homeless = sample(*army)
    assert raider.strike(homeless, intel, army, _CATALOGUE, True) == ()


def test_an_empty_army_raids_nothing() -> None:
    intel = _seen(Intel(), enemy(9, "extractorT1", x=900.0))
    raider = Raider(size=2)
    assert raider.strike(_world(), intel, (), _CATALOGUE, True) == ()
    assert raider.party() == frozenset()
