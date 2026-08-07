"""The finisher channel: fund, stand, arm, and fire at what they value most.

Every rule here is one the live probes paid for: the 45,000 saves through
the withhold or it never accumulates, the launch flag lies about ammo so
launches are refired, and the warhead goes to the priciest structure in
sight because an 11,000-credit strike on a scout is a mistake
(`runs/nuke-probe*.out`, log 2026-08-05).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.budget import Budget
from rw_bot.policy.nuker import (
    ARM_RETRY_SAMPLES,
    LAUNCH_RETRY_SAMPLES,
    LAUNCHER_TYPE,
    Nuker,
    best_target,
)
from rw_bot.policy.siting import PLACEMENT_RING
from rw_bot.policy.workforce import EXPAND_RETRY_SAMPLES, Workforce
from rw_bot.wire.state import Sample
from tests.campaign_fixtures import unit_stats
from tests.wire_fixtures import enemy, entity, option, player, sample

CATALOGUE: dict[str, UnitStats] = {
    "commandCenter": unit_stats("commandCenter", speed=0.0, armed=False, price=0),
    "builder": unit_stats("builder", speed=0.6, armed=False, price=200),
    "c_tank": unit_stats("c_tank"),
    "landFactory": unit_stats("landFactory", speed=0.0, armed=False, price=1000),
    "commandCenterT2": unit_stats("commandCenterT2", speed=0.0, armed=False, price=6000),
    LAUNCHER_TYPE: unit_stats(LAUNCHER_TYPE, speed=0.0, armed=False, price=45000),
}

_CENTRE = entity(213, "commandCenter")
_BUILDER = entity(214, "builder", x=50.0)
_LAUNCHER = entity(500, LAUNCHER_TYPE, x=120.0)
_RISING = entity(500, LAUNCHER_TYPE, x=120.0, complete=False)

_PLACEABLE = option(214, LAUNCHER_TYPE, key="u_nuke", placed=True)
_BUILD_NUKE = option(500, "", key="c_buildNuke", index=1, price=11000)
_LAUNCH_NUKE = option(500, "", key="c_launchNuke", index=2, price=0)
_LAUNCH_GATED = option(500, "", key="c_launchNuke", index=2, price=0, available=False)


def _workforce() -> Workforce:
    return Workforce(EXPAND_RETRY_SAMPLES)


_EARNING = (
    player(0, index=0, local=True, hostile=False, income=60),
    player(1, index=1, income=60),
)
_STARVING = (
    player(0, index=0, local=True, hostile=False, income=34),
    player(1, index=1, income=60),
)


def _funding_world(credits_held: int) -> Sample:
    return sample(_CENTRE, _BUILDER, credits=credits_held, options=(_PLACEABLE,), players=_EARNING)


def test_the_doctrine_off_asks_for_nothing() -> None:
    nuker = Nuker()
    budget = Budget(90_000, reserve=0)
    orders = nuker.advance(
        _funding_world(90_000), CATALOGUE, budget, (_BUILDER,), _workforce(), 0, True
    )
    assert orders == {"build": None, "arm": None, "launch": None}
    assert budget.spent() == 0


def test_a_funded_launcher_is_placed_and_the_worker_assigned() -> None:
    nuker = Nuker()
    budget = Budget(90_000, reserve=0)
    workforce = _workforce()
    orders = nuker.advance(
        _funding_world(90_000), CATALOGUE, budget, (_BUILDER,), workforce, 1, True
    )
    build = orders["build"]
    if build is None:
        raise AssertionError("the funded launcher was not placed")
    assert build["type_name"] == LAUNCHER_TYPE
    assert build["unit_id"] == 214
    assert budget.spent() == 45_000
    assert workforce.claims() == ((build["x"], build["y"]),)


def test_a_refused_launcher_withholds_its_whole_price() -> None:
    """The probe's first law: the save never accumulates unless it binds
    every cheaper spender below it (`runs/nuke-probe.out`)."""
    nuker = Nuker()
    budget = Budget(30_000, reserve=0)
    orders = nuker.advance(
        _funding_world(30_000), CATALOGUE, budget, (_BUILDER,), _workforce(), 1, True
    )
    assert orders["build"] is None
    # The withhold binds even a protected later claimant this tick.
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is False


def test_a_standing_launcher_satisfies_the_headcount() -> None:
    nuker = Nuker()
    budget = Budget(90_000, reserve=0)
    world = sample(_CENTRE, _BUILDER, _LAUNCHER, credits=90_000, options=(_PLACEABLE,))
    orders = nuker.advance(world, CATALOGUE, budget, (_BUILDER,), _workforce(), 1, True)
    assert orders["build"] is None
    assert budget.spent() == 0


def test_no_free_builder_neither_places_nor_saves() -> None:
    """Before a worker exists there is nothing to save toward -- the same
    rule the flame conversion holds for its own refusals."""
    nuker = Nuker()
    budget = Budget(30_000, reserve=0)
    orders = nuker.advance(_funding_world(30_000), CATALOGUE, budget, (), _workforce(), 1, True)
    assert orders["build"] is None
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is True


def test_a_complete_launcher_arms_when_the_warhead_is_affordable() -> None:
    nuker = Nuker()
    budget = Budget(20_000, reserve=0)
    world = sample(_CENTRE, _LAUNCHER, credits=20_000, options=(_BUILD_NUKE, _LAUNCH_GATED))
    orders = nuker.advance(world, CATALOGUE, budget, (), _workforce(), 1)
    arm = orders["arm"]
    if arm is None:
        raise AssertionError("the affordable warhead was not ordered")
    assert arm == {"kind": "ability", "unit_id": 500, "key": "c_buildNuke"}
    assert budget.spent() == 11_000


def test_a_refused_warhead_withholds_only_while_a_launcher_stands() -> None:
    nuker = Nuker()
    budget = Budget(5_000, reserve=0)
    world = sample(_CENTRE, _LAUNCHER, credits=5_000, options=(_BUILD_NUKE, _LAUNCH_GATED))
    orders = nuker.advance(world, CATALOGUE, budget, (), _workforce(), 1)
    assert orders["arm"] is None
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is False


def test_the_arm_debounce_holds_between_stockpile_orders() -> None:
    """A dispatched row is still offered next tick; paying it twice in
    consecutive ticks is two warheads where one was decided."""
    nuker = Nuker()
    world = sample(_CENTRE, _LAUNCHER, credits=90_000, options=(_BUILD_NUKE, _LAUNCH_GATED))
    armed = {"kind": "ability", "unit_id": 500, "key": "c_buildNuke"}
    first = nuker.advance(world, CATALOGUE, Budget(90_000, reserve=0), (), _workforce(), 1)
    assert first["arm"] == armed
    for _ in range(ARM_RETRY_SAMPLES - 1):
        again = nuker.advance(world, CATALOGUE, Budget(90_000, reserve=0), (), _workforce(), 1)
        assert again["arm"] is None
    resumed = nuker.advance(world, CATALOGUE, Budget(90_000, reserve=0), (), _workforce(), 1)
    assert resumed["arm"] == armed


def test_an_incomplete_launcher_is_not_asked_to_arm() -> None:
    nuker = Nuker()
    budget = Budget(90_000, reserve=0)
    world = sample(_CENTRE, _RISING, credits=90_000, options=(_BUILD_NUKE,))
    orders = nuker.advance(world, CATALOGUE, budget, (), _workforce(), 1)
    assert orders["arm"] is None
    assert orders["launch"] is None


def test_the_launch_goes_to_the_richest_blast_circle() -> None:
    """An area weapon aims at clusters. A factory standing 200 from a
    command centre shares its blast circle (7,000 erased together); a
    lone command centre across the map scores only itself (6,000). The
    sum wins -- targeting the priciest single structure would aim at
    either centre and waste the circle."""
    nuker = Nuker()
    world = sample(
        _CENTRE,
        _LAUNCHER,
        enemy(80, "landFactory", x=900.0),
        enemy(81, "commandCenterT2", x=1100.0),
        enemy(82, "c_tank", x=700.0),
        enemy(84, "commandCenterT2", x=3000.0, y=2000.0),
        credits=1_000,
        options=(_LAUNCH_NUKE,),
    )
    orders = nuker.advance(world, CATALOGUE, Budget(1_000, reserve=0), (), _workforce(), 1)
    launch = orders["launch"]
    if launch is None:
        raise AssertionError("the standing cluster was not fired at")
    assert launch == {
        "kind": "ability_at",
        "unit_id": 500,
        "key": "c_launchNuke",
        "x": 900.0,
        "y": 0.0,
    }


def test_the_launch_refire_holds_for_its_window_then_fires_again() -> None:
    """The probe's second law: the launch flag reads available at zero
    ammo, so an unanswered launch is refired rather than trusted
    (`runs/nuke-probe4.out`: dud at s239, kill on the s539 refire)."""
    nuker = Nuker()
    world = sample(
        _CENTRE,
        _LAUNCHER,
        enemy(80, "landFactory", x=900.0),
        credits=1_000,
        options=(_LAUNCH_NUKE,),
    )
    fired = {
        "kind": "ability_at",
        "unit_id": 500,
        "key": "c_launchNuke",
        "x": 900.0,
        "y": 0.0,
    }
    first = nuker.advance(world, CATALOGUE, Budget(1_000, reserve=0), (), _workforce(), 1)
    assert first["launch"] == fired
    for _ in range(LAUNCH_RETRY_SAMPLES - 1):
        held = nuker.advance(world, CATALOGUE, Budget(1_000, reserve=0), (), _workforce(), 1)
        assert held["launch"] is None
    refired = nuker.advance(world, CATALOGUE, Budget(1_000, reserve=0), (), _workforce(), 1)
    assert refired["launch"] == fired


def test_no_visible_structure_holds_the_launch() -> None:
    """An 11,000-credit warhead is not spent on a tank or on fog."""
    nuker = Nuker()
    world = sample(
        _CENTRE,
        _LAUNCHER,
        enemy(82, "c_tank", x=700.0),
        credits=1_000,
        options=(_LAUNCH_NUKE,),
    )
    orders = nuker.advance(world, CATALOGUE, Budget(1_000, reserve=0), (), _workforce(), 1)
    assert orders["launch"] is None


def test_a_gated_launch_row_is_not_fired_into() -> None:
    nuker = Nuker()
    world = sample(
        _CENTRE,
        _LAUNCHER,
        enemy(80, "landFactory", x=900.0),
        credits=1_000,
        options=(_LAUNCH_GATED,),
    )
    orders = nuker.advance(world, CATALOGUE, Budget(1_000, reserve=0), (), _workforce(), 1)
    assert orders["launch"] is None


def test_best_target_ignores_what_the_catalogue_cannot_price() -> None:
    world = sample(_CENTRE, enemy(90, "modded_mystery", x=500.0))
    assert best_target(world, CATALOGUE) is None


def test_a_full_ring_neither_places_nor_saves() -> None:
    """Every ring slot occupied is a fact about the base, not the balance:
    nothing is claimed and nothing withheld until a site exists to build
    on -- the engine would refuse the collision silently anyway."""
    crowded = sample(
        _CENTRE,
        _BUILDER,
        *(
            entity(600 + index, "landFactory", x=offset[0], y=offset[1])
            for index, offset in enumerate(PLACEMENT_RING)
        ),
        credits=90_000,
        options=(_PLACEABLE,),
        players=_EARNING,
    )
    nuker = Nuker()
    budget = Budget(90_000, reserve=0)
    orders = nuker.advance(crowded, CATALOGUE, budget, (_BUILDER,), _workforce(), 1, True)
    assert orders["build"] is None
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is True


def test_below_the_income_floor_the_launcher_neither_funds_nor_saves() -> None:
    """The second screen's law: withheld from tick one, three Impossible
    matches starved before their fortress stood -- worth never above the
    starting 3,500 (`runs/sweeps/imp-nuke`). The save is earned by the
    46-duel income line, not assumed by the doctrine being on."""
    starving = sample(_CENTRE, _BUILDER, credits=90_000, options=(_PLACEABLE,), players=_STARVING)
    nuker = Nuker()
    budget = Budget(90_000, reserve=0)
    orders = nuker.advance(starving, CATALOGUE, budget, (_BUILDER,), _workforce(), 1, True)
    assert orders["build"] is None
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is True


def test_a_world_with_no_scoreboard_earns_nothing() -> None:
    """Absence of data is not an economy: a scripted or pre-scoreboard
    sample must not release a 45,000 save on a blank."""
    blank = sample(_CENTRE, _BUILDER, credits=90_000, options=(_PLACEABLE,))
    nuker = Nuker()
    budget = Budget(90_000, reserve=0)
    orders = nuker.advance(blank, CATALOGUE, budget, (_BUILDER,), _workforce(), 1, True)
    assert orders["build"] is None
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is True


def test_a_launcher_underway_is_not_ordered_twice() -> None:
    """The vh-nuke bug, pinned: while the first builder walks, standing
    reads zero -- and a headcount blind to the walk assigned a fresh
    builder to the same job every tick, eight granted launchers and
    360,000 credits for one structure (`runs/sweeps/vh-nuke`)."""
    nuker = Nuker()
    workforce = _workforce()
    first = nuker.advance(
        _funding_world(90_000),
        CATALOGUE,
        Budget(90_000, reserve=0),
        (_BUILDER,),
        workforce,
        1,
        True,
    )
    if first["build"] is None:
        raise AssertionError("the funded launcher was not placed")
    second_builder = entity(215, "builder", x=80.0)
    offered_again = sample(
        _CENTRE,
        _BUILDER,
        second_builder,
        credits=90_000,
        options=(_PLACEABLE, option(215, LAUNCHER_TYPE, key="u_nuke", index=1, placed=True)),
        players=_EARNING,
    )
    budget = Budget(90_000, reserve=0)
    again = nuker.advance(offered_again, CATALOGUE, budget, (second_builder,), workforce, 1, True)
    assert again["build"] is None
    assert budget.spent() == 0


def test_no_commitment_means_no_funding_and_no_withhold() -> None:
    """The gate's other half: income alone released the save mid-fight and
    a baseline win bled out around it (`runs/sweeps/vh-nuke`, 90210). Until
    the closer stands committed, the channel neither claims nor binds."""
    nuker = Nuker()
    budget = Budget(90_000, reserve=0)
    orders = nuker.advance(
        _funding_world(90_000), CATALOGUE, budget, (_BUILDER,), _workforce(), 1, False
    )
    assert orders["build"] is None
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is True
