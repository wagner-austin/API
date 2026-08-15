"""The battery's three steps: fund early, walk the shore, convert in place.

The fake world grants land by growing a turret and refuses it by silence
-- the same sensor the live probe proved (log 2026-08-13) -- and the
conversion is priced by the option row the engine publishes, never by a
guess. Funding is its own step because the fifth pilot starved a
tail-of-tick claim through 4,866 refusals (log 2026-08-14).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.policy.battery import (
    BATTERY_TYPE,
    FRACTIONS,
    PATIENCE,
    TURRET_TYPE,
    Battery,
)
from rw_bot.policy.budget import Budget
from rw_bot.wire.state import Sample
from tests.wire_fixtures import entity, option, pool, sample


def _stats(name: str, price: int) -> UnitStats:
    return UnitStats(
        type_name=name,
        display_name=name,
        description="",
        price=price,
        hp=100,
        speed=0.0 if name in ("commandCenter", TURRET_TYPE) else 1.0,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE = {
    "commandCenter": _stats("commandCenter", 0),
    "builder": _stats("builder", 500),
    TURRET_TYPE: _stats(TURRET_TYPE, 500),
}

_ANCHOR = entity(1, "commandCenter", x=0.0, y=0.0)
_BUILDER = entity(2, "builder", x=10.0, y=0.0)
_POOL = pool(x=500.0, y=0.0)

#: Where the first candidate lands: anchor (0,0), mirror (1000,0), so the
#: water-most fraction 0.22 offers x=220.
_FIRST_X = FRACTIONS[0] * 1000.0


def _world() -> Sample:
    return sample(_ANCHOR, _BUILDER, pools=(_POOL,), credits=4000)


def _step(battery: Battery, world: Sample, budget: Budget) -> tuple[str, ...]:
    """One tick of the channel: fund, then walk; the quartermaster's order."""
    battery.fund(world, _CATALOGUE, budget, True)
    return tuple(o["type_name"] for o in battery.establish(world, _CATALOGUE, budget, True))


def _walk_to_acceptance(battery: Battery) -> None:
    """Fund and drive the walk one tick so the first candidate is offered."""
    _step(battery, _world(), Budget(4000, 0))


def _standing(complete: bool = True, type_name: str = TURRET_TYPE) -> Sample:
    """A world where the walk's turret stands at the offered site."""
    return sample(
        _ANCHOR,
        _BUILDER,
        entity(9, type_name, x=_FIRST_X + 8.0, y=0.0, complete=complete),
        pools=(_POOL,),
        credits=4000,
        options=(option(9, BATTERY_TYPE, price=1600),),
    )


def test_the_walk_offers_the_water_most_fraction_first() -> None:
    """The turret wants the LAST land before the water, so the walk
    descends from the shore where the shipyard ascended into it."""
    battery = Battery()
    budget = Budget(4000, 0)
    battery.fund(_world(), _CATALOGUE, budget, True)
    orders = battery.establish(_world(), _CATALOGUE, budget, True)
    assert [(o["type_name"], o["unit_id"], o["x"], o["y"]) for o in orders] == [
        (TURRET_TYPE, 2, _FIRST_X, 0.0)
    ]


def test_an_unfunded_walk_sends_nothing_and_withholds() -> None:
    """The saving pattern, now early in the tick where it binds: refused,
    the price is withheld so lesser claims cannot snipe the accrual --
    and the walk sends nothing until the claim lands."""
    battery = Battery()
    budget = Budget(300, 0)
    assert _step(battery, _world(), budget) == ()
    lesser = budget.claim("produce:c_tank", 300)
    assert lesser["granted"] is False


def test_one_claim_then_the_order_resends_and_patience_advances() -> None:
    """The shipyard's two measured laws, inherited whole: the price is
    claimed exactly once, the order re-sends every tick, and silence for
    PATIENCE samples advances the fraction (log 2026-08-10)."""
    battery = Battery()
    world = _world()
    claims = 0
    sent = 0
    for _ in range(2 * PATIENCE + 1):
        budget = Budget(4000, 0)
        sent += len(_step(battery, world, budget))
        follow_up = budget.claim("produce:c_tank", 4000)
        claims += 0 if follow_up["granted"] else 1
    assert claims == 1
    assert sent == 2 * PATIENCE + 1


def test_the_walk_gives_up_after_the_last_fraction() -> None:
    battery = Battery()
    world = _world()
    for _ in range(len(FRACTIONS) * (PATIENCE + 2) + 5):
        _step(battery, world, Budget(4000, 0))
    assert _step(battery, world, Budget(4000, 0)) == ()


def test_a_turret_at_the_site_ends_the_walk_and_a_cover_turret_does_not() -> None:
    """The walk's turret is identified by WHERE it stands: a cover
    doctrine's base turret must not satisfy -- or be converted by -- the
    shore channel."""
    battery = Battery()
    _walk_to_acceptance(battery)
    at_site = _standing()
    assert _step(battery, at_site, Budget(4000, 0)) == ()
    at_base = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, TURRET_TYPE, x=15.0, y=0.0, complete=True),
        pools=(_POOL,),
        credits=4000,
    )
    assert _step(battery, at_base, Budget(4000, 0)) == (TURRET_TYPE,)


def test_the_walk_avoids_the_builder_another_walk_pinned() -> None:
    """Two live walks re-sending against one builder would override each
    other every tick; the battery takes the newest builder NOT pinned by
    the shipyard."""
    battery = Battery()
    world = sample(
        _ANCHOR, _BUILDER, entity(7, "builder", x=20.0, y=0.0), pools=(_POOL,), credits=4000
    )
    budget = Budget(4000, 0)
    battery.fund(world, _CATALOGUE, budget, True)
    orders = battery.establish(world, _CATALOGUE, budget, True, avoid_builder=7)
    assert [o["unit_id"] for o in orders] == [2]
    lone = Battery()
    lone_budget = Budget(4000, 0)
    lone.fund(_world(), _CATALOGUE, lone_budget, True)
    assert lone.establish(_world(), _CATALOGUE, lone_budget, True, avoid_builder=2) == ()


def test_the_fork_funds_once_from_the_option_row_and_the_order_resends() -> None:
    """The fork price is the option row's, claimed once and early; the
    produce re-sends every tick because conversion never fills the queue
    -- and the re-send is what wins a holder the flame converter also
    claimed ([[policy-holding-ground]])."""
    battery = Battery()
    _walk_to_acceptance(battery)
    standing = _standing()
    claims = 0
    sent = 0
    for _ in range(3):
        budget = Budget(4000, 0)
        battery.fund(standing, _CATALOGUE, budget, True)
        sent += len(battery.convert(standing, budget, True))
        follow_up = budget.claim("produce:c_tank", 2600)
        claims += 0 if follow_up["granted"] else 1
    assert claims == 1
    assert sent == 3


def test_an_unfunded_fork_withholds_and_sends_nothing() -> None:
    battery = Battery()
    _walk_to_acceptance(battery)
    budget = Budget(300, 0)
    battery.fund(_standing(), _CATALOGUE, budget, True)
    assert battery.convert(_standing(), budget, True) == ()
    lesser = budget.claim("produce:c_tank", 300)
    assert lesser["granted"] is False


def test_the_fork_waits_for_completion_and_for_the_offer() -> None:
    battery = Battery()
    _walk_to_acceptance(battery)
    growing = _standing(complete=False)
    budget = Budget(4000, 0)
    battery.fund(growing, _CATALOGUE, budget, True)
    assert battery.convert(growing, budget, True) == ()
    unoffered = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, TURRET_TYPE, x=_FIRST_X + 8.0, y=0.0, complete=True),
        pools=(_POOL,),
        credits=4000,
    )
    other = Budget(4000, 0)
    battery.fund(unoffered, _CATALOGUE, other, True)
    assert battery.convert(unoffered, other, True) == ()
    # Neither wait spent anything: a full follow-up claim still fits.
    assert other.claim("produce:c_tank", 4000)["granted"] is True


def test_a_standing_fork_closes_the_channel_for_the_match() -> None:
    """One battery per match: a fork that stood and later died is a loss
    the panel measures, not an unfunded rebuild."""
    battery = Battery()
    _walk_to_acceptance(battery)
    forked = _standing(type_name=BATTERY_TYPE)
    assert battery.holder_id(_standing()) == 9
    assert battery.convert(forked, Budget(4000, 0), True) == ()
    afterwards = _world()
    assert _step(battery, afterwards, Budget(4000, 0)) == ()
    assert battery.convert(afterwards, Budget(4000, 0), True) == ()
    # A closed channel speaks for no holder.
    assert battery.holder_id(_standing()) is None


def test_the_knob_off_a_missing_type_or_a_lost_base_stay_silent() -> None:
    battery = Battery()
    off = Budget(4000, 0)
    battery.fund(_world(), _CATALOGUE, off, False)
    assert battery.establish(_world(), _CATALOGUE, off, False) == ()
    assert battery.convert(_world(), off, False) == ()
    bare = {name: stats for name, stats in _CATALOGUE.items() if name != TURRET_TYPE}
    poor = Budget(4000, 0)
    battery.fund(_world(), bare, poor, True)
    assert battery.establish(_world(), bare, poor, True) == ()
    builderless = sample(_ANCHOR, pools=(_POOL,), credits=4000)
    funded = Budget(4000, 0)
    battery.fund(builderless, _CATALOGUE, funded, True)
    assert battery.establish(builderless, _CATALOGUE, funded, True) == ()
    poolless = sample(_ANCHOR, _BUILDER, credits=4000)
    assert battery.establish(poolless, _CATALOGUE, Budget(4000, 0), True) == ()
    # Before any site is offered there is nothing to convert toward.
    assert Battery().convert(_standing(), Budget(4000, 0), True) == ()
    # A battery that never walked has pinned nobody; one that walked has.
    walked = Battery()
    _walk_to_acceptance(walked)
    assert walked.pinned_builder() == 2


def test_a_dead_turret_re_funds_its_rebuild() -> None:
    """The engine charges per attempt, so the books must too: a turret
    that stood and died re-claims both halves on the rebuild -- while a
    turret merely not yet built keeps the original claim (log
    2026-08-14, the pilot's unaccounted second turret)."""
    battery = Battery()
    claims = 0
    budget = Budget(4000, 0)
    _step(battery, _world(), budget)
    claims += 0 if budget.claim("produce:c_tank", 4000)["granted"] else 1
    stood = _standing()
    assert _step(battery, stood, Budget(4000, 0)) == ()
    razed = Budget(4000, 0)
    orders = _step(battery, _world(), razed)
    claims += 0 if razed.claim("produce:c_tank", 4000)["granted"] else 1
    assert orders == (TURRET_TYPE,)
    assert claims == 2


def test_a_late_acceptance_at_an_abandoned_fraction_is_still_ours() -> None:
    """The third pilot's flaw: the builder was still walking when
    patience moved the walk to the next fraction, the turret then stood
    at the ABANDONED point, and a last-site-only check never recognized
    it -- so the fork never claimed (log 2026-08-14). Every offered
    point stays the walk's own."""
    battery = Battery()
    world = _world()
    for _ in range(PATIENCE + 2):
        _step(battery, world, Budget(4000, 0))
    late = sample(
        _ANCHOR,
        _BUILDER,
        entity(9, TURRET_TYPE, x=_FIRST_X + 8.0, y=0.0),
        pools=(_POOL,),
        credits=4000,
        options=(option(9, BATTERY_TYPE, price=1600),),
    )
    budget = Budget(4000, 0)
    assert _step(battery, late, budget) == ()
    orders = battery.convert(late, budget, True)
    assert [(o["type_name"], o["unit_id"]) for o in orders] == [(BATTERY_TYPE, 9)]


def test_an_incomplete_site_turret_holds_the_builder_on_construction() -> None:
    """Pilot six's defect: the expander re-tasks the builder to distant
    pools the moment the walk goes silent, and the abandoned turret dies
    unfinished -- three stood, none completed (log 2026-08-14). While
    the site turret is incomplete the walk re-sends the build at the
    STANDING turret, winning the builder back every tick; completion
    releases it."""
    battery = Battery()
    _walk_to_acceptance(battery)
    growing = _standing(complete=False)
    orders = battery.establish(growing, _CATALOGUE, Budget(4000, 0), True)
    assert [(o["type_name"], o["unit_id"], o["x"], o["y"]) for o in orders] == [
        (TURRET_TYPE, 2, _FIRST_X + 8.0, 0.0)
    ]
    finished = _standing()
    assert battery.establish(finished, _CATALOGUE, Budget(4000, 0), True) == ()
