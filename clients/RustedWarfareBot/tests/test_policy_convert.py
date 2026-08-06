"""The conversion channel: one saving upgrade at a time until the count holds.

The ground turret's tier-two upgrade is a four-way fork the extractor walk
deliberately skips, and the flame branch is the community's named anti-horde
static ([[community-play-strategies]]). A conversion never fills the queue,
so the channel remembers each holder it ordered -- the duplicate that lands
after completion names an action that no longer exists and crashes the match
([[policy-holding-ground]]).
"""

from __future__ import annotations

from rw_bot.policy.budget import Budget
from rw_bot.policy.convert import FLAME_TYPE, Converter, TurretLadder
from rw_bot.wire.state import Sample
from tests.wire_fixtures import entity, option, sample


def _world(
    *, flames_alive: int = 0, holders: int = 1, offer: bool = True, converting: bool = False
) -> Sample:
    entities = [entity(213, "commandCenter")]
    for index in range(holders):
        entities.append(entity(700 + index, "c_turret_t1", x=100.0 + index))
    for index in range(flames_alive):
        entities.append(entity(800 + index, FLAME_TYPE, x=300.0 + index))
    options = (
        tuple(
            option(700 + index, FLAME_TYPE, key=f"u_flame_{index}", index=index, price=1000)
            for index in range(holders)
        )
        if offer or converting
        else ()
    )
    return sample(*entities, credits=4000, options=options)


def test_a_funded_conversion_goes_to_the_idle_offering_holder() -> None:
    flamer = Converter()
    budget = Budget(10_000, reserve=0)
    orders = flamer.convert(_world(), budget, 1)
    assert [o["unit_id"] for o in orders] == [700]
    assert [o["type_name"] for o in orders] == [FLAME_TYPE]
    assert budget.spent() == 1000


def test_a_refused_conversion_withholds_its_price() -> None:
    """The saving pattern: later spenders see that much less this tick."""
    flamer = Converter()
    budget = Budget(800, reserve=0)
    assert flamer.convert(_world(), budget, 1) == ()
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is False


def test_a_holder_is_never_ordered_twice() -> None:
    """Converting never fills the queue, so the holder keeps offering the
    upgrade it is performing -- and a duplicate landing after completion
    crashed the match when the extractors met this first."""
    flamer = Converter()
    assert len(flamer.convert(_world(), Budget(10_000, reserve=0), 2)) == 1
    # Same world next tick: the holder still stands as a T1, still offering.
    assert flamer.convert(_world(), Budget(10_000, reserve=0), 2) == ()
    # A second holder appears: the channel moves on to it, never back.
    orders = flamer.convert(_world(holders=2), Budget(10_000, reserve=0), 2)
    assert [o["unit_id"] for o in orders] == [701]


def test_a_finished_conversion_counts_and_the_channel_stops() -> None:
    flamer = Converter()
    assert len(flamer.convert(_world(), Budget(10_000, reserve=0), 1)) == 1
    # The holder became the flame turret: headcount met, nothing more sent.
    done = _world(flames_alive=1, holders=1)
    assert flamer.convert(done, Budget(10_000, reserve=0), 1) == ()


def test_a_full_headcount_converts_nothing() -> None:
    flamer = Converter()
    assert flamer.convert(_world(flames_alive=2), Budget(10_000, reserve=0), 2) == ()
    assert flamer.convert(_world(), Budget(10_000, reserve=0), 0) == ()


def test_hostile_and_unfinished_flames_do_not_count_toward_the_headcount() -> None:
    """The count is ours and standing: an enemy's flamethrower satisfies
    nothing, and one still under construction is not yet holding a line."""
    world = sample(
        entity(213, "commandCenter"),
        entity(700, "c_turret_t1", x=100.0),
        entity(900, FLAME_TYPE, mine=False, hostile=True, x=800.0),
        entity(901, FLAME_TYPE, complete=False, x=400.0),
        credits=4000,
        options=(option(700, FLAME_TYPE, key="u_flame_0", price=1000),),
    )
    orders = Converter().convert(world, Budget(10_000, reserve=0), 1)
    assert [o["unit_id"] for o in orders] == [700]


def test_no_offer_means_no_order_and_no_saving() -> None:
    """Before a turret stands there is nothing to save toward."""
    flamer = Converter()
    budget = Budget(800, reserve=0)
    assert flamer.convert(_world(offer=False), budget, 1) == ()
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is True


def _ladder_world(
    *,
    base: int = 0,
    mids: int = 0,
    tops: int = 0,
    mid_offers_top: bool = True,
    credits_held: int = 20_000,
) -> Sample:
    """A turret roster where every idle holder offers its next gun tier."""
    entities = [entity(213, "commandCenter")]
    options = []
    for index in range(base):
        uid = 700 + index
        entities.append(entity(uid, "c_turret_t1", x=100.0 + index))
        options.append(option(uid, "c_turret_t2_gun", key=f"u_t2_{index}", index=index, price=1000))
    for index in range(mids):
        uid = 800 + index
        entities.append(entity(uid, "c_turret_t2_gun", x=300.0 + index))
        if mid_offers_top:
            options.append(
                option(uid, "c_turret_t3_gun", key=f"u_t3_{index}", index=50 + index, price=11000)
            )
    for index in range(tops):
        entities.append(entity(900 + index, "c_turret_t3_gun", x=500.0 + index))
    return sample(*entities, credits=credits_held, options=tuple(options))


def test_the_ladder_converts_an_offering_mid_to_the_top_first() -> None:
    ladder = TurretLadder()
    budget = Budget(20_000, reserve=0)
    orders = ladder.convert(_ladder_world(base=2, mids=1), budget, 1)
    assert [o["unit_id"] for o in orders] == [800]
    assert [o["type_name"] for o in orders] == ["c_turret_t3_gun"]
    assert budget.spent() == 11_000


def test_the_ladder_feeds_the_pipeline_when_no_mid_offers() -> None:
    """Base turrets step up only while the top's shortfall demands feed."""
    ladder = TurretLadder()
    budget = Budget(20_000, reserve=0)
    orders = ladder.convert(_ladder_world(base=2), budget, 1)
    assert [o["type_name"] for o in orders] == ["c_turret_t2_gun"]
    assert budget.spent() == 1000
    # The pipeline is bounded: one mid in flight covers a shortfall of one,
    # so the second base turret is left standing as cover.
    again = ladder.convert(_ladder_world(base=2), budget, 1)
    assert again == ()


def test_a_fed_mid_takes_its_second_step_up_the_chain() -> None:
    """The pair key, not the holder key: a turret told to become the mid
    tier must be re-orderable for the top -- keyed by id alone the chain
    stopped where the extractor walk once did ([[policy-holding-ground]])."""
    ladder = TurretLadder()
    assert len(ladder.convert(_ladder_world(base=1), Budget(20_000, reserve=0), 1)) == 1
    # The same holder finished converting: unit 700 now stands as the mid.
    grown = sample(
        entity(213, "commandCenter"),
        entity(700, "c_turret_t2_gun", x=100.0),
        credits=20_000,
        options=(option(700, "c_turret_t3_gun", key="u_t3", price=11000),),
    )
    orders = ladder.convert(grown, Budget(20_000, reserve=0), 1)
    assert [(o["unit_id"], o["type_name"]) for o in orders] == [(700, "c_turret_t3_gun")]


def test_a_refused_ladder_step_withholds_its_own_price() -> None:
    ladder = TurretLadder()
    budget = Budget(5_000, reserve=0)
    assert ladder.convert(_ladder_world(base=0, mids=1, credits_held=5_000), budget, 1) == ()
    # The 11,000 top step was withheld: nothing else fits this tick.
    assert budget.claim("produce:c_tank", 350, protected=True)["granted"] is False


def test_a_full_top_count_walks_nothing() -> None:
    ladder = TurretLadder()
    budget = Budget(20_000, reserve=0)
    assert ladder.convert(_ladder_world(base=2, tops=1), budget, 1) == ()
    assert ladder.convert(_ladder_world(base=2), budget, 0) == ()


def test_a_top_conversion_in_flight_counts_toward_the_headcount() -> None:
    ladder = TurretLadder()
    assert len(ladder.convert(_ladder_world(mids=1), Budget(20_000, reserve=0), 1)) == 1
    # Same world next tick: the mid still stands, still offering -- the
    # in-flight conversion must not be ordered twice nor a base fed.
    assert ladder.convert(_ladder_world(base=1, mids=1), Budget(20_000, reserve=0), 1) == ()


def test_an_empty_field_offers_no_step() -> None:
    ladder = TurretLadder()
    budget = Budget(20_000, reserve=0)
    assert ladder.convert(_ladder_world(), budget, 2) == ()
    assert budget.spent() == 0
