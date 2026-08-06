"""Upgrading what already stands, driven through the loop.

An extractor that converts itself in place is income bought without a builder,
a pool or a walk -- and it competes for the same credits as the army and the
next pool. What it must outrank, what it must yield to, and that it is ordered
once rather than every observation, are all answers about the spending order,
so they are driven through whole ticks.

Split from ``test_campaign_economy``, which keeps claiming and defending
ground; the roster both stand on is :mod:`tests.campaign_fixtures`.
"""

from __future__ import annotations

from rw_bot.control.channel import AgentChannel
from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.placement import TypePlacement
from rw_bot.policy.campaign import play
from tests.campaign_fixtures import (
    BUILDER,
    CATALOGUE,
    CENTRE,
    PROFILES,
    ScriptedPeer,
    run_campaign,
    unit_stats,
    verb,
)
from tests.wire_fixtures import (
    entity,
    lines,
    option,
    pool,
    sample,
)


def _upgrade_world() -> tuple[dict[str, UnitStats], dict[str, TypePlacement]]:
    """A catalogue that prices the upgrade the way the engine prices it.

    **The two prices are deliberately different, because in the game they are.**
    This fixture used to give ``extractorT2`` a build price of 1,400 -- the cost
    of the *conversion* -- which made the loop's old reading, claiming the
    target's build price, look correct. The engine's dump says ``Price: $2100``
    for a tier two and ``T2 Upgrade Price: $1400`` on the tier one, so the
    fixture described a world the game cannot produce and the test could not
    have caught the substitution ([[mechanics-unit-value]]).
    """
    catalogue = {
        **CATALOGUE,
        "extractorT1": unit_stats("extractorT1", speed=0.0, price=700, upgrade_prices=(1400,)),
        "extractorT2": unit_stats("extractorT2", speed=0.0, price=2100),
    }
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=name.startswith("extractor"))
        for i, name in enumerate(catalogue)
    }
    return catalogue, placements


def test_an_extractor_is_told_to_upgrade_itself() -> None:
    """The income the map cannot take away.

    An extractor converting itself needs no builder, crosses no contested
    ground and claims no new pool -- which is what matters on a map where the
    opponents finish holding 44 of the 46 pools and 247 expansion orders leave
    the bot with one extractor. It was invisible until the agent stopped
    dropping actions that neither place nor "make something"
    ([[policy-holding-ground]]).
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        credits=4000,
        options=(option(400, "extractorT2", placed=False, makes_something=False),),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 1)
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']


def test_an_owned_extractor_is_upgraded_before_a_free_pool_is_claimed() -> None:
    """The order the arithmetic argues against and the matches preferred.

    A new extractor is 700 for +8 credits a second; converting one is 1,400 for
    +4 and then 4,000 for +8, so pools are six times better per credit --
    ``#price per credit: $87`` against ``$800`` in the game's own assets -- and
    [[policy-economy]] states the rule outright: take every free pool before
    upgrading anything.

    **Reordered on exactly that arithmetic, it measured worse.** Twelve seeds at
    Very Hard: 7 won with upgrades first against 5 with expansion first, the
    same two losses, routs 3 -> 2, median win 2,207 -> 2,362. That sits inside
    the noise floor, so it refutes nothing -- but it is not the improvement the
    per-credit figure promised either, and two weak signals pointing the same
    way is what this decision rests on.

    **What the arithmetic leaves out is risk**, and risk is the one thing every
    rung of this ladder turns on: matches are decided by extractors *lost*, with
    winners dropping nought to four and the rest six or more
    ([[policy-holding-ground]]). A new extractor is income that can be
    destroyed; a conversion is income on ground already held. Six times the
    price for income that cannot be taken away is a different trade from six
    times the price for nothing.

    Nothing pinned this order before -- swapping the two calls broke no test at
    all -- which is why it is pinned here now, with the measurement behind it.
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        BUILDER,
        entity(400, "extractorT1", x=900.0, y=0.0),
        credits=1500,
        pools=(pool(x=300.0),),
        options=(
            option(214, "extractorT1", placed=True),
            option(400, "extractorT2", placed=False, makes_something=False),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 1)
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']
    # Expansion is not disabled, only outranked: 1,500 covered the conversion
    # and what was left could not also cover a 700 extractor.
    assert verb(peer, "build") == []


def test_an_upgrade_outranks_replacing_the_army() -> None:
    """Income before army, the third ordering this pair has held.

    Production-first left the tier-three conversion asked ~1,800 times a
    match and granted never, while produce drained every credit above the
    reserve into units that traded even against a 1.8x income and
    equilibrated (log 2026-07-31). A conversion pays back inside the match
    on ground already held, and the reserve still protects the replacement
    of a loss -- upgrades claim past it -- so production is deferred, not
    starved. Nothing pinned this order before; it is pinned here with the
    reasoning.
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        entity(500, "landFactory"),
        credits=1400,
        options=(
            option(400, "extractorT2", placed=False, makes_something=False),
            option(500, "c_tank"),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(
        AgentChannel(peer),
        (),
        catalogue,
        placements,
        PROFILES,
        1,
        reinforce=("c_tank",),
    )
    # 1,400 covered the conversion; what remained could not also cover a
    # 350-credit tank, so the army waits one observation.
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']


def test_a_refused_upgrade_does_not_hold_the_armys_credits() -> None:
    """The saving that was built, measured, and taken back out.

    A refused conversion once withheld its price from every later claim, and
    the mechanism did what it promised: six T2 and six T3 conversions a
    match, income doubled to 98/s -- and the army pauses it forced let the
    enemy's economy live untouched, rival scores rising to the worst of the
    whole screening arc ([[policy-budget]], log 2026-07-31). Income that
    displaces early pressure is income for a longer strangle. So a refusal
    is again just a refusal, and the tank the credits do cover is bought.
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        entity(500, "landFactory"),
        credits=500,
        options=(
            option(400, "extractorT2", placed=False, makes_something=False),
            option(500, "c_tank"),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(
        AgentChannel(peer),
        (),
        catalogue,
        placements,
        PROFILES,
        1,
        reinforce=("c_tank",),
    )
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":500,"type":"c_tank"}']


def test_tech_unlocks_the_factory_and_off_leaves_it_locked() -> None:
    """The tech verb end to end: a factory offering its no-type unlock is
    told to fire it, once, at the price the wire itself carries -- and the
    same world with the flag off sends nothing ([[mechanics-build-actions]]).
    """
    catalogue = {
        **CATALOGUE,
        "landFactory": unit_stats("landFactory", speed=0.0, price=700, upgrade_prices=(2000,)),
    }
    placements = {
        name: TypePlacement(index=i, type_name=name, needs_pool=name.startswith("extractor"))
        for i, name in enumerate(catalogue)
    }
    world = sample(
        CENTRE,
        entity(500, "landFactory"),
        credits=4000,
        options=(option(500, "", key="c_2", placed=False, makes_something=False, price=2000),),
    )
    peer = ScriptedPeer(lines(world, world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 2, tech=1)
    fired = [line for line in peer.sent if '"ability"' in line]
    # Once, not once per observation: the unlock never fills the queue.
    assert fired == ['{"kind":"ability","unit_id":500,"key":"c_2"}']

    held = ScriptedPeer(lines(world))
    play(AgentChannel(held), (), catalogue, placements, PROFILES, 1)
    assert [line for line in held.sent if '"ability"' in line] == []


def test_an_upgrade_is_ordered_once_rather_than_every_observation() -> None:
    """A conversion never fills the production queue.

    ``queued`` stays at zero for as long as the conversion runs, so the
    structure keeps offering the upgrade it is already performing. Re-ordering
    it every observation sent a stream of duplicates, and one arrived after the
    conversion had finished -- addressed to a unit that was now an
    ``extractorT2`` and could only make an ``extractorT3``. The agent refuses an
    order naming something its subject cannot make, and that refusal crashed the
    match ([[policy-holding-ground]]).
    """
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        credits=40000,
        options=(option(400, "extractorT2", placed=False, makes_something=False),),
    )
    peer = ScriptedPeer(lines(world, world, world, world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 4)
    assert verb(peer, "produce") == ['{"kind":"produce","unit_id":400,"type":"extractorT2"}']


def test_upgrades_stop_at_the_first_refusal_rather_than_overdrawing() -> None:
    """Every upgrade costs the same, so a refusal means the budget is out."""
    catalogue, placements = _upgrade_world()
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        entity(401, "extractorT1"),
        credits=1400,
        options=(
            option(400, "extractorT2", placed=False, makes_something=False),
            option(401, "extractorT2", placed=False, makes_something=False),
        ),
    )
    peer = ScriptedPeer(lines(world))
    play(AgentChannel(peer), (), catalogue, placements, PROFILES, 1)
    assert len(verb(peer, "produce")) == 1


def test_an_upgrade_the_catalogue_cannot_price_is_never_ordered() -> None:
    """Unpriced means unbudgetable, and spending blind is what the budget
    prevents ([[policy-budget]]).
    """
    world = sample(
        CENTRE,
        entity(400, "extractorT1"),
        credits=4000,
        options=(option(400, "extractorT2", placed=False, makes_something=False),),
    )
    # CATALOGUE does not price extractorT2.
    _, peer = run_campaign(world, times=1)
    assert verb(peer, "produce") == []
