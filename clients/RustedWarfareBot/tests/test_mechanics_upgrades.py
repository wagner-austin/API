"""What an owned structure offers, read as the pure function it is.

The question these answer is whether the engine ever hands the policy layer an
upgrade. The build tree says an ``extractorT1`` produces an ``extractorT2``; the
option stream is what decides whether that edge is reachable, and the two are
not the same thing ([[policy-holding-ground]]).
"""

from __future__ import annotations

from rw_bot.mechanics.catalogue import UnitStats
from rw_bot.mechanics.upgrades import format_offers, next_tier, satisfies, upgrade_offers
from tests.wire_fixtures import entity, option, sample


def _unit(type_name: str, *, speed: float) -> UnitStats:
    return UnitStats(
        type_name=type_name,
        display_name=type_name,
        description="",
        price=700,
        hp=100,
        speed=speed,
        turn_speed=0.0,
        mass=1,
        upgrade_prices=(),
        weapon=None,
    )


_CATALOGUE = {
    "extractorT1": _unit("extractorT1", speed=0.0),
    "commandCenter": _unit("commandCenter", speed=0.0),
    "landFactory": _unit("landFactory", speed=0.0),
    "builder": _unit("builder", speed=0.6),
    "c_tank": _unit("c_tank", speed=1.1),
}


def test_a_structures_offer_is_reported_with_what_the_engine_says_about_it() -> None:
    """Placed and available are carried because they decide who can own the
    order: a placed option cannot come from a queue, and availability is where
    tech gating already lives ([[mechanics-build-actions]]).
    """
    world = sample(
        entity(1, "extractorT1"),
        options=(option(1, "extractorT2", placed=True, available=False),),
    )
    assert upgrade_offers(world, _CATALOGUE) == (
        {
            "unit_id": 1,
            "holder_type": "extractorT1",
            "produces": "extractorT2",
            "placed": True,
            "available": False,
        },
    )


def test_a_mobile_producer_is_not_asked_what_it_upgrades_into() -> None:
    """A factory offering a tank is ordinary production and already understood;
    the question is what a *structure* offers.
    """
    world = sample(
        entity(1, "builder"),
        entity(2, "c_tank"),
        options=(option(1, "extractorT1"), option(2, "somethingElse")),
    )
    assert upgrade_offers(world, _CATALOGUE) == ()


def test_an_unfinished_structure_offers_nothing_yet() -> None:
    world = sample(
        entity(1, "extractorT1", complete=False),
        options=(option(1, "extractorT2"),),
    )
    assert upgrade_offers(world, _CATALOGUE) == ()


def test_a_type_the_catalogue_does_not_describe_is_treated_as_mobile() -> None:
    """Refusing is the safe direction: an unknown type reported as a structure
    would put a mystery unit into an answer about upgrades.
    """
    world = sample(
        entity(1, "mysteryThing"),
        options=(option(1, "mysteryUpgrade"),),
    )
    assert upgrade_offers(world, _CATALOGUE) == ()


def test_no_offers_at_all_is_itself_the_answer() -> None:
    """An empty result is a finding, so it is printed as one rather than as a
    bare table with no rows.

    This is what the probe first measured: four extractors standing in a live
    match offered nothing. The reading taken from it -- that the upgrade path
    was unreachable -- turned out to be wrong, because the agent was dropping
    every ``convertTo`` action before it reached the wire. The observation was
    sound and the conclusion was not, which is the reason the line says what
    was seen instead of what it means ([[policy-holding-ground]]).
    """
    assert format_offers(()) == ("no owned structure offered any option at all",)


def test_a_later_tier_satisfies_an_earlier_one_along_the_path() -> None:
    """Every tier above the asked-for one answers it, or an upgrade would
    un-satisfy the plan entry that built the thing ([[policy-holding-ground]]).
    """
    for held in ("extractorT2", "extractorT3", "extractorT3_overclocked"):
        assert satisfies(held, "extractorT1")
    assert satisfies("extractorT3_reinforced", "extractorT2")


def test_the_relation_is_one_way() -> None:
    """Holding a tier one is not an answer to a plan asking for a tier two."""
    assert not satisfies("extractorT1", "extractorT2")
    assert not satisfies("extractorT3", "extractorT3_reinforced")


def test_the_two_tier_three_branches_do_not_satisfy_each_other() -> None:
    """Overclocked and reinforced are siblings, not two grades of one thing.

    ``extractorT3.ini`` declares both as separate ``convertTo`` actions off the
    tier three, and neither leads to the other -- each carries only an
    ``action_refund`` back down. Listing them as one linear chain claimed one
    was an upgrade of the other, in both directions.

    They are genuinely different purchases: overclocked pays 30 credits a
    second at 1,100 hit points, reinforced pays 20 at 4,700 with a shield.
    Neither substitutes for the other, so neither satisfies a plan asking for
    it.
    """
    assert not satisfies("extractorT3_overclocked", "extractorT3_reinforced")
    assert not satisfies("extractorT3_reinforced", "extractorT3_overclocked")


def test_the_walk_advances_one_tier_where_the_paths_agree() -> None:
    """Both paths say the same thing about the steps below the fork."""
    assert next_tier("extractorT1") == "extractorT2"
    assert next_tier("extractorT2") == "extractorT3"


def test_the_walk_stops_at_the_fork_rather_than_picking_a_branch() -> None:
    """Two successors is an unanswered question, not a default.

    Overclocking and reinforcing are different purchases -- 30 credits a second
    at 1,100 hit points against 20 at 4,700 with a shield -- and which is worth
    more depends on whether the ground is contested. Returning either one here
    would be a preference invented in a constant rather than measured
    ([[policy-holding-ground]]).
    """
    assert next_tier("extractorT3") is None


def test_a_leaf_and_a_stranger_both_convert_into_nothing() -> None:
    """Distinguishing them is not this function's job; both mean "no step"."""
    assert next_tier("extractorT3_overclocked") is None
    assert next_tier("extractorT3_reinforced") is None
    assert next_tier("landFactory") is None


def test_a_type_outside_every_path_satisfies_only_itself() -> None:
    assert satisfies("landFactory", "landFactory")
    assert not satisfies("landFactory", "extractorT1")
    assert not satisfies("extractorT3", "landFactory")


def test_offers_are_rendered_one_per_line_under_a_header() -> None:
    world = sample(
        entity(213, "commandCenter"),
        options=(option(213, "builder"),),
    )
    rendered = format_offers(upgrade_offers(world, _CATALOGUE))
    assert rendered[0].startswith("holder")
    assert rendered[1].split() == ["commandCenter", "213", "builder", "False", "True"]
    assert len(rendered) == 2
