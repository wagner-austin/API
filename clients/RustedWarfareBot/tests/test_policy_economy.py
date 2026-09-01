"""Claiming resource pools, exercised as the pure function it is.

No socket and no game: a world state goes in and either a site or a stated
reason comes out. The reasons are asserted as carefully as the sites, because a
match that never expanded has to be able to say which of the five causes it hit
-- and a bare "expansions: 0" cannot.

Claiming ground and upgrading what stands on it; holding it is
``test_policy_defence`` and converting surplus into queues is
``test_policy_throughput``. The world all three argue over is
:mod:`tests.economy_fixtures`.
"""

from __future__ import annotations

from rw_bot.policy.economy import (
    EXTRACTOR_TYPE,
    count_extractors,
    expand_economy,
    upgradeable,
)
from tests.economy_fixtures import (
    BUILDER,
    CAN_PLACE,
    CATALOGUE,
    PROFILES,
    free,
    option,
    pool_at,
    sample,
    unit,
)


def test_an_upgraded_extractor_still_counts_as_an_extractor() -> None:
    """A figure that quietly means something else is how a reading goes wrong.

    Counting the named type alone was right until the bot could upgrade. The
    moment it could, a run holding three tier-two extractors and earning 54
    credits a second reported ``extractors 0 -> 0``
    ([[policy-holding-ground]]).
    """
    world = sample(
        unit(400, "extractorT1"),
        unit(401, "extractorT2"),
        unit(402, "extractorT3"),
        unit(403, "landFactory"),
    )
    assert count_extractors(world) == 3


def test_an_extractor_offering_to_upgrade_itself_is_found() -> None:
    """The action that was invisible for the whole life of the bot.

    An upgrade is declared in the asset as ``convertTo`` and the engine reports
    it with no placement type and no "makes something" flag -- exactly the shape
    the agent used to drop. With every action published, all four standing
    extractors offer ``extractorT2`` and the engine calls it available, with no
    tier-2 builder and no prerequisite chain ([[policy-holding-ground]]).
    """
    world = sample(
        unit(400, EXTRACTOR_TYPE),
        unit(401, EXTRACTOR_TYPE),
        options=(
            option(400, "extractorT2", placed=False),
            option(401, "extractorT2", placed=False),
        ),
    )
    assert upgradeable(world) == (
        {"unit_id": 400, "produces": "extractorT2"},
        {"unit_id": 401, "produces": "extractorT2"},
    )


def test_each_structure_is_offered_its_own_next_tier() -> None:
    """The walk that used to stop at tier two.

    A single wanted type cannot describe a roster holding both tiers, so the
    tier two was never asked to become a tier three -- 20 credits a second
    against 12, beside a bank of 62,146 that had nothing to buy
    ([[mechanics-unit-value]]).
    """
    world = sample(
        unit(400, EXTRACTOR_TYPE),
        unit(401, "extractorT2"),
        options=(
            option(400, "extractorT2", placed=False),
            option(401, "extractorT3", placed=False),
        ),
    )
    assert upgradeable(world) == (
        {"unit_id": 400, "produces": "extractorT2"},
        {"unit_id": 401, "produces": "extractorT3"},
    )


def test_a_structure_at_the_fork_is_left_alone() -> None:
    """The tier three offers two conversions and neither is an upgrade of the
    other, so choosing between them here would be a preference nobody measured
    ([[mechanics-unit-value]]).
    """
    world = sample(
        unit(400, "extractorT3"),
        options=(
            option(400, "extractorT3_overclocked", placed=False),
            option(400, "extractorT3_reinforced", placed=False),
        ),
    )
    assert upgradeable(world) == ()


def test_an_extractor_already_upgrading_is_left_alone() -> None:
    """A queued conversion is in progress; ordering it again spends twice."""
    world = sample(
        unit(400, EXTRACTOR_TYPE, queued=1),
        options=(option(400, "extractorT2", placed=False),),
    )
    assert upgradeable(world) == ()


def test_an_upgrade_the_engine_has_not_unlocked_is_not_ordered() -> None:
    """Availability is the engine's own predicate, so tech gating and the unit
    cap are already accounted for rather than modelled here
    ([[mechanics-build-actions]]).
    """
    world = sample(
        unit(400, EXTRACTOR_TYPE),
        options=(option(400, "extractorT2", placed=False, available=False),),
    )
    assert upgradeable(world) == ()


def test_a_placed_option_is_not_an_upgrade() -> None:
    """Placing a second extractor somewhere is expansion, not conversion."""
    world = sample(
        unit(400, EXTRACTOR_TYPE),
        options=(option(400, "extractorT2", placed=True),),
    )
    assert upgradeable(world) == ()


def test_a_type_that_heads_no_chain_offers_no_upgrade() -> None:
    """The chain says which tier is next; a type outside it has no next."""
    world = sample(
        unit(400, "landFactory"),
        options=(option(400, "c_tank", placed=False),),
    )
    assert upgradeable(world) == ()


def test_only_finished_extractors_count_as_income() -> None:
    """A structure joins the roster when construction starts, not when it pays."""
    world = sample(
        unit(1, EXTRACTOR_TYPE),
        unit(2, EXTRACTOR_TYPE, complete=False),
        unit(3, EXTRACTOR_TYPE, mine=False),
    )
    assert count_extractors(world) == 1


def test_a_free_pool_is_claimed() -> None:
    world = sample(
        BUILDER,
        unit(213, "commandCenter", 0.0, 0.0),
        pools=(pool_at(300.0, 0.0),),
        options=(CAN_PLACE,),
    )
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert plan["build"] is True
    assert plan["unit_id"] == 214
    assert plan["type_name"] == EXTRACTOR_TYPE
    assert (plan["x"], plan["y"]) == (300.0, 0.0)
    assert plan["reason"] == f"{EXTRACTOR_TYPE} #1 at (300, 0)"


def test_the_reason_counts_the_extractors_already_standing() -> None:
    """So a run log reads as a sequence rather than a repeated sentence."""
    world = sample(
        BUILDER,
        unit(1, EXTRACTOR_TYPE, 500.0, 500.0),
        unit(2, EXTRACTOR_TYPE, 600.0, 600.0),
        pools=(pool_at(300.0, 0.0),),
        options=(CAN_PLACE,),
    )
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert plan["reason"] == f"{EXTRACTOR_TYPE} #3 at (300, 0)"
    assert plan["owned"] == 2


def test_a_worker_the_loop_has_not_offered_is_not_used() -> None:
    """Availability has one owner, and it is not this module.

    A worker walking to a site, or already building one, is not in the free
    list the loop supplies -- which is what stopped two expansion rules
    re-tasking the same worker off each other ([[policy-loop]]).
    """
    world = sample(BUILDER, pools=(pool_at(300.0, 0.0),), options=(CAN_PLACE,))
    plan = expand_economy(world, CATALOGUE, PROFILES, reserve=0, free=(), claimed=(), refused=())
    assert plan["build"] is False
    assert plan["reason"] == f"no free worker can place {EXTRACTOR_TYPE}"


def test_a_second_worker_can_claim_a_pool_while_the_first_builds() -> None:
    """The point of tracking workers rather than "the" builder.

    One worker mid-build no longer blocks the other. Whether each is busy is
    the loop's judgement, and it hands over only the ones that are not
    ([[policy-loop]]).
    """
    second = unit(215, "builder", 100.0, 0.0)
    world = sample(
        BUILDER,
        second,
        unit(9, EXTRACTOR_TYPE, 300.0, 0.0, complete=False),
        pools=(pool_at(900.0, 0.0),),
        options=(CAN_PLACE, option(215)),
    )
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=(second,), claimed=(), refused=()
    )
    assert plan["build"] is True
    assert plan["unit_id"] == 215


def test_an_enemy_extractor_going_up_does_not_block_ours() -> None:
    """Ownership is what makes the rising check about our own construction."""
    world = sample(
        BUILDER,
        unit(9, EXTRACTOR_TYPE, 5000.0, 5000.0, mine=False, complete=False),
        pools=(pool_at(300.0, 0.0),),
        options=(CAN_PLACE,),
    )
    assert expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )["build"]


def test_no_builder_alive_stops_the_economy_and_says_so() -> None:
    """The caller reads this reason to know it should make another builder."""
    world = sample(unit(213, "commandCenter"), pools=(pool_at(300.0, 0.0),))
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert plan["build"] is False
    assert plan["reason"] == f"no free worker can place {EXTRACTOR_TYPE}"


def test_an_unavailable_action_is_not_a_placer() -> None:
    """The action exists but cannot be used, which is a wait rather than a site."""
    world = sample(
        BUILDER,
        pools=(pool_at(300.0, 0.0),),
        options=(option(214, available=False),),
    )
    assert (
        expand_economy(
            world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
        )["build"]
        is False
    )


def test_an_option_from_a_unit_not_in_the_roster_is_not_a_placer() -> None:
    """A dead producer's option can outlive it in a partial observation."""
    world = sample(BUILDER, pools=(pool_at(300.0, 0.0),), options=(option(999),))
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert plan["build"] is False


def test_the_map_editor_placeholder_never_places_anything() -> None:
    """It offers nearly every type in the game and is parked off-map.

    Ordering against it spends nothing and builds nothing, which is the silent
    failure the placeholder exclusion exists to prevent.
    """
    world = sample(
        unit(217, "editorOrBuilder", -1000.0, -1000.0),
        pools=(pool_at(300.0, 0.0),),
        options=(option(217),),
    )
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert plan["build"] is False
    assert plan["reason"] == f"no free worker can place {EXTRACTOR_TYPE}"


def test_a_type_the_catalogue_cannot_price_is_refused_by_name() -> None:
    """A stale dump names itself here rather than freezing the economy silently."""
    world = sample(BUILDER, pools=(pool_at(300.0, 0.0),), options=(option(214, "someModRig"),))
    plan = expand_economy(
        world,
        CATALOGUE,
        PROFILES,
        reserve=0,
        free=free(world),
        claimed=(),
        refused=(),
        type_name="someModRig",
    )
    assert plan["build"] is False
    assert plan["reason"] == "someModRig is not in the catalogue"


def test_the_reserve_is_held_back_for_the_army() -> None:
    """Expansion competes with reinforcement for the same credits."""
    world = sample(
        BUILDER,
        pools=(pool_at(300.0, 0.0),),
        options=(CAN_PLACE,),
        credits_held=1000,
    )
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=350, free=free(world), claimed=(), refused=()
    )
    assert plan["build"] is False
    assert plan["reason"] == "1000 credits, need 1050 to expand past a 350 reserve"


def test_the_reserve_boundary_is_inclusive() -> None:
    """Exactly enough is enough; refusing here would bank a credit forever."""
    world = sample(
        BUILDER,
        pools=(pool_at(300.0, 0.0),),
        options=(CAN_PLACE,),
        credits_held=1050,
    )
    assert expand_economy(
        world, CATALOGUE, PROFILES, reserve=350, free=free(world), claimed=(), refused=()
    )["build"]


def test_every_pool_taken_is_reported_with_its_counts() -> None:
    """ "No pool" has several causes and they call for opposite responses."""
    world = sample(
        BUILDER,
        unit(9, EXTRACTOR_TYPE, 300.0, 0.0, mine=False),
        pools=(pool_at(300.0, 0.0), pool_at(900.0, 0.0, group_land=7)),
        options=(CAN_PLACE,),
    )
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert plan["build"] is False
    assert plan["reason"] == "no pool free of 2: 1 occupied, 1 unreachable, 0 exposed"
    assert plan["occupied"] == 1


def test_a_pool_under_hostile_guns_is_refused() -> None:
    """The builder dies in transit, so the route is what is judged."""
    world = sample(
        BUILDER,
        unit(9, "turret", 150.0, 0.0, mine=False),
        pools=(pool_at(300.0, 0.0),),
        options=(CAN_PLACE,),
    )
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert plan["build"] is False
    assert plan["exposed"] == 1


def test_the_economy_grows_outward_from_the_base_not_from_the_builder() -> None:
    """Distance is measured from the anchor, so the base stays defensible.

    The builder ends every job standing on the pool it just took, so ranking by
    distance from the builder would walk it steadily off the map.
    """
    centre = unit(213, "commandCenter", 0.0, 0.0)
    strayed = unit(214, "builder", 900.0, 0.0)
    world = sample(
        strayed,
        centre,
        pools=(pool_at(200.0, 0.0), pool_at(1000.0, 0.0)),
        options=(CAN_PLACE,),
    )
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert (plan["x"], plan["y"]) == (200.0, 0.0)


def test_a_player_holding_no_structure_measures_from_the_builder() -> None:
    """The opening fallback, and the build plan's own."""
    world = sample(
        unit(214, "builder", 900.0, 0.0),
        pools=(pool_at(200.0, 0.0), pool_at(1000.0, 0.0)),
        options=(CAN_PLACE,),
    )
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert (plan["x"], plan["y"]) == (1000.0, 0.0)


def test_no_pool_in_sight_is_not_an_error() -> None:
    world = sample(BUILDER, options=(CAN_PLACE,))
    plan = expand_economy(
        world, CATALOGUE, PROFILES, reserve=0, free=free(world), claimed=(), refused=()
    )
    assert plan["build"] is False
    assert plan["reason"] == "no pool free of 0: 0 occupied, 0 unreachable, 0 exposed"
