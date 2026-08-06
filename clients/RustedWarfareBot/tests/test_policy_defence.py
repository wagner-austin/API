"""Covering what has been claimed, exercised as the pure function it is.

A structure with no turret in reach of it is a structure the opponent gets for
free. This asks two questions of a world: which structure is bare, and where a
turret covering it may stand.

Split from ``test_policy_economy`` because claiming ground and holding it fail
for unrelated reasons -- and because the module that held both, plus the
throughput rule, was 932 lines. The world all three argue over is
:mod:`tests.economy_fixtures`.
"""

from __future__ import annotations

from rw_bot.policy.defence import expand_defence, undefended
from rw_bot.policy.economy import EXTRACTOR_TYPE
from rw_bot.policy.siting import COVER_RING, RING_SLOT_RADIUS
from rw_bot.wire.state import Entity, Sample
from tests.economy_fixtures import BUILDER, CATALOGUE, PROFILES, free, option, sample, unit

#: The option by which the engine says the Builder can place a turret.
CAN_DEFEND = option(214, "turret")


def defence_world(*entities: Entity, cover_base: bool = True) -> Sample:
    """A base with a Command Center, a Builder, and by default a turret on it.

    The Command Center is itself a structure with no cover, so a world without
    the base turret answers "cover the base" to every question. Covering it in
    the fixture is what lets a test isolate the structure it is actually about.
    """
    base = (unit(299, "turret", 0.0, 0.0),) if cover_base else ()
    return sample(
        unit(213, "commandCenter", 0.0, 0.0),
        BUILDER,
        *base,
        *entities,
        options=(CAN_DEFEND,),
    )


def bare(world: Sample) -> Entity:
    """The structure needing cover, failing loudly when there is none.

    Written as a helper so the assertions below can name the structure they
    expect rather than merely checking that *something* came back.
    """
    found = undefended(world, CATALOGUE, PROFILES, "turret")
    if found is None:
        raise AssertionError("expected a structure needing cover, got none")
    return found


def test_a_structure_with_no_turret_near_it_is_the_one_to_cover() -> None:
    """The finding this exists for: played to a verdict the bot is defeated, and
    the build policy names the cause -- 44 of the map's 46 pools end up owned by
    the opponents, one run spending 275 expansion orders for a single surviving
    extractor. A turret costs less than the extractor it covers
    ([[policy-holding-ground]]).
    """
    world = defence_world(unit(400, EXTRACTOR_TYPE, 900.0, 0.0))
    assert bare(world)["unit_id"] == 400


def test_cover_is_the_turrets_own_reach_rather_than_a_chosen_radius() -> None:
    """A structure is covered when a turret could actually shoot something
    standing on it, which the registry answers ([[mechanics-combat-profile]]).
    """
    # The fixture turret reaches 100.
    covered = defence_world(
        unit(400, EXTRACTOR_TYPE, 900.0, 0.0),
        unit(401, "turret", 950.0, 0.0),
    )
    assert undefended(covered, CATALOGUE, PROFILES, "turret") is None

    out_of_reach = defence_world(
        unit(400, EXTRACTOR_TYPE, 900.0, 0.0),
        unit(401, "turret", 1100.0, 0.0),
    )
    assert bare(out_of_reach)["unit_id"] == 400


def test_the_base_is_covered_before_the_frontier() -> None:
    """Nearest to the anchor first, and it is kept because covering the
    extractors instead was **measured worse**.

    The argument against this ordering was the strongest the defence policy has
    had. Extractor losses decide every duel; the per-loss table puts each death
    688-1,766 world units from the army's own fighting cloud; and not one unit
    died within 900 units of the base across two traced runs, so nearest-first
    reads as spending where nothing is ever attacked. Restricting cover to the
    extractors, same twelve seeds and same rung: **wins 4 -> 0, drops 21 -> 24,
    and the first two defeats in fifty-two duels** ([[policy-holding-ground]]).

    So the ordering stands on the measurement rather than on the reasoning, and
    the reasoning is recorded because it was good and still lost.
    """
    world = defence_world(
        unit(400, EXTRACTOR_TYPE, 4000.0, 0.0),
        unit(401, EXTRACTOR_TYPE, 400.0, 0.0),
    )
    assert bare(world)["unit_id"] == 401


def test_a_mobile_unit_is_not_a_structure_to_cover() -> None:
    """A turret cannot follow a tank, so covering one is not a thing to buy."""
    world = defence_world(unit(400, "c_tank", 900.0, 0.0), cover_base=False)
    # The Command Center and the Builder are the only other candidates, and the
    # Builder moves; the Command Center is at the anchor and is itself bare.
    assert bare(world)["type_name"] == "commandCenter"


def test_nothing_to_anchor_on_means_nothing_to_cover() -> None:
    world = sample(BUILDER, options=(CAN_DEFEND,))
    assert undefended(world, CATALOGUE, PROFILES, "turret") is None


def test_defence_places_a_turret_beside_what_it_covers() -> None:
    """Beside the structure rather than on the base ring: a turret at the base
    does not defend a pool on the far side of the map, and the pools are what is
    being lost.
    """
    world = defence_world(unit(400, EXTRACTOR_TYPE, 900.0, 40.0))
    plan = expand_defence(
        world, CATALOGUE, PROFILES, available=4000, free=free(world), turret_type="turret"
    )
    assert plan["build"]
    assert plan["type_name"] == "turret"
    assert plan["unit_id"] == 214
    assert (plan["x"], plan["y"]) == (900.0 + RING_SLOT_RADIUS, 40.0)


def test_the_editor_placeholder_is_never_a_structure_to_cover() -> None:
    """The placeholder trap, sprung in a new consumer: an owned immobile
    entity parked off-map with 170,000 hp is eternally bare, and the moment
    cover WORKED everywhere real, it became the only bare structure left --
    defence poured turrets at an unbuildable point for whole matches and
    walked 46 of 51 builders to their deaths (log: 2026-07-31)."""
    world = defence_world(
        unit(999, "editorOrBuilder", -1000.0, -1000.0),
    )
    assert undefended(world, CATALOGUE, PROFILES, "turret") is None


def test_defence_steps_around_an_occupied_cover_slot() -> None:
    """The check defence never had: the old bare offset was reached for
    without looking, the engine refuses an occupied site silently, and one
    scorecard priced the habit at 27 paid orders for about five turrets
    standing. An occupied first slot now means the second, not a refusal."""
    world = defence_world(
        unit(400, EXTRACTOR_TYPE, 900.0, 40.0),
        unit(402, EXTRACTOR_TYPE, 960.0, 40.0),
        unit(403, "turret", 960.0, 140.0),
    )
    plan = expand_defence(
        world, CATALOGUE, PROFILES, available=4000, free=free(world), turret_type="turret"
    )
    assert plan["build"]
    # (60, 0) is occupied by the neighbouring extractor, so (0, 60) is next.
    assert (plan["x"], plan["y"]) == (900.0, 100.0)


def test_defence_waits_when_every_cover_slot_is_taken() -> None:
    """A refusal with a reason, where the old offset was a silent one."""
    # Hostile, because an opponent's building fills a position exactly as
    # firmly as ours -- and because our own would themselves ask for cover.
    blockers = tuple(
        unit(500 + i, EXTRACTOR_TYPE, 900.0 + dx, 40.0 + dy, mine=False)
        for i, (dx, dy) in enumerate(COVER_RING)
    )
    world = defence_world(unit(400, EXTRACTOR_TYPE, 900.0, 40.0), *blockers)
    plan = expand_defence(
        world, CATALOGUE, PROFILES, available=4000, free=free(world), turret_type="turret"
    )
    assert not plan["build"]
    assert plan["reason"] == "no clear cover position around extractorT1 at 900,40"


def test_defence_waits_when_it_cannot_afford_the_turret() -> None:
    """And says credits were the ONLY obstacle, so the expander saves toward
    it -- a priced-out wait now implies a bare structure exists, because
    demand is checked before price (log 2026-08-01)."""
    world = defence_world(unit(400, EXTRACTOR_TYPE, 900.0, 0.0))
    plan = expand_defence(
        world, CATALOGUE, PROFILES, available=100, free=free(world), turret_type="turret"
    )
    assert not plan["build"]
    assert "300 of 100" in plan["reason"]
    assert plan["priced_out"] is True


def test_a_broke_tick_with_everything_covered_is_not_a_deficit() -> None:
    """The reorder's point: no bare structure means nothing to save toward,
    however empty the balance -- withholding here would be a permanent tax."""
    world = defence_world(
        unit(400, EXTRACTOR_TYPE, 900.0, 0.0),
        unit(401, "turret", 900.0, 0.0),
        unit(3, "turret", 0.0, 0.0),
    )
    plan = expand_defence(
        world, CATALOGUE, PROFILES, available=100, free=free(world), turret_type="turret"
    )
    assert not plan["build"]
    assert plan["reason"] == "every structure already has cover"
    assert plan["priced_out"] is False


def test_defence_waits_when_everything_already_has_cover() -> None:
    world = defence_world(
        unit(400, EXTRACTOR_TYPE, 900.0, 0.0),
        unit(401, "turret", 900.0, 0.0),
        unit(3, "turret", 0.0, 0.0),
    )
    plan = expand_defence(
        world, CATALOGUE, PROFILES, available=4000, free=free(world), turret_type="turret"
    )
    assert not plan["build"]
    assert plan["reason"] == "every structure already has cover"


def test_defence_waits_when_no_worker_can_place_one() -> None:
    world = defence_world(unit(400, EXTRACTOR_TYPE, 900.0, 0.0))
    plan = expand_defence(world, CATALOGUE, PROFILES, available=4000, free=(), turret_type="turret")
    assert not plan["build"]
    assert "no free worker" in plan["reason"]


def test_a_turret_the_catalogue_cannot_price_is_never_ordered() -> None:
    """Unpriced means unbudgetable, and spending blind is what the budget
    prevents ([[policy-budget]]).
    """
    world = defence_world(unit(400, EXTRACTOR_TYPE, 900.0, 0.0))
    plan = expand_defence(
        world, CATALOGUE, PROFILES, available=4000, free=free(world), turret_type="mysteryGun"
    )
    assert not plan["build"]
    assert "does not price" in plan["reason"]
