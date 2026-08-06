"""The build-order policy, exercised as the pure function it is.

No socket, no game, no clock. Every case here is a world state in and a
decision out, which is the point of keeping the deciding half pure.

What the plan wants next and whether it can be afforded, ordered or reached at
all. Where the answer may legally stand is the other half of the same call and
lives in ``test_policy_siting``; the world both argue over is
:mod:`tests.build_fixtures`.
"""

from __future__ import annotations

from rw_bot.policy.build_order import completed_count, decide, next_unsatisfied_index
from tests.build_fixtures import (
    ANCHOR,
    BUILDER,
    CATALOGUE,
    PLACEMENTS,
    PROFILES,
    free,
    option,
    pool_at,
    sample,
    unit,
    unit_stats,
)


def test_an_empty_plan_is_immediately_done() -> None:
    world = sample(BUILDER)
    assert decide(world, (), CATALOGUE, PLACEMENTS, PROFILES, free(world))["action"] == "done"


def test_progress_is_read_from_the_roster_not_from_a_counter() -> None:
    """A structure already standing counts, whoever built it."""
    world = sample(BUILDER, unit(300, "landFactory"))
    decision = decide(
        world, ("landFactory", "airFactory"), CATALOGUE, PLACEMENTS, PROFILES, free(world)
    )
    assert decision["type_name"] == "airFactory"


def test_a_destroyed_structure_is_rebuilt_rather_than_counted() -> None:
    """Counting from the roster is what makes this fall out for free."""
    world = sample(BUILDER, credits=4000)
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["type_name"] == "landFactory"


def test_a_finished_plan_reports_done() -> None:
    world = sample(BUILDER, unit(300, "landFactory"), unit(301, "airFactory"))
    decision = decide(
        world, ("landFactory", "airFactory"), CATALOGUE, PLACEMENTS, PROFILES, free(world)
    )
    assert decision["action"] == "done"
    assert decision["reason"] == "all 2 plan entries satisfied"


def test_insufficient_credits_waits_rather_than_ordering() -> None:
    world = sample(BUILDER, credits=899)
    decision = decide(world, ("airFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "wait"
    assert decision["reason"] == "airFactory costs 900, holding 899"
    assert decision["type_name"] == ""
    # The shortfall rides as a number, because the savings clock cannot judge
    # a save it cannot see and parsing it back out of the reason would be the
    # same figure laundered through prose ([[policy-economy]]).
    assert decision["deficit"] == 1


def test_exactly_enough_credits_orders() -> None:
    """The boundary matters: the engine spends in whole units."""
    world = sample(BUILDER, credits=900)
    assert (
        decide(world, ("airFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))["action"]
        == "build"
    )


def test_an_unfinished_structure_does_not_satisfy_the_plan() -> None:
    """A building joins the roster when construction starts, not when it ends.

    Counting on presence reported a plan finished while a factory was still a
    shell -- and a shell produces nothing, so the next entry could be ordered
    against a building that could not accept it.
    """
    shell = unit(300, "landFactory", 4450.0, 2730.0, complete=False)
    world = sample(BUILDER, ANCHOR, shell)
    assert completed_count(world, ("landFactory",)) == 0
    assert next_unsatisfied_index(world, ("landFactory",)) == 0
    assert (
        decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))["action"]
        == "build"
    )


def test_a_finished_structure_does_satisfy_the_plan() -> None:
    """The same world, one flag different, is the whole of the distinction."""
    done = unit(300, "landFactory", 4450.0, 2730.0)
    world = sample(BUILDER, ANCHOR, done)
    assert completed_count(world, ("landFactory",)) == 1
    assert (
        decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))["action"]
        == "done"
    )


def test_a_type_nothing_owned_can_make_is_blocked() -> None:
    """The laboratory failure, caught before an order is spent.

    A builder has no action producing a laboratory. The engine refuses the
    waypoint and says so only in its own log, so the old planner ordered it and
    then reported "building laboratory" for three hundred samples. The build
    tree answers the question up front instead.
    """
    decision = decide(
        sample(BUILDER),
        ("laboratory",),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        free(sample(BUILDER)),
    )
    assert decision["action"] == "blocked"
    assert "nothing the player owns can make laboratory" in decision["reason"]


def test_the_editor_placeholder_never_counts_as_a_producer() -> None:
    """The map editor's unit answers for nearly every type in the game.

    It is owned, it is in every sample, and in the live capture it offers 108
    types against the real Builder's 13 -- a superset, plus 95 more including
    the laboratory. Counting it would make the check above pass for types
    nothing playable can build, and the resulting order would go to a unit
    parked at (-1000, -1000) and do nothing at all.
    """
    placeholder = unit(217, "editorOrBuilder", -1000.0, -1000.0)
    world = sample(
        BUILDER,
        placeholder,
        options=(option(217, "laboratory"),),
    )
    assert (
        decide(world, ("laboratory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))["action"]
        == "blocked"
    )


def test_an_action_that_exists_but_is_unavailable_waits() -> None:
    """Present-but-unavailable is a world state, not a dead plan.

    A prerequisite can still be built and tech can still be researched, so this
    resolves on its own -- unlike an action that does not exist at all.
    """
    world = sample(BUILDER, options=(option(214, "landFactory", available=False),))
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "wait"
    assert "not available yet" in decision["reason"]


def test_the_first_unavailable_option_is_the_one_reported() -> None:
    """Two units both offer it and neither can act yet.

    The wait is reported against one of them rather than collapsing to "nothing
    can make this", which is the answer that would end the run. Which one is
    named is the first seen, so the message stays stable across samples while
    the roster does.
    """
    second = unit(215, "builder", 4300.0, 2610.0)
    world = sample(
        BUILDER,
        second,
        options=(
            option(214, "landFactory", available=False),
            option(215, "landFactory", available=False),
        ),
    )
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "wait"
    assert "unit 214" in decision["reason"]


def test_an_available_action_is_preferred_over_an_unavailable_one() -> None:
    """Two units offer the same type; only one can act on it now."""
    second = unit(215, "builder", 4300.0, 2610.0)
    world = sample(
        BUILDER,
        second,
        options=(
            option(214, "landFactory", available=False),
            option(215, "landFactory"),
        ),
    )
    assert (
        decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))["unit_id"]
        == 215
    )


def test_a_unit_that_rolls_out_is_produced_rather_than_placed() -> None:
    """The engine decides where a produced unit appears, so no site is chosen.

    ``placed`` is the engine's own distinction between the two verbs, read from
    the action rather than guessed from the type's speed.
    """
    centre = unit(213, "commandCenter", 4250.0, 2550.0)
    world = sample(centre, options=(option(213, "builder", placed=False),))
    decision = decide(world, ("builder",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "produce"
    assert decision["unit_id"] == 213
    assert decision["type_name"] == "builder"
    assert (decision["x"], decision["y"]) == (0.0, 0.0)


def test_a_produced_unit_still_has_to_be_afforded() -> None:
    centre = unit(213, "commandCenter", 4250.0, 2550.0)
    world = sample(centre, credits=499, options=(option(213, "builder", placed=False),))
    decision = decide(world, ("builder",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "wait"
    assert decision["reason"] == "builder costs 500, holding 499"
    assert decision["deficit"] == 1


def test_only_the_price_wait_carries_a_shortfall() -> None:
    """A full ring and a missing action are waits the world can end on its
    own; only a deficit is judged for convergence, so only the price wait
    carries one."""
    unready = sample(BUILDER, credits=9000, options=(option(214, "airFactory", available=False),))
    decision = decide(unready, ("airFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(unready))
    assert decision["action"] == "wait"
    assert decision["deficit"] == 0


def test_no_builder_is_blocked_not_a_wait() -> None:
    """Waiting implies it could resolve on its own; this one cannot."""
    world = sample(unit(213, "commandCenter"))
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "blocked"
    assert decision["reason"] == (
        "nothing the player owns can make landFactory; the plan is not playable from here"
    )


def test_a_structure_missing_from_the_catalogue_is_blocked() -> None:
    decision = decide(
        sample(BUILDER),
        ("teleporter",),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        free(sample(BUILDER)),
    )
    assert decision["action"] == "blocked"
    assert "not in the unit catalogue" in decision["reason"]


def test_credits_are_checked_before_a_builder_is_required() -> None:
    """A plan naming an unknown structure fails on the plan, not the roster."""
    decision = decide(sample(), ("teleporter",), CATALOGUE, PLACEMENTS, PROFILES, free(sample()))
    assert decision["action"] == "blocked"
    assert "catalogue" in decision["reason"]


def test_a_type_absent_from_the_placement_dump_is_blocked() -> None:
    """Where it may stand is unknown, which is not the same as unconstrained."""
    catalogue = dict(CATALOGUE)
    catalogue["teleporter"] = unit_stats("teleporter", 100)
    placements = {n: p for n, p in PLACEMENTS.items() if n != "teleporter"}
    decision = decide(sample(BUILDER), ("teleporter",), catalogue, placements, PROFILES)
    assert decision["action"] == "blocked"
    assert decision["reason"] == (
        "'teleporter' is not in the placement dump, so where it may stand is unknown"
    )


def test_completed_count_matches_each_plan_entry_only_once() -> None:
    """Two factories satisfy two plan entries, not one entry twice."""
    world = sample(BUILDER, unit(300, "landFactory"), unit(301, "landFactory"))
    assert completed_count(world, ("landFactory", "landFactory")) == 2
    assert completed_count(world, ("landFactory",)) == 1


def test_a_structure_outside_the_plan_does_not_count_as_progress() -> None:
    world = sample(BUILDER, unit(300, "laboratory"))
    assert completed_count(world, ("landFactory",)) == 0


def test_an_entry_with_nowhere_to_stand_defers_to_the_next() -> None:
    """The fault that lost two duels while the bot was level on worth.

    The opening is a sequence -- extractors, then the factory, then the army --
    and it waited on whichever entry it had reached. With every pool taken the
    wait was permanent: the factory was never built, no army was ever produced,
    and a match ended with five idle builders, 60,676 credits and nothing to
    fight with, against an opponent left to do as it liked
    ([[policy-holding-ground]]).

    The extractor here has no pool, so the plan reaches past it to the factory
    rather than stopping.
    """
    world = sample(BUILDER, ANCHOR, pools=())
    decision = decide(
        world,
        ("extractorT1", "landFactory"),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        free(world),
    )
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"


def test_a_deferred_entry_is_still_wanted_once_a_pool_frees() -> None:
    """Deferring is not dropping: progress is read off the roster every
    observation, so the extractor is taken the moment a pool is claimable.
    """
    world = sample(BUILDER, ANCHOR, pools=(pool_at(0, 15, 0),))
    decision = decide(
        world,
        ("extractorT1", "landFactory"),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        free(world),
    )
    assert decision["type_name"] == "extractorT1"


def test_every_entry_unplaceable_still_waits_and_says_why() -> None:
    """Deferring past the last entry is not a decision, so it reports the
    reason the first one could not be placed.
    """
    world = sample(BUILDER, ANCHOR, pools=())
    decision = decide(world, ("extractorT1",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "wait"
    assert "resource pool" in decision["reason"]


def test_an_unaffordable_entry_still_stops_the_plan() -> None:
    """Only *placement* defers. Being short of credits is a condition on the
    whole plan, so reaching past it would spend the next entry's credits on the
    strength of this one's being short.
    """
    world = sample(BUILDER, ANCHOR, credits=10, pools=(pool_at(0, 15, 0),))
    decision = decide(
        world,
        ("extractorT1", "landFactory"),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        free(world),
    )
    assert decision["action"] == "wait"
    assert "costs 700" in decision["reason"]


def test_an_upgraded_structure_still_satisfies_the_entry_that_built_it() -> None:
    """The regression that cost a match, asked of both readings of progress.

    The plan asks for three ``extractorT1``. They upgrade themselves in place.
    Matching type names exactly, the plan then saw none of them, ordered three
    more, and did that forever -- so the one builder was never free, expansion
    never ran, and the match ended defeated with 41,559 credits banked and the
    plan reading 0 of 8 ([[policy-holding-ground]]).

    Both functions are asserted because they answer different questions off the
    same roster, and it was the disagreement between them that hid the fault:
    a count that advances while the index does not is a plan that reports
    progress it will not act on.
    """
    upgraded = sample(
        BUILDER,
        unit(300, "extractorT2", 100.0, 100.0),
        unit(301, "extractorT3", 200.0, 200.0),
    )
    plan = ("extractorT1", "extractorT1")
    assert completed_count(upgraded, plan) == 2
    assert next_unsatisfied_index(upgraded, plan) == 2


def test_an_earlier_tier_does_not_satisfy_a_later_one() -> None:
    """The relation is one-way, or the plan would skip work it has not done.

    Holding a tier one is not an answer to a plan asking for a tier two. Were
    it symmetric, a plan naming ``extractorT2`` would call itself satisfied by
    the tier one already standing and never upgrade anything.
    """
    world = sample(BUILDER, unit(300, "extractorT1", 100.0, 100.0))
    assert completed_count(world, ("extractorT2",)) == 0
    assert next_unsatisfied_index(world, ("extractorT2",)) == 0


def test_an_upgrade_satisfies_one_entry_rather_than_every_entry_it_outranks() -> None:
    """One structure, one entry -- the tier rule must not double-count.

    A single ``extractorT3`` outranks both entries of a two-extractor plan. It
    is still one structure earning one structure's income, so it satisfies the
    first entry and leaves the second wanting.
    """
    world = sample(BUILDER, unit(300, "extractorT3", 100.0, 100.0))
    plan = ("extractorT1", "extractorT1")
    assert completed_count(world, plan) == 1
    assert next_unsatisfied_index(world, plan) == 1


def test_no_free_worker_means_no_placed_order() -> None:
    """Which workers are free is the loop's judgement, and this is what it buys.

    A structure is only ever ordered from a worker the loop reports as free, so
    two rules can no longer re-task the same one off each other
    ([[policy-loop]]).

    **A busy workforce is a wait now, not a block.** Ruling it "not playable
    from here" cost every Hard win in a batch: defence kept all eight workers
    employed, the factory never met a free one, and the plan stayed blocked
    over a state the world leaves the moment a worker frees
    (log: 2026-07-31).
    """
    world = sample(BUILDER)
    busy = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, ())
    assert busy["action"] == "wait"
    assert busy["reason"] == "every unit that can make landFactory is busy"
    assert busy["unit_id"] == 0
    assert (
        decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))["action"]
        == "build"
    )


def test_an_enemy_structure_does_not_advance_the_plan() -> None:
    """The stream carries enemies, so ownership is what makes progress mine."""
    world = sample(BUILDER, unit(900, "landFactory", mine=False))
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"


def test_an_enemy_builder_is_never_selected() -> None:
    """Ordering a unit we do not own would be rejected by the engine anyway."""
    world = sample(unit(901, "builder", 1.0, 2.0, mine=False))
    assert free(world) == ()
    assert (
        decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))["action"]
        == "blocked"
    )


def test_an_owned_structure_still_counts_when_an_enemy_has_one_too() -> None:
    world = sample(
        BUILDER,
        unit(300, "landFactory"),
        unit(900, "landFactory", mine=False),
    )
    assert completed_count(world, ("landFactory", "landFactory")) == 1


def test_a_plan_naming_a_starting_unit_does_not_skip_earlier_entries() -> None:
    """The count-as-index conflation, which silently skipped a whole entry.

    Every game starts with a builder. Under the old reading, the plan
    ``("landFactory", "builder")`` counted as one-satisfied, jumped to index 1,
    built a second builder and never built the factory at all.
    """
    world = sample(unit(214, "builder"), credits=10_000)
    plan = ("landFactory", "builder")

    assert completed_count(world, plan) == 1
    assert next_unsatisfied_index(world, plan) == 0

    decision = decide(world, plan, CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"


def test_the_first_unsatisfied_entry_is_found_past_a_satisfied_one() -> None:
    world = sample(unit(214, "builder"), unit(300, "landFactory"), credits=10_000)
    assert next_unsatisfied_index(world, ("landFactory", "builder")) == 2
    assert next_unsatisfied_index(world, ("landFactory", "builder", "landFactory")) == 2


def test_an_enemy_unit_never_satisfies_a_plan_entry() -> None:
    world = sample(unit(900, "landFactory", mine=False), credits=10_000)
    assert next_unsatisfied_index(world, ("landFactory",)) == 0
