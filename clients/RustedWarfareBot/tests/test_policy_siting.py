"""Where a structure may stand, exercised as the pure function it is.

The other half of the build decision. ``test_policy_build_order`` asks what the
plan wants next; this asks where the answer can legally go -- which anchor the
ring is measured from, which ring slots are free, and which resource pools a
builder can actually reach and survive the walk to.

Split from that module because the two fail for unrelated reasons: a plan bug
is arithmetic over the roster, a siting bug is geometry over the map, and a
module holding both was 1,170 lines in which neither could be read alone.
"""

from __future__ import annotations

from rw_bot.policy.build_order import decide
from rw_bot.policy.siting import PLACEMENT_RING, POOL_OCCUPIED_RADIUS, find_anchor, survey_pools
from tests.build_fixtures import (
    ANCHOR,
    ARMED,
    BUILDER,
    CATALOGUE,
    ISLAND,
    PLACEMENTS,
    PROFILES,
    TURRET_RANGE,
    free,
    pool_at,
    sample,
    unit,
)


def test_the_first_structure_is_ordered_from_the_builders_position() -> None:
    decision = decide(
        sample(BUILDER),
        ("landFactory",),
        CATALOGUE,
        PLACEMENTS,
        PROFILES,
        free(sample(BUILDER)),
    )
    assert decision["action"] == "build"
    assert decision["type_name"] == "landFactory"
    assert decision["unit_id"] == 214
    assert (decision["x"], decision["y"]) == (
        4250.0 + PLACEMENT_RING[0][0],
        2610.0 + PLACEMENT_RING[0][1],
    )


def test_successive_structures_take_successive_ring_positions() -> None:
    world = sample(ANCHOR, BUILDER, unit(300, "landFactory", 4450.0, 2670.0))
    decision = decide(
        world, ("landFactory", "airFactory"), CATALOGUE, PLACEMENTS, PROFILES, free(world)
    )
    assert (decision["x"], decision["y"]) == (
        ANCHOR["x"] + PLACEMENT_RING[1][0],
        ANCHOR["y"] + PLACEMENT_RING[1][1],
    )


def test_a_ring_position_already_built_on_is_passed_over() -> None:
    """The site is the first *free* position, read from the world.

    Two schemes used to index this ring by counting -- the plan by its own
    position, the economy by how many immobile structures were standing, which
    counts extractors sitting on pools nowhere near it. Neither answers "which
    position is free", and two counters can land on the same slot, which the
    engine refuses silently ([[policy-loop]]).
    """
    first = (ANCHOR["x"] + PLACEMENT_RING[0][0], ANCHOR["y"] + PLACEMENT_RING[0][1])
    world = sample(ANCHOR, BUILDER, unit(400, "laboratory", first[0], first[1]))
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert (decision["x"], decision["y"]) == (
        ANCHOR["x"] + PLACEMENT_RING[1][0],
        ANCHOR["y"] + PLACEMENT_RING[1][1],
    )


def test_an_enemy_building_fills_a_ring_position_too() -> None:
    """It occupies the ground exactly as firmly, and ordering onto it spends
    the credits for nothing.
    """
    first = (ANCHOR["x"] + PLACEMENT_RING[0][0], ANCHOR["y"] + PLACEMENT_RING[0][1])
    theirs = unit(900, "laboratory", first[0], first[1], mine=False)
    world = sample(ANCHOR, BUILDER, theirs)
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert (decision["x"], decision["y"]) != first


def test_a_mobile_unit_standing_on_a_ring_position_does_not_fill_it() -> None:
    """It moves; a building does not. Counting it would hide a usable slot."""
    first = (ANCHOR["x"] + PLACEMENT_RING[0][0], ANCHOR["y"] + PLACEMENT_RING[0][1])
    world = sample(ANCHOR, BUILDER, unit(401, "c_tank", first[0], first[1]))
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert (decision["x"], decision["y"]) == first


def test_a_full_ring_waits_rather_than_ordering_onto_a_building() -> None:
    """A structure destroyed frees its slot, so the world can leave this state."""
    filled = [
        unit(400 + i, "laboratory", ANCHOR["x"] + dx, ANCHOR["y"] + dy)
        for i, (dx, dy) in enumerate(PLACEMENT_RING)
    ]
    world = sample(ANCHOR, BUILDER, *filled)
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "wait"
    assert "all are taken" in decision["reason"]


def test_a_ring_exhausted_by_refusals_stalls_the_plan_loudly() -> None:
    """No event in the world un-refuses a slot: a destroyed structure frees a
    TAKEN one, but a refused one was empty all along. Waiting here is the
    armyless match the Hard panel measured (wiki log 2026-08-31), so the plan
    ends and says exactly what stood in its way."""
    world = sample(ANCHOR, BUILDER)
    refused = tuple((ANCHOR["x"] + dx, ANCHOR["y"] + dy) for dx, dy in PLACEMENT_RING)
    decision = decide(
        world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world), (), refused
    )
    assert decision["action"] == "stalled"
    assert decision["reason"] == (
        "landFactory has nowhere the engine will take: "
        "8 free ring position(s) were refused silently and the rest are occupied"
    )


def test_the_free_slot_is_found_through_a_partly_filled_ring() -> None:
    """Not merely the first or the last: the first one actually free."""
    filled = [
        unit(400 + i, "laboratory", ANCHOR["x"] + dx, ANCHOR["y"] + dy)
        for i, (dx, dy) in enumerate(PLACEMENT_RING[:3])
    ]
    world = sample(ANCHOR, BUILDER, *filled)
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert (decision["x"], decision["y"]) == (
        ANCHOR["x"] + PLACEMENT_RING[3][0],
        ANCHOR["y"] + PLACEMENT_RING[3][1],
    )


def test_placement_is_measured_from_a_fixed_structure_not_the_builder() -> None:
    """The ring only spreads if its centre holds still.

    Measuring from the builder collapsed the spread, because the builder walks
    to each site it is sent to. Observed live: the first factory landed at
    (4450, 2730) and the third order went to (4451, 2646) -- 84 apart, close
    enough to overlap, and the engine silently refused it.
    """
    command_centre = unit(213, "commandCenter", 4250.0, 2550.0)
    builder = unit(214, "builder", 4250.0, 2610.0)
    plan = ("landFactory", "landFactory", "landFactory")

    owned = [command_centre, builder]
    sites: list[tuple[float, float]] = []
    for _step in range(3):
        world = sample(*owned, credits=99_999)
        decision = decide(world, plan, CATALOGUE, PLACEMENTS, PROFILES, free(world))
        assert decision["action"] == "build"
        sites.append((decision["x"], decision["y"]))
        # The builder ends each build standing at the site it just built.
        owned = [
            command_centre,
            unit(214, "builder", decision["x"], decision["y"]),
            *[unit(300 + i, "landFactory", x, y) for i, (x, y) in enumerate(sites)],
        ]

    assert len(set(sites)) == 3
    first_to_third = abs(sites[0][1] - sites[2][1]) + abs(sites[0][0] - sites[2][0])
    assert first_to_third == 240.0


def test_the_anchor_is_the_oldest_owned_immobile_structure() -> None:
    """The factory is listed first, so first-seen would pick the wrong one."""
    command_centre = unit(213, "commandCenter", 1.0, 2.0)
    world = sample(
        unit(400, "landFactory", 9.0, 9.0),
        command_centre,
        unit(214, "builder", 5.0, 5.0),
        credits=10_000,
    )
    assert find_anchor(world, CATALOGUE) == command_centre


def test_a_mobile_unit_is_never_the_anchor() -> None:
    """Immobility is read from the catalogue, not guessed from the type name."""
    world = sample(unit(214, "builder", 5.0, 5.0), credits=10_000)
    assert find_anchor(world, CATALOGUE) is None


def test_an_enemy_structure_is_never_the_anchor() -> None:
    world = sample(
        unit(1, "commandCenter", 0.0, 0.0, mine=False),
        unit(214, "builder", 5.0, 5.0),
        credits=10_000,
    )
    assert find_anchor(world, CATALOGUE) is None


def test_with_no_structure_owned_the_builder_is_the_reference() -> None:
    """A player who has lost every building must still be able to rebuild."""
    world = sample(BUILDER, credits=10_000)
    decision = decide(world, ("landFactory",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "build"
    assert (decision["x"], decision["y"]) == (
        BUILDER["x"] + PLACEMENT_RING[0][0],
        BUILDER["y"] + PLACEMENT_RING[0][1],
    )


def test_an_extractor_is_placed_on_a_pool_and_not_on_the_ring() -> None:
    """The ring is not a legal site for it, so offering one would be refused."""
    pool = pool_at(0, 200, 130)
    world = sample(ANCHOR, BUILDER, pools=(pool,), credits=10_000)
    decision = decide(world, ("extractorT1",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "build"
    assert decision["type_name"] == "extractorT1"
    assert (decision["x"], decision["y"]) == (pool["x"], pool["y"])


def test_the_nearest_free_pool_to_the_anchor_is_chosen() -> None:
    """Nearest to the base, so the economy grows outward rather than wandering."""
    near = pool_at(0, 220, 130)
    far = pool_at(1, 10, 10)
    world = sample(ANCHOR, BUILDER, pools=(far, near), credits=10_000)
    decision = decide(world, ("extractorT1",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert (decision["x"], decision["y"]) == (near["x"], near["y"])


def test_a_pool_under_a_structure_is_not_offered_again() -> None:
    taken = pool_at(0, 220, 130)
    spare = pool_at(1, 230, 130)
    standing = unit(400, "extractorT1", taken["x"], taken["y"])
    world = sample(ANCHOR, BUILDER, standing, pools=(taken, spare), credits=10_000)
    decision = decide(
        world, ("extractorT1", "extractorT1"), CATALOGUE, PLACEMENTS, PROFILES, free(world)
    )
    assert (decision["x"], decision["y"]) == (spare["x"], spare["y"])


def test_an_enemy_extractor_holds_a_pool_just_as_firmly() -> None:
    """Ownership is irrelevant to whether the ground is free."""
    taken = pool_at(0, 220, 130)
    enemy = unit(900, "extractorT1", taken["x"], taken["y"], mine=False)
    world = sample(ANCHOR, BUILDER, enemy, pools=(taken,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, (), ()) == {
        "pool": None,
        "visible": 1,
        "occupied": 1,
        "unreachable": 0,
        "exposed": 0,
        "refused_blocked": 0,
        "embargoed_blocked": 0,
    }


def test_an_embargoed_pool_is_withheld_and_counted() -> None:
    """A razed pool waits for the wave that took it to break: withheld
    from the offer while the caller says so, offered again the moment the
    caller passes nothing -- unlike a refusal, the exclusion is temporary
    ([[impossible-economy-problem]])."""
    razed = pool_at(0, 220, 130)
    spare = pool_at(1, 600, 130)
    world = sample(ANCHOR, BUILDER, pools=(razed, spare), credits=10_000)
    held = survey_pools(
        world, ANCHOR, BUILDER, CATALOGUE, PROFILES, (), (), ((razed["x"], razed["y"]),)
    )
    assert held["pool"] == spare
    assert held["embargoed_blocked"] == 1
    released = survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, (), (), ())
    assert released["pool"] == razed
    assert released["embargoed_blocked"] == 0


def test_a_pool_another_worker_is_walking_to_is_not_free() -> None:
    """The defect that ate nineteen expansion orders in twenty.

    Occupancy is judged by what is *standing* on a pool, so one a builder is
    merely walking toward reads as free. With a single free worker that was
    nearly harmless -- one order in flight at a time. The moment several workers
    were freed ([[policy-economy]]) each was offered the same nearest pool on
    successive observations, because none had arrived yet: an instrumented run
    granted **23 extractor orders, lost nothing at all, and finished with four
    extractors** ([[policy-holding-ground]]).
    """
    taken = pool_at(0, 220, 130)
    spare = pool_at(1, 600, 130)
    world = sample(ANCHOR, BUILDER, pools=(taken, spare), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, (), ())["pool"] == taken
    # Somebody is already on their way to the near one, so the next worker is
    # offered the far one rather than the same site again.
    claimed = ((taken["x"], taken["y"]),)
    survey = survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, claimed, ())
    assert survey["pool"] == spare
    # Counted as occupied rather than given a category of its own: from here,
    # "somebody already has this pool" is one fact.
    assert survey["occupied"] == 1


def test_every_pool_claimed_leaves_nothing_to_take() -> None:
    """The complement, and what stops the fix hiding the map from itself."""
    only = pool_at(0, 220, 130)
    world = sample(ANCHOR, BUILDER, pools=(only,), credits=10_000)
    survey = survey_pools(
        world, ANCHOR, BUILDER, CATALOGUE, PROFILES, ((only["x"], only["y"]),), ()
    )
    assert survey["pool"] is None
    assert survey["occupied"] == 1


def test_a_claim_far_from_a_pool_leaves_it_free() -> None:
    """A worker building a factory on the ring must not burn a pool it is near."""
    pool = pool_at(0, 220, 130)
    world = sample(ANCHOR, BUILDER, pools=(pool,), credits=10_000)
    elsewhere = ((pool["x"] + POOL_OCCUPIED_RADIUS + 0.5, pool["y"]),)
    assert survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, elsewhere, ())["pool"] == pool


def test_a_builder_parked_on_a_pool_does_not_occupy_it() -> None:
    """It stands there after every build; counting it would burn the pool."""
    pool = pool_at(0, 220, 130)
    parked = unit(214, "builder", pool["x"], pool["y"])
    world = sample(ANCHOR, parked, pools=(pool,), credits=10_000)
    assert survey_pools(world, ANCHOR, parked, CATALOGUE, PROFILES, (), ())["pool"] == pool


def test_a_structure_outside_the_occupancy_radius_leaves_a_pool_free() -> None:
    pool = pool_at(0, 220, 130)
    beyond = unit(400, "landFactory", pool["x"] + POOL_OCCUPIED_RADIUS + 0.5, pool["y"])
    world = sample(ANCHOR, BUILDER, beyond, pools=(pool,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, (), ())["pool"] == pool


def test_a_structure_exactly_at_the_occupancy_radius_takes_the_pool() -> None:
    """The boundary is inclusive, and which side it falls on is a real choice."""
    pool = pool_at(0, 220, 130)
    astride = unit(400, "landFactory", pool["x"] + POOL_OCCUPIED_RADIUS, pool["y"])
    world = sample(ANCHOR, BUILDER, astride, pools=(pool,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, (), ())["pool"] is None


def test_a_type_the_catalogue_does_not_know_leaves_the_pool_free() -> None:
    """A wrong 'free' costs one refused order; a wrong 'taken' hides it for good."""
    pool = pool_at(0, 220, 130)
    unknown = unit(400, "someModStructure", pool["x"], pool["y"])
    world = sample(ANCHOR, BUILDER, unknown, pools=(pool,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, (), ())["pool"] == pool


def test_a_pool_across_water_is_not_offered_to_a_land_builder() -> None:
    """The failure the whole reachability check exists for.

    Twelve of the forty-six pools on the archived map sit in components the
    mainland cannot walk to ([[mechanics-movement-layers]]). Distance alone
    would send a builder at one of them the moment the near ground filled up.
    """
    here = pool_at(0, 220, 130)
    across = pool_at(1, 12, 52)
    across["group_land"] = ISLAND
    world = sample(ANCHOR, BUILDER, pools=(across, here), credits=10_000)
    survey = survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, (), ())
    assert survey["pool"] == here
    assert survey["unreachable"] == 1


def test_a_pool_with_no_land_component_at_all_is_not_offered() -> None:
    """A negative id is the engine saying there is no component, not an id.

    Comparing two of them for equality is how the engine's own predicate
    answers true for a point it could not place at all; refusing every negative
    is the more conservative reading.
    """
    nowhere = pool_at(0, 220, 130)
    nowhere["group_land"] = -1
    world = sample(ANCHOR, BUILDER, pools=(nowhere,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, CATALOGUE, PROFILES, (), ())["unreachable"] == 1


def test_a_builder_off_the_land_grid_is_not_offered_a_pool_it_cannot_be_judged_for() -> None:
    """Its own component id belongs to a different grid, so it has none here."""
    stranded = unit(214, "builder", 4250.0, 2610.0, group=-1)
    world = sample(ANCHOR, stranded, pools=(pool_at(0, 220, 130),), credits=10_000)
    assert survey_pools(world, ANCHOR, stranded, CATALOGUE, PROFILES, (), ())["unreachable"] == 1


def test_a_builder_on_another_layer_refuses_the_pool_rather_than_guessing() -> None:
    """No special case, and none needed.

    A hover unit's component id indexes the hover grid, so it matches no land
    component and the pool is simply refused. The safe direction falls out of
    the comparison instead of being arranged by a branch.
    """
    hover = unit(214, "builder", 4250.0, 2610.0, movement="HOVER", group=99)
    world = sample(ANCHOR, hover, pools=(pool_at(0, 220, 130),), credits=10_000)
    survey = survey_pools(world, ANCHOR, hover, CATALOGUE, PROFILES, (), ())
    assert survey["pool"] is None
    assert survey["unreachable"] == 1


def test_a_pool_inside_an_enemy_gun_is_not_offered() -> None:
    pool = pool_at(0, 220, 130)
    turret = unit(900, "turret", pool["x"] + 50.0, pool["y"], mine=False)
    world = sample(ANCHOR, BUILDER, turret, pools=(pool,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, ARMED, PROFILES, (), ()) == {
        "pool": None,
        "visible": 1,
        "occupied": 0,
        "unreachable": 0,
        "exposed": 1,
        "refused_blocked": 0,
        "embargoed_blocked": 0,
    }


def test_a_pool_is_rejected_for_the_walk_even_when_the_pool_itself_is_safe() -> None:
    """The failure this rule exists for: the builder died in transit, not on arrival.

    The turret sits beside the midpoint of the walk and nowhere near either end,
    so a check that only looked at the destination would send the builder
    straight past it.
    """
    pool = pool_at(0, 220, 130)
    midpoint = ((BUILDER["x"] + pool["x"]) / 2, (BUILDER["y"] + pool["y"]) / 2)
    ambush = unit(900, "turret", midpoint[0], midpoint[1] + 50.0, mine=False)
    world = sample(ANCHOR, BUILDER, ambush, pools=(pool,), credits=10_000)

    assert survey_pools(world, ANCHOR, BUILDER, ARMED, PROFILES, (), ())["pool"] is None
    # ... and the same turret standing at the same distance from the pool, but
    # behind the builder rather than between the two, rules out nothing.
    behind = unit(900, "turret", BUILDER["x"], BUILDER["y"] - 400.0, mine=False)
    clear = sample(ANCHOR, BUILDER, behind, pools=(pool,), credits=10_000)
    assert survey_pools(clear, ANCHOR, BUILDER, ARMED, PROFILES, (), ())["pool"] == pool


def test_a_pool_exactly_at_the_edge_of_a_gun_is_rejected() -> None:
    """The boundary is inclusive: a unit at maximum range is a unit in range."""
    pool = pool_at(0, 220, 130)
    turret = unit(900, "turret", pool["x"] + TURRET_RANGE, pool["y"], mine=False)
    world = sample(ANCHOR, BUILDER, turret, pools=(pool,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, ARMED, PROFILES, (), ())["pool"] is None


def test_a_pool_beyond_every_gun_is_offered() -> None:
    pool = pool_at(0, 220, 130)
    turret = unit(900, "turret", pool["x"] + TURRET_RANGE + 0.5, pool["y"], mine=False)
    world = sample(ANCHOR, BUILDER, turret, pools=(pool,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, ARMED, PROFILES, (), ())["pool"] == pool


def test_an_unarmed_enemy_standing_on_the_route_is_not_a_threat() -> None:
    """An enemy builder is an obstacle, not a gun, and ruling out ground it
    happens to stand on would concede the map to something that cannot shoot."""
    pool = pool_at(0, 220, 130)
    harmless = unit(900, "builder", pool["x"] + 10.0, pool["y"], mine=False)
    world = sample(ANCHOR, BUILDER, harmless, pools=(pool,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, ARMED, PROFILES, (), ())["pool"] == pool


def test_an_ally_is_not_a_threat_even_though_it_is_not_mine() -> None:
    """Hostility is the engine's answer, not the negation of ownership."""
    pool = pool_at(0, 220, 130)
    ally = unit(900, "turret", pool["x"] + 50.0, pool["y"], mine=False, hostile=False)
    world = sample(ANCHOR, BUILDER, ally, pools=(pool,), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, ARMED, PROFILES, (), ())["pool"] == pool


def test_the_nearest_safe_pool_beats_a_nearer_exposed_one() -> None:
    """Threat filters before distance ranks, which is the whole ordering.

    The two pools are deliberately not collinear with the builder. When they
    are, the nearer one lies on the walk to the farther one and covering the
    first necessarily covers the route to the second — so a test laid out that
    way could never distinguish the rule from a blanket refusal.
    """
    near = pool_at(0, 220, 130)
    far = pool_at(1, 220, 90)
    turret = unit(900, "turret", near["x"], near["y"] + 50.0, mine=False)
    world = sample(ANCHOR, BUILDER, turret, pools=(near, far), credits=10_000)
    assert survey_pools(world, ANCHOR, BUILDER, ARMED, PROFILES, (), ())["pool"] == far


def test_an_extractor_with_every_pool_taken_waits_rather_than_blocking() -> None:
    """Fog lifts and extractors die, so the world can resolve this on its own."""
    taken = pool_at(0, 220, 130)
    standing = unit(400, "extractorT1", taken["x"], taken["y"])
    world = sample(ANCHOR, BUILDER, standing, pools=(taken,), credits=10_000)
    decision = decide(
        world, ("extractorT1", "extractorT1"), CATALOGUE, PLACEMENTS, PROFILES, free(world)
    )
    assert decision["action"] == "wait"
    assert decision["reason"] == (
        "extractorT1 needs a resource pool: of the 1 in sight, 1 are built on, "
        "0 cannot be walked to and 0 can only be reached through enemy fire"
    )


def test_the_wait_reason_separates_a_taken_pool_from_a_covered_one() -> None:
    """Two different games. One is progress, the other is losing ground."""
    taken = pool_at(0, 220, 130)
    covered = pool_at(1, 260, 130)
    standing = unit(400, "extractorT1", taken["x"], taken["y"])
    turret = unit(900, "turret", covered["x"], covered["y"] + 50.0, mine=False)
    world = sample(ANCHOR, BUILDER, standing, turret, pools=(taken, covered), credits=10_000)
    decision = decide(
        world, ("extractorT1", "extractorT1"), ARMED, PLACEMENTS, PROFILES, free(world)
    )
    assert decision["action"] == "wait"
    assert decision["reason"] == (
        "extractorT1 needs a resource pool: of the 2 in sight, 1 are built on, "
        "0 cannot be walked to and 1 can only be reached through enemy fire"
    )


def test_an_extractor_with_no_pool_visible_waits() -> None:
    world = sample(ANCHOR, BUILDER, credits=10_000)
    decision = decide(world, ("extractorT1",), CATALOGUE, PLACEMENTS, PROFILES, free(world))
    assert decision["action"] == "wait"
    assert decision["reason"] == "extractorT1 needs a resource pool and none is visible yet"


def test_a_pool_placement_ignores_the_ring_index_entirely() -> None:
    """Two extractors in a row must not drift apart the way ring entries do."""
    first = pool_at(0, 220, 130)
    second = pool_at(1, 221, 130)
    world = sample(ANCHOR, BUILDER, pools=(first, second), credits=10_000)
    plan = ("landFactory", "extractorT1")
    built = sample(
        ANCHOR,
        BUILDER,
        unit(300, "landFactory", 9000.0, 9000.0),
        pools=(first, second),
        credits=10_000,
    )
    decision = decide(built, plan, CATALOGUE, PLACEMENTS, PROFILES, free(built))
    assert (decision["x"], decision["y"]) == (first["x"], first["y"])
    assert (
        decide(world, plan, CATALOGUE, PLACEMENTS, PROFILES, free(world))["type_name"]
        == "landFactory"
    )
