"""Coverage: both directions of standing reach, judged where units stand.

Every rule under test is the engine's arithmetic composed -- hostility
from the wire, layer reach from the engine's own test, range from the
registry -- so the fixtures here vary exactly those three axes.
"""

from __future__ import annotations

from rw_bot.policy.field import coverage, guns_covering
from tests.wire_fixtures import enemy, entity, profile, sample

_PROFILES = {
    "commandCenter": profile("commandCenter", 0.0),
    "extractorT1": profile("extractorT1", 0.0),
    "extractorT2": profile("extractorT2", 0.0),
    "c_tank": profile("c_tank", 130.0),
    "c_artillery": profile("c_artillery", 290.0),
    "antiAirTurret": profile("antiAirTurret", 250.0, land=False, air=True),
    "builder": profile("builder", 0.0),
}

_FAMILY = frozenset({"extractorT1", "extractorT2"})


def test_a_gun_covers_what_stands_inside_its_reach_and_layer() -> None:
    """One artillery at 290 covers the extractor at 200 and not the one at
    400; the anti-air turret covers neither however close, because its
    weapon does not reach the ground layer."""
    world = sample(
        entity(1, "extractorT1", x=200.0, y=0.0),
        entity(2, "extractorT1", x=400.0, y=0.0),
        enemy(9, "c_artillery", x=0.0, y=0.0),
        enemy(10, "antiAirTurret", x=190.0, y=0.0),
    )
    near = world["entities"][0]
    far = world["entities"][1]
    assert guns_covering(world, _PROFILES, near) == 1
    assert guns_covering(world, _PROFILES, far) == 0


def test_unarmed_hostiles_and_non_hostiles_never_cover() -> None:
    """An enemy builder on top of us is an obstacle, not a gun; our own
    artillery next door is not a hostile."""
    world = sample(
        entity(1, "extractorT1", x=0.0, y=0.0),
        entity(2, "c_artillery", x=10.0, y=0.0),
        enemy(9, "builder", x=5.0, y=0.0),
    )
    ours = world["entities"][0]
    assert guns_covering(world, _PROFILES, ours) == 0


def test_coverage_counts_both_directions_and_the_extractor_family() -> None:
    """One enemy tank sits on our extractor: the extractor is covered
    (eco and own), the command center out of reach is not -- and the tank
    itself stands inside our artillery's longer reach, so the reverse
    count carries it once however many of our guns cover it."""
    world = sample(
        entity(1, "commandCenter", x=1000.0, y=0.0),
        entity(2, "extractorT2", x=0.0, y=0.0),
        entity(3, "c_artillery", x=100.0, y=0.0),
        entity(4, "c_tank", x=150.0, y=0.0),
        enemy(9, "c_tank", x=50.0, y=0.0),
        enemy(11, "c_tank", x=2000.0, y=0.0),
    )
    covered = coverage(world, _PROFILES, _FAMILY)
    # The near enemy tank (reach 130) covers the extractor at 50 and our
    # tank at 100; the artillery at 50 is covered too; the base at 950 is
    # not. The far tank at 2000 is inside nobody's reach in either
    # direction: our guns engage its layer but do not span the distance.
    assert covered["eco_covered"] == 1
    assert covered["own_covered"] == 3
    assert covered["foe_covered"] == 1


def test_an_unfinished_structure_is_not_counted_as_ours_yet() -> None:
    world = sample(
        entity(1, "extractorT1", x=0.0, y=0.0, complete=False),
        enemy(9, "c_tank", x=50.0, y=0.0),
    )
    covered = coverage(world, _PROFILES, _FAMILY)
    assert covered == {"eco_covered": 0, "own_covered": 0, "foe_covered": 0}
