"""Releasing a wave and gathering the reserve, as the pure functions they are.

The ladder says how many units a wave is worth waiting for, ``muster`` decides
when the reserve has reached it, and ``rally`` says where the units that are
not fighting stand. All three are about *whether and when* to commit, which is
a separate question from what to attack once committed
(``test_policy_combat``).

Split from that module, which was 610 lines holding both; the skirmish they
share is :mod:`tests.combat_fixtures`.
"""

from __future__ import annotations

from math import hypot

from rw_bot.policy.combat import (
    FIRST_WAVE,
    RALLY_RADIUS,
    WAVE_SIZES,
    ladder_to,
    muster,
    rally,
    wave_size,
)
from rw_bot.wire.state import Entity
from tests.combat_fixtures import unit


def _wave(size: int) -> tuple[Entity, ...]:
    return tuple(unit(unit_id, "c_tank") for unit_id in range(1, size + 1))


def test_the_wave_ladder_is_the_engines() -> None:
    """Three, then five, then seven, the last rung repeating."""
    assert [wave_size(n) for n in range(8)] == [3, 3, 5, 5, 5, 7, 7, 7]


def test_massing_more_changes_only_the_sustained_wave() -> None:
    """The early rungs govern the opening, when holding three units back is the
    difference between a first attack and none at all. The final rung governs
    the other twenty-eight minutes, and it is the one worth a question -- an
    experiment that moved both could not say which end mattered
    ([[policy-combat]]).
    """
    assert ladder_to(25) == (3, 3, 5, 5, 5, 25)
    assert [wave_size(n, ladder_to(25)) for n in range(8)] == [3, 3, 5, 5, 5, 25, 25, 25]


def test_the_shipped_ladder_is_reachable_rather_than_a_special_case() -> None:
    assert ladder_to(7) == WAVE_SIZES


def test_massing_less_than_the_fixed_rungs_cannot_lower_them() -> None:
    """A mass below the ladder's own body would make the sustained wave smaller
    than the opening ones, which is the trickle the gate exists to prevent.
    """
    assert ladder_to(1) == (3, 3, 5, 5, 5, 5)


def test_a_bigger_wave_holds_units_back_until_it_is_full() -> None:
    """The behaviour the mass argument buys: an army short of the mass gathers
    rather than trickling in.
    """
    state = muster(_wave(9), frozenset(), 5, ladder_to(25))
    assert state["released"] == frozenset()
    assert state["gathering"] == 9
    assert state["wanted"] == 25


def test_a_bigger_wave_releases_once_it_is_full() -> None:
    state = muster(_wave(25), frozenset(), 5, ladder_to(25))
    assert len(state["released"]) == 25
    assert state["waves"] == 6


def test_a_forced_muster_releases_below_the_rung_but_not_below_a_wave() -> None:
    """The riposte's floor: the ladder yields, the anti-trickle rule does not.

    Four units against a rung of five release when forced -- the window
    after the enemy's attack burned out is worth more than the fifth unit --
    but two units are not a punch at any moment, forced or not
    ([[ai-opponent-strategy]]).
    """
    four = _wave(4)
    forced = muster(four, frozenset(), waves=2, force=True)
    assert len(forced["released"]) == 4
    unforced = muster(four, frozenset(), waves=2)
    assert unforced["released"] == frozenset()
    two = _wave(2)
    assert muster(two, frozenset(), waves=2, force=True)["released"] == frozenset()


def test_a_reserve_short_of_a_wave_releases_nobody() -> None:
    state = muster(_wave(2), frozenset(), 0)
    assert state["released"] == frozenset()
    assert state["gathering"] == 2
    assert state["wanted"] == 3
    assert state["waves"] == 0


def test_a_full_reserve_is_released_as_one_wave() -> None:
    state = muster(_wave(3), frozenset(), 0)
    assert state["released"] == frozenset({1, 2, 3})
    assert state["gathering"] == 0
    assert state["waves"] == 1
    assert state["reason"] == "wave 1 of 3 released"


def test_reinforcements_gather_instead_of_joining_the_fight_alone() -> None:
    """The failure the membership model exists for.

    A plain "have we started" flag latched on the first wave and let every
    later unit walk in one at a time -- 45 reinforcements for a net army growth
    of one, measured over 1,500 samples.
    """
    state = muster(_wave(4), frozenset({1, 2, 3}), 1)
    assert state["released"] == frozenset({1, 2, 3})
    assert state["gathering"] == 1
    assert state["wanted"] == 3


def test_the_second_wave_needs_its_own_full_reserve() -> None:
    state = muster(_wave(6), frozenset({1, 2, 3}), 1)
    assert state["released"] == frozenset({1, 2, 3, 4, 5, 6})
    assert state["waves"] == 2
    assert state["wanted"] == WAVE_SIZES[2]


def test_a_wave_still_worth_the_name_keeps_its_clearance() -> None:
    """Losses do not disband a wave that is still a wave."""
    intact = muster(_wave(3), frozenset({1, 2, 3, 4, 5}), 1)
    assert intact["released"] == frozenset({1, 2, 3})
    assert intact["gathering"] == 0


def test_a_decimated_wave_returns_to_the_reserve() -> None:
    """The trickle this gate exists to prevent, happening on the way out.

    Of 48 units lost in a 1500-sample match, 46 died more than 2,000 world units
    from home and not one died within 900 -- nothing was attacking the base. The
    last survivor of each wave kept its clearance and walked in after the rest,
    alone ([[policy-combat]]). Below the ladder's own first rung the survivors
    go back to the reserve, rally home, and go out with the next wave.
    """
    survivors = muster((unit(2, "c_tank"),), frozenset({1, 2, 3}), 1, strength=3)
    assert survivors["released"] == frozenset()
    assert survivors["gathering"] == 1


def test_the_disband_threshold_is_the_ladders_own_first_rung() -> None:
    """Reused rather than reinvented, so there is no new number to justify --
    and since the retreat allele, the IDENTITY the margin defaults to."""
    assert WAVE_SIZES[0] == FIRST_WAVE
    at_threshold = muster(_wave(FIRST_WAVE), frozenset({1, 2, 3, 4, 5}), 1, strength=5)
    assert at_threshold["released"] == frozenset({1, 2, 3})
    below = muster(_wave(FIRST_WAVE - 1), frozenset({1, 2, 3, 4, 5}), 1, strength=5)
    assert below["released"] == frozenset()


def test_the_retreat_margin_disbands_a_bled_wave_early() -> None:
    """The retreat-regroup allele ([[impossible-tactical-genome]]): a wave
    released at ten and bled to four disbands at margin eight where the
    identity would have held it to two survivors -- the GROUP goes home to
    join the next wave instead of dying by ones."""
    four = _wave(4)
    ladder = (3, 3, 5, 5, 5, 25)
    held = muster(four, frozenset(range(1, 11)), 5, ladder, strength=10)
    assert held["released"] == frozenset({1, 2, 3, 4})
    early = muster(four, frozenset(range(1, 11)), 5, ladder, strength=10, retreat=8)
    assert early["released"] == frozenset()
    assert early["strength"] == 0
    assert early["gathering"] == 4


def test_a_margin_larger_than_a_small_wave_keeps_the_identity_floor() -> None:
    """min(retreat, strength): a wave of three under margin eight keeps
    today's floor rather than churning -- released whole, it stands."""
    three = _wave(3)
    state = muster(three, frozenset({1, 2, 3}), 1, strength=3, retreat=8)
    assert state["released"] == frozenset({1, 2, 3})
    assert state["strength"] == 3


def test_retreat_zero_fights_to_the_last_unit() -> None:
    """The margin's off-state: one survivor of a ten-wave keeps clearance."""
    lone = (unit(7, "c_tank"),)
    state = muster(lone, frozenset(range(1, 11)), 1, strength=10, retreat=0)
    assert state["released"] == frozenset({7})


def test_a_release_records_its_strength() -> None:
    state = muster(_wave(5), frozenset(), 2, ladder=(3, 5))
    assert state["released"] == frozenset({1, 2, 3, 4, 5})
    assert state["strength"] == 5


def test_a_wiped_wave_leaves_nothing_released() -> None:
    """And the next reserve then re-gathers rather than trickling in."""
    assert muster((), frozenset({1, 2, 3}), 1)["released"] == frozenset()


def test_a_scattered_reserve_is_sent_to_the_rally_point() -> None:
    """Units that rolled out of a factory are wherever the factory is."""
    scattered = (unit(4, "c_tank", 900.0, 0.0), unit(5, "c_tank", 0.0, 900.0))
    moves = rally(scattered, (0.0, 0.0))
    assert [m["unit_id"] for m in moves] == [4, 5]
    assert {(m["x"], m["y"]) for m in moves} == {(0.0, 0.0)}


def test_a_unit_already_at_the_rally_point_is_not_re_ordered() -> None:
    """The engine runs a waypoint until it is replaced.

    Re-issuing every sample would reset the walk at the sampling rate and
    nothing would ever arrive -- the failure the attack path already learned.
    """
    assert rally((unit(4, "c_tank", 10.0, 10.0),), (0.0, 0.0)) == ()


def test_the_rally_boundary_is_the_engines_own_arrival_test() -> None:
    """Sixty world units, which is where its rally group drops a member."""
    just_outside = unit(4, "c_tank", RALLY_RADIUS + 0.5, 0.0)
    just_inside = unit(5, "c_tank", RALLY_RADIUS - 0.5, 0.0)
    assert [m["unit_id"] for m in rally((just_outside, just_inside), (0.0, 0.0))] == [4]


def test_an_empty_reserve_is_sent_nowhere() -> None:
    assert rally((), (0.0, 0.0)) == ()


def test_spacing_gives_each_unit_its_own_station_and_holds_it() -> None:
    """The spacing allele ([[impossible-tactical-genome]]): distinct
    stations on the ring, distance equal to the allele, pure in the unit
    id -- the same call yields the same stations, so an arrived unit is
    left alone exactly as at the point rally."""
    scattered = (unit(4, "c_tank", 900.0, 0.0), unit(5, "c_tank", 0.0, 900.0))
    moves = rally(scattered, (0.0, 0.0), spacing=120)
    assert [m["unit_id"] for m in moves] == [4, 5]
    stations = {(m["x"], m["y"]) for m in moves}
    assert len(stations) == 2
    for x, y in stations:
        assert abs(hypot(x, y) - 120.0) < 1e-6
    again = rally(scattered, (0.0, 0.0), spacing=120)
    assert moves == again
    # A unit standing on its own station is not re-ordered.
    x4, y4 = next((m["x"], m["y"]) for m in moves if m["unit_id"] == 4)
    parked = unit(4, "c_tank", x4, y4)
    assert rally((parked,), (0.0, 0.0), spacing=120) == ()


def test_spacing_zero_is_the_point_rally_exactly() -> None:
    """The identity: no station arithmetic, the same deployments as ever."""
    scattered = (unit(4, "c_tank", 900.0, 0.0),)
    assert rally(scattered, (0.0, 0.0), spacing=0) == rally(scattered, (0.0, 0.0))
