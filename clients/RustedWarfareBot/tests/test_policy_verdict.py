"""Reading the engine's verdict, which is the only unconfounded figure we have.

Every other number on the battle report measures effort. These tests are about
the one that measures result, so the cases that matter are the ones where two
readings of the same observation disagree -- losing and winning can both look
true at once, and the order the flags are checked in is what decides which the
report claims.
"""

from __future__ import annotations

from rw_bot.policy.verdict import (
    GRADE_DEFEATED,
    GRADE_SURVIVED,
    GRADE_WIPED,
    GRADE_WON,
    eliminated,
    grade,
)
from rw_bot.wire.state import Sample


def _sample(*, defeated: bool = False, wiped: bool = False, players_left: int = 6) -> Sample:
    return Sample(
        frame=1,
        clock_ms=10,
        credits=4000,
        defeated=defeated,
        wiped=wiped,
        players_left=players_left,
        entities=(),
        pools=(),
        players=(),
        options=(),
        refusals=(),
    )


def test_a_match_still_running_is_survived_not_won() -> None:
    """Not a win and not a loss. The budget stopped it; nothing decided it."""
    assert grade(_sample(players_left=6)) == GRADE_SURVIVED


def test_one_player_left_is_a_win() -> None:
    """The engine ends the match at one player remaining."""
    assert grade(_sample(players_left=1)) == GRADE_WON


def test_being_defeated_outranks_the_survivor_count() -> None:
    """The case that would have graded a loss as a win.

    When we are eliminated the remaining-player count falls towards one as
    well, so an observation can satisfy "one player left" and "we were
    defeated" at the same time. Reading the count first would report a victory
    for the run that just lost.
    """
    assert grade(_sample(defeated=True, players_left=1)) == GRADE_DEFEATED


def test_being_wiped_out_outranks_being_defeated() -> None:
    """Both flags describe the same event; wiped is the stronger statement."""
    assert grade(_sample(defeated=True, wiped=True, players_left=1)) == GRADE_WIPED


def test_a_wipe_is_reported_even_with_players_still_in_the_match() -> None:
    assert grade(_sample(wiped=True, players_left=4)) == GRADE_WIPED


def test_a_defeat_is_reported_even_with_players_still_in_the_match() -> None:
    assert grade(_sample(defeated=True, players_left=4)) == GRADE_DEFEATED


def test_an_emptied_match_still_reads_as_a_win() -> None:
    """Zero remaining is not a state the engine should reach, but <= 1 covers it."""
    assert grade(_sample(players_left=0)) == GRADE_WON


def test_eliminations_are_the_fall_in_the_player_count() -> None:
    assert eliminated(6, 4) == 2


def test_no_eliminations_is_zero_rather_than_absent() -> None:
    assert eliminated(6, 6) == 0


def test_a_player_count_that_rose_reports_no_eliminations() -> None:
    """Flattened rather than reported negative.

    A rising count is not something the engine should produce, and "-2
    eliminated" would be a stranger claim in a run report than "none".
    """
    assert eliminated(4, 6) == 0
