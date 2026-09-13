"""The rival income read, which must describe the same opponent as the worth.

The trace carries ``rival`` and ``rival_income`` side by side, and the pair is
only a rivalry if both figures come off one scoreboard row. These pin the
selection to :func:`~rw_bot.policy.scoreboard.best_rival`'s -- strongest worth,
not largest income -- and the zero on a board with nothing hostile on it.
"""

from __future__ import annotations

from rw_bot.policy.scoreboard import best_rival, rival_attrition, rival_income, strongest_rival
from tests.wire_fixtures import player, sample


def test_an_empty_scoreboard_reads_zero_income() -> None:
    """A stream that predates the player record, or a board swept clean."""
    assert rival_income(sample()) == 0


def test_a_board_with_only_allies_reads_zero_income() -> None:
    us = player(0, index=0, local=True, hostile=False, income=54)
    ally = player(0, index=1, hostile=False, income=90)
    assert rival_income(sample(players=(us, ally))) == 0


def test_the_income_belongs_to_the_strongest_rival_by_worth() -> None:
    """A poorer opponent with the bigger army is the one the race is against,
    even when a richer one is compounding faster off a smaller base."""
    rich = player(1, index=1, income=200, army_value=1000, building_value=500)
    strong = player(2, index=2, income=40, army_value=8000, building_value=2000)
    board = sample(players=(rich, strong))
    assert rival_income(board) == 40
    assert best_rival(board) == 10000


def test_the_first_of_equal_rivals_is_read_like_best_rival_reads_it() -> None:
    """Ties break to enumeration order, the order ``max`` breaks them in."""
    first = player(1, index=1, income=25, army_value=3000)
    second = player(2, index=2, income=75, army_value=3000)
    assert rival_income(sample(players=(first, second))) == 25


def test_a_weaker_rival_after_the_strongest_does_not_displace_it() -> None:
    strong = player(1, index=1, income=60, army_value=5000)
    weak = player(2, index=2, income=300, army_value=100)
    assert rival_income(sample(players=(strong, weak))) == 60


def test_the_attrition_pair_comes_off_the_same_row_as_the_income() -> None:
    """The kill ledger of the strongest rival by worth, not of the rival that
    has lost the most: a poorer opponent bleeding units is not the race."""
    bleeding = player(1, index=1, income=200, army_value=1000, units_lost=90, buildings_lost=12)
    strong = player(2, index=2, income=40, army_value=8000, units_lost=17, buildings_lost=3)
    board = sample(players=(bleeding, strong))
    assert strongest_rival(board) is strong
    assert rival_attrition(board) == (17, 3)


def test_attrition_reads_zero_zero_with_nothing_hostile() -> None:
    us = player(0, index=0, local=True, hostile=False, units_lost=5)
    assert strongest_rival(sample(players=(us,))) is None
    assert rival_attrition(sample(players=(us,))) == (0, 0)
    assert rival_attrition(sample()) == (0, 0)
