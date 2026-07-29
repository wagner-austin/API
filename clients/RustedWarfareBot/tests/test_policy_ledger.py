"""Keeping the reasoning the loop used to throw away.

The budget records what every claim wanted and why it was refused, and the loop
reduced a whole tick of that to a count of refusals. These are the two records
that keep it: what was asked for, and which spender was even asked.
"""

from __future__ import annotations

from rw_bot.harness.sweep import LABEL_WIDTH
from rw_bot.policy.budget import Budget
from rw_bot.policy.ledger import Outlays, Reaches, format_outlays, format_reaches


def _spent(credits: int, reserve: int = 0) -> Budget:
    """A budget to make real claims against, so the rows are not hand-built."""
    return Budget(credits, reserve)


def test_a_purpose_accumulates_across_observations() -> None:
    """One row per purpose for the whole match, not one per tick."""
    outlays = Outlays()
    for _ in range(3):
        budget = _spent(1000)
        budget.claim("produce:c_tank", 350)
        outlays.add(budget.ledger())
    row = outlays.rows()[0]
    assert row["purpose"] == "produce:c_tank"
    assert (row["asked"], row["granted"], row["spent"]) == (3, 3, 1050)


def test_a_refusal_keeps_the_budget_s_own_words() -> None:
    """The sentence is the whole point: "wanted 700 of 305 available" says which
    of the five causes it was, where a bare count says only that something failed.
    """
    outlays = Outlays()
    budget = _spent(305)
    budget.claim("expand:extractorT1", 700)
    outlays.add(budget.ledger())
    row = outlays.rows()[0]
    assert (row["asked"], row["granted"], row["spent"]) == (1, 0, 0)
    assert "wanted 700 of 305 available" in row["refusal"]


def test_the_last_refusal_wins_over_the_first() -> None:
    """A purpose that starts unaffordable and stays so says the same thing every
    time; one that fails late has usually failed for a new reason.
    """
    outlays = Outlays()
    first = _spent(10)
    first.claim("expand:extractorT1", 700)
    outlays.add(first.ledger())
    second = _spent(600)
    second.claim("expand:extractorT1", 700)
    outlays.add(second.ledger())
    assert "of 600 available" in outlays.rows()[0]["refusal"]


def test_a_purpose_that_later_succeeds_keeps_the_refusal_it_had() -> None:
    """Granting does not erase the history: "it was refused eleven times and then
    bought once" is a different match from "it was bought once".
    """
    outlays = Outlays()
    poor = _spent(10)
    poor.claim("expand:extractorT1", 700)
    outlays.add(poor.ledger())
    rich = _spent(4000)
    rich.claim("expand:extractorT1", 700)
    outlays.add(rich.ledger())
    row = outlays.rows()[0]
    assert (row["asked"], row["granted"], row["spent"]) == (2, 1, 700)
    assert row["refusal"] != ""


def test_purposes_are_reported_dearest_first() -> None:
    """Two runs that spent the same way report it identically, so a diff between
    them is a difference in the match rather than in dictionary iteration.
    """
    outlays = Outlays()
    budget = _spent(10_000)
    budget.claim("produce:c_tank", 350)
    budget.claim("upgrade:extractorT2", 1400)
    budget.claim("expand:extractorT1", 700)
    outlays.add(budget.ledger())
    assert [row["purpose"] for row in outlays.rows()] == [
        "upgrade:extractorT2",
        "expand:extractorT1",
        "produce:c_tank",
    ]


def test_purposes_that_spent_the_same_are_ordered_by_name() -> None:
    """The tie-break, without which two equal rows could swap between runs."""
    outlays = Outlays()
    budget = _spent(10_000)
    budget.claim("produce:hoverTank", 450)
    budget.claim("produce:c_tank", 450)
    outlays.add(budget.ledger())
    assert [row["purpose"] for row in outlays.rows()] == ["produce:c_tank", "produce:hoverTank"]


def test_a_stage_counts_reaching_separately_from_acting() -> None:
    """The distinction the whole census exists for: a stage that declined three
    thousand times and one that was never asked leave the same trace otherwise,
    and defence was refuted on exactly that ambiguity
    ([[policy-holding-ground]]).
    """
    reaches = Reaches()
    reaches.reached("defence", False, "every structure already has cover")
    reaches.reached("defence", True, "")
    reaches.reached("defence", False, "no free worker can place c_turret_t1")
    row = reaches.rows()[0]
    assert (row["reached"], row["acted"]) == (3, 1)
    assert row["reason"] == "no free worker can place c_turret_t1"


def test_acting_leaves_the_previous_reason_alone() -> None:
    """A stage that acted has no refusal to report, so it must not blank the one
    that explains the times it did not.
    """
    reaches = Reaches()
    reaches.reached("income", False, "no pool free of 9: 9 occupied")
    reaches.reached("income", True, "")
    assert reaches.rows()[0]["reason"] == "no pool free of 9: 9 occupied"


def test_stages_report_in_the_order_they_were_first_reached() -> None:
    """The order **is** the policy -- plan, losses, income, defence, throughput --
    so reading the counts down the page shows where the chain stops
    ([[policy-budget]]).
    """
    reaches = Reaches()
    reaches.reached("income", False, "no pool")
    reaches.reached("defence", False, "covered")
    reaches.reached("throughput", False, "not short")
    reaches.reached("income", True, "")
    assert [row["stage"] for row in reaches.rows()] == ["income", "defence", "throughput"]


def test_the_rendered_spend_line_carries_the_refusal() -> None:
    outlays = Outlays()
    budget = _spent(305)
    budget.claim("expand:extractorT1", 700)
    outlays.add(budget.ledger())
    line = format_outlays(outlays.rows())[0]
    assert "expand:extractorT1" in line
    assert "asked     1" in line
    assert "held: " in line


def test_a_purpose_never_refused_renders_without_a_held_clause() -> None:
    outlays = Outlays()
    budget = _spent(1000)
    budget.claim("produce:c_tank", 350)
    outlays.add(budget.ledger())
    assert "held:" not in format_outlays(outlays.rows())[0]


def test_the_rendered_reach_line_carries_the_last_reason() -> None:
    reaches = Reaches()
    reaches.reached("defence", False, "every structure already has cover")
    line = format_reaches(reaches.rows())[0]
    assert "defence" in line
    assert "reached     1" in line
    assert "last: every structure already has cover" in line


def test_a_stage_that_always_acted_renders_without_a_last_clause() -> None:
    reaches = Reaches()
    reaches.reached("income", True, "")
    assert "last:" not in format_reaches(reaches.rows())[0]


def test_an_empty_record_says_so_rather_than_rendering_nothing() -> None:
    """A blank block reads as a measurement that failed to happen, which is a
    different thing from a match that claimed nothing.
    """
    assert format_outlays(()) == ("spend          nothing was ever claimed",)
    assert format_reaches(()) == ("reach          the economy never ran",)


def test_both_blocks_sit_in_the_report_s_own_label_column() -> None:
    """The sweep files a result line only when the character at the label width
    is not a space, so a block one character wide is dropped from every filed
    result without a word being said ([[harness-parallel-matches]]). Both of
    these were a character out on the first draft.
    """
    outlays = Outlays()
    budget = _spent(1000)
    budget.claim("produce:c_tank", 350)
    outlays.add(budget.ledger())
    reaches = Reaches()
    reaches.reached("income", True, "")
    for line in (*format_outlays(outlays.rows()), *format_reaches(reaches.rows())):
        assert line[LABEL_WIDTH] != " ", line
        assert line[:LABEL_WIDTH].strip().isalpha()
