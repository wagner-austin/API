"""One authority over a tick's credits.

The defect this module exists for was not a rule that failed to fire. It was two
rules that each fired correctly against the same balance: production budgeted
across every idle producer using ``sample["credits"]``, and expansion then asked
the same field whether it could afford an extractor. Both were right alone and
the pair committed one credit twice ([[policy-budget]]).
"""

from __future__ import annotations

import pytest

from rw_bot.policy.budget import Budget, BudgetError, format_ledger


def test_a_claim_that_fits_is_granted_and_committed() -> None:
    budget = Budget(1000, reserve=0)
    claim = budget.claim("produce:c_tank", 350)
    assert claim["granted"] is True
    assert budget.spent() == 350
    assert budget.remaining() == 650


def test_a_second_claim_cannot_spend_the_first_ones_credits() -> None:
    """The whole point. Two spenders, one balance, and no double commitment."""
    budget = Budget(1000, reserve=0)
    assert budget.claim("produce:c_tank", 700)["granted"] is True
    refused = budget.claim("expand:extractorT1", 700)
    assert refused["granted"] is False
    assert budget.spent() == 700


def test_a_refusal_says_what_was_wanted_and_what_was_left() -> None:
    """A refusal is the more informative half, so it carries its own reasoning."""
    budget = Budget(500, reserve=0)
    budget.claim("plan:landFactory", 400)
    refused = budget.claim("expand:extractorT1", 700)
    assert "700" in refused["reason"]
    assert "100" in refused["reason"]
    assert "400 already committed" in refused["reason"]


def test_the_reserve_is_invisible_to_an_unprotected_claim() -> None:
    """Investment may not take the credits held for replacing a loss."""
    budget = Budget(1000, reserve=400)
    assert budget.spendable() == 600
    assert budget.claim("expand:extractorT1", 700)["granted"] is False
    assert budget.claim("expand:extractorT1", 600)["granted"] is True


def test_a_protected_claim_may_cross_the_reserve() -> None:
    """That is what the reserve is for: the army, not the bank."""
    budget = Budget(1000, reserve=900)
    assert budget.spendable() == 100
    assert budget.claim("produce:c_tank", 950, protected=True)["granted"] is True


def test_a_withholding_hides_credits_from_every_later_claim() -> None:
    """The saving mechanism: a refused conversion keeps its price back.

    Same-tick income can never reach a 4,000-credit conversion when every
    tick's spenders drain the balance first -- the tier three was asked
    3,788 times and granted never ([[policy-budget]], log 2026-07-31). The
    saving binds protected claims too, deliberately: replacing losses is
    protected and drains the balance to zero each tick, so a saving that
    only investment respected would never fill. What still fits past the
    withholding is still spendable either way.
    """
    budget = Budget(2000, reserve=0)
    budget.withhold(1400)
    assert budget.spendable() == 600
    assert budget.claim("produce:c_tank", 700)["granted"] is False
    assert budget.claim("produce:c_tank", 700, protected=True)["granted"] is False
    assert budget.claim("produce:c_tank", 600, protected=True)["granted"] is True


def test_withholdings_accumulate() -> None:
    budget = Budget(5000, reserve=500)
    budget.withhold(1400)
    budget.withhold(1400)
    assert budget.spendable() == 5000 - 500 - 2800


def test_a_negative_withholding_is_refused_loudly() -> None:
    """A negative would free credits rather than save them."""
    budget = Budget(1000, reserve=0)
    with pytest.raises(BudgetError) as caught:
        budget.withhold(-1)
    assert caught.value.code == "RW-BUDGET-003"


def test_a_release_hands_a_withholding_back_to_later_claims() -> None:
    """The late claimant's half of the saving pair.

    Defence claims last, so a withholding it placed early in the tick would
    bind its own claim too -- the deficit is withheld early, binding produce
    and upgrades, and released where the expander runs (log 2026-08-01).
    """
    budget = Budget(800, reserve=0)
    budget.withhold(500)
    assert budget.claim("produce:c_tank", 350)["granted"] is False
    budget.release(500)
    assert budget.claim("expand:c_turret_t1", 500)["granted"] is True


def test_a_release_never_frees_more_than_stands_withheld() -> None:
    """Over-releasing must not mint credits a later tick never withheld."""
    budget = Budget(1000, reserve=0)
    budget.withhold(300)
    budget.release(500)
    assert budget.spendable() == 1000


def test_a_negative_release_is_refused_loudly() -> None:
    """A negative would withhold credits rather than free them."""
    budget = Budget(1000, reserve=0)
    with pytest.raises(BudgetError) as caught:
        budget.release(-1)
    assert caught.value.code == "RW-BUDGET-003"


def test_a_reserve_larger_than_the_balance_is_an_ordinary_early_state() -> None:
    """Not an error -- it means every credit is spoken for."""
    budget = Budget(100, reserve=500)
    assert budget.spendable() == 0
    assert budget.claim("expand:extractorT1", 1)["granted"] is False
    assert budget.claim("produce:c_tank", 100, protected=True)["granted"] is True


def test_a_claim_of_nothing_is_granted_and_still_recorded() -> None:
    """An order that costs nothing still wants a ledger line saying it happened."""
    budget = Budget(0, reserve=0)
    assert budget.claim("plan:free", 0)["granted"] is True
    assert len(budget.ledger()) == 1


def test_the_ledger_keeps_refusals_in_order() -> None:
    budget = Budget(500, reserve=0)
    budget.claim("plan:landFactory", 400)
    budget.claim("expand:extractorT1", 700)
    budget.claim("produce:c_tank", 100, protected=True)
    assert [c["purpose"] for c in budget.ledger()] == [
        "plan:landFactory",
        "expand:extractorT1",
        "produce:c_tank",
    ]
    assert [c["granted"] for c in budget.ledger()] == [True, False, True]


def test_the_ledger_renders_as_report_lines() -> None:
    budget = Budget(500, reserve=0)
    budget.claim("plan:landFactory", 400)
    budget.claim("expand:extractorT1", 700)
    lines = format_ledger(budget.ledger())
    assert lines[0].startswith("took")
    assert lines[1].startswith("held")


def test_an_exhausted_budget_refuses_everything_after() -> None:
    budget = Budget(350, reserve=0)
    assert budget.claim("produce:c_tank", 350)["granted"] is True
    assert budget.remaining() == 0
    assert budget.claim("produce:c_tank", 1)["granted"] is False


def test_a_negative_balance_is_a_decode_fault_rather_than_a_debt() -> None:
    with pytest.raises(BudgetError) as caught:
        Budget(-1, reserve=0)
    assert caught.value.code == "RW-BUDGET-001"


def test_a_negative_reserve_is_rejected() -> None:
    with pytest.raises(BudgetError) as caught:
        Budget(100, reserve=-1)
    assert caught.value.code == "RW-BUDGET-002"


def test_a_negative_claim_is_rejected_rather_than_refunding_the_budget() -> None:
    """It would let a later claim spend credits the player never held."""
    budget = Budget(100, reserve=0)
    with pytest.raises(BudgetError) as caught:
        budget.claim("produce:c_tank", -50)
    assert caught.value.code == "RW-BUDGET-003"
    assert budget.spent() == 0
