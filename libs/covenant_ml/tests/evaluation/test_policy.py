"""Tests for payoff tables, the Bayes decision rule and policy value."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.evaluation.policy import (
    ALERT_ACTIONS,
    CONTRACT_ACTIONS,
    alert_payoffs,
    binary_contract_payoffs,
    choose_actions,
    evaluate_actions,
    evaluate_policy,
    payoff_table,
    realized_utility,
)
from covenant_ml.evaluation.resampling import build_resampling
from covenant_ml.evaluation.types import (
    ERR_INVALID_COST,
    ERR_INVALID_PRICE,
    ERR_LENGTH_MISMATCH,
    ERR_PAYOFF_SHAPE,
    AlertCosts,
    IntervalMethod,
    ResamplingUnit,
)

from ._fixtures import COSTS, at, floats, ints


def test_a_table_needs_two_distinct_actions() -> None:
    payoffs: NDArray[np.float64] = np.zeros((2, 1, 2), dtype=np.float64)
    with pytest.raises(ValueError, match=ERR_PAYOFF_SHAPE):
        payoff_table(("only",), payoffs)
    with pytest.raises(ValueError, match=ERR_PAYOFF_SHAPE):
        payoff_table(("same", "same"), np.zeros((2, 2, 2), dtype=np.float64))


def test_a_table_must_be_row_by_action_by_outcome() -> None:
    with pytest.raises(ValueError, match=ERR_PAYOFF_SHAPE):
        payoff_table(("a", "b"), np.zeros((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match=ERR_PAYOFF_SHAPE):
        payoff_table(("a", "b"), np.zeros((2, 3, 2), dtype=np.float64))
    with pytest.raises(ValueError, match=ERR_PAYOFF_SHAPE):
        payoff_table(("a", "b"), np.zeros((2, 2, 3), dtype=np.float64))


def test_alert_payoffs_place_each_cost_in_its_cell() -> None:
    table = alert_payoffs(3, COSTS)
    assert table.action_names == ALERT_ACTIONS
    hold_negative: NDArray[np.float64] = table.payoffs[:, 0, 0]
    hold_positive: NDArray[np.float64] = table.payoffs[:, 0, 1]
    alert_negative: NDArray[np.float64] = table.payoffs[:, 1, 0]
    alert_positive: NDArray[np.float64] = table.payoffs[:, 1, 1]
    assert np.array_equal(hold_negative, floats([0.0, 0.0, 0.0]))
    assert np.array_equal(hold_positive, floats([-5.0, -5.0, -5.0]))
    assert np.array_equal(alert_negative, floats([-1.0, -1.0, -1.0]))
    assert np.array_equal(alert_positive, floats([10.0, 10.0, 10.0]))


def test_a_negative_cost_is_refused() -> None:
    costs: AlertCosts = {"true_alert_value": 1.0, "false_alert_cost": -1.0, "missed_cost": 1.0}
    with pytest.raises(ValueError, match=ERR_INVALID_COST):
        alert_payoffs(2, costs)


def test_the_alert_threshold_is_where_expected_payoffs_cross() -> None:
    # Alert pays 10p - (1-p), hold pays -5p: they cross at p = 1/16. At the
    # crossing exactly the tie goes to hold.
    table = alert_payoffs(3, COSTS)
    actions = choose_actions(floats([0.05, 0.0625, 0.07]), table)
    assert np.array_equal(actions, ints([0, 0, 1]))


def test_choose_actions_needs_one_probability_per_row() -> None:
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        choose_actions(floats([0.5]), alert_payoffs(2, COSTS))


def _contract() -> tuple[NDArray[np.float64], ...]:
    return (
        floats([0.50, 0.50, 0.30]),
        floats([0.52, 0.52, 0.72]),
        floats([0.02, 0.02, 0.01]),
        floats([0.02, 0.02, 0.01]),
    )


def test_contract_payoffs_net_the_price_and_fee() -> None:
    yes_ask, no_ask, yes_fee, no_fee = _contract()
    table = binary_contract_payoffs(yes_ask, no_ask, yes_fee, no_fee)
    assert table.action_names == CONTRACT_ACTIONS
    yes_if_happens: NDArray[np.float64] = table.payoffs[:, 1, 1]
    yes_if_not: NDArray[np.float64] = table.payoffs[:, 1, 0]
    no_if_not: NDArray[np.float64] = table.payoffs[:, 2, 0]
    no_if_happens: NDArray[np.float64] = table.payoffs[:, 2, 1]
    assert at(yes_if_happens, 0) == pytest.approx(0.48)
    assert at(yes_if_not, 0) == pytest.approx(-0.52)
    assert at(no_if_not, 2) == pytest.approx(0.27)
    assert at(no_if_happens, 2) == pytest.approx(-0.73)


def test_a_model_buys_the_side_it_thinks_is_cheap_and_passes_otherwise() -> None:
    table = binary_contract_payoffs(*_contract())
    # 0.7 against a 0.52 all-in yes: buy yes. 0.5: both sides lose to fees.
    # 0.1 against a 0.73 all-in no that pays 0.9 of the time: buy no.
    actions = choose_actions(floats([0.7, 0.5, 0.1]), table)
    assert np.array_equal(actions, ints([1, 0, 2]))


def test_contract_inputs_must_align() -> None:
    yes_ask, no_ask, yes_fee, _ = _contract()
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        binary_contract_payoffs(yes_ask, no_ask, yes_fee, floats([0.0]))


@pytest.mark.parametrize("bad", [0.0, 1.0])
def test_a_price_outside_the_open_unit_interval_is_refused(bad: float) -> None:
    yes_ask, no_ask, yes_fee, no_fee = _contract()
    with pytest.raises(ValueError, match=ERR_INVALID_PRICE):
        binary_contract_payoffs(floats([bad, 0.5, 0.5]), no_ask, yes_fee, no_fee)
    with pytest.raises(ValueError, match=ERR_INVALID_PRICE):
        binary_contract_payoffs(yes_ask, floats([0.5, bad, 0.5]), yes_fee, no_fee)


def test_a_negative_fee_is_refused() -> None:
    yes_ask, no_ask, yes_fee, _ = _contract()
    with pytest.raises(ValueError, match=ERR_INVALID_PRICE):
        binary_contract_payoffs(yes_ask, no_ask, yes_fee, floats([0.0, -0.01, 0.0]))


def test_realized_utility_reads_the_cell_the_outcome_picks() -> None:
    table = alert_payoffs(4, COSTS)
    utilities = realized_utility(ints([0, 0, 1, 1]), ints([0, 1, 0, 1]), table)
    assert np.array_equal(utilities, floats([0.0, -5.0, -1.0, 10.0]))


def test_realized_utility_needs_aligned_actions_and_labels() -> None:
    table = alert_payoffs(2, COSTS)
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        realized_utility(ints([0]), ints([0, 1]), table)
    with pytest.raises(ValueError, match=ERR_LENGTH_MISMATCH):
        realized_utility(ints([0, 1]), ints([0]), table)


def test_policy_value_counts_actions_and_intervals_the_mean_payoff() -> None:
    table = alert_payoffs(4, COSTS)
    resampling = build_resampling(ResamplingUnit.ROW, 4, ints([0] * 4), 1)
    evaluation = evaluate_policy(
        "model", floats([0.01, 0.9, 0.5, 0.02]), ints([0, 1, 0, 1]), table, resampling, 0.95
    )
    outcome = evaluation.outcome
    assert outcome["arm"] == "model"
    assert outcome["action_counts"] == [2, 2]
    # hold/neg 0, alert/pos 10, alert/neg -1, hold/pos -5.
    assert np.array_equal(evaluation.utilities, floats([0.0, 10.0, -1.0, -5.0]))
    assert outcome["value"]["estimate"] == pytest.approx(1.0)
    assert outcome["value"]["method"] is IntervalMethod.PAIRED_T


def test_a_fixed_action_is_scored_like_any_policy() -> None:
    table = alert_payoffs(3, COSTS)
    resampling = build_resampling(ResamplingUnit.ROW, 3, ints([0] * 3), 1)
    never = evaluate_actions("never", ints([0, 0, 0]), ints([1, 0, 1]), table, resampling, 0.95)
    assert never.outcome["action_counts"] == [3, 0]
    assert never.outcome["value"]["estimate"] == pytest.approx(-10.0 / 3)
