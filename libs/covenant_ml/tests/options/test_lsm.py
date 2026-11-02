"""Longstaff-Schwartz fitting and the decisions a fitted policy makes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.options.lsm import (
    basis,
    date_discount,
    exercise_choices,
    fit_policy,
    simulated_value,
)
from covenant_ml.options.paths import decision_inputs, gbm_paths, put_payoffs
from covenant_ml.options.types import (
    BASIS_TERMS,
    ERR_TOO_FEW_IN_THE_MONEY,
    LsmPolicy,
    decode_policy,
    encode_policy,
)

from ._fixtures import BASE, SMALL, floats, listed, listed_ints, matrix, mean

#: Continuation is the constant 5 on the first date and 1 on the second.
CONSTANT = LsmPolicy(
    contract=SMALL,
    rate=0.0,
    volatility=0.2,
    training_paths=1,
    seed=0,
    coefficients=((5.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
)


def test_basis_columns_follow_the_named_terms() -> None:
    assert listed(basis(floats([2.0]), floats([3.0]))) == [1.0, 2.0, 3.0, 4.0, 9.0, 6.0]
    assert len(BASIS_TERMS) == 6


def test_one_interval_is_discounted_at_the_rate() -> None:
    assert date_discount(SMALL, 0.06) == pytest.approx(math.exp(-0.02))


def test_exercise_is_the_first_date_whose_payoff_beats_continuation() -> None:
    averages = matrix([[0.9, 0.85, 0.8], [0.97, 0.9, 0.8], [0.97, 0.98, 0.8], [1.1, 1.2, 1.3]])
    spots: NDArray[np.float64] = np.ones_like(averages)
    # Payoffs on the first two dates are (10, 15), (3, 10), (3, 2) and (0, 0),
    # against continuation 5 on the first and 1 on the second.
    assert listed_ints(exercise_choices(CONSTANT, spots, averages)) == [0, 1, 1, 2]


def test_a_fitted_policy_has_one_row_per_date_before_maturity() -> None:
    policy = fit_policy(SMALL, 0.05, 0.2, 2_000, 5)
    assert len(policy["coefficients"]) == SMALL["exercise_dates"] - 1
    assert all(len(row) == len(BASIS_TERMS) for row in policy["coefficients"])
    assert decode_policy(encode_policy(policy)) == policy
    assert fit_policy(SMALL, 0.05, 0.2, 2_000, 5) == policy


def test_a_date_with_too_few_paths_in_the_money_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_TOO_FEW_IN_THE_MONEY):
        fit_policy(SMALL, 0.05, 0.2, 5, 5)


def test_base_policy_beats_holding_on_fresh_simulated_paths() -> None:
    policy = fit_policy(BASE, 0.05, 0.2, 20_000, 11)
    paths = gbm_paths(BASE, 0.05, 0.2, 20_000, 99)
    spots, averages = decision_inputs(BASE, paths)
    chosen = exercise_choices(policy, spots, averages)
    payoffs = put_payoffs(BASE, averages)
    rows: NDArray[np.intp] = np.arange(len(chosen))
    years: NDArray[np.float64] = (chosen + 1).astype(np.float64) / 50.0
    exercised: NDArray[np.float64] = payoffs[rows, chosen]
    held: NDArray[np.float64] = payoffs[:, -1]
    differences: NDArray[np.float64] = exercised * np.exp(-0.05 * years) - held * math.exp(-0.05)
    average = mean(differences)
    spread: NDArray[np.float64] = differences - average
    error = math.sqrt(mean(spread * spread) / len(differences))
    assert average > 4.0 * error
    assert 0 < int(np.count_nonzero(chosen < 49)) < len(chosen)


def test_base_value_is_pinned_on_fresh_paths() -> None:
    # Measured with this implementation (daily sessions, average including the
    # start): 3.98 to 4.03 across fresh seeds at 50,000 paths. The parent study
    # reports 3.9531 for the same contract with its own simulation grid.
    policy = fit_policy(BASE, 0.05, 0.2, 20_000, 11)
    assert simulated_value(policy, 20_000, 101) == pytest.approx(4.0, abs=0.12)
