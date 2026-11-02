"""A fitted policy on realized paths, paired with holding the same contract."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.options.realized import realized_outcomes
from covenant_ml.options.types import ERR_PATH_SHAPE, ERR_RATE, LsmPolicy

from ._fixtures import SMALL, floats, listed, listed_ints, matrix

#: Continuation is the constant 5 on the first date and 1 on the second.
CONSTANT = LsmPolicy(
    contract=SMALL,
    rate=0.0,
    volatility=0.2,
    training_paths=1,
    seed=0,
    coefficients=((5.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
)

#: Averages on the three dates: (90, 90, 92.14...) then (100, 105, 110).
PATHS = matrix(
    [
        [100.0, 90.0, 80.0, 110.0, 70.0, 100.0, 95.0],
        [100.0, 100.0, 100.0, 110.0, 115.0, 120.0, 125.0],
    ]
)


def test_an_early_payoff_is_carried_to_maturity_at_the_contract_rate() -> None:
    outcomes = realized_outcomes(CONSTANT, PATHS, floats([0.03, 0.03]))
    assert listed_ints(outcomes.chosen) == [0, 2]
    # Exercised on the first of three dates, so two thirds of a year remain.
    carried: list[float] = [10.0 * math.exp(0.03 * 2.0 / 3.0), 0.0]
    held: list[float] = [100.0 - 645.0 / 7.0, 0.0]
    assert listed(outcomes.policy_payoffs) == pytest.approx(carried)
    assert listed(outcomes.hold_payoffs) == pytest.approx(held)


@pytest.mark.parametrize("rates", [floats([0.03]), matrix([[0.03, 0.03]])])
def test_rates_must_align_with_the_contracts(rates: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match=ERR_PATH_SHAPE):
        realized_outcomes(CONSTANT, PATHS, rates)


def test_an_unusable_rate_is_refused() -> None:
    with pytest.raises(ValueError, match=ERR_RATE):
        realized_outcomes(CONSTANT, PATHS, floats([0.03, math.nan]))
