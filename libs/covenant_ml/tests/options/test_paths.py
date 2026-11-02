"""Path geometry, simulation and compounding of realized returns."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.options.paths import (
    decision_inputs,
    exercise_sessions,
    gbm_paths,
    paths_from_returns,
    put_payoffs,
    require_paths,
    require_rate,
    require_rates,
    running_averages,
    total_sessions,
)
from covenant_ml.options.types import ERR_PATH_SHAPE, ERR_RATE

from ._fixtures import SMALL, floats, listed, listed_ints, matrix, mean

#: One hand-written path of SMALL's six sessions.
PATH = matrix([[100.0, 90.0, 80.0, 110.0, 70.0, 100.0, 95.0]])


def test_the_session_grid_puts_maturity_last() -> None:
    assert total_sessions(SMALL) == 6
    assert listed_ints(exercise_sessions(SMALL)) == [2, 4, 6]


def test_running_average_includes_the_start() -> None:
    expected = [100.0, 95.0, 90.0, 95.0, 90.0, 550.0 / 6.0, 645.0 / 7.0]
    assert listed(running_averages(PATH)) == pytest.approx(expected)


def test_decision_inputs_read_each_date_session() -> None:
    spots, averages = decision_inputs(SMALL, PATH)
    expected_spots: list[float] = [0.8, 0.7, 0.95]
    expected_averages: list[float] = [0.9, 0.9, 6.45 / 7.0]
    assert listed(spots) == pytest.approx(expected_spots)
    assert listed(averages) == pytest.approx(expected_averages)


def test_payoffs_are_the_put_on_the_average_in_price_units() -> None:
    expected: list[float] = [10.0, 0.0, 0.0]
    assert listed(put_payoffs(SMALL, floats([0.9, 1.1, 1.0]))) == pytest.approx(expected)


@pytest.mark.parametrize(
    "paths",
    [
        floats([100.0] * 7),
        np.empty((0, 7), dtype=np.float64),
        matrix([[100.0] * 6]),
        matrix([[100.0, 90.0, math.nan, 1.0, 1.0, 1.0, 1.0]]),
        matrix([[100.0, 90.0, 0.0, 1.0, 1.0, 1.0, 1.0]]),
        matrix([[99.0, 90.0, 80.0, 1.0, 1.0, 1.0, 1.0]]),
    ],
)
def test_malformed_paths_are_refused(paths: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match=ERR_PATH_SHAPE):
        require_paths(SMALL, paths)


@pytest.mark.parametrize("rate", [math.nan, -1.0, -2.0])
def test_unusable_rates_are_refused(rate: float) -> None:
    with pytest.raises(ValueError, match=ERR_RATE):
        require_rate(rate)
    with pytest.raises(ValueError, match=ERR_RATE):
        require_rates(floats([0.01, rate]))


def test_simulation_is_seeded_and_valid() -> None:
    first = gbm_paths(SMALL, 0.05, 0.2, 4, 11)
    require_paths(SMALL, first)
    assert np.array_equal(first, gbm_paths(SMALL, 0.05, 0.2, 4, 11))
    assert not np.array_equal(first, gbm_paths(SMALL, 0.05, 0.2, 4, 12))


def test_simulated_growth_matches_the_risk_neutral_drift() -> None:
    finals: NDArray[np.float64] = gbm_paths(SMALL, 0.05, 0.2, 200_000, 3)[:, -1]
    assert mean(finals) == pytest.approx(100.0 * math.exp(0.05), rel=2e-3)


@pytest.mark.parametrize(
    ("rate", "volatility", "n_paths", "code"),
    [
        (0.05, 0.0, 4, ERR_RATE),
        (0.05, math.inf, 4, ERR_RATE),
        (-1.5, 0.2, 4, ERR_RATE),
        (0.05, 0.2, 0, ERR_PATH_SHAPE),
    ],
)
def test_simulation_refuses_bad_settings(
    rate: float, volatility: float, n_paths: int, code: str
) -> None:
    with pytest.raises(ValueError, match=code):
        gbm_paths(SMALL, rate, volatility, n_paths, 1)


def test_returns_compound_from_the_spot() -> None:
    path = paths_from_returns(SMALL, matrix([[-0.1, 0.0, 0.5, -0.5, 0.0, 0.0]]))
    expected: list[float] = [100.0, 90.0, 90.0, 135.0, 67.5, 67.5, 67.5]
    assert listed(path) == pytest.approx(expected)


@pytest.mark.parametrize(
    "returns",
    [
        floats([0.0] * 6),
        np.empty((0, 6), dtype=np.float64),
        matrix([[0.0] * 5]),
        matrix([[0.0, 0.0, math.inf, 0.0, 0.0, 0.0]]),
        matrix([[0.0, 0.0, -1.0, 0.0, 0.0, 0.0]]),
    ],
)
def test_malformed_returns_are_refused(returns: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match=ERR_PATH_SHAPE):
        paths_from_returns(SMALL, returns)
