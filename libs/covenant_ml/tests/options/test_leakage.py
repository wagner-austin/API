"""A decision on an exercise date reads nothing after that date.

The checks mutate the inputs the study actually consumes, the realized
session returns, and compare what a decision sees. Changing returns after a
date must leave that date's inputs and every choice up to it unchanged;
changing returns before it must change them. The same future-invariance check
is run against a deliberately leaky reader, which it must catch, so a check
that could never fail does not pass for one that holds.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from covenant_ml.options.lsm import exercise_choices, fit_policy
from covenant_ml.options.paths import decision_inputs, paths_from_returns
from covenant_ml.options.types import AsianPutContract

from ._fixtures import BASE

DATE = 20
SESSION = (DATE + 1) * BASE["sessions_per_date"]


def _returns(seed: int) -> NDArray[np.float64]:
    """Realized-looking session returns for 64 contracts."""
    rng = np.random.default_rng(seed)
    returns: NDArray[np.float64] = rng.normal(0.0, 0.02, size=(64, 250))
    return returns


def _future_changed(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Double every return after the decision date's session."""
    changed: NDArray[np.float64] = returns.copy()
    changed[:, SESSION:] = changed[:, SESSION:] * 2.0
    return changed


def _past_changed(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Double every return up to the decision date's session."""
    changed: NDArray[np.float64] = returns.copy()
    changed[:, :SESSION] = changed[:, :SESSION] * 2.0
    return changed


def _leaky_inputs(
    contract: AsianPutContract, paths: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A reader that wrongly averages each whole path, maturity included."""
    spots, _ = decision_inputs(contract, paths)
    whole: NDArray[np.float64] = np.mean(paths, axis=1) / contract["strike"]
    averages: NDArray[np.float64] = np.repeat(whole[:, None], spots.shape[1], axis=1)
    return spots, averages


def _sees_no_future(
    reader: Callable[
        [AsianPutContract, NDArray[np.float64]],
        tuple[NDArray[np.float64], NDArray[np.float64]],
    ],
    returns: NDArray[np.float64],
) -> bool:
    """Whether date DATE's inputs survive a change to every later return."""
    before = reader(BASE, paths_from_returns(BASE, returns))
    after = reader(BASE, paths_from_returns(BASE, _future_changed(returns)))
    pairs = zip(before, after, strict=True)
    return all(np.array_equal(a[:, : DATE + 1], b[:, : DATE + 1]) for a, b in pairs)


def test_future_returns_do_not_reach_a_decision() -> None:
    assert _sees_no_future(decision_inputs, _returns(1))


def test_the_check_catches_a_reader_that_sees_the_future() -> None:
    assert not _sees_no_future(_leaky_inputs, _returns(1))


def test_past_returns_do_reach_a_decision() -> None:
    returns = _returns(2)
    spots, averages = decision_inputs(BASE, paths_from_returns(BASE, returns))
    changed_spots, changed_averages = decision_inputs(
        BASE, paths_from_returns(BASE, _past_changed(returns))
    )
    assert not np.array_equal(spots[:, DATE], changed_spots[:, DATE])
    assert not np.array_equal(averages[:, DATE], changed_averages[:, DATE])


def test_choices_up_to_a_date_ignore_later_returns() -> None:
    policy = fit_policy(BASE, 0.05, 0.2, 5_000, 4)
    returns = _returns(3)
    before = exercise_choices(policy, *decision_inputs(BASE, paths_from_returns(BASE, returns)))
    future = paths_from_returns(BASE, _future_changed(returns))
    after = exercise_choices(policy, *decision_inputs(BASE, future))
    decided: NDArray[np.bool_] = before <= DATE
    assert bool(np.any(decided))
    assert np.array_equal(before[decided], after[decided])
    assert bool(np.all(after[~decided] > DATE))
