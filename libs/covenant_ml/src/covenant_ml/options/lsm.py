"""Longstaff-Schwartz exercise policies for the Bermudan Asian put.

On every exercise date before maturity the holder compares the payoff of
exercising now with the estimated value of continuing. The continuation value
is a least-squares regression, fitted on in-the-money simulated paths only, of
the discounted cash flow the policy goes on to realize, on a quadratic basis
in spot over strike and running average over strike (:data:`BASIS_TERMS`).
Dates are fitted backward from maturity, so each regression sees the policy
already fitted for the dates after it.

A fitted policy is its coefficient table. :func:`exercise_choices` applies it
to any paths' decision inputs, simulated or realized, and never reads a price
after the date it decides.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .paths import decision_inputs, exp_array, gbm_paths, put_payoffs
from .types import BASIS_TERMS, ERR_TOO_FEW_IN_THE_MONEY, AsianPutContract, LsmPolicy


class _LeastSquaresProtocol(Protocol):
    """``numpy.linalg.lstsq`` with the float64 result its stubs leave untyped."""

    def __call__(
        self, a: NDArray[np.float64], b: NDArray[np.float64], rcond: None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], int, NDArray[np.float64]]: ...


_linalg = __import__("numpy.linalg", fromlist=["lstsq"])
_least_squares: _LeastSquaresProtocol = _linalg.lstsq


def basis(spots: NDArray[np.float64], averages: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the regression basis.

    Args:
        spots: Spot over strike, one value per path.
        averages: Running average over strike, aligned with ``spots``.

    Returns:
        One row per path, one column per :data:`BASIS_TERMS` term.
    """
    columns = (
        np.ones_like(spots),
        spots,
        averages,
        spots * spots,
        averages * averages,
        spots * averages,
    )
    design: NDArray[np.float64] = np.column_stack(columns)
    return design


def date_discount(contract: AsianPutContract, rate: float) -> float:
    """Discount one exercise interval at a continuously compounded rate.

    Args:
        contract: Sets the interval, ``years / exercise_dates``.
        rate: The rate.

    Returns:
        ``exp(-rate * years / exercise_dates)``.
    """
    return math.exp(-rate * contract["years"] / contract["exercise_dates"])


def fit_policy(
    contract: AsianPutContract, rate: float, volatility: float, n_paths: int, seed: int
) -> LsmPolicy:
    """Fit continuation regressions on simulated geometric Brownian motion.

    Args:
        contract: The contract to decide for.
        rate: Continuously compounded risk-free rate of the training model.
        volatility: Annual volatility of the training model.
        n_paths: Simulated training paths.
        seed: Seed of those paths.

    Returns:
        The fitted policy.

    Raises:
        ValueError: CVML-OPT-004 when a date has fewer in-the-money paths than
            basis terms, so its regression is not identified; the simulation's
            own refusals propagate.
    """
    spots, averages = decision_inputs(
        contract, gbm_paths(contract, rate, volatility, n_paths, seed)
    )
    payoffs = put_payoffs(contract, averages)
    discount = date_discount(contract, rate)
    cash: NDArray[np.float64] = payoffs[:, -1].copy()
    rows: list[tuple[float, ...]] = []
    for date in range(contract["exercise_dates"] - 2, -1, -1):
        cash *= discount
        in_the_money: NDArray[np.intp] = np.flatnonzero(payoffs[:, date] > 0.0)
        if len(in_the_money) < len(BASIS_TERMS):
            raise ValueError(
                f"[{ERR_TOO_FEW_IN_THE_MONEY}] date {date} has {len(in_the_money)} "
                f"in-the-money paths, fewer than {len(BASIS_TERMS)} basis terms"
            )
        date_spots: NDArray[np.float64] = spots[in_the_money, date]
        date_averages: NDArray[np.float64] = averages[in_the_money, date]
        design = basis(date_spots, date_averages)
        targets: NDArray[np.float64] = cash[in_the_money]
        coefficients, _, _, _ = _least_squares(design, targets, rcond=None)
        continuation: NDArray[np.float64] = design @ coefficients
        immediate: NDArray[np.float64] = payoffs[in_the_money, date]
        better: NDArray[np.bool_] = immediate > continuation
        exercised: NDArray[np.intp] = in_the_money[better]
        cash[exercised] = immediate[better]
        rows.append(tuple(float(coefficients.flat[term]) for term in range(len(BASIS_TERMS))))
    rows.reverse()
    return LsmPolicy(
        contract=contract,
        rate=rate,
        volatility=volatility,
        training_paths=n_paths,
        seed=seed,
        coefficients=tuple(rows),
    )


def exercise_choices(
    policy: LsmPolicy, spots: NDArray[np.float64], averages: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Choose each path's exercise date.

    A path is exercised on the first date before maturity where its payoff is
    positive and exceeds the fitted continuation value; otherwise it is held to
    maturity. Date ``j``'s choice reads column ``j`` of the inputs alone.

    Args:
        policy: The fitted policy.
        spots: Spot over strike, one row per path, one column per date.
        averages: Running average over strike, the same shape.

    Returns:
        The chosen date index per path, ``exercise_dates - 1`` for maturity.
    """
    contract = policy["contract"]
    payoffs = put_payoffs(contract, averages)
    chosen: NDArray[np.int64] = np.full(
        spots.shape[0], contract["exercise_dates"] - 1, dtype=np.int64
    )
    open_paths: NDArray[np.bool_] = np.ones(spots.shape[0], dtype=np.bool_)
    for date, row in enumerate(policy["coefficients"]):
        coefficients: NDArray[np.float64] = np.asarray(row, dtype=np.float64)
        continuation: NDArray[np.float64] = basis(spots[:, date], averages[:, date]) @ coefficients
        payoff: NDArray[np.float64] = payoffs[:, date]
        exercise: NDArray[np.bool_] = open_paths & (payoff > 0.0) & (payoff > continuation)
        chosen[exercise] = date
        open_paths &= ~exercise
    return chosen


def simulated_value(policy: LsmPolicy, n_paths: int, seed: int) -> float:
    """Value the policy on fresh paths of its own training model.

    Args:
        policy: The fitted policy.
        n_paths: Fresh paths to draw.
        seed: Their seed, which should differ from the training seed.

    Returns:
        The mean payoff at each path's chosen date, discounted to the start.
    """
    contract = policy["contract"]
    rate = policy["rate"]
    paths = gbm_paths(contract, rate, policy["volatility"], n_paths, seed)
    spots, averages = decision_inputs(contract, paths)
    chosen = exercise_choices(policy, spots, averages)
    rows: NDArray[np.intp] = np.arange(n_paths)
    realized: NDArray[np.float64] = put_payoffs(contract, averages)[rows, chosen]
    dates: NDArray[np.float64] = (chosen + 1).astype(np.float64)
    years: NDArray[np.float64] = dates * (contract["years"] / contract["exercise_dates"])
    discounted: NDArray[np.float64] = realized * exp_array(-rate * years)
    return float(np.sum(discounted)) / n_paths


__all__ = ["basis", "date_discount", "exercise_choices", "fit_policy", "simulated_value"]
