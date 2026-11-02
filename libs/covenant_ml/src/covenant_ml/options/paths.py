"""Price paths of a Bermudan Asian put: simulated, realized, and what a policy may see.

A path is one row of ``sessions + 1`` prices, column 0 the normalized spot at
the contract's start and column ``k`` the close ``k`` sessions later. The
arithmetic running average at session ``k`` includes the start, so it is the
mean of columns ``0..k``. A decision on exercise date ``j`` (session
``j * sessions_per_date``) sees that session's spot and running average and
nothing later; :func:`decision_inputs` is the only reader of a path that a
policy's decision goes through.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .types import ERR_PATH_SHAPE, ERR_RATE, AsianPutContract, require_contract


def total_sessions(contract: AsianPutContract) -> int:
    """Count the sessions from the start to maturity.

    Args:
        contract: The contract.

    Returns:
        ``exercise_dates * sessions_per_date``.
    """
    return contract["exercise_dates"] * contract["sessions_per_date"]


def exercise_sessions(contract: AsianPutContract) -> NDArray[np.int64]:
    """Name the session of every exercise date, maturity last.

    Args:
        contract: The contract.

    Returns:
        ``sessions_per_date * (1, 2, ..., exercise_dates)``.
    """
    dates: NDArray[np.int64] = np.arange(1, contract["exercise_dates"] + 1, dtype=np.int64)
    sessions: NDArray[np.int64] = dates * contract["sessions_per_date"]
    return sessions


def require_paths(contract: AsianPutContract, paths: NDArray[np.float64]) -> None:
    """Refuse paths a contract cannot be evaluated on.

    Args:
        contract: The contract the paths belong to.
        paths: One row per path.

    Raises:
        ValueError: CVML-OPT-002 unless the array has one row or more of
            ``sessions + 1`` finite positive prices starting at the spot.
    """
    require_contract(contract)
    width = total_sessions(contract) + 1
    if not _is_table(paths, width):
        raise ValueError(f"[{ERR_PATH_SHAPE}] paths must be rows of {width} prices")
    finite: NDArray[np.bool_] = np.isfinite(paths)
    positive: NDArray[np.bool_] = paths > 0.0
    if not bool(np.all(finite)) or not bool(np.all(positive)):
        raise ValueError(f"[{ERR_PATH_SHAPE}] every price must be finite and positive")
    starts: NDArray[np.float64] = paths[:, 0]
    at_spot: NDArray[np.bool_] = starts == contract["spot"]
    if not bool(np.all(at_spot)):
        raise ValueError(f"[{ERR_PATH_SHAPE}] every path must start at the spot")


def _is_table(values: NDArray[np.float64], width: int) -> bool:
    """Whether an array is one or more rows of exactly ``width`` columns.

    Args:
        values: The array.
        width: Required columns.

    Returns:
        True for a two-dimensional array with a row and ``width`` columns.
    """
    return values.ndim == 2 and int(values.shape[0]) >= 1 and int(values.shape[1]) == width


def running_averages(paths: NDArray[np.float64]) -> NDArray[np.float64]:
    """Average each path from its start through every session.

    Args:
        paths: Validated paths.

    Returns:
        Same shape; column ``k`` is the mean of columns ``0..k``.
    """
    counts: NDArray[np.float64] = np.arange(1, int(paths.shape[1]) + 1, dtype=np.float64)
    averages: NDArray[np.float64] = np.cumsum(paths, axis=1) / counts
    return averages


def decision_inputs(
    contract: AsianPutContract, paths: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read what a policy may know on each exercise date.

    Args:
        contract: The contract.
        paths: Paths :func:`require_paths` accepts.

    Returns:
        Spot over strike and running average over strike, each with one row
        per path and one column per exercise date. Column ``j`` depends only on
        prices up to that date's session.

    Raises:
        ValueError: CVML-OPT-002 for paths :func:`require_paths` refuses.
    """
    require_paths(contract, paths)
    sessions = exercise_sessions(contract)
    strike = contract["strike"]
    spots: NDArray[np.float64] = paths[:, sessions] / strike
    averages: NDArray[np.float64] = running_averages(paths)[:, sessions] / strike
    return spots, averages


def put_payoffs(contract: AsianPutContract, averages: NDArray[np.float64]) -> NDArray[np.float64]:
    """Value immediate exercise from averages already divided by the strike.

    Args:
        contract: The contract.
        averages: Running average over strike, any shape.

    Returns:
        ``max(strike - average, 0)`` in price units, same shape.
    """
    strike = contract["strike"]
    payoffs: NDArray[np.float64] = np.maximum(strike - averages * strike, 0.0)
    return payoffs


def require_rates(rates: NDArray[np.float64]) -> None:
    """Refuse rates no discounting can use.

    Args:
        rates: Continuously compounded annual rates.

    Raises:
        ValueError: CVML-OPT-003 for a nonfinite rate or one at or below -100%.
    """
    finite: NDArray[np.bool_] = np.isfinite(rates)
    above: NDArray[np.bool_] = rates > -1.0
    if not bool(np.all(finite)) or not bool(np.all(above)):
        raise ValueError(f"[{ERR_RATE}] a rate must be finite and above -1")


def require_rate(rate: float) -> None:
    """Refuse a single rate :func:`require_rates` would refuse.

    Args:
        rate: A continuously compounded annual rate.

    Raises:
        ValueError: CVML-OPT-003 as :func:`require_rates` does.
    """
    require_rates(np.full(1, rate, dtype=np.float64))


def gbm_paths(
    contract: AsianPutContract, rate: float, volatility: float, n_paths: int, seed: int
) -> NDArray[np.float64]:
    """Simulate risk-neutral geometric Brownian motion paths.

    Args:
        contract: Sets the spot, the life and the session grid.
        rate: Continuously compounded risk-free rate (the drift).
        volatility: Annual volatility, positive.
        n_paths: Paths to draw, at least one.
        seed: Generator seed; the same seed gives the same paths.

    Returns:
        ``n_paths`` rows :func:`require_paths` accepts.

    Raises:
        ValueError: CVML-OPT-003 for a rate :func:`require_rate` refuses or a
            nonpositive volatility; CVML-OPT-002 for no paths; CVML-OPT-001 for
            the contract.
    """
    require_contract(contract)
    require_rate(rate)
    if not (math.isfinite(volatility) and volatility > 0.0):
        raise ValueError(f"[{ERR_RATE}] volatility must be finite and positive")
    if n_paths < 1:
        raise ValueError(f"[{ERR_PATH_SHAPE}] simulate at least one path")
    steps = total_sessions(contract)
    step = contract["years"] / steps
    shocks: NDArray[np.float64] = np.random.default_rng(seed).standard_normal((n_paths, steps))
    drift = (rate - 0.5 * volatility * volatility) * step
    increments: NDArray[np.float64] = drift + volatility * math.sqrt(step) * shocks
    logs: NDArray[np.float64] = np.cumsum(increments, axis=1)
    paths: NDArray[np.float64] = np.empty((n_paths, steps + 1), dtype=np.float64)
    paths[:, 0] = contract["spot"]
    paths[:, 1:] = contract["spot"] * np.exp(logs)
    return paths


def paths_from_returns(
    contract: AsianPutContract, returns: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Compound realized session returns into paths starting at the spot.

    A security that stops trading inside a contract carries its terminal value
    to maturity as zero returns; that is the caller's to encode, because only
    it knows the terminal value.

    Args:
        contract: The contract.
        returns: One row per contract of ``sessions`` simple returns, each
            above -1.

    Returns:
        Paths :func:`require_paths` accepts.

    Raises:
        ValueError: CVML-OPT-002 for a row of the wrong length, a nonfinite
            return or one at or below -1.
    """
    require_contract(contract)
    steps = total_sessions(contract)
    if not _is_table(returns, steps):
        raise ValueError(f"[{ERR_PATH_SHAPE}] returns must be rows of {steps} sessions")
    finite: NDArray[np.bool_] = np.isfinite(returns)
    above: NDArray[np.bool_] = returns > -1.0
    if not bool(np.all(finite)) or not bool(np.all(above)):
        raise ValueError(f"[{ERR_PATH_SHAPE}] returns must be finite and above -1")
    growth: NDArray[np.float64] = 1.0 + returns
    paths: NDArray[np.float64] = np.empty((int(returns.shape[0]), steps + 1), dtype=np.float64)
    paths[:, 0] = contract["spot"]
    paths[:, 1:] = contract["spot"] * np.cumprod(growth, axis=1, dtype=np.float64)
    return paths


__all__ = [
    "decision_inputs",
    "exercise_sessions",
    "gbm_paths",
    "paths_from_returns",
    "put_payoffs",
    "require_paths",
    "require_rate",
    "require_rates",
    "running_averages",
    "total_sessions",
]
