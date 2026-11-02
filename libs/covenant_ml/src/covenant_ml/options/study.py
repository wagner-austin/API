"""The realized-path study: one policy, many contracts, intervalled over months.

Contracts start on the first session of consecutive months, one per eligible
security per month. Contracts starting within a year of each other share most
of their path, and contracts in one month share the market, so neither a
contract nor a month is an independent unit. The estimand is the mean over
months of each month's average policy-minus-hold difference, and its interval
is the shared paired moving-block bootstrap over consecutive months
(:func:`~covenant_ml.evaluation.paired.compare_moving_block`), whose blocks
should span at least the contracts' one-year overlap.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..evaluation.paired import compare_moving_block
from .paths import paths_from_returns
from .realized import realized_outcomes
from .types import ERR_MONTHS, LsmPolicy, StudyResult

#: The arm and reference names a study's comparison carries.
POLICY_ARM = "exercise-policy"
HOLD_ARM = "hold-to-maturity"


class RealizedContracts(NamedTuple):
    """Hypothetical contracts on realized paths, one row each.

    Attributes:
        securities: Underlying security identifier per contract.
        months: Start month per contract as ``12 * year + month - 1``.
        returns: The contract's session returns, start to maturity.
        carry_rates: Continuously compounded rate known at the contract's start.
    """

    securities: NDArray[np.int64]
    months: NDArray[np.int64]
    returns: NDArray[np.float64]
    carry_rates: NDArray[np.float64]


class StudySettings(NamedTuple):
    """How the monthly interval is drawn.

    Attributes:
        block_length: Consecutive months per bootstrap block.
        n_resamples: Bootstrap draws.
        seed: Draw seed.
        confidence: Interval coverage.
    """

    block_length: int
    n_resamples: int
    seed: int
    confidence: float


def month_label(month: int) -> str:
    """Render a month ordinal.

    Args:
        month: ``12 * year + month - 1``.

    Returns:
        ``YYYY-MM``.
    """
    return f"{month // 12:04d}-{month % 12 + 1:02d}"


def monthly_means(
    months: NDArray[np.int64], values: NDArray[np.float64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Average values within each start month.

    Args:
        months: Start month per contract.
        values: One value per contract.

    Returns:
        The distinct months in order and each one's mean value.
    """
    distinct, inverse = np.unique(months, return_inverse=True)
    counts: NDArray[np.float64] = np.bincount(inverse).astype(np.float64)
    sums: NDArray[np.float64] = np.bincount(inverse, weights=values).astype(np.float64)
    ordered: NDArray[np.int64] = distinct.astype(np.int64)
    means: NDArray[np.float64] = sums / counts
    return ordered, means


def run_study(
    policy: LsmPolicy, policy_name: str, contracts: RealizedContracts, settings: StudySettings
) -> StudyResult:
    """Apply a policy to realized contracts and compare it with holding.

    Args:
        policy: The fitted policy, applied unchanged.
        policy_name: Its name in the result.
        contracts: The realized contracts.
        settings: The monthly interval's settings.

    Returns:
        Counts, the month span and the paired monthly comparison.

    Raises:
        ValueError: CVML-OPT-006 when the contract fields are not aligned;
            the path, rate and moving-block refusals propagate, including
            CVML-EVAL-018 when the start months are not consecutive.
    """
    count = contracts.returns.shape[0]
    if contracts.securities.shape != (count,) or contracts.months.shape != (count,):
        raise ValueError(f"[{ERR_MONTHS}] securities and months must have one entry per contract")
    paths = paths_from_returns(policy["contract"], contracts.returns)
    outcomes = realized_outcomes(policy, paths, contracts.carry_rates)
    months, policy_means = monthly_means(contracts.months, outcomes.policy_payoffs)
    _, hold_means = monthly_means(contracts.months, outcomes.hold_payoffs)
    comparison = compare_moving_block(
        POLICY_ARM,
        HOLD_ARM,
        policy_means,
        hold_means,
        months,
        months,
        block_length=settings.block_length,
        n_resamples=settings.n_resamples,
        seed=settings.seed,
        confidence=settings.confidence,
    )
    maturity = policy["contract"]["exercise_dates"] - 1
    return StudyResult(
        policy_name=policy_name,
        contracts=count,
        securities=len(np.unique(contracts.securities)),
        first_month=month_label(int(months.flat[0])),
        last_month=month_label(int(months.flat[len(months) - 1])),
        exercised_early=int(np.count_nonzero(outcomes.chosen < maturity)),
        comparison=comparison,
    )


__all__ = [
    "HOLD_ARM",
    "POLICY_ARM",
    "RealizedContracts",
    "StudySettings",
    "month_label",
    "monthly_means",
    "run_study",
]
