"""Apply a fitted policy to hypothetical contracts on realized paths.

Each contract is the policy's contract written on one realized path. The
policy exercises on the date :func:`~covenant_ml.options.lsm.exercise_choices`
picks, seeing only that date's spot and running average. An early payoff is
carried to maturity at the contract's own risk-free rate, known at its start,
so it can be paired with the payoff of holding the same contract to maturity.
Neither is an option price or a trading profit.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .lsm import exercise_choices
from .paths import decision_inputs, put_payoffs, require_rates
from .types import ERR_PATH_SHAPE, LsmPolicy


class RealizedOutcomes(NamedTuple):
    """Per-contract results of one policy on realized paths.

    Attributes:
        chosen: Exercise date index per contract, ``exercise_dates - 1`` when
            held to maturity.
        policy_payoffs: The policy's payoff, carried to maturity.
        hold_payoffs: The payoff of holding the same contract to maturity.
    """

    chosen: NDArray[np.int64]
    policy_payoffs: NDArray[np.float64]
    hold_payoffs: NDArray[np.float64]


def realized_outcomes(
    policy: LsmPolicy, paths: NDArray[np.float64], carry_rates: NDArray[np.float64]
) -> RealizedOutcomes:
    """Score a policy against holding on the same realized contracts.

    Args:
        policy: The fitted policy, applied unchanged.
        paths: One realized path per contract, normalized to the spot.
        carry_rates: Continuously compounded annual rate per contract, known at
            its start, that carries an early payoff to maturity.

    Returns:
        The chosen dates and both payoffs per contract.

    Raises:
        ValueError: CVML-OPT-002 for paths the contract refuses or a rate
            vector not aligned with them; CVML-OPT-003 for rates
            :func:`~covenant_ml.options.paths.require_rates` refuses.
    """
    contract = policy["contract"]
    spots, averages = decision_inputs(contract, paths)
    count = int(paths.shape[0])
    if carry_rates.ndim != 1 or len(carry_rates) != count:
        raise ValueError(f"[{ERR_PATH_SHAPE}] one carry rate per contract")
    require_rates(carry_rates)
    chosen = exercise_choices(policy, spots, averages)
    payoffs = put_payoffs(contract, averages)
    rows: NDArray[np.intp] = np.arange(count)
    exercised: NDArray[np.float64] = payoffs[rows, chosen]
    interval = contract["years"] / contract["exercise_dates"]
    dates_left: NDArray[np.float64] = (contract["exercise_dates"] - 1 - chosen).astype(np.float64)
    remaining: NDArray[np.float64] = dates_left * interval
    carried: NDArray[np.float64] = exercised * np.exp(carry_rates * remaining)
    held: NDArray[np.float64] = payoffs[:, -1].copy()
    return RealizedOutcomes(chosen=chosen, policy_payoffs=carried, hold_payoffs=held)


__all__ = ["RealizedOutcomes", "realized_outcomes"]
