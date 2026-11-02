"""The realized-path study over consecutive start months."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_ml.evaluation.types import ERR_INVALID_ARGUMENT, IntervalMethod
from covenant_ml.options.lsm import fit_policy
from covenant_ml.options.paths import gbm_paths, paths_from_returns
from covenant_ml.options.realized import realized_outcomes
from covenant_ml.options.study import (
    HOLD_ARM,
    POLICY_ARM,
    RealizedContracts,
    StudySettings,
    month_label,
    monthly_means,
    run_study,
)
from covenant_ml.options.types import ERR_MONTHS, decode_study_result, encode_study_result

from ._fixtures import SMALL, floats, listed, listed_ints, mean

SETTINGS = StudySettings(block_length=12, n_resamples=200, seed=7, confidence=0.95)
MONTHS = 30
SECURITIES = 4
FIRST_MONTH = 12 * 2000


def _contracts(months: NDArray[np.int64]) -> RealizedContracts:
    """Simulated returns for SECURITIES contracts in each given month."""
    count = len(months) * SECURITIES
    paths = gbm_paths(SMALL, 0.02, 0.3, count, 21)
    returns: NDArray[np.float64] = paths[:, 1:] / paths[:, :-1] - 1.0
    return RealizedContracts(
        securities=np.tile(np.arange(SECURITIES, dtype=np.int64), len(months)),
        months=np.repeat(months, SECURITIES),
        returns=returns,
        carry_rates=np.full(count, 0.02, dtype=np.float64),
    )


def test_month_labels_and_means() -> None:
    assert month_label(FIRST_MONTH) == "2000-01"
    assert month_label(FIRST_MONTH + 23) == "2001-12"
    listing: list[int] = [5, 3, 5, 3]
    months: NDArray[np.int64] = np.asarray(listing, dtype=np.int64)
    distinct, means = monthly_means(months, floats([1.0, 2.0, 3.0, 6.0]))
    assert listed_ints(distinct) == [3, 5]
    assert listed(means) == [4.0, 2.0]


def test_study_pairs_monthly_means_and_counts_what_it_scored() -> None:
    policy = fit_policy(SMALL, 0.02, 0.3, 2_000, 3)
    contracts = _contracts(np.arange(FIRST_MONTH, FIRST_MONTH + MONTHS, dtype=np.int64))
    result = run_study(policy, "small", contracts, SETTINGS)
    outcomes = realized_outcomes(
        policy, paths_from_returns(SMALL, contracts.returns), contracts.carry_rates
    )
    _, policy_means = monthly_means(contracts.months, outcomes.policy_payoffs)
    _, hold_means = monthly_means(contracts.months, outcomes.hold_payoffs)
    comparison = result["comparison"]
    assert result["contracts"] == MONTHS * SECURITIES
    assert result["securities"] == SECURITIES
    assert (result["first_month"], result["last_month"]) == ("2000-01", "2002-06")
    assert result["exercised_early"] == int(np.count_nonzero(outcomes.chosen < 2))
    assert (comparison["arm"], comparison["reference"]) == (POLICY_ARM, HOLD_ARM)
    assert comparison["n_rows"] == MONTHS
    assert comparison["value_arm"] == pytest.approx(mean(policy_means))
    assert comparison["value_reference"] == pytest.approx(mean(hold_means))
    interval = comparison["difference"]
    assert interval["method"] is IntervalMethod.MOVING_BLOCK_BOOTSTRAP
    assert interval["ci_low"] <= interval["estimate"] <= interval["ci_high"]
    assert decode_study_result(encode_study_result(result)) == result


def test_a_gap_in_start_months_is_refused() -> None:
    policy = fit_policy(SMALL, 0.02, 0.3, 2_000, 3)
    months: NDArray[np.int64] = np.arange(FIRST_MONTH, FIRST_MONTH + MONTHS + 1, dtype=np.int64)
    contracts = _contracts(np.delete(months, 10))
    with pytest.raises(ValueError, match=ERR_INVALID_ARGUMENT):
        run_study(policy, "small", contracts, SETTINGS)


def test_misaligned_contract_fields_are_refused() -> None:
    policy = fit_policy(SMALL, 0.02, 0.3, 2_000, 3)
    contracts = _contracts(np.arange(FIRST_MONTH, FIRST_MONTH + MONTHS, dtype=np.int64))
    with pytest.raises(ValueError, match=ERR_MONTHS):
        run_study(policy, "small", contracts._replace(months=contracts.months[1:]), SETTINGS)
    with pytest.raises(ValueError, match=ERR_MONTHS):
        run_study(
            policy, "small", contracts._replace(securities=contracts.securities[1:]), SETTINGS
        )
