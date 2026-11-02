"""The option study's records survive JSON and refuse what they cannot hold."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from covenant_ml.evaluation.types import ComparisonMetric, IntervalMethod
from covenant_ml.options.types import (
    BASIS_TERMS,
    ERR_CONTRACT,
    ERR_POLICY,
    AsianPutContract,
    LsmPolicy,
    StudyResult,
    decode_contract,
    decode_policy,
    decode_study_result,
    encode_contract,
    encode_policy,
    encode_study_result,
    require_contract,
)

from ._fixtures import SMALL, contract_with

POLICY = LsmPolicy(
    contract=SMALL,
    rate=0.05,
    volatility=0.2,
    training_paths=1000,
    seed=7,
    coefficients=((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (0.5, -1.0, 0.0, 2.5, 1.0, -3.0)),
)


def _saved_policy(**changes: JSONValue) -> JSONObject:
    """The encoded :data:`POLICY` with fields replaced."""
    saved = encode_policy(POLICY)
    saved.update(changes)
    return saved


def test_contract_roundtrips() -> None:
    assert decode_contract(encode_contract(SMALL)) == SMALL


@pytest.mark.parametrize(
    "contract",
    [
        contract_with(spot=0.0),
        contract_with(strike=-1.0),
        contract_with(years=float("nan")),
        contract_with(years=float("inf")),
        AsianPutContract(spot=1.0, strike=1.0, years=1.0, exercise_dates=1, sessions_per_date=1),
        AsianPutContract(spot=1.0, strike=1.0, years=1.0, exercise_dates=2, sessions_per_date=0),
    ],
)
def test_unusable_contracts_are_refused(contract: AsianPutContract) -> None:
    with pytest.raises(ValueError, match=ERR_CONTRACT):
        require_contract(contract)
    with pytest.raises(ValueError, match=ERR_CONTRACT):
        decode_contract(encode_contract(contract))


def test_policy_roundtrips_with_its_basis_named() -> None:
    saved = encode_policy(POLICY)
    assert saved["basis"] == list(BASIS_TERMS)
    assert decode_policy(saved) == POLICY


def test_integer_coefficients_are_read_as_floats() -> None:
    rows: list[JSONValue] = [[1, 2, 3, 4, 5, 6], [0, 0, 0, 0, 0, 0]]
    policy = decode_policy(_saved_policy(coefficients=rows))
    assert policy["coefficients"][0] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"basis": ["1", "s", "a"]}, ERR_POLICY),
        ({"coefficients": [[1, 2, 3, 4, 5, 6]]}, ERR_POLICY),
        ({"coefficients": [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]}, ERR_POLICY),
        ({"volatility": 0.0}, ERR_POLICY),
        ({"volatility": float("inf")}, ERR_POLICY),
        ({"training_paths": 0}, ERR_POLICY),
    ],
)
def test_malformed_policies_are_refused(changes: dict[str, JSONValue], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        decode_policy(_saved_policy(**changes))


def test_nonfinite_coefficients_are_refused() -> None:
    rows: list[JSONValue] = [[1, 2, 3, 4, 5, float("nan")], [0, 0, 0, 0, 0, 0]]
    with pytest.raises(ValueError, match=ERR_POLICY):
        decode_policy(_saved_policy(coefficients=rows))


@pytest.mark.parametrize(
    "coefficients",
    [["row", [0, 0, 0, 0, 0, 0]], [[1, 2, 3, 4, 5, True], [0, 0, 0, 0, 0, 0]]],
)
def test_mistyped_coefficients_are_refused(coefficients: list[JSONValue]) -> None:
    with pytest.raises(JSONTypeError):
        decode_policy(_saved_policy(coefficients=coefficients))


def test_study_result_roundtrips() -> None:
    result = StudyResult(
        policy_name="fixed-gbm",
        contracts=10,
        securities=3,
        first_month="2000-01",
        last_month="2001-12",
        exercised_early=4,
        comparison={
            "metric": ComparisonMetric.UTILITY,
            "arm": "exercise-policy",
            "reference": "hold-to-maturity",
            "value_arm": 2.0,
            "value_reference": 1.5,
            "difference": {
                "method": IntervalMethod.MOVING_BLOCK_BOOTSTRAP,
                "estimate": 0.5,
                "std_error": 0.1,
                "ci_low": 0.3,
                "ci_high": 0.7,
                "confidence": 0.95,
                "p_value": 0.01,
                "n_units": 2,
            },
            "n_rows": 24,
        },
    )
    assert decode_study_result(encode_study_result(result)) == result
