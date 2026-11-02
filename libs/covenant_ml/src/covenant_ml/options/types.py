"""Records of the Asian-put exercise study, with their JSON codecs.

Every record crosses a file boundary (a fitted policy is saved before it is
applied; a study result is saved beside its report), so each has an encoder
and a decoder that validates every field.
"""

from __future__ import annotations

import math

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    narrow_json_to_float,
    narrow_json_to_list,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
    require_str_list,
)
from typing_extensions import TypedDict

from ..evaluation.codec_records import decode_comparison, encode_comparison
from ..evaluation.types import PairedComparison

ERR_CONTRACT = "CVML-OPT-001"
ERR_PATH_SHAPE = "CVML-OPT-002"
ERR_RATE = "CVML-OPT-003"
ERR_TOO_FEW_IN_THE_MONEY = "CVML-OPT-004"
ERR_POLICY = "CVML-OPT-005"
ERR_MONTHS = "CVML-OPT-006"

#: Columns of the regression basis in (spot / strike, average / strike).
BASIS_TERMS: tuple[str, ...] = ("1", "s", "a", "s^2", "a^2", "s*a")


class AsianPutContract(TypedDict, total=True):
    """A Bermudan arithmetic-average Asian put on one normalized path.

    Args:
        spot: Initial price every path is normalized to.
        strike: Strike of the put on the running average.
        years: Life of the contract.
        exercise_dates: Exercise opportunities, the last one at maturity.
        sessions_per_date: Trading sessions between consecutive dates.
    """

    spot: float
    strike: float
    years: float
    exercise_dates: int
    sessions_per_date: int


class LsmPolicy(TypedDict, total=True):
    """Continuation-value regressions fitted under one simulated model.

    Args:
        contract: The contract the policy decides for.
        rate: Continuously compounded risk-free rate of the training model.
        volatility: Annual volatility of the training model.
        training_paths: Simulated paths the regressions were fitted on.
        seed: Seed of those paths.
        coefficients: One row of :data:`BASIS_TERMS` coefficients per exercise
            date before maturity, in date order.
    """

    contract: AsianPutContract
    rate: float
    volatility: float
    training_paths: int
    seed: int
    coefficients: tuple[tuple[float, ...], ...]


class StudyResult(TypedDict, total=True):
    """A policy applied to realized contracts and compared with holding.

    Args:
        policy_name: Which fitted policy was applied.
        contracts: Realized contracts scored.
        securities: Distinct underlying securities among them.
        first_month: First contract-start month, ``YYYY-MM``.
        last_month: Last contract-start month, ``YYYY-MM``.
        exercised_early: Contracts the policy exercised before maturity.
        comparison: Monthly mean policy payoff against monthly mean
            hold-to-maturity payoff, with a moving-block interval over months.
    """

    policy_name: str
    contracts: int
    securities: int
    first_month: str
    last_month: str
    exercised_early: int
    comparison: PairedComparison


def require_contract(contract: AsianPutContract) -> None:
    """Refuse a contract no path can be evaluated under.

    Args:
        contract: The contract.

    Raises:
        ValueError: CVML-OPT-001 for a nonpositive or nonfinite spot, strike or
            life, fewer than two exercise dates or no session between dates.
    """
    values = (contract["spot"], contract["strike"], contract["years"])
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(f"[{ERR_CONTRACT}] spot, strike and years must be finite and positive")
    if contract["exercise_dates"] < 2 or contract["sessions_per_date"] < 1:
        raise ValueError(f"[{ERR_CONTRACT}] a contract needs two dates and a session between")


def encode_contract(contract: AsianPutContract) -> JSONObject:
    """Lower a contract to JSON.

    Args:
        contract: The contract.

    Returns:
        A JSON object mirroring :class:`AsianPutContract`.
    """
    return {
        "spot": contract["spot"],
        "strike": contract["strike"],
        "years": contract["years"],
        "exercise_dates": contract["exercise_dates"],
        "sessions_per_date": contract["sessions_per_date"],
    }


def decode_contract(obj: JSONObject) -> AsianPutContract:
    """Raise a contract out of untrusted JSON.

    Args:
        obj: The parsed object.

    Returns:
        The validated contract.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
        ValueError: CVML-OPT-001 for a contract :func:`require_contract` refuses.
    """
    contract = AsianPutContract(
        spot=require_float(obj, "spot"),
        strike=require_float(obj, "strike"),
        years=require_float(obj, "years"),
        exercise_dates=require_int(obj, "exercise_dates"),
        sessions_per_date=require_int(obj, "sessions_per_date"),
    )
    require_contract(contract)
    return contract


def encode_policy(policy: LsmPolicy) -> JSONObject:
    """Lower a fitted policy to JSON.

    Args:
        policy: The policy.

    Returns:
        A JSON object mirroring :class:`LsmPolicy`.
    """
    rows: list[JSONValue] = [list(row) for row in policy["coefficients"]]
    return {
        "contract": encode_contract(policy["contract"]),
        "rate": policy["rate"],
        "volatility": policy["volatility"],
        "training_paths": policy["training_paths"],
        "seed": policy["seed"],
        "basis": list(BASIS_TERMS),
        "coefficients": rows,
    }


def _coefficient_row(value: JSONValue) -> tuple[float, ...]:
    """Validate one date's coefficients.

    Args:
        value: One element of the saved coefficient list.

    Returns:
        The finite coefficients.

    Raises:
        JSONTypeError: If the row is not a list of numbers.
        ValueError: CVML-OPT-005 for a row of the wrong width or a nonfinite value.
    """
    row = [narrow_json_to_float(item) for item in narrow_json_to_list(value)]
    if len(row) != len(BASIS_TERMS) or not all(math.isfinite(item) for item in row):
        raise ValueError(f"[{ERR_POLICY}] each date needs {len(BASIS_TERMS)} finite coefficients")
    return tuple(row)


def decode_policy(value: JSONValue) -> LsmPolicy:
    """Raise a fitted policy out of untrusted JSON.

    Args:
        value: The parsed document.

    Returns:
        The validated policy.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
        ValueError: CVML-OPT-005 for a different basis, a coefficient table
            whose length is not one row per date before maturity, or a
            nonpositive volatility or path count; CVML-OPT-001 for the contract.
    """
    obj = narrow_json_to_dict(value)
    contract = decode_contract(require_dict(obj, "contract"))
    basis = tuple(require_str_list(obj, "basis"))
    if basis != BASIS_TERMS:
        raise ValueError(f"[{ERR_POLICY}] the saved basis is {basis}, not {BASIS_TERMS}")
    rows = tuple(_coefficient_row(item) for item in require_list(obj, "coefficients"))
    if len(rows) != contract["exercise_dates"] - 1:
        raise ValueError(f"[{ERR_POLICY}] one coefficient row per date before maturity")
    volatility = require_float(obj, "volatility")
    training_paths = require_int(obj, "training_paths")
    if not (math.isfinite(volatility) and volatility > 0.0) or training_paths < 1:
        raise ValueError(f"[{ERR_POLICY}] volatility and training paths must be positive")
    return LsmPolicy(
        contract=contract,
        rate=require_float(obj, "rate"),
        volatility=volatility,
        training_paths=training_paths,
        seed=require_int(obj, "seed"),
        coefficients=rows,
    )


def encode_study_result(result: StudyResult) -> JSONObject:
    """Lower a study result to JSON.

    Args:
        result: The result.

    Returns:
        A JSON object mirroring :class:`StudyResult`.
    """
    return {
        "policy_name": result["policy_name"],
        "contracts": result["contracts"],
        "securities": result["securities"],
        "first_month": result["first_month"],
        "last_month": result["last_month"],
        "exercised_early": result["exercised_early"],
        "comparison": encode_comparison(result["comparison"]),
    }


def decode_study_result(value: JSONValue) -> StudyResult:
    """Raise a study result out of untrusted JSON.

    Args:
        value: The parsed document.

    Returns:
        The validated result.

    Raises:
        JSONTypeError: If a field is missing or mistyped.
    """
    obj = narrow_json_to_dict(value)
    return StudyResult(
        policy_name=require_str(obj, "policy_name"),
        contracts=require_int(obj, "contracts"),
        securities=require_int(obj, "securities"),
        first_month=require_str(obj, "first_month"),
        last_month=require_str(obj, "last_month"),
        exercised_early=require_int(obj, "exercised_early"),
        comparison=decode_comparison(require_dict(obj, "comparison")),
    )


__all__ = [
    "BASIS_TERMS",
    "ERR_CONTRACT",
    "ERR_MONTHS",
    "ERR_PATH_SHAPE",
    "ERR_POLICY",
    "ERR_RATE",
    "ERR_TOO_FEW_IN_THE_MONEY",
    "AsianPutContract",
    "LsmPolicy",
    "StudyResult",
    "decode_contract",
    "decode_policy",
    "decode_study_result",
    "encode_contract",
    "encode_policy",
    "encode_study_result",
    "require_contract",
]
