"""A forward test's frozen plan, read only from the exact bytes its board record hashed.

The plan names which campaign answers for it, which saved arm is the model and
which is the market reference, the future event dates it scores, and the costs and
statistics of its comparison. It is recorded before its first contract settles; the
scorer refuses any file whose SHA-256 is not the one pinned beside it, and any cost
the quoted taker policy does not implement, so the declared plan and the executed one
cannot drift apart.
"""

from __future__ import annotations

from hashlib import sha256
from itertools import pairwise
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_int,
    require_str,
    require_str_list,
)
from typing_extensions import ReadOnly, TypedDict

from .forecast_policy import QUOTED_FEE_BPS, QUOTED_HURDLE, QUOTED_STRESS
from .forecast_records import CITY_COORDINATES
from .settlement_records import ARMS
from .validation import require_day, require_nonempty


class ForwardCampaign(TypedDict):
    """The frozen collector campaign whose saved captures and settlements are scored."""

    name: ReadOnly[str]
    manifest_sha256: ReadOnly[str]
    collector_implementation_sha256: ReadOnly[str]
    protocol_sha256: ReadOnly[str]


class ForwardModel(TypedDict):
    """The saved prediction arm under test and the fitted report that produced it."""

    arm: ReadOnly[str]
    forecast_hour: ReadOnly[int]
    report_sha256: ReadOnly[str]
    features: ReadOnly[tuple[str, ...]]
    description: ReadOnly[str]


class ForwardPlan(TypedDict):
    """Every declared choice of one forward test, fixed before its outcomes exist."""

    name: ReadOnly[str]
    frozen_at: ReadOnly[str]
    board_task: ReadOnly[str]
    purpose: ReadOnly[str]
    campaign: ReadOnly[ForwardCampaign]
    model: ReadOnly[ForwardModel]
    reference_arm: ReadOnly[str]
    inputs: ReadOnly[str]
    series: ReadOnly[tuple[str, ...]]
    contract_set: ReadOnly[str]
    event_dates: ReadOnly[tuple[str, ...]]
    fee_coefficient_bps: ReadOnly[int]
    edge_hurdle: ReadOnly[int]
    stress_per_trade: ReadOnly[int]
    confidence: ReadOnly[float]
    alpha: ReadOnly[float]
    smallest_effect_of_interest: ReadOnly[float]
    scoring_rule: ReadOnly[str]
    decision_rule: ReadOnly[str]
    missing_policy: ReadOnly[str]
    scope: ReadOnly[str]


def _require_digest(document: JSONObject, key: str) -> str:
    """Read a lowercase hexadecimal SHA-256 digest.

    Args:
        document: Decoded record.
        key: Required field.

    Returns:
        The 64-character digest.

    Raises:
        ValueError: FORWARD-002 when the value is not a SHA-256 hex digest.
    """
    value = require_str(document, key)
    if len(value) != 64 or value.strip("0123456789abcdef"):
        raise ValueError(f"[FORWARD-002] {key} must be a lowercase SHA-256 hex digest")
    return value


def decode_forward_campaign(document: JSONObject) -> ForwardCampaign:
    """Validate the campaign identity a plan binds.

    Args:
        document: Decoded ``campaign`` object.

    Returns:
        Immutable campaign identity.

    Raises:
        ValueError: FORWARD-002 for an empty name or a malformed digest.
    """
    return ForwardCampaign(
        name=require_nonempty(document, "name"),
        manifest_sha256=_require_digest(document, "manifest_sha256"),
        collector_implementation_sha256=_require_digest(
            document, "collector_implementation_sha256"
        ),
        protocol_sha256=_require_digest(document, "protocol_sha256"),
    )


def decode_forward_model(document: JSONObject) -> ForwardModel:
    """Validate the arm under test, its forecast cycle and its fitted report.

    Args:
        document: Decoded ``model`` object.

    Returns:
        Immutable model declaration.

    Raises:
        ValueError: FORWARD-002 for an unknown arm, an unsupported cycle, no
            features or a malformed digest.
    """
    arm = require_str(document, "arm")
    hour = require_int(document, "forecast_hour")
    features = tuple(require_str_list(document, "features"))
    if arm not in ARMS or hour not in (0, 12) or not features:
        raise ValueError("[FORWARD-002] model needs a saved arm, a 0 or 12 cycle and features")
    return ForwardModel(
        arm=arm,
        forecast_hour=hour,
        report_sha256=_require_digest(document, "report_sha256"),
        features=features,
        description=require_nonempty(document, "description"),
    )


def decode_forward_plan(document: JSONObject) -> ForwardPlan:
    """Validate a complete forward plan against what the scorer implements.

    Args:
        document: Decoded plan file.

    Returns:
        Immutable plan.

    Raises:
        ValueError: FORWARD-002 for an unknown schema, a reference that is not a
            different saved arm, unknown or repeated series, dates that are not
            strictly increasing or disagree with their declared count, or invalid
            statistical settings; FORWARD-003 for costs the quoted taker policy
            does not implement.
    """
    if require_int(document, "schema_version") != 1:
        raise ValueError("[FORWARD-002] unsupported forward plan schema")
    model = decode_forward_model(require_dict(document, "model"))
    reference = require_str(document, "reference_arm")
    series = tuple(require_str_list(document, "series"))
    dates = tuple(
        require_day({"day": day}, "day") for day in require_str_list(document, "event_dates")
    )
    confidence = require_float(document, "confidence")
    alpha = require_float(document, "alpha")
    effect = require_float(document, "smallest_effect_of_interest")
    if (
        reference not in ARMS
        or reference == model["arm"]
        or not series
        or len(set(series)) != len(series)
        or any(name not in CITY_COORDINATES for name in series)
        or not dates
        or any(later <= earlier for earlier, later in pairwise(dates))
        or require_int(document, "event_date_count") != len(dates)
        or not 0 < confidence < 1
        or not 0 < alpha < 1
        or not effect > 0
    ):
        raise ValueError(
            "[FORWARD-002] plan needs a distinct saved reference, known unique series, "
            "strictly increasing dates matching their count and valid statistics"
        )
    fee = require_int(document, "fee_coefficient_bps")
    hurdle = require_int(document, "edge_hurdle")
    stress = require_int(document, "stress_per_trade")
    if (fee, hurdle, stress) != (QUOTED_FEE_BPS, QUOTED_HURDLE, QUOTED_STRESS):
        raise ValueError("[FORWARD-003] plan costs differ from the quoted taker policy's")
    return ForwardPlan(
        name=require_nonempty(document, "name"),
        frozen_at=require_day(document, "frozen_at"),
        board_task=require_nonempty(document, "board_task"),
        purpose=require_nonempty(document, "purpose"),
        campaign=decode_forward_campaign(require_dict(document, "campaign")),
        model=model,
        reference_arm=reference,
        inputs=require_nonempty(document, "inputs"),
        series=series,
        contract_set=require_nonempty(document, "contract_set"),
        event_dates=dates,
        fee_coefficient_bps=fee,
        edge_hurdle=hurdle,
        stress_per_trade=stress,
        confidence=confidence,
        alpha=alpha,
        smallest_effect_of_interest=effect,
        scoring_rule=require_nonempty(document, "scoring_rule"),
        decision_rule=require_nonempty(document, "decision_rule"),
        missing_policy=require_nonempty(document, "missing_policy"),
        scope=require_nonempty(document, "scope"),
    )


def encode_forward_plan(plan: ForwardPlan) -> JSONObject:
    """Encode every declared field in the plan file's own layout.

    Args:
        plan: Validated plan.

    Returns:
        JSON object that :func:`decode_forward_plan` reads back unchanged.
    """
    campaign = plan["campaign"]
    model = plan["model"]
    features: list[JSONValue] = list(model["features"])
    series: list[JSONValue] = list(plan["series"])
    dates: list[JSONValue] = list(plan["event_dates"])
    return {
        "schema_version": 1,
        "name": plan["name"],
        "frozen_at": plan["frozen_at"],
        "board_task": plan["board_task"],
        "purpose": plan["purpose"],
        "campaign": {
            "name": campaign["name"],
            "manifest_sha256": campaign["manifest_sha256"],
            "collector_implementation_sha256": campaign["collector_implementation_sha256"],
            "protocol_sha256": campaign["protocol_sha256"],
        },
        "model": {
            "arm": model["arm"],
            "forecast_hour": model["forecast_hour"],
            "report_sha256": model["report_sha256"],
            "features": features,
            "description": model["description"],
        },
        "reference_arm": plan["reference_arm"],
        "inputs": plan["inputs"],
        "series": series,
        "contract_set": plan["contract_set"],
        "event_dates": dates,
        "event_date_count": len(plan["event_dates"]),
        "fee_coefficient_bps": plan["fee_coefficient_bps"],
        "edge_hurdle": plan["edge_hurdle"],
        "stress_per_trade": plan["stress_per_trade"],
        "confidence": plan["confidence"],
        "alpha": plan["alpha"],
        "smallest_effect_of_interest": plan["smallest_effect_of_interest"],
        "scoring_rule": plan["scoring_rule"],
        "decision_rule": plan["decision_rule"],
        "missing_policy": plan["missing_policy"],
        "scope": plan["scope"],
    }


def load_forward_plan(path: Path, digest: str) -> ForwardPlan:
    """Read a plan only when its exact bytes carry the externally pinned digest.

    Args:
        path: Plan file.
        digest: SHA-256 recorded on the board before the first scored settlement.

    Returns:
        Validated plan.

    Raises:
        ValueError: FORWARD-001 when the bytes differ from the pinned digest, or a
            decoding error from :func:`decode_forward_plan`.
        OSError: The plan file cannot be read.
    """
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != digest:
        raise ValueError("[FORWARD-001] plan bytes differ from the pinned SHA-256")
    return decode_forward_plan(narrow_json_to_dict(load_json_str(payload.decode("utf-8"))))


__all__ = [
    "ForwardCampaign",
    "ForwardModel",
    "ForwardPlan",
    "decode_forward_campaign",
    "decode_forward_model",
    "decode_forward_plan",
    "encode_forward_plan",
    "load_forward_plan",
]
