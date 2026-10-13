"""Immutable multi-date acquisition manifests and explicit attempt dispositions."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from platform_core.json_utils import JSONObject, narrow_json_to_dict, require_list
from typing_extensions import ReadOnly, TypedDict

from .paper_records import PaperPlan, decode_paper_plan, encode_paper_plan
from .validation import parse_timestamp, require_bounded_int, require_nonempty


class FrozenModel(TypedDict):
    """One fitted model report bound to its declared forecast cycle."""

    hour: ReadOnly[int]
    path: ReadOnly[str]
    digest: ReadOnly[str]


class Campaign(TypedDict):
    """All acquisition windows and immutable input identities fixed before collection."""

    name: ReadOnly[str]
    created_at: ReadOnly[str]
    implementation_digest: ReadOnly[str]
    protocol_path: ReadOnly[str]
    protocol_digest: ReadOnly[str]
    models: ReadOnly[tuple[FrozenModel, ...]]
    plans: ReadOnly[tuple[PaperPlan, ...]]


class AttemptState(StrEnum):
    """Terminal dispositions; a failed attempt is never automatically repeated."""

    COMPLETE = "complete"
    MISSED = "missed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    DEPENDENCY_MISSING = "dependency_missing"


class Attempt(TypedDict):
    """Permanent result for one capture or settlement acquisition window."""

    name: ReadOnly[str]
    operation: ReadOnly[str]
    state: ReadOnly[AttemptState]
    observed_at: ReadOnly[str]
    returncode: ReadOnly[int | None]
    report_digest: ReadOnly[str | None]


def require_digest(document: JSONObject, key: str) -> str:
    """Require a lowercase SHA-256 content identity.

    Args:
        document: JSON object containing the field.
        key: Required field name.

    Returns:
        Validated digest string.

    Raises:
        ValueError: CAMPAIGN-001 for malformed content identity.
    """
    value = require_nonempty(document, key)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("[CAMPAIGN-001] require lowercase SHA-256 identity")
    return value


def require_absolute_path(document: JSONObject, key: str) -> str:
    """Require an absolute input path independent of the scheduler's directory.

    Args:
        document: Input object.
        key: Path field.

    Returns:
        Absolute path string.

    Raises:
        ValueError: CAMPAIGN-002 for a relative path.
    """
    value = require_nonempty(document, key)
    if not Path(value).is_absolute():
        raise ValueError("[CAMPAIGN-002] scheduled inputs require absolute paths")
    return value


def encode_frozen_model(model: FrozenModel) -> JSONObject:
    """Encode a model artifact.

    Args:
        model: Validated artifact.

    Returns:
        JSON object.
    """
    return {"hour": model["hour"], "path": model["path"], "digest": model["digest"]}


def decode_frozen_model(document: JSONObject) -> FrozenModel:
    """Validate an explicit cycle and model artifact identity.

    Args:
        document: Decoded artifact.

    Returns:
        Immutable artifact.

    Raises:
        ValueError: Invalid cycle, path or digest.
    """
    hour = require_bounded_int(document, "hour", 0, 12)
    if hour not in (0, 12):
        raise ValueError("[CAMPAIGN-003] campaign retains exactly the 00 and 12 UTC cycles")
    return FrozenModel(
        hour=hour,
        path=require_absolute_path(document, "path"),
        digest=require_digest(document, "digest"),
    )


def encode_campaign(campaign: Campaign) -> JSONObject:
    """Encode the complete frozen acquisition manifest.

    Args:
        campaign: Validated campaign.

    Returns:
        JSON manifest.
    """
    return {
        "name": campaign["name"],
        "created_at": campaign["created_at"],
        "implementation_digest": campaign["implementation_digest"],
        "protocol_path": campaign["protocol_path"],
        "protocol_digest": campaign["protocol_digest"],
        "models": [encode_frozen_model(model) for model in campaign["models"]],
        "plans": [encode_paper_plan(plan) for plan in campaign["plans"]],
    }


def decode_campaign(document: JSONObject) -> Campaign:
    """Require unique safe run names and all windows after the manifest was frozen.

    Args:
        document: Decoded manifest.

    Returns:
        Immutable campaign.

    Raises:
        ValueError: CAMPAIGN-004 for identity, timing or input-binding inconsistency.
    """
    models = tuple(
        decode_frozen_model(narrow_json_to_dict(v)) for v in require_list(document, "models")
    )
    plans = tuple(
        decode_paper_plan(narrow_json_to_dict(v)) for v in require_list(document, "plans")
    )
    created = require_nonempty(document, "created_at")
    created_ms = parse_timestamp(created)
    protocol = require_digest(document, "protocol_digest")
    if tuple(model["hour"] for model in models) != (0, 12) or not plans:
        raise ValueError(
            "[CAMPAIGN-004] require both ordered cycles and nonempty acquisition plans"
        )
    model_digests = {model["hour"]: model["digest"] for model in models}
    names = {plan["name"] for plan in plans}
    if len(names) != len(plans) or any(
        not all(c.isascii() and (c.isalnum() or c in "-_") for c in plan["name"])
        or plan["forecast_hour"] not in model_digests
        or plan["model_digest"] != model_digests[plan["forecast_hour"]]
        or plan["protocol_digest"] != protocol
        or parse_timestamp(plan["window_start"]) <= created_ms
        for plan in plans
    ):
        raise ValueError(
            "[CAMPAIGN-004] unsafe duplicate names, changed inputs or nonfuture windows"
        )
    return Campaign(
        name=require_nonempty(document, "name"),
        created_at=created,
        implementation_digest=require_digest(document, "implementation_digest"),
        protocol_path=require_absolute_path(document, "protocol_path"),
        protocol_digest=protocol,
        models=models,
        plans=plans,
    )


def encode_attempt(attempt: Attempt) -> JSONObject:
    """Encode a permanent attempt result.

    Args:
        attempt: Validated result.

    Returns:
        JSON record.
    """
    return {
        "name": attempt["name"],
        "operation": attempt["operation"],
        "state": attempt["state"].value,
        "observed_at": attempt["observed_at"],
        "returncode": attempt["returncode"],
        "report_digest": attempt["report_digest"],
    }


def decode_attempt(document: JSONObject) -> Attempt:
    """Require consistent explicit completion, failure or omission evidence.

    Args:
        document: Decoded attempt record.

    Returns:
        Immutable terminal result.

    Raises:
        ValueError: CAMPAIGN-005 for incompatible outcome fields.
    """
    state = AttemptState(require_nonempty(document, "state"))
    operation = require_nonempty(document, "operation")
    observed = require_nonempty(document, "observed_at")
    parse_timestamp(observed)
    code = (
        None
        if document["returncode"] is None
        else require_bounded_int(document, "returncode", -(2**31), 2**32 - 1)
    )
    digest = (
        None if document["report_digest"] is None else require_digest(document, "report_digest")
    )
    valid = (
        code == 0 and digest is not None
        if state is AttemptState.COMPLETE
        else code is not None and code != 0 and digest is None
        if state is AttemptState.FAILED
        else code is None and digest is None
    )
    if operation not in ("capture", "settlement") or not valid:
        raise ValueError("[CAMPAIGN-005] inconsistent operation or attempt result")
    return Attempt(
        name=require_nonempty(document, "name"),
        operation=operation,
        state=state,
        observed_at=observed,
        returncode=code,
        report_digest=digest,
    )


__all__ = [
    "Attempt",
    "AttemptState",
    "Campaign",
    "FrozenModel",
    "decode_attempt",
    "decode_campaign",
    "decode_frozen_model",
    "encode_attempt",
    "encode_campaign",
    "encode_frozen_model",
]
