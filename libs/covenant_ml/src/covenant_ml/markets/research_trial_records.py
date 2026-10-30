"""Frozen monthly validation settings and compute request codecs."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_float
from typing_extensions import ReadOnly, TypedDict

from ..evaluation.intervals import require_confidence
from .crsp_protocol import CrspProtocol, decode_crsp_protocol, encode_crsp_protocol
from .research_records import (
    ResearchPolicy,
    ResearchProposal,
    decode_research_policy,
    decode_research_proposal,
    encode_research_policy,
    encode_research_proposal,
)
from .validation import require_bounded_int, require_nonempty


class ResearchEvaluation(TypedDict):
    """The frozen monthly CRSP design and its validation fold schedule."""

    design: ReadOnly[CrspProtocol]
    stopping_months: ReadOnly[int]
    fold_months: ReadOnly[int]
    confidence: ReadOnly[float]


class ResearchTrial(TypedDict):
    """One immutable admitted request staged for a compute node."""

    policy: ReadOnly[ResearchPolicy]
    proposal: ReadOnly[ResearchProposal]
    evaluation: ReadOnly[ResearchEvaluation]
    number: ReadOnly[int]
    manifest: ReadOnly[str]


def encode_research_evaluation(value: ResearchEvaluation) -> JSONObject:
    """Encode every frozen evaluation choice.

    Args:
        value: Validated monthly validation protocol.

    Returns:
        Canonical protocol document.
    """
    return {
        "design": encode_crsp_protocol(value["design"]),
        "stopping_months": value["stopping_months"],
        "fold_months": value["fold_months"],
        "confidence": value["confidence"],
    }


def decode_research_evaluation(value: JSONObject) -> ResearchEvaluation:
    """Require the frozen monthly design and a bounded expanding fold schedule.

    Args:
        value: Untrusted protocol.

    Returns:
        Validated fixed evaluation settings.

    Raises:
        ValueError: RESEARCH_TRIAL_PROTOCOL for unknown fields or settings;
            CRSP-001 and interval validation propagate from the shared decoders.
    """
    if set(value) != set(ResearchEvaluation.__annotations__):
        raise ValueError("RESEARCH_TRIAL_PROTOCOL: exact protocol fields required")
    confidence = require_float(value, "confidence")
    require_confidence(confidence)
    return ResearchEvaluation(
        design=decode_crsp_protocol(require_dict(value, "design")),
        stopping_months=require_bounded_int(value, "stopping_months", 1, 60),
        fold_months=require_bounded_int(value, "fold_months", 1, 60),
        confidence=confidence,
    )


def evaluation_digest(value: ResearchEvaluation) -> str:
    """Bind the study semantics as well as its frozen settings.

    Args:
        value: Validated fixed evaluation choices.

    Returns:
        Digest naming the validation-only monthly holdings study.
    """
    document: JSONObject = {
        "kind": "crsp-monthly-holdings-validation-v1",
        "settings": encode_research_evaluation(value),
    }
    return sha256(dump_json_str(document).encode()).hexdigest()


def encode_research_trial(value: ResearchTrial) -> JSONObject:
    """Encode the complete compute request.

    Args:
        value: Validated request.

    Returns:
        Portable request document, with an explicit compute-node manifest path.
    """
    return {
        "policy": encode_research_policy(value["policy"]),
        "proposal": encode_research_proposal(value["proposal"]),
        "evaluation": encode_research_evaluation(value["evaluation"]),
        "number": value["number"],
        "manifest": value["manifest"],
    }


def decode_research_trial(value: JSONObject) -> ResearchTrial:
    """Validate all nested inputs before a compute operation.

    Args:
        value: Untrusted staged request.

    Returns:
        Strict request; admission history remains the controller's responsibility.

    Raises:
        ValueError: RESEARCH_TRIAL_REQUEST for unknown fields or relative paths.
    """
    if set(value) != set(ResearchTrial.__annotations__):
        raise ValueError("RESEARCH_TRIAL_REQUEST: exact request fields required")
    manifest = require_nonempty(value, "manifest")
    if not Path(manifest).is_absolute():
        raise ValueError("RESEARCH_TRIAL_REQUEST: absolute manifest path required")
    return ResearchTrial(
        policy=decode_research_policy(require_dict(value, "policy")),
        proposal=decode_research_proposal(require_dict(value, "proposal")),
        evaluation=decode_research_evaluation(require_dict(value, "evaluation")),
        number=require_bounded_int(value, "number", 1, 2**31 - 1),
        manifest=manifest,
    )


__all__ = [
    "ResearchEvaluation",
    "ResearchTrial",
    "decode_research_evaluation",
    "decode_research_trial",
    "encode_research_evaluation",
    "encode_research_trial",
    "evaluation_digest",
]
