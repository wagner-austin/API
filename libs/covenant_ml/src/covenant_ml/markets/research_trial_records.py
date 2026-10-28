"""Frozen validation diagnostic settings and compute request codecs."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_float
from typing_extensions import ReadOnly, TypedDict

from ..evaluation.arguments import resampling_settings
from ..evaluation.codec_records import (
    decode_alert_costs,
    decode_resampling,
    encode_alert_costs,
    encode_resampling,
)
from ..evaluation.types import AlertCosts, ResamplingSettings, ResamplingUnit
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
    """A diagnostic protocol, distinct from the substantive monthly experiment."""

    reference: ReadOnly[str]
    inner_train_fraction: ReadOnly[float]
    resampling: ReadOnly[ResamplingSettings]
    costs: ReadOnly[AlertCosts]


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
        value: Validated diagnostic protocol.

    Returns:
        Canonical protocol document.
    """
    return {
        "reference": value["reference"],
        "inner_train_fraction": value["inner_train_fraction"],
        "resampling": encode_resampling(value["resampling"]),
        "costs": encode_alert_costs(value["costs"]),
    }


def decode_research_evaluation(value: JSONObject) -> ResearchEvaluation:
    """Require a chronological validation protocol with finite sampling work.

    Args:
        value: Untrusted protocol.

    Returns:
        Validated fixed diagnostic settings.

    Raises:
        ValueError: RESEARCH_TRIAL_PROTOCOL for unsupported or excessive settings.
    """
    if set(value) != set(ResearchEvaluation.__annotations__):
        raise ValueError("RESEARCH_TRIAL_PROTOCOL: exact protocol fields required")
    reference = require_nonempty(value, "reference")
    inner = require_float(value, "inner_train_fraction")
    settings = decode_resampling(require_dict(value, "resampling"))
    settings = resampling_settings(
        settings["unit"],
        settings["block_length"],
        settings["n_resamples"],
        settings["resample_seed"],
        settings["confidence"],
    )
    costs = decode_alert_costs(require_dict(value, "costs"))
    if (
        reference != "lightgbm"
        or not 0 < inner < 1
        or settings["unit"] != ResamplingUnit.BLOCK
        or settings["n_resamples"] > 10000
    ):
        raise ValueError(
            "RESEARCH_TRIAL_PROTOCOL: require bounded block validation against lightgbm"
        )
    return ResearchEvaluation(
        reference=reference, inner_train_fraction=inner, resampling=settings, costs=costs
    )


def evaluation_digest(value: ResearchEvaluation) -> str:
    """Bind the diagnostic semantics as well as its numerical settings.

    Args:
        value: Validated fixed evaluation choices.

    Returns:
        Digest naming a validation-only next-return-sign diagnostic.
    """
    document: JSONObject = {
        "kind": "crsp-next-return-sign-validation-v1",
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
