"""Frozen research limits and typed proposals using the existing benchmark config."""

from __future__ import annotations

from hashlib import sha256

from platform_core.json_utils import JSONObject, dump_json_str, require_dict, require_list
from typing_extensions import ReadOnly, TypedDict

from ..benchmarking.types import BenchmarkConfig
from ..benchmarking.types_codec import decode_benchmark_config, encode_benchmark_config
from .campaign_records import require_digest
from .validation import require_bounded_int, require_nonempty


class ResearchPolicy(TypedDict):
    """Pinned evidence and admission ceilings; never supplied by a proposer."""

    protocol_digest: ReadOnly[str]
    dataset_digest: ReadOnly[str]
    code_digest: ReadOnly[str]
    image_digest: ReadOnly[str]
    features: ReadOnly[tuple[str, ...]]
    seeds: ReadOnly[tuple[int, ...]]
    max_candidates: ReadOnly[int]
    max_seconds: ReadOnly[int]
    max_trees: ReadOnly[int]
    max_leaves: ReadOnly[int]


class ResearchProposal(TypedDict):
    """One candidate and seed on the validation partition."""

    hypothesis: ReadOnly[str]
    parent: ReadOnly[str]
    protocol_digest: ReadOnly[str]
    partition: ReadOnly[str]
    features: ReadOnly[tuple[str, ...]]
    seed: ReadOnly[int]
    seconds: ReadOnly[int]
    config: ReadOnly[BenchmarkConfig]


def _strings(document: JSONObject, key: str) -> tuple[str, ...]:
    """Require a nonempty, unique ordered list of feature names.

    Args:
        document: Untrusted object.
        key: Required list field.

    Returns:
        Immutable feature names.

    Raises:
        ValueError: RESEARCH-001 for empty or repeated names.
    """
    values = tuple(require_nonempty({"item": item}, "item") for item in require_list(document, key))
    if not values or len(set(values)) != len(values):
        raise ValueError("[RESEARCH-001] features must be nonempty and unique")
    return values


def encode_research_policy(policy: ResearchPolicy) -> JSONObject:
    """Encode all frozen policy fields.

    Args:
        policy: Validated policy.

    Returns:
        JSON object.
    """
    return {
        "protocol_digest": policy["protocol_digest"],
        "dataset_digest": policy["dataset_digest"],
        "code_digest": policy["code_digest"],
        "image_digest": policy["image_digest"],
        "features": list(policy["features"]),
        "seeds": list(policy["seeds"]),
        "max_candidates": policy["max_candidates"],
        "max_seconds": policy["max_seconds"],
        "max_trees": policy["max_trees"],
        "max_leaves": policy["max_leaves"],
    }


def decode_research_policy(document: JSONObject) -> ResearchPolicy:
    """Validate exact fields, content identities and finite resource ceilings.

    Args:
        document: Frozen policy JSON.

    Returns:
        Immutable policy.

    Raises:
        ValueError: RESEARCH-002 for unknown fields or duplicate/empty seeds.
        TypeError: Incorrect field types.
    """
    if set(document) != set(ResearchPolicy.__annotations__):
        raise ValueError("[RESEARCH-002] require exact policy fields")
    seeds = tuple(
        require_bounded_int({"seed": item}, "seed", 0, 2**31 - 1)
        for item in require_list(document, "seeds")
    )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("[RESEARCH-002] seeds must be nonempty and unique")
    return ResearchPolicy(
        protocol_digest=require_digest(document, "protocol_digest"),
        dataset_digest=require_digest(document, "dataset_digest"),
        code_digest=require_digest(document, "code_digest"),
        image_digest=require_digest(document, "image_digest"),
        features=_strings(document, "features"),
        seeds=seeds,
        max_candidates=require_bounded_int(document, "max_candidates", 1, 20),
        max_seconds=require_bounded_int(document, "max_seconds", 1, 259200),
        max_trees=require_bounded_int(document, "max_trees", 1, 10000),
        max_leaves=require_bounded_int(document, "max_leaves", 2, 1024),
    )


def encode_research_proposal(proposal: ResearchProposal) -> JSONObject:
    """Encode a proposal without adding data paths or executable commands.

    Args:
        proposal: Validated proposal.

    Returns:
        JSON object.
    """
    return {
        "hypothesis": proposal["hypothesis"],
        "parent": proposal["parent"],
        "protocol_digest": proposal["protocol_digest"],
        "partition": proposal["partition"],
        "features": list(proposal["features"]),
        "seed": proposal["seed"],
        "seconds": proposal["seconds"],
        "config": encode_benchmark_config(proposal["config"]),
    }


def decode_research_proposal(document: JSONObject) -> ResearchProposal:
    """Decode a bounded proposal; policy admission is a separate locked decision.

    Args:
        document: Proposer input.

    Returns:
        Typed proposal.

    Raises:
        ValueError: RESEARCH-003 for unknown fields.
        TypeError: Missing or wrongly typed fields.
    """
    if set(document) != set(ResearchProposal.__annotations__):
        raise ValueError("[RESEARCH-003] require exact proposal fields")
    config = require_dict(document, "config")
    if set(config) != set(BenchmarkConfig.__annotations__):
        raise ValueError("[RESEARCH-003] require exact benchmark config fields")
    return ResearchProposal(
        hypothesis=require_nonempty(document, "hypothesis"),
        parent=require_nonempty(document, "parent"),
        protocol_digest=require_digest(document, "protocol_digest"),
        partition=require_nonempty(document, "partition"),
        features=_strings(document, "features"),
        seed=require_bounded_int(document, "seed", 0, 2**31 - 1),
        seconds=require_bounded_int(document, "seconds", 1, 259200),
        config=decode_benchmark_config(config, "config"),
    )


def candidate_digest(proposal: ResearchProposal) -> str:
    """Identify a model independently of renamed hypotheses, seeds and feature order.

    Args:
        proposal: Typed proposal.

    Returns:
        Canonical candidate SHA-256.
    """
    return sha256(
        dump_json_str(
            {
                "config": encode_benchmark_config(proposal["config"]),
                "features": sorted(proposal["features"]),
            }
        ).encode()
    ).hexdigest()


__all__ = [
    "ResearchPolicy",
    "ResearchProposal",
    "candidate_digest",
    "decode_research_policy",
    "decode_research_proposal",
    "encode_research_policy",
    "encode_research_proposal",
]
