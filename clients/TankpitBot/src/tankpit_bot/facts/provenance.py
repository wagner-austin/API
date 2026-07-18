"""Provenance chains: where a belief came from, structurally.

A fact's provenance is its origin channel plus the list of prior
sources it was derived from. Observations have an empty derivation
list; inferences must cite at least one prior source. The rootedness
rule is enforced at fact construction (see
:mod:`tankpit_bot.facts.fact`).
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_list,
)
from typing_extensions import TypedDict

from tankpit_bot.facts.source import FactSource, require_fact_source


class SourceRefDict(TypedDict):
    """Reference to one prior source a derived fact relied on.

    Attributes:
        source: Channel the prior observation came from.
        observed_ms: When the prior observation was made.
    """

    source: FactSource
    observed_ms: int


class ProvenanceChainDict(TypedDict):
    """Origin channel plus derivation references for one fact.

    Attributes:
        origin: Channel this fact arrived on (or was inferred by).
        derived_from: Prior sources cited by a derivation. Empty for
            direct observations.
    """

    origin: FactSource
    derived_from: list[SourceRefDict]


def make_source_ref(source: FactSource, observed_ms: int) -> SourceRefDict:
    """Create a source reference.

    Args:
        source: Channel the prior observation came from.
        observed_ms: When the prior observation was made.

    Returns:
        SourceRefDict with the provided values.
    """
    return SourceRefDict(source=source, observed_ms=observed_ms)


def make_provenance(
    origin: FactSource,
    derived_from: list[SourceRefDict],
) -> ProvenanceChainDict:
    """Create a provenance chain.

    Args:
        origin: Channel this fact arrived on (or was inferred by).
        derived_from: Prior sources cited by a derivation.

    Returns:
        ProvenanceChainDict with the provided values.
    """
    return ProvenanceChainDict(origin=origin, derived_from=derived_from)


def encode_source_ref(ref: SourceRefDict) -> JSONObject:
    """Encode SourceRefDict to a JSON-serializable dict.

    Args:
        ref: SourceRefDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "source": ref["source"],
        "observed_ms": ref["observed_ms"],
    }


def decode_source_ref(data: JSONObject) -> SourceRefDict:
    """Decode SourceRefDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated SourceRefDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return SourceRefDict(
        source=require_fact_source(data, "source"),
        observed_ms=require_int(data, "observed_ms"),
    )


def encode_provenance(chain: ProvenanceChainDict) -> JSONObject:
    """Encode ProvenanceChainDict to a JSON-serializable dict.

    Args:
        chain: ProvenanceChainDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    encoded_refs: list[JSONValue] = [encode_source_ref(ref) for ref in chain["derived_from"]]
    return {
        "origin": chain["origin"],
        "derived_from": encoded_refs,
    }


def decode_provenance(data: JSONObject) -> ProvenanceChainDict:
    """Decode ProvenanceChainDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ProvenanceChainDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    raw_refs = require_list(data, "derived_from")
    refs: list[SourceRefDict] = [decode_source_ref(narrow_json_to_dict(raw)) for raw in raw_refs]
    return ProvenanceChainDict(
        origin=require_fact_source(data, "origin"),
        derived_from=refs,
    )


__all__ = [
    "ProvenanceChainDict",
    "SourceRefDict",
    "decode_provenance",
    "decode_source_ref",
    "encode_provenance",
    "encode_source_ref",
    "make_provenance",
    "make_source_ref",
]
