"""The generic ``Fact[T]``: a value plus complete source metadata.

Phase 1a of the self-observing architecture: ``Fact[T]`` exists
alongside the raw world-state types; retrofits (Phases 1b-1d) migrate
``ContainerStateDict``, ``TankStateDict``, and the rest onto it.

Construction goes through :func:`make_fact`, which enforces the three
fact contracts: no unsourced facts, confidence in bounds, and
provenance rootedness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_dict,
    require_float,
    require_int,
)
from typing_extensions import TypedDict

from tankpit_bot.contracts.base import ProvenanceRootednessError
from tankpit_bot.contracts.enforcement import require
from tankpit_bot.facts._contracts import FactConstructionContract
from tankpit_bot.facts.provenance import (
    ProvenanceChainDict,
    decode_provenance,
    encode_provenance,
)
from tankpit_bot.facts.source import FactSource, is_observation_source, require_fact_source

T = TypeVar("T")


class Fact(TypedDict, Generic[T]):
    """A belief with complete source metadata.

    Attributes:
        value: The believed value.
        source: Channel the belief came from.
        observed_ms: When the belief was observed (or inferred).
        confidence: Trust in the belief, [0.0, 1.0].
        provenance: Origin plus derivation references.
    """

    value: T
    source: FactSource
    observed_ms: int
    confidence: float
    provenance: ProvenanceChainDict


_CONSTRUCTION_CONTRACT = FactConstructionContract()


def make_fact(
    *,
    value: T,
    source: FactSource,
    observed_ms: int,
    confidence: float,
    provenance: ProvenanceChainDict,
) -> Fact[T]:
    """Create a fact, enforcing the fact-construction contracts.

    The contract check is invoked explicitly rather than via
    ``@enforce_contract`` because decorating a generic function erases
    its type variable under mypy; the decorator is the mechanism for
    the non-generic ``apply_*`` / ``record_*`` mutations of later
    phases (and the guard rule targets those names, not ``make_*``).

    Args:
        value: The believed value.
        source: Channel the belief came from.
        observed_ms: When the belief was observed (or inferred).
        confidence: Trust in the belief, [0.0, 1.0].
        provenance: Origin plus derivation references.

    Returns:
        Fact with the provided values.

    Raises:
        NoUnsourcedFactError: If the source metadata is incomplete.
        ConfidenceOutOfBoundsError: If confidence is out of range.
        ProvenanceRootednessError: If the provenance origin disagrees
            with the source, or an inference cites no prior sources.
    """
    _CONSTRUCTION_CONTRACT.check(
        source=source,
        observed_ms=observed_ms,
        confidence=confidence,
    )
    require(
        provenance["origin"] == source,
        ProvenanceRootednessError,
        source=source,
        origin=provenance["origin"],
    )
    require(
        is_observation_source(source) or len(provenance["derived_from"]) > 0,
        ProvenanceRootednessError,
        source=source,
        derived_from="empty",
    )
    return Fact(
        value=value,
        source=source,
        observed_ms=observed_ms,
        confidence=confidence,
        provenance=provenance,
    )


def encode_fact(fact: Fact[T], encode_value: Callable[[T], JSONValue]) -> JSONObject:
    """Encode a fact to a JSON-serializable dict.

    Args:
        fact: Fact to encode.
        encode_value: Encoder for the fact's value type.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "value": encode_value(fact["value"]),
        "source": fact["source"],
        "observed_ms": fact["observed_ms"],
        "confidence": fact["confidence"],
        "provenance": encode_provenance(fact["provenance"]),
    }


def decode_fact(data: JSONObject, decode_value: Callable[[JSONValue], T]) -> Fact[T]:
    """Decode a fact from JSON with validation.

    Runs the same construction contracts as :func:`make_fact`, so a
    stored fact that violates a contract fails at load, not at use.

    Args:
        data: JSON object to decode.
        decode_value: Decoder for the fact's value type.

    Returns:
        Validated fact.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
        NoUnsourcedFactError: If the source metadata is incomplete.
        ConfidenceOutOfBoundsError: If confidence is out of range.
        ProvenanceRootednessError: If the provenance is not rooted.
    """
    if "value" not in data:
        raise JSONTypeError("value is required")
    return make_fact(
        value=decode_value(data["value"]),
        source=require_fact_source(data, "source"),
        observed_ms=require_int(data, "observed_ms"),
        confidence=require_float(data, "confidence"),
        provenance=decode_provenance(require_dict(data, "provenance")),
    )


__all__ = [
    "Fact",
    "decode_fact",
    "encode_fact",
    "make_fact",
]
