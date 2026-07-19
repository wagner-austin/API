"""Tests for Fact[T] construction, contracts, and encode/decode."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_int,
)

from tankpit_bot.contracts.base import (
    ConfidenceOutOfBoundsError,
    NoUnsourcedFactError,
    ProvenanceRootednessError,
)
from tankpit_bot.facts._contracts import FactConstructionContract
from tankpit_bot.facts.fact import Fact, decode_fact, encode_fact, make_fact
from tankpit_bot.facts.provenance import make_provenance, make_source_ref


def _encode_int(value: int) -> JSONValue:
    """Encode an int fact value."""
    return value


def _decode_int(value: JSONValue) -> int:
    """Decode an int fact value."""
    return narrow_json_to_int(value)


def test_make_fact_observation_happy_path() -> None:
    """An observation fact carries its metadata unchanged."""
    fact: Fact[int] = make_fact(
        value=420,
        source="wire_0x4F_radar_response",
        observed_ms=1000,
        confidence=0.9,
        provenance=make_provenance("wire_0x4F_radar_response", []),
    )
    assert fact["value"] == 420
    assert fact["source"] == "wire_0x4F_radar_response"
    assert fact["observed_ms"] == 1000
    assert fact["confidence"] == 0.9
    assert fact["provenance"]["derived_from"] == []


def test_make_fact_inference_requires_citations() -> None:
    """An inference with no derivations violates rootedness."""
    with pytest.raises(ProvenanceRootednessError) as exc:
        make_fact(
            value=1,
            source="client_side_inference",
            observed_ms=0,
            confidence=0.5,
            provenance=make_provenance("client_side_inference", []),
        )
    assert exc.value.details == {"source": "client_side_inference", "derived_from": "empty"}


def test_make_fact_inference_with_citation_is_rooted() -> None:
    """An inference citing a prior source constructs fine."""
    fact: Fact[int] = make_fact(
        value=1,
        source="client_side_inference",
        observed_ms=50,
        confidence=0.4,
        provenance=make_provenance(
            "client_side_inference",
            [make_source_ref("wire_0x3D_movement", 40)],
        ),
    )
    assert fact["provenance"]["derived_from"][0]["source"] == "wire_0x3D_movement"


def test_make_fact_rejects_origin_source_mismatch() -> None:
    """The provenance origin must equal the fact source."""
    with pytest.raises(ProvenanceRootednessError) as exc:
        make_fact(
            value=1,
            source="wire_0x3D_movement",
            observed_ms=0,
            confidence=0.5,
            provenance=make_provenance("wire_0x4F_radar_response", []),
        )
    assert exc.value.details == {
        "source": "wire_0x3D_movement",
        "origin": "wire_0x4F_radar_response",
    }


def test_make_fact_rejects_out_of_bounds_confidence() -> None:
    """Confidence outside [0, 1] violates the bounds contract."""
    with pytest.raises(ConfidenceOutOfBoundsError):
        make_fact(
            value=1,
            source="wire_0x3D_movement",
            observed_ms=0,
            confidence=1.5,
            provenance=make_provenance("wire_0x3D_movement", []),
        )


def test_make_fact_rejects_negative_observed_ms() -> None:
    """A negative observation time violates no-unsourced-fact."""
    with pytest.raises(NoUnsourcedFactError) as exc:
        make_fact(
            value=1,
            source="wire_0x3D_movement",
            observed_ms=-5,
            confidence=0.5,
            provenance=make_provenance("wire_0x3D_movement", []),
        )
    assert exc.value.details == {"observed_ms": "-5", "source": "wire_0x3D_movement"}


def test_construction_contract_rejects_negative_observed_ms() -> None:
    """The contract names the bad observed_ms and its source."""
    contract = FactConstructionContract()
    assert contract.name == "fact_construction"
    with pytest.raises(NoUnsourcedFactError) as exc:
        contract.check(source="wire_0x3D_movement", observed_ms=-1, confidence=0.5)
    assert exc.value.details == {"observed_ms": "-1", "source": "wire_0x3D_movement"}


def test_construction_contract_rejects_out_of_bounds_confidence() -> None:
    """The contract enforces confidence bounds."""
    contract = FactConstructionContract()
    with pytest.raises(ConfidenceOutOfBoundsError) as exc:
        contract.check(source="wire_0x3D_movement", observed_ms=0, confidence=1.5)
    assert exc.value.details == {"value": "1.5"}


def test_fact_round_trip() -> None:
    """A fact survives encode/decode unchanged."""
    fact: Fact[int] = make_fact(
        value=7,
        source="dom_registry_scrape",
        observed_ms=123,
        confidence=0.75,
        provenance=make_provenance("dom_registry_scrape", []),
    )
    encoded = encode_fact(fact, _encode_int)
    assert decode_fact(encoded, _decode_int) == fact


def test_decode_fact_requires_value_field() -> None:
    """Decoding without a value field raises JSONTypeError."""
    data: JSONObject = {
        "source": "wire_0x3D_movement",
        "observed_ms": 0,
        "confidence": 0.5,
        "provenance": {"origin": "wire_0x3D_movement", "derived_from": []},
    }
    with pytest.raises(JSONTypeError, match="value is required"):
        decode_fact(data, _decode_int)


def test_decode_fact_enforces_contracts_at_load() -> None:
    """A stored fact violating a contract fails at decode."""
    data: JSONObject = {
        "value": 7,
        "source": "client_side_inference",
        "observed_ms": 0,
        "confidence": 0.5,
        "provenance": {"origin": "client_side_inference", "derived_from": []},
    }
    with pytest.raises(ProvenanceRootednessError):
        decode_fact(data, _decode_int)
