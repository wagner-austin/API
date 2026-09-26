"""Tests for the fact source vocabulary and its observation split."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.facts.provenance import decode_source_ref, encode_source_ref, make_source_ref
from tankpit_bot.facts.source import FactSource, is_observation_source


def test_fact_sources_cover_the_twenty_three_channels() -> None:
    """20 wire channels + DOM scrape + inference + the fleet report.

    ``fleet_report`` joined 2026-08-14 ([[fleet-coordination]]): a
    teammate's published belief merged through the observation
    pathway — an observation source, but not a wire channel of this
    session.
    """
    assert len(FactSource) == 23
    wire = [source for source in FactSource if source.value.startswith("wire_")]
    assert len(wire) == 20
    assert {source for source in FactSource if source not in wire} == {
        FactSource.DOM_REGISTRY_SCRAPE,
        FactSource.CLIENT_SIDE_INFERENCE,
        FactSource.FLEET_REPORT,
    }


@pytest.mark.parametrize("source", list(FactSource))
def test_every_source_round_trips_through_its_word(source: FactSource) -> None:
    """Each source encodes as its own word and decodes back to the member."""
    encoded = encode_source_ref(make_source_ref(source, 10))
    assert encoded["source"] == source.value
    assert decode_source_ref(encoded)["source"] is source


def test_an_unknown_source_is_refused_by_name() -> None:
    """An unknown source word raises JSONTypeError naming the word."""
    with pytest.raises(JSONTypeError, match="Invalid source 'wire_0xFF_unknown'"):
        decode_source_ref({"source": "wire_0xFF_unknown", "observed_ms": 0})


def test_a_non_string_source_is_refused() -> None:
    """A non-string source raises JSONTypeError."""
    with pytest.raises(JSONTypeError):
        decode_source_ref({"source": 7, "observed_ms": 0})


@pytest.mark.parametrize("source", list(FactSource))
def test_only_inference_is_not_an_observation(source: FactSource) -> None:
    """Every source but CLIENT_SIDE_INFERENCE is an observation."""
    assert is_observation_source(source) is (source is not FactSource.CLIENT_SIDE_INFERENCE)
