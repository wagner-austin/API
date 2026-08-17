"""Tests for fact source literals and validation."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.facts.source import (
    FACT_SOURCES,
    INFERENCE_SOURCE,
    FactSource,
    is_observation_source,
    require_fact_source,
)


def test_fact_sources_cover_the_twenty_three_channels() -> None:
    """20 wire channels + DOM scrape + inference + the fleet report.

    ``fleet_report`` joined 2026-08-14 ([[fleet-coordination]]): a
    teammate's published belief merged through the observation
    pathway — an observation source, but not a wire channel of this
    session.
    """
    assert len(FACT_SOURCES) == 23
    wire = [name for name in FACT_SOURCES if name.startswith("wire_")]
    assert len(wire) == 20
    assert "dom_registry_scrape" in FACT_SOURCES
    assert "client_side_inference" in FACT_SOURCES
    assert "fleet_report" in FACT_SOURCES


@pytest.mark.parametrize("name", FACT_SOURCES)
def test_require_fact_source_accepts_every_valid_source(name: str) -> None:
    """Every listed source round-trips through validation."""
    assert require_fact_source({"source": name}, "source") == name


def test_require_fact_source_rejects_unknown_source() -> None:
    """An unknown source name raises JSONTypeError."""
    with pytest.raises(JSONTypeError, match="source must be one of"):
        require_fact_source({"source": "wire_0xFF_unknown"}, "source")


def test_require_fact_source_rejects_non_string() -> None:
    """A non-string source raises JSONTypeError."""
    with pytest.raises(JSONTypeError):
        require_fact_source({"source": 7}, "source")


def test_is_observation_source_splits_inference_from_observation() -> None:
    """Only client_side_inference is a non-observation source."""
    assert is_observation_source("wire_0x3D_movement") is True
    assert is_observation_source("dom_registry_scrape") is True
    inference: FactSource = INFERENCE_SOURCE
    assert is_observation_source(inference) is False
