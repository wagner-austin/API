"""Tests for the Phase 1b/1c entity fact retrofits and projections."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.facts.container_facts import container_fact
from tankpit_bot.facts.provenance import make_provenance, make_source_ref
from tankpit_bot.facts.tank_facts import tank_fact
from tankpit_bot.state.mutations import apply_tank_observation
from tankpit_bot.state.types.constants import ContainerRefreshKind, EntitySource
from tankpit_bot.state.types.container import (
    container_fact_source,
    decode_container_state,
    encode_container_state,
    make_container_state,
)
from tankpit_bot.state.types.tank import (
    decode_tank_state,
    encode_tank_state,
    make_tank_state,
    tank_default_fact_source,
)
from tankpit_bot.state.types.tank_observation import (
    decode_tank_observation,
    encode_tank_observation,
    make_tank_observation,
)
from tankpit_bot.state.types.world import make_empty_world_state


@pytest.mark.parametrize(
    ("refresh_kind", "expected"),
    [
        ("radar_response", "wire_0x4F_radar_response"),
        ("radar_cache_refresh", "wire_0x43_cache_update"),
        ("radar_known_resources", "wire_0x4F_radar_response"),
        ("viewport_patch", "wire_0x5A_viewport_patch"),
        ("world_state", "wire_0x4C_map_data"),
    ],
)
def test_container_fact_source_maps_every_refresh_kind(
    refresh_kind: ContainerRefreshKind, expected: str
) -> None:
    """Each refresh kind resolves to its wire channel."""
    assert container_fact_source(refresh_kind) == expected


def test_make_container_state_derives_provenance_from_refresh_kind() -> None:
    """The default provenance origin tracks the refresh kind."""
    state = make_container_state(10, 20, True, 400, source="radar", timestamp_ms=500)
    assert state["confidence"] == 1.0
    assert state["provenance"] == make_provenance("wire_0x4F_radar_response", [])


def test_make_container_state_accepts_explicit_provenance() -> None:
    """An explicit provenance chain is stored unchanged."""
    chain = make_provenance(
        "client_side_inference", [make_source_ref("wire_0x4F_radar_response", 100)]
    )
    state = make_container_state(10, 20, True, 400, confidence=0.6, provenance=chain)
    assert state["confidence"] == 0.6
    assert state["provenance"] == chain


def test_container_round_trip_preserves_fact_metadata() -> None:
    """Encode/decode keeps confidence and provenance intact."""
    state = make_container_state(
        10, 20, False, 0, source="viewport", timestamp_ms=900, confidence=0.8
    )
    assert decode_container_state(encode_container_state(state)) == state


def test_container_decode_without_new_keys_matches_derived_defaults() -> None:
    """A pre-Phase-1b snapshot decodes to what a new encoder writes."""
    legacy: JSONObject = {
        "x": 10,
        "y": 20,
        "is_fuel": True,
        "volume": 400,
        "source": "radar",
        "refresh_kind": "radar_cache_refresh",
        "timestamp_ms": 500,
        "failed_pickups": 1,
    }
    decoded = decode_container_state(legacy)
    assert decoded == make_container_state(
        10,
        20,
        True,
        400,
        source="radar",
        refresh_kind="radar_cache_refresh",
        timestamp_ms=500,
        failed_pickups=1,
    )
    assert decoded["provenance"]["origin"] == "wire_0x43_cache_update"


def test_container_fact_projection() -> None:
    """The Fact view exposes the container value and flat metadata."""
    state = make_container_state(
        10, 20, True, 400, source="viewport", timestamp_ms=900, confidence=0.9
    )
    fact = container_fact(state)
    assert fact["value"]["x"] == 10
    assert fact["value"]["volume"] == 400
    assert fact["value"]["refresh_kind"] == "viewport_patch"
    assert fact["source"] == "wire_0x5A_viewport_patch"
    assert fact["observed_ms"] == 900
    assert fact["confidence"] == 0.9
    assert fact["provenance"] == state["provenance"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("viewport", "wire_0x28_tank_entry"),
        ("radar", "wire_0x48_enemy_detect"),
        ("world_state", "wire_0x4C_map_data"),
    ],
)
def test_tank_default_fact_source_maps_every_entity_source(
    source: EntitySource, expected: str
) -> None:
    """Each coarse entity source resolves to its canonical channel."""
    assert tank_default_fact_source(source) == expected


def test_make_tank_state_derives_provenance_and_round_trips() -> None:
    """Default provenance tracks the coarse source; encode/decode holds."""
    tank = make_tank_state(7, 10, 20, 1, 3, 0, "enemy", False, False, timestamp_ms=100)
    assert tank["confidence"] == 1.0
    assert tank["provenance"] == make_provenance("wire_0x28_tank_entry", [])
    assert decode_tank_state(encode_tank_state(tank)) == tank


def test_tank_decode_without_new_keys_matches_derived_defaults() -> None:
    """A pre-Phase-1c snapshot decodes to what a new encoder writes."""
    tank = make_tank_state(7, 10, 20, 1, 3, 0, "enemy", False, False, source="radar")
    legacy = encode_tank_state(tank)
    del legacy["confidence"]
    del legacy["provenance"]
    assert decode_tank_state(legacy) == tank


def test_tank_fact_projection() -> None:
    """The Fact view exposes the tank value and flat metadata."""
    tank = make_tank_state(
        7,
        10,
        20,
        1,
        3,
        2,
        "enemy",
        False,
        False,
        source="world_state",
        timestamp_ms=800,
        confidence=0.7,
    )
    fact = tank_fact(tank)
    assert fact["value"]["tank_id"] == 7
    assert fact["value"]["damage_state"] == 2
    assert fact["value"]["liveness"] == "alive"
    assert fact["source"] == "wire_0x4C_map_data"
    assert fact["observed_ms"] == 800
    assert fact["confidence"] == 0.7


def test_tank_observation_carries_explicit_fact_source() -> None:
    """An explicit fact source survives construction and round trip."""
    obs = make_tank_observation(
        7,
        100,
        True,
        "viewport",
        fact_source="wire_0x3D_movement",
        position=(10, 20),
    )
    assert obs["fact_source"] == "wire_0x3D_movement"
    assert decode_tank_observation(encode_tank_observation(obs)) == obs


def test_tank_observation_derives_default_fact_source() -> None:
    """Without an explicit channel, the coarse default applies."""
    obs = make_tank_observation(7, 100, False, "world_state")
    assert obs["fact_source"] == "wire_0x4C_map_data"


def test_tank_observation_decode_without_fact_source_derives_default() -> None:
    """A pre-Phase-1c encoded observation decodes with the coarse default."""
    obs = make_tank_observation(7, 100, True, "viewport", fact_source="wire_0x47_movement")
    legacy = encode_tank_observation(obs)
    del legacy["fact_source"]
    decoded = decode_tank_observation(legacy)
    assert decoded["fact_source"] == "wire_0x28_tank_entry"


def test_apply_tank_observation_records_provenance_origin() -> None:
    """The mutator writes the observation's channel as provenance origin."""
    world = make_empty_world_state()
    obs = make_tank_observation(
        7,
        100,
        True,
        "viewport",
        fact_source="wire_0x53_shoot_event",
        position=(10, 20),
    )
    updated = apply_tank_observation(world, obs)
    tank = updated["tanks"]["7"]
    assert tank["provenance"] == make_provenance("wire_0x53_shoot_event", [])
    assert tank["confidence"] == 1.0
