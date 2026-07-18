"""Tests for the Phase 1d retrofits: self, mine, terrain, viewport."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject

from tankpit_bot.facts.provenance import make_provenance
from tankpit_bot.facts.world_facts import (
    mine_fact,
    self_fact,
    terrain_tile_fact,
    viewport_fact,
)
from tankpit_bot.state.container_mutations import add_mine
from tankpit_bot.state.mutations import set_self_fuel, set_self_rank, update_self_position
from tankpit_bot.state.types.constants import EntitySource
from tankpit_bot.state.types.mine import (
    decode_mine_state,
    encode_mine_state,
    make_mine_state,
    mine_default_fact_source,
)
from tankpit_bot.state.types.self_state import (
    decode_self_state,
    encode_self_state,
    make_self_state,
)
from tankpit_bot.state.types.terrain import (
    decode_terrain_tile,
    encode_terrain_tile,
    make_terrain_tile,
)
from tankpit_bot.state.types.viewport import (
    decode_viewport_state,
    encode_viewport_state,
    make_viewport_state,
)
from tankpit_bot.state.types.world import make_empty_world_state
from tests.world_state.helpers import get_self_state


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("viewport", "wire_0x5A_viewport_patch"),
        ("radar", "wire_0x4F_radar_response"),
        ("world_state", "wire_0x4C_map_data"),
    ],
)
def test_mine_default_fact_source_maps_every_entity_source(
    source: EntitySource, expected: str
) -> None:
    """Each coarse mine source resolves to its tile-observation channel."""
    assert mine_default_fact_source(source) == expected


def test_mine_round_trip_and_legacy_decode() -> None:
    """Mine metadata survives round trip; legacy decode converges."""
    mine = make_mine_state(10, 20, 2, 55, 1, source="radar", timestamp_ms=300)
    assert mine["provenance"] == make_provenance("wire_0x4F_radar_response", [])
    encoded = encode_mine_state(mine)
    assert decode_mine_state(encoded) == mine
    legacy = dict(encoded)
    del legacy["confidence"]
    del legacy["provenance"]
    assert decode_mine_state(legacy) == mine


def test_add_mine_records_placement_channel() -> None:
    """A witnessed placement carries the 0x4B provenance origin."""
    world = make_empty_world_state()
    updated = add_mine(world, 10, 20, 2, 55, 1, 500)
    mine = updated["mines"]["10,20"]
    assert mine["provenance"] == make_provenance("wire_0x4B_mine_placement", [])


def test_mine_fact_projection() -> None:
    """The mine Fact view exposes the value and flat metadata."""
    mine = make_mine_state(10, 20, 2, 55, 1, source="viewport", timestamp_ms=300)
    fact = mine_fact(mine)
    assert fact["value"]["team"] == 1
    assert fact["source"] == "wire_0x5A_viewport_patch"
    assert fact["observed_ms"] == 300


def test_self_state_round_trip_and_legacy_decode() -> None:
    """Self metadata survives round trip; legacy decode converges."""
    state = make_self_state(1, 100, 100, 0, 4, 800, 5, observed_ms=700)
    encoded = encode_self_state(state)
    assert decode_self_state(encoded) == state
    legacy = dict(encoded)
    del legacy["observed_ms"]
    del legacy["confidence"]
    del legacy["provenance"]
    assert decode_self_state(legacy) == make_self_state(1, 100, 100, 0, 4, 800, 5)


def test_update_self_position_records_channel() -> None:
    """The position mutator stamps observed_ms and the given channel."""
    world = make_empty_world_state()
    updated = update_self_position(world, 50, 60, 900, "wire_0x47_movement")
    self_state = get_self_state(updated)
    assert self_state["observed_ms"] == 900
    assert self_state["provenance"] == make_provenance("wire_0x47_movement", [])


def test_set_self_fuel_and_rank_record_channels() -> None:
    """Fuel and rank mutators stamp their message channels."""
    world = update_self_position(make_empty_world_state(), 50, 60, 100)
    fueled = set_self_fuel(world, 500, 200, "wire_0x64_fuel_total")
    fueled_self = get_self_state(fueled)
    assert fueled_self["provenance"] == make_provenance("wire_0x64_fuel_total", [])
    assert fueled_self["observed_ms"] == 200
    ranked = set_self_rank(fueled, 3, 300)
    ranked_self = get_self_state(ranked)
    assert ranked_self["provenance"] == make_provenance("wire_0x2B_promotion", [])


def test_self_fact_projection() -> None:
    """The self Fact view exposes the value and flat metadata."""
    state = make_self_state(1, 100, 100, 0, 4, 800, 5, observed_ms=700)
    fact = self_fact(state)
    assert fact["value"]["fuel"] == 800
    assert fact["source"] == "wire_0x3D_movement"
    assert fact["observed_ms"] == 700


def test_terrain_round_trip_and_legacy_decode() -> None:
    """Terrain metadata survives round trip; legacy decode converges."""
    tile = make_terrain_tile(10, 20, 1, observed_ms=400)
    encoded = encode_terrain_tile(tile)
    assert decode_terrain_tile(encoded) == tile
    legacy = dict(encoded)
    del legacy["observed_ms"]
    del legacy["confidence"]
    del legacy["provenance"]
    assert decode_terrain_tile(legacy) == make_terrain_tile(10, 20, 1)


def test_terrain_explicit_provenance_for_terrain_update() -> None:
    """The 0x4A terrain-update channel is stored when passed."""
    tile = make_terrain_tile(
        10,
        20,
        1,
        observed_ms=400,
        provenance=make_provenance("wire_0x4A_terrain_update", []),
    )
    fact = terrain_tile_fact(tile)
    assert fact["source"] == "wire_0x4A_terrain_update"
    assert fact["value"]["terrain_type"] == 1


def test_viewport_round_trip_and_legacy_decode() -> None:
    """Viewport metadata survives round trip; legacy decode converges."""
    state = make_viewport_state(100, 50, 16, 16, observed_ms=600)
    assert state["provenance"] == make_provenance("wire_0x5A_viewport_patch", [])
    encoded = encode_viewport_state(state)
    assert decode_viewport_state(encoded) == state
    legacy: JSONObject = {"left": 100, "top": 50, "width": 16, "height": 16}
    assert decode_viewport_state(legacy) == make_viewport_state(100, 50, 16, 16)


def test_viewport_fact_projection() -> None:
    """The viewport Fact view exposes the value and flat metadata."""
    state = make_viewport_state(100, 50, 16, 16, observed_ms=600)
    fact = viewport_fact(state)
    assert fact["value"]["left"] == 100
    assert fact["source"] == "wire_0x5A_viewport_patch"
    assert fact["observed_ms"] == 600
