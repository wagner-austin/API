"""Round-trip tests for the tile-inspector TypedDicts."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.diagnostics.tile_inspector_types import (
    NeighborTileDict,
    TileInspectionDict,
    decode_neighbor_tile,
    decode_tile_inspection,
    encode_neighbor_tile,
    encode_tile_inspection,
)


def _round_trip(encoded: JSONObject) -> JSONObject:
    """Round-trip a dict through ``dump_json_str`` / ``load_json_str``."""
    return narrow_json_to_dict(load_json_str(dump_json_str(encoded)))


def test_neighbor_tile_round_trip() -> None:
    """``NeighborTileDict`` round-trips through JSON encoding."""
    neighbor = NeighborTileDict(
        direction="NE",
        x=145,
        y=123,
        terrain=".",
        passable=True,
        in_bounds=True,
    )

    decoded = decode_neighbor_tile(_round_trip(encode_neighbor_tile(neighbor)))

    assert decoded == neighbor


def test_neighbor_tile_rejects_non_bool_passable() -> None:
    """A non-bool ``passable`` field raises ``JSONTypeError`` at decode."""
    raw: JSONObject = {
        "direction": "N",
        "x": 144,
        "y": 123,
        "terrain": ".",
        "passable": "yes",
        "in_bounds": True,
    }

    with pytest.raises(JSONTypeError, match=r"passable.*must be bool"):
        decode_neighbor_tile(raw)


def test_neighbor_tile_rejects_missing_in_bounds() -> None:
    """A missing ``in_bounds`` field raises ``JSONTypeError`` at decode."""
    raw: JSONObject = {
        "direction": "N",
        "x": 144,
        "y": 123,
        "terrain": ".",
        "passable": True,
    }

    with pytest.raises(JSONTypeError, match="in_bounds"):
        decode_neighbor_tile(raw)


def test_tile_inspection_round_trip() -> None:
    """``TileInspectionDict`` round-trips through JSON encoding."""
    inspection = TileInspectionDict(
        field_image="field01_r.gif",
        target_x=144,
        target_y=124,
        target_terrain=".",
        target_passable=True,
        target_in_bounds=True,
        neighbors=[
            NeighborTileDict(
                direction="N", x=144, y=123, terrain=".", passable=True, in_bounds=True
            ),
            NeighborTileDict(
                direction="NE", x=145, y=123, terrain=".", passable=True, in_bounds=True
            ),
        ],
        landing_tile_x=144,
        landing_tile_y=124,
        landing_resolution="target_is_passable",
        from_x=131,
        from_y=110,
        reachable=True,
    )

    decoded = decode_tile_inspection(_round_trip(encode_tile_inspection(inspection)))

    assert decoded == inspection


def test_tile_inspection_decode_rejects_non_object_neighbor() -> None:
    """A non-object element in the ``neighbors`` list raises at decode."""
    raw: JSONObject = {
        "field_image": "field01_r.gif",
        "target_x": 1,
        "target_y": 2,
        "target_terrain": ".",
        "target_passable": True,
        "target_in_bounds": True,
        "neighbors": ["bad"],
        "landing_tile_x": 1,
        "landing_tile_y": 2,
        "landing_resolution": "target_is_passable",
        "from_x": 0,
        "from_y": 0,
        "reachable": True,
    }

    with pytest.raises(JSONTypeError, match=r"neighbors\[0\] must be object"):
        decode_tile_inspection(raw)
