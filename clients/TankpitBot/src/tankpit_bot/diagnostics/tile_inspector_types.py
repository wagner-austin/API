"""Strict TypedDict payloads for the terrain tile inspector.

Each inspection report has the same shape regardless of the inspected
tile so the renderer, the JSON encoder, and any downstream consumer can
reason about a single explicit contract.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict


class NeighborTileDict(TypedDict):
    """One of the eight adjacent tiles around an inspected coordinate.

    Attributes:
        direction: Compass-relative label (``N``, ``NE``, ``E``, ``SE``,
            ``S``, ``SW``, ``W``, ``NW``).
        x: Absolute X coordinate.
        y: Absolute Y coordinate.
        terrain: Terrain character returned by
            :meth:`tankpit_bot.terrain.TerrainMap.get_terrain`
            (``"#"`` rock, ``"."`` ground, ``"W"`` water,
            ``" "`` out-of-bounds).
        passable: Whether the tile is passable (ground and on the map).
        in_bounds: Whether the tile is on the 256x256 grid.
    """

    direction: str
    x: int
    y: int
    terrain: str
    passable: bool
    in_bounds: bool


class TileInspectionDict(TypedDict):
    """Structured terrain inspection for one coordinate.

    Attributes:
        field_image: Field image file the inspection was run against.
        target_x: X coordinate the user asked about.
        target_y: Y coordinate the user asked about.
        target_terrain: Terrain character at the target tile.
        target_passable: Whether the target tile is passable.
        target_in_bounds: Whether the target is on the 256x256 grid.
        neighbors: All eight adjacent tiles in compass order.
        landing_tile_x: X of the tile
            :func:`tankpit_bot.bot.ai.equipment.find_teleport_landing_tile`
            would resolve as the landing for a teleport TO this tile,
            assuming the bot is at ``from_x, from_y``. ``-1`` when no
            landing is found.
        landing_tile_y: Y of the chosen landing tile, or ``-1``.
        landing_resolution: Plain-English summary of how the landing was
            chosen (``"target_is_passable"``, ``"adjacent:NE"``, etc.).
        from_x: Origin X used for the reachability check (``-1`` when the
            caller did not supply one).
        from_y: Origin Y used for the reachability check (``-1`` when the
            caller did not supply one).
        reachable: Whether
            :func:`tankpit_bot.bot.ai.equipment.is_reachable`
            returned True for ``(from_x, from_y)`` -> ``(target_x,
            target_y)``. ``False`` when no origin was supplied.
    """

    field_image: str
    target_x: int
    target_y: int
    target_terrain: str
    target_passable: bool
    target_in_bounds: bool
    neighbors: list[NeighborTileDict]
    landing_tile_x: int
    landing_tile_y: int
    landing_resolution: str
    from_x: int
    from_y: int
    reachable: bool


def encode_neighbor_tile(neighbor: NeighborTileDict) -> JSONObject:
    """Encode a neighbor tile to JSON.

    Args:
        neighbor: Neighbor tile record.

    Returns:
        JSON-compatible representation.
    """
    return {
        "direction": neighbor["direction"],
        "x": neighbor["x"],
        "y": neighbor["y"],
        "terrain": neighbor["terrain"],
        "passable": neighbor["passable"],
        "in_bounds": neighbor["in_bounds"],
    }


def _require_bool(data: JSONObject, key: str) -> bool:
    """Extract a required bool field with strict type checking.

    Args:
        data: JSON object to read.
        key: Field name.

    Returns:
        Validated bool value.

    Raises:
        JSONTypeError: When ``key`` is absent or not a bool.
    """
    if key not in data:
        raise JSONTypeError(f"Missing required field {key!r}")
    value = data[key]
    if not isinstance(value, bool):
        raise JSONTypeError(f"Field {key!r} must be bool, got {type(value).__name__}")
    return value


def decode_neighbor_tile(data: JSONObject) -> NeighborTileDict:
    """Decode a neighbor tile from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated neighbor tile.
    """
    return NeighborTileDict(
        direction=require_str(data, "direction"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        terrain=require_str(data, "terrain"),
        passable=_require_bool(data, "passable"),
        in_bounds=_require_bool(data, "in_bounds"),
    )


def encode_tile_inspection(report: TileInspectionDict) -> JSONObject:
    """Encode a tile inspection report to JSON.

    Args:
        report: Inspection report.

    Returns:
        JSON-compatible representation.
    """
    neighbors_encoded: list[JSONValue] = [encode_neighbor_tile(n) for n in report["neighbors"]]
    return {
        "field_image": report["field_image"],
        "target_x": report["target_x"],
        "target_y": report["target_y"],
        "target_terrain": report["target_terrain"],
        "target_passable": report["target_passable"],
        "target_in_bounds": report["target_in_bounds"],
        "neighbors": neighbors_encoded,
        "landing_tile_x": report["landing_tile_x"],
        "landing_tile_y": report["landing_tile_y"],
        "landing_resolution": report["landing_resolution"],
        "from_x": report["from_x"],
        "from_y": report["from_y"],
        "reachable": report["reachable"],
    }


def decode_tile_inspection(data: JSONObject) -> TileInspectionDict:
    """Decode a tile inspection report from JSON.

    Args:
        data: JSON object to decode.

    Returns:
        Validated inspection report.
    """
    raw_neighbors = require_list(data, "neighbors")
    neighbors: list[NeighborTileDict] = []
    for index, raw in enumerate(raw_neighbors):
        if not isinstance(raw, dict):
            raise JSONTypeError(f"neighbors[{index}] must be object, got {type(raw).__name__}")
        neighbors.append(decode_neighbor_tile(raw))
    return TileInspectionDict(
        field_image=require_str(data, "field_image"),
        target_x=require_int(data, "target_x"),
        target_y=require_int(data, "target_y"),
        target_terrain=require_str(data, "target_terrain"),
        target_passable=_require_bool(data, "target_passable"),
        target_in_bounds=_require_bool(data, "target_in_bounds"),
        neighbors=neighbors,
        landing_tile_x=require_int(data, "landing_tile_x"),
        landing_tile_y=require_int(data, "landing_tile_y"),
        landing_resolution=require_str(data, "landing_resolution"),
        from_x=require_int(data, "from_x"),
        from_y=require_int(data, "from_y"),
        reachable=_require_bool(data, "reachable"),
    )


__all__ = [
    "NeighborTileDict",
    "TileInspectionDict",
    "decode_neighbor_tile",
    "decode_tile_inspection",
    "encode_neighbor_tile",
    "encode_tile_inspection",
]
