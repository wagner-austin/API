"""Terrain tile TypedDict + factory + encode/decode.

A tile stores only its terrain type (ground/rock/water/ferry/etc).
Container and mine layers live in their own world-state registries
(``world.containers`` and ``world.mines``) populated by the per-tile
mutators in :mod:`tankpit_bot.state.container_mutations`.

Phase 1d of the self-observing architecture: the tile carries the
fact metadata flat. Terrain arrives on two channels -- 0x5A viewport
patch grids (the default) and 0x4A terrain updates. The Fact[T]
projection lives in :mod:`tankpit_bot.facts.world_facts`.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, require_dict, require_float, require_int
from typing_extensions import TypedDict

from tankpit_bot.facts.provenance import (
    ProvenanceChainDict,
    decode_provenance,
    encode_provenance,
    make_provenance,
)
from tankpit_bot.facts.source import FactSource

DEFAULT_TERRAIN_FACT_SOURCE: FactSource = "wire_0x5A_viewport_patch"
"""Default terrain channel: the 0x5A patch grid carries per-tile terrain."""


class TerrainTileDict(TypedDict):
    """State of a terrain tile.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        terrain_type: Terrain/structure type (0=ground, 1-3=rock variants, 5=ferry, 7=ferry+rock).
        observed_ms: When the tile was last confirmed. Zero for
            fixtures constructed without a clock.
        confidence: Trust in this belief, [0.0, 1.0].
        provenance: Origin wire channel plus derivation references.
    """

    x: int
    y: int
    terrain_type: int
    observed_ms: int
    confidence: float
    provenance: ProvenanceChainDict


def make_terrain_tile(
    x: int,
    y: int,
    terrain_type: int,
    observed_ms: int = 0,
    confidence: float = 1.0,
    provenance: ProvenanceChainDict | None = None,
) -> TerrainTileDict:
    """Create a terrain tile.

    Args:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        terrain_type: Terrain type (0-7).
        observed_ms: When the tile was confirmed.
        confidence: Trust in this belief. Fresh observations use 1.0.
        provenance: Origin plus derivation references. When omitted,
            defaults to the 0x5A patch-grid channel; the 0x4A terrain
            update path passes its own.

    Returns:
        TerrainTileDict with the provided values.
    """
    resolved_provenance = (
        make_provenance(DEFAULT_TERRAIN_FACT_SOURCE, []) if provenance is None else provenance
    )
    return TerrainTileDict(
        x=x,
        y=y,
        terrain_type=terrain_type,
        observed_ms=observed_ms,
        confidence=confidence,
        provenance=resolved_provenance,
    )


def encode_terrain_tile(tile: TerrainTileDict) -> JSONObject:
    """Encode TerrainTileDict to JSON-serializable dict.

    Args:
        tile: TerrainTileDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "x": tile["x"],
        "y": tile["y"],
        "terrain_type": tile["terrain_type"],
        "observed_ms": tile["observed_ms"],
        "confidence": tile["confidence"],
        "provenance": encode_provenance(tile["provenance"]),
    }


def decode_terrain_tile(data: JSONObject) -> TerrainTileDict:
    """Decode TerrainTileDict from JSON with validation.

    The fact-metadata fields were added after the on-disk format
    stabilised (Phase 1d); older snapshots lacking the keys decode to
    the same defaults ``make_terrain_tile`` derives.

    Args:
        data: JSON object to decode.

    Returns:
        Validated TerrainTileDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    observed_ms = require_int(data, "observed_ms") if "observed_ms" in data else 0
    confidence = require_float(data, "confidence") if "confidence" in data else 1.0
    provenance = (
        decode_provenance(require_dict(data, "provenance"))
        if "provenance" in data
        else make_provenance(DEFAULT_TERRAIN_FACT_SOURCE, [])
    )
    return TerrainTileDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        terrain_type=require_int(data, "terrain_type"),
        observed_ms=observed_ms,
        confidence=confidence,
        provenance=provenance,
    )


__all__ = [
    "DEFAULT_TERRAIN_FACT_SOURCE",
    "TerrainTileDict",
    "decode_terrain_tile",
    "encode_terrain_tile",
    "make_terrain_tile",
]
