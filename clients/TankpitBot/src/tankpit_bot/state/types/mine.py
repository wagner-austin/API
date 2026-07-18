"""Mine state TypedDict + factory + encode/decode.

Phase 1d of the self-observing architecture: the mine carries the full
fact metadata flat -- ``source``/``timestamp_ms`` (pre-existing) plus
``confidence`` and ``provenance``. The Fact[T] projection lives in
:mod:`tankpit_bot.facts.world_facts`.
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
from tankpit_bot.state.types.constants import EntitySource, require_entity_source

_MINE_FACT_SOURCE_BY_ENTITY_SOURCE: dict[EntitySource, FactSource] = {
    "viewport": "wire_0x5A_viewport_patch",
    "radar": "wire_0x4F_radar_response",
    "world_state": "wire_0x4C_map_data",
}


def mine_default_fact_source(source: EntitySource) -> FactSource:
    """Return the default fact source for a coarse mine source.

    Covers the tile-observation channels (viewport patch, radar, map
    data). A witnessed placement passes its explicit
    ``wire_0x4B_mine_placement`` provenance instead (see
    ``container_mutations.add_mine``).

    Args:
        source: Coarse observed source.

    Returns:
        Canonical fact source for that coarse label.
    """
    return _MINE_FACT_SOURCE_BY_ENTITY_SOURCE[source]


class MineStateDict(TypedDict):
    """State of a placed mine.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        mine_type: Type of mine (from protocol). 0 if unknown (radar-discovered).
        tank_id: ID of tank that placed the mine. -1 if unknown (radar-discovered).
        team: Team that owns the mine (0=red, 1=purple, 2=blue, 3=orange).
        source: Which observed source most recently confirmed this mine.
        timestamp_ms: When this mine was last confirmed by the server.
        confidence: Trust in this mine belief, [0.0, 1.0]. Fresh
            observations record 1.0.
        provenance: Origin wire channel plus derivation references.
    """

    x: int
    y: int
    mine_type: int
    tank_id: int
    team: int
    source: EntitySource
    timestamp_ms: int
    confidence: float
    provenance: ProvenanceChainDict


def make_mine_state(
    x: int,
    y: int,
    mine_type: int,
    tank_id: int,
    team: int,
    source: EntitySource = "viewport",
    timestamp_ms: int = 0,
    confidence: float = 1.0,
    provenance: ProvenanceChainDict | None = None,
) -> MineStateDict:
    """Create a mine state.

    Args:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        mine_type: Type of mine. 0 if unknown (radar-discovered).
        tank_id: ID of placing tank. -1 if unknown (radar-discovered).
        team: Team that owns the mine (0=red, 1=purple, 2=blue, 3=orange).
        source: Which observed source confirmed this mine.
        timestamp_ms: When this mine was confirmed.
        confidence: Trust in this belief. Fresh observations use 1.0.
        provenance: Origin plus derivation references. When omitted,
            derived from ``source`` via :func:`mine_default_fact_source`.

    Returns:
        MineStateDict with the provided values.
    """
    resolved_provenance = (
        make_provenance(mine_default_fact_source(source), []) if provenance is None else provenance
    )
    return MineStateDict(
        x=x,
        y=y,
        mine_type=mine_type,
        tank_id=tank_id,
        team=team,
        source=source,
        timestamp_ms=timestamp_ms,
        confidence=confidence,
        provenance=resolved_provenance,
    )


def encode_mine_state(state: MineStateDict) -> JSONObject:
    """Encode MineStateDict to JSON-serializable dict.

    Args:
        state: MineStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "x": state["x"],
        "y": state["y"],
        "mine_type": state["mine_type"],
        "tank_id": state["tank_id"],
        "team": state["team"],
        "source": state["source"],
        "timestamp_ms": state["timestamp_ms"],
        "confidence": state["confidence"],
        "provenance": encode_provenance(state["provenance"]),
    }


def decode_mine_state(data: JSONObject) -> MineStateDict:
    """Decode MineStateDict from JSON with validation.

    ``confidence`` and ``provenance`` were added after the on-disk
    format stabilised (Phase 1d); older snapshots lacking the keys
    decode to exactly what a contemporary encoder writes (fresh
    confidence, provenance derived from the coarse source).

    Args:
        data: JSON object to decode.

    Returns:
        Validated MineStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    source = require_entity_source(data, "source")
    confidence = require_float(data, "confidence") if "confidence" in data else 1.0
    provenance = (
        decode_provenance(require_dict(data, "provenance"))
        if "provenance" in data
        else make_provenance(mine_default_fact_source(source), [])
    )
    return MineStateDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        mine_type=require_int(data, "mine_type"),
        tank_id=require_int(data, "tank_id"),
        team=require_int(data, "team"),
        source=source,
        timestamp_ms=require_int(data, "timestamp_ms"),
        confidence=confidence,
        provenance=provenance,
    )


__all__ = [
    "MineStateDict",
    "decode_mine_state",
    "encode_mine_state",
    "make_mine_state",
    "mine_default_fact_source",
]
