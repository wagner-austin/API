"""Container (fuel/equipment) state TypedDict + factory + encode/decode.

Phase 1b of the self-observing architecture: the container carries the
full fact metadata flat -- ``source``/``refresh_kind``/``timestamp_ms``
(pre-existing) plus ``confidence`` and ``provenance``. The Fact[T]
projection lives in :mod:`tankpit_bot.state.projections.container`.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_dict,
    require_float,
    require_int,
)
from typing_extensions import TypedDict

from tankpit_bot.facts.provenance import (
    ProvenanceChainDict,
    decode_provenance,
    encode_provenance,
    make_provenance,
)
from tankpit_bot.facts.source import FactSource
from tankpit_bot.types.constants import (
    ContainerRefreshKind,
    EntitySource,
    decode_container_refresh_kind,
    encode_container_refresh_kind,
    require_entity_source,
)

_FACT_SOURCE_BY_REFRESH_KIND: dict[ContainerRefreshKind, FactSource] = {
    "radar_response": "wire_0x4F_radar_response",
    "radar_cache_refresh": "wire_0x43_cache_update",
    "radar_known_resources": "wire_0x4F_radar_response",
    "viewport_patch": "wire_0x5A_viewport_patch",
    "world_state": "wire_0x4C_map_data",
    "fleet_report": "fleet_report",
}


def container_fact_source(refresh_kind: ContainerRefreshKind) -> FactSource:
    """Map a container refresh kind to the wire channel it arrived on.

    ``radar_known_resources`` maps to the radar response channel: the
    known-resources promotion happens inside the radar-ack cycle and
    re-confirms the radar envelope's containers.

    Args:
        refresh_kind: Container refresh kind.

    Returns:
        The wire fact source behind that refresh kind.
    """
    return _FACT_SOURCE_BY_REFRESH_KIND[refresh_kind]


class ContainerStateDict(TypedDict):
    """State of a fuel or equipment container.

    Attributes:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        is_fuel: True if fuel container, False if equipment.
        volume: Fuel amount (0 for equipment).
        source: Which observed source most recently confirmed this container.
        refresh_kind: Specific confirmation path that most recently refreshed
            this container.
        timestamp_ms: When this container was last confirmed by the
            server (radar, viewport, or world state). Used for
            freshness-based target selection.
        failed_pickups: How many pickup attempts failed for this
            container. Incremented on stall timeout, reset when the
            container is re-confirmed by a fresh source.
        confidence: Trust in this container belief, [0.0, 1.0]. Fresh
            observations record 1.0; decay by age is a consumer policy
            (Phase 3), not baked into storage.
        provenance: Origin wire channel plus derivation references.
            The origin tracks ``refresh_kind`` via
            :func:`container_fact_source`.
    """

    x: int
    y: int
    is_fuel: bool
    volume: int
    source: EntitySource
    refresh_kind: ContainerRefreshKind
    timestamp_ms: int
    failed_pickups: int
    confidence: float
    provenance: ProvenanceChainDict


def _default_container_refresh_kind(source: EntitySource) -> ContainerRefreshKind:
    """Return the canonical refresh kind for a coarse container source.

    Args:
        source: Coarse observed source.

    Returns:
        Canonical refresh kind matching the source.
    """
    if source == "radar":
        return "radar_response"
    if source == "viewport":
        return "viewport_patch"
    return "world_state"


def make_container_state(
    x: int,
    y: int,
    is_fuel: bool,
    volume: int,
    source: EntitySource = "radar",
    refresh_kind: ContainerRefreshKind | None = None,
    timestamp_ms: int = 0,
    failed_pickups: int = 0,
    confidence: float = 1.0,
    provenance: ProvenanceChainDict | None = None,
) -> ContainerStateDict:
    """Create a container state.

    Args:
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        is_fuel: True if fuel, False if equipment.
        volume: Fuel amount (0 for equipment).
        source: Which observed source confirmed this container.
        refresh_kind: Specific refresh path that confirmed this container.
        timestamp_ms: When this container was confirmed.
        failed_pickups: How many pickup attempts failed.
        confidence: Trust in this belief. Fresh observations use 1.0.
        provenance: Origin plus derivation references. When omitted,
            derived from the resolved refresh kind via
            :func:`container_fact_source` with no derivations (a
            direct observation).

    Returns:
        ContainerStateDict with the provided values.
    """
    resolved_refresh_kind = (
        _default_container_refresh_kind(source) if refresh_kind is None else refresh_kind
    )
    resolved_provenance = (
        make_provenance(container_fact_source(resolved_refresh_kind), [])
        if provenance is None
        else provenance
    )
    return ContainerStateDict(
        x=x,
        y=y,
        is_fuel=is_fuel,
        volume=volume,
        source=source,
        refresh_kind=resolved_refresh_kind,
        timestamp_ms=timestamp_ms,
        failed_pickups=failed_pickups,
        confidence=confidence,
        provenance=resolved_provenance,
    )


def encode_container_state(state: ContainerStateDict) -> JSONObject:
    """Encode ContainerStateDict to JSON-serializable dict.

    Args:
        state: ContainerStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "x": state["x"],
        "y": state["y"],
        "is_fuel": state["is_fuel"],
        "volume": state["volume"],
        "source": state["source"],
        "refresh_kind": encode_container_refresh_kind(state["refresh_kind"]),
        "timestamp_ms": state["timestamp_ms"],
        "failed_pickups": state["failed_pickups"],
        "confidence": state["confidence"],
        "provenance": encode_provenance(state["provenance"]),
    }


def decode_container_state(data: JSONObject) -> ContainerStateDict:
    """Decode ContainerStateDict from JSON with validation.

    ``confidence`` and ``provenance`` were added after the on-disk
    format stabilised (Phase 1b); older snapshots / fixtures lack the
    keys and decode with a fresh-observation confidence (1.0) and a
    provenance derived from their refresh kind -- exactly what a
    contemporary encoder would have written.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ContainerStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    refresh_kind = decode_container_refresh_kind(data, "refresh_kind")
    confidence = require_float(data, "confidence") if "confidence" in data else 1.0
    provenance = (
        decode_provenance(require_dict(data, "provenance"))
        if "provenance" in data
        else make_provenance(container_fact_source(refresh_kind), [])
    )
    return ContainerStateDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        is_fuel=require_bool(data, "is_fuel"),
        volume=require_int(data, "volume"),
        source=require_entity_source(data, "source"),
        refresh_kind=refresh_kind,
        timestamp_ms=require_int(data, "timestamp_ms"),
        failed_pickups=require_int(data, "failed_pickups"),
        confidence=confidence,
        provenance=provenance,
    )


__all__ = [
    "ContainerStateDict",
    "container_fact_source",
    "decode_container_state",
    "encode_container_state",
    "make_container_state",
]
