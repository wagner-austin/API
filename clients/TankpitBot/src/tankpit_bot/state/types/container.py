"""Container (fuel/equipment) state TypedDict + factory + encode/decode."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_bool,
    require_int,
)
from typing_extensions import TypedDict

from tankpit_bot.state.types.constants import (
    ContainerRefreshKind,
    EntitySource,
    decode_container_refresh_kind,
    encode_container_refresh_kind,
    require_entity_source,
)


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
    """

    x: int
    y: int
    is_fuel: bool
    volume: int
    source: EntitySource
    refresh_kind: ContainerRefreshKind
    timestamp_ms: int
    failed_pickups: int


def _default_container_refresh_kind(source: EntitySource) -> ContainerRefreshKind:
    """Return the canonical refresh kind for a coarse container source.

    Args:
        source: Coarse observed source.

    Returns:
        Canonical refresh kind matching the source.
    """
    if source == "radar":
        return "radar_response"
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

    Returns:
        ContainerStateDict with the provided values.
    """
    resolved_refresh_kind = (
        _default_container_refresh_kind(source) if refresh_kind is None else refresh_kind
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
    }


def decode_container_state(data: JSONObject) -> ContainerStateDict:
    """Decode ContainerStateDict from JSON with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ContainerStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    return ContainerStateDict(
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        is_fuel=require_bool(data, "is_fuel"),
        volume=require_int(data, "volume"),
        source=require_entity_source(data, "source"),
        refresh_kind=decode_container_refresh_kind(data, "refresh_kind"),
        timestamp_ms=require_int(data, "timestamp_ms"),
        failed_pickups=require_int(data, "failed_pickups"),
    )


__all__ = [
    "ContainerStateDict",
    "decode_container_state",
    "encode_container_state",
    "make_container_state",
]
