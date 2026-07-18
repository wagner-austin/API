"""Viewport bounds TypedDict + encode/decode.

Phase 1d of the self-observing architecture: the viewport carries the
fact metadata flat. Its provenance origin is always
``wire_0x5A_viewport_patch`` -- 0x5A ViewportUpdate is the only wire
message that sets the viewport (see wiki ``viewport-shift-protocol``).
The Fact[T] projection lives in :mod:`tankpit_bot.facts.world_facts`.
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

VIEWPORT_FACT_SOURCE: FactSource = "wire_0x5A_viewport_patch"
"""The one channel that sets the viewport (0x5A ViewportUpdate)."""


class ViewportStateDict(TypedDict):
    """Current viewport state.

    Attributes:
        left: Left edge X coordinate of viewport.
        top: Top edge Y coordinate of viewport.
        width: Visible viewport width in tiles (typically 16).
        height: Visible viewport height in tiles (typically 16).
        observed_ms: When the viewport was last set by a 0x5A update.
            Zero for fixtures constructed without a clock.
        confidence: Trust in this belief, [0.0, 1.0].
        provenance: Origin channel (always the 0x5A viewport patch)
            plus derivation references.
    """

    left: int
    top: int
    width: int
    height: int
    observed_ms: int
    confidence: float
    provenance: ProvenanceChainDict


def make_viewport_state(
    left: int,
    top: int,
    width: int,
    height: int,
    observed_ms: int = 0,
    confidence: float = 1.0,
) -> ViewportStateDict:
    """Create a viewport state.

    Args:
        left: Left edge X coordinate.
        top: Top edge Y coordinate.
        width: Viewport width in tiles.
        height: Viewport height in tiles.
        observed_ms: When the viewport was set.
        confidence: Trust in this belief. Fresh observations use 1.0.

    Returns:
        ViewportStateDict with the provided values.
    """
    return ViewportStateDict(
        left=left,
        top=top,
        width=width,
        height=height,
        observed_ms=observed_ms,
        confidence=confidence,
        provenance=make_provenance(VIEWPORT_FACT_SOURCE, []),
    )


def encode_viewport_state(state: ViewportStateDict) -> JSONObject:
    """Encode ViewportStateDict to JSON-serializable dict.

    Args:
        state: ViewportStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "left": state["left"],
        "top": state["top"],
        "width": state["width"],
        "height": state["height"],
        "observed_ms": state["observed_ms"],
        "confidence": state["confidence"],
        "provenance": encode_provenance(state["provenance"]),
    }


def decode_viewport_state(data: JSONObject) -> ViewportStateDict:
    """Decode ViewportStateDict from JSON with validation.

    The fact-metadata fields were added after the on-disk format
    stabilised (Phase 1d); older snapshots lacking the keys decode to
    the same defaults ``make_viewport_state`` derives.

    Args:
        data: JSON object to decode.

    Returns:
        Validated ViewportStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    observed_ms = require_int(data, "observed_ms") if "observed_ms" in data else 0
    confidence = require_float(data, "confidence") if "confidence" in data else 1.0
    provenance = (
        decode_provenance(require_dict(data, "provenance"))
        if "provenance" in data
        else make_provenance(VIEWPORT_FACT_SOURCE, [])
    )
    return ViewportStateDict(
        left=require_int(data, "left"),
        top=require_int(data, "top"),
        width=require_int(data, "width"),
        height=require_int(data, "height"),
        observed_ms=observed_ms,
        confidence=confidence,
        provenance=provenance,
    )


__all__ = [
    "VIEWPORT_FACT_SOURCE",
    "ViewportStateDict",
    "decode_viewport_state",
    "encode_viewport_state",
    "make_viewport_state",
]
