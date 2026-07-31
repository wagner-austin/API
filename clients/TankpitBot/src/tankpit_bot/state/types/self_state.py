"""Self-tank state TypedDict + factory + encode/decode.

Phase 1d of the self-observing architecture: the self state carries
the fact metadata flat -- ``observed_ms``, ``confidence`` and
``provenance``. The Fact[T] projection lives in
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

DEFAULT_SELF_FACT_SOURCE: FactSource = "wire_0x3D_movement"
"""Synthetic default channel for direct ``make_self_state`` calls.

0x3D MovementResponse is the canonical self-position message. The
production mutators (``update_self_position``, ``set_self_fuel``,
``set_self_rank``) always pass the true channel explicitly; this
default only serves direct construction in tests and fixtures.
"""


class SelfStateDict(TypedDict):
    """State of the player's own tank.

    Attributes:
        tank_id: Player's tank ID.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0 recruit .. 8 general).
        fuel: Current fuel (also health).
        leaderboard_position: Position on leaderboard.
        observed_ms: When the self state was last refreshed by a wire
            message. Zero for fixtures constructed without a clock.
        confidence: Trust in this belief, [0.0, 1.0]. Fresh
            observations record 1.0.
        provenance: Origin wire channel plus derivation references.
    """

    tank_id: int
    x: int
    y: int
    team: int
    rank: int
    fuel: int
    leaderboard_position: int
    observed_ms: int
    confidence: float
    provenance: ProvenanceChainDict


def make_self_state(
    tank_id: int,
    x: int,
    y: int,
    team: int,
    rank: int,
    fuel: int,
    leaderboard_position: int,
    observed_ms: int = 0,
    confidence: float = 1.0,
    provenance: ProvenanceChainDict | None = None,
) -> SelfStateDict:
    """Create self state.

    Args:
        tank_id: Player's tank ID.
        x: X coordinate (0-255).
        y: Y coordinate (0-255).
        team: Team ID (0-3).
        rank: Military rank (0 recruit .. 8 general).
        fuel: Current fuel amount.
        leaderboard_position: Leaderboard position.
        observed_ms: When the self state was refreshed.
        confidence: Trust in this belief. Fresh observations use 1.0.
        provenance: Origin plus derivation references. When omitted,
            defaults to the canonical self-position channel
            (``DEFAULT_SELF_FACT_SOURCE``); production mutators always
            pass the true channel.

    Returns:
        SelfStateDict with the provided values.
    """
    resolved_provenance = (
        make_provenance(DEFAULT_SELF_FACT_SOURCE, []) if provenance is None else provenance
    )
    return SelfStateDict(
        tank_id=tank_id,
        x=x,
        y=y,
        team=team,
        rank=rank,
        fuel=fuel,
        leaderboard_position=leaderboard_position,
        observed_ms=observed_ms,
        confidence=confidence,
        provenance=resolved_provenance,
    )


def encode_self_state(state: SelfStateDict) -> JSONObject:
    """Encode SelfStateDict to JSON-serializable dict.

    Args:
        state: SelfStateDict to encode.

    Returns:
        JSON-serializable dict representation.
    """
    return {
        "tank_id": state["tank_id"],
        "x": state["x"],
        "y": state["y"],
        "team": state["team"],
        "rank": state["rank"],
        "fuel": state["fuel"],
        "leaderboard_position": state["leaderboard_position"],
        "observed_ms": state["observed_ms"],
        "confidence": state["confidence"],
        "provenance": encode_provenance(state["provenance"]),
    }


def decode_self_state(data: JSONObject) -> SelfStateDict:
    """Decode SelfStateDict from JSON with validation.

    The fact-metadata fields were added after the on-disk format
    stabilised (Phase 1d); older snapshots lacking the keys decode to
    the same defaults a contemporary ``make_self_state`` would derive.

    Args:
        data: JSON object to decode.

    Returns:
        Validated SelfStateDict.

    Raises:
        JSONTypeError: If required fields are missing or invalid.
    """
    observed_ms = require_int(data, "observed_ms") if "observed_ms" in data else 0
    confidence = require_float(data, "confidence") if "confidence" in data else 1.0
    provenance = (
        decode_provenance(require_dict(data, "provenance"))
        if "provenance" in data
        else make_provenance(DEFAULT_SELF_FACT_SOURCE, [])
    )
    return SelfStateDict(
        tank_id=require_int(data, "tank_id"),
        x=require_int(data, "x"),
        y=require_int(data, "y"),
        team=require_int(data, "team"),
        rank=require_int(data, "rank"),
        fuel=require_int(data, "fuel"),
        leaderboard_position=require_int(data, "leaderboard_position"),
        observed_ms=observed_ms,
        confidence=confidence,
        provenance=provenance,
    )


__all__ = [
    "DEFAULT_SELF_FACT_SOURCE",
    "SelfStateDict",
    "decode_self_state",
    "encode_self_state",
    "make_self_state",
]
