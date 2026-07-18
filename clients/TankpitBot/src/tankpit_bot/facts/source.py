"""Fact source literals: every belief names the channel it came from.

The eleven sources are the complete set of observation and inference
channels the bot has today: nine wire message types, the game-log DOM
scrape, and client-side inference. A fact whose source is
``client_side_inference`` is a *derivation* and must cite prior
sources in its provenance chain (see
:mod:`tankpit_bot.facts.provenance`); every other source is a direct
*observation*.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONObject, JSONTypeError, require_str

FactSource = Literal[
    "wire_0x2E_tank_status",
    "wire_0x5A_viewport_patch",
    "wire_0x43_cache_update",
    "wire_0x4F_radar_response",
    "wire_0x4C_map_data",
    "wire_0x3D_movement",
    "wire_0x41_deactivation",
    "wire_0x53_shoot_event",
    "wire_0x52_supervisor",
    "game_log_scrape",
    "client_side_inference",
]
"""Channel a fact was observed on (or inferred from)."""

_FACT_SOURCE_BY_NAME: dict[str, FactSource] = {
    "wire_0x2E_tank_status": "wire_0x2E_tank_status",
    "wire_0x5A_viewport_patch": "wire_0x5A_viewport_patch",
    "wire_0x43_cache_update": "wire_0x43_cache_update",
    "wire_0x4F_radar_response": "wire_0x4F_radar_response",
    "wire_0x4C_map_data": "wire_0x4C_map_data",
    "wire_0x3D_movement": "wire_0x3D_movement",
    "wire_0x41_deactivation": "wire_0x41_deactivation",
    "wire_0x53_shoot_event": "wire_0x53_shoot_event",
    "wire_0x52_supervisor": "wire_0x52_supervisor",
    "game_log_scrape": "game_log_scrape",
    "client_side_inference": "client_side_inference",
}

FACT_SOURCES: tuple[str, ...] = tuple(_FACT_SOURCE_BY_NAME)
"""All valid fact source names, for validation messages."""

INFERENCE_SOURCE: FactSource = "client_side_inference"
"""The one derivation source; every other source is an observation.

Deviation from the Phase 1 spec text ("non-derived Facts must have a
wire-originating source"): ``game_log_scrape`` counts as an
observation origin here. The game log is scraped from the page DOM --
an external channel the bot reads, not something it derives from
prior beliefs -- so a game-log fact with an empty derivation list is
rooted. Only ``client_side_inference`` requires citations.
"""


def is_observation_source(source: FactSource) -> bool:
    """Report whether ``source`` is a direct observation channel.

    Args:
        source: Fact source to classify.

    Returns:
        True for wire and game-log sources; False for inference.
    """
    return source != INFERENCE_SOURCE


def require_fact_source(data: JSONObject, key: str) -> FactSource:
    """Validate and extract a fact source from JSON.

    Args:
        data: JSON object containing the field.
        key: Key to extract.

    Returns:
        Validated fact source value.

    Raises:
        JSONTypeError: If the value is not a supported fact source.
    """
    raw = require_str(data, key)
    source = _FACT_SOURCE_BY_NAME.get(raw)
    if source is None:
        raise JSONTypeError(f"{key} must be one of {FACT_SOURCES}, got {raw!r}")
    return source


__all__ = [
    "FACT_SOURCES",
    "INFERENCE_SOURCE",
    "FactSource",
    "is_observation_source",
    "require_fact_source",
]
