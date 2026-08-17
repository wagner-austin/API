"""Fact source literals: every belief names the channel it came from.

The sources are the complete set of observation and inference channels
the bot has today: twenty wire message types, one DOM scrape channel
(the JS client's tank registry), and client-side inference. A fact
whose source is ``client_side_inference`` is a *derivation* and must
cite prior sources in its provenance chain (see
:mod:`tankpit_bot.facts.provenance`); every other source is a direct
*observation*.

The DOM game log is deliberately NOT a source: capture replay
2026-07-19 proved every line it renders is the client's presentation
of a wire message the bot already decodes (0x41 for kills, 0x52 error
codes for rejections), so it acts on nothing and is recorded only as
a capture witness.

Deviation from the Phase 1 handoff spec (11 sources): the spec's list
missed the wire channels that demonstrably update the tank registry
(0x21 TankInfo, 0x28 TankEntry, 0x3E TankStatus, 0x42 BuildPickup,
0x47 Movement, 0x48 EnemyDetect) and the registry DOM scrape; the
spec's ``wire_0x2E_tank_status`` is named ``wire_0x2E_tank_status_sync``
here to distinguish it from 0x3E TankStatus.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONObject, JSONTypeError, require_str

FactSource = Literal[
    "wire_0x21_tank_info",
    "wire_0x28_tank_entry",
    "wire_0x2B_promotion",
    "wire_0x2E_tank_status_sync",
    "wire_0x3D_movement",
    "wire_0x3E_tank_status",
    "wire_0x41_deactivation",
    "wire_0x42_build_pickup",
    "wire_0x43_cache_update",
    "wire_0x44_fuel_gain",
    "wire_0x47_movement",
    "wire_0x48_enemy_detect",
    "wire_0x4A_terrain_update",
    "wire_0x4B_mine_placement",
    "wire_0x4C_map_data",
    "wire_0x4F_radar_response",
    "wire_0x52_supervisor",
    "wire_0x53_shoot_event",
    "wire_0x5A_viewport_patch",
    "wire_0x64_fuel_total",
    "dom_registry_scrape",
    "client_side_inference",
    "fleet_report",
]
"""Channel a fact was observed on (or inferred from)."""

_FACT_SOURCE_BY_NAME: dict[str, FactSource] = {
    "wire_0x21_tank_info": "wire_0x21_tank_info",
    "wire_0x28_tank_entry": "wire_0x28_tank_entry",
    "wire_0x2B_promotion": "wire_0x2B_promotion",
    "wire_0x2E_tank_status_sync": "wire_0x2E_tank_status_sync",
    "wire_0x3D_movement": "wire_0x3D_movement",
    "wire_0x3E_tank_status": "wire_0x3E_tank_status",
    "wire_0x41_deactivation": "wire_0x41_deactivation",
    "wire_0x42_build_pickup": "wire_0x42_build_pickup",
    "wire_0x43_cache_update": "wire_0x43_cache_update",
    "wire_0x44_fuel_gain": "wire_0x44_fuel_gain",
    "wire_0x47_movement": "wire_0x47_movement",
    "wire_0x48_enemy_detect": "wire_0x48_enemy_detect",
    "wire_0x4A_terrain_update": "wire_0x4A_terrain_update",
    "wire_0x4B_mine_placement": "wire_0x4B_mine_placement",
    "wire_0x4C_map_data": "wire_0x4C_map_data",
    "wire_0x4F_radar_response": "wire_0x4F_radar_response",
    "wire_0x52_supervisor": "wire_0x52_supervisor",
    "wire_0x53_shoot_event": "wire_0x53_shoot_event",
    "wire_0x5A_viewport_patch": "wire_0x5A_viewport_patch",
    "wire_0x64_fuel_total": "wire_0x64_fuel_total",
    "dom_registry_scrape": "dom_registry_scrape",
    "fleet_report": "fleet_report",
    "client_side_inference": "client_side_inference",
}

FACT_SOURCES: tuple[str, ...] = tuple(_FACT_SOURCE_BY_NAME)
"""All valid fact source names, for validation messages."""

INFERENCE_SOURCE: FactSource = "client_side_inference"
"""The one derivation source; every other source is an observation.

Deviation from the Phase 1 spec text ("non-derived Facts must have a
wire-originating source"): the ``dom_registry_scrape`` channel counts
as an observation origin here. The page DOM is a second wire the bot
reads, not something it derives from prior beliefs -- so a
DOM-scraped fact with an empty derivation list is rooted. Only
``client_side_inference`` requires citations.
"""


def is_observation_source(source: FactSource) -> bool:
    """Report whether ``source`` is a direct observation channel.

    Args:
        source: Fact source to classify.

    Returns:
        True for wire and DOM-scrape sources; False for inference.
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
