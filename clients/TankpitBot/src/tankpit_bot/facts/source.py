"""Fact sources: every belief names the channel it came from.

The sources are the complete set of observation and inference channels
the bot has today: twenty wire message types, one DOM scrape channel
(the JS client's tank registry), the fleet report, and client-side
inference. A fact whose source is ``client_side_inference`` is a
*derivation* and must cite prior sources in its provenance chain (see
:mod:`tankpit_bot.facts.provenance`); every other source is a direct
*observation*. Decoders narrow the words with
:func:`platform_core.members.require_member`.

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

from enum import StrEnum


class FactSource(StrEnum):
    """Channel a fact was observed on (or inferred from)."""

    WIRE_0X21_TANK_INFO = "wire_0x21_tank_info"
    WIRE_0X28_TANK_ENTRY = "wire_0x28_tank_entry"
    WIRE_0X2B_PROMOTION = "wire_0x2B_promotion"
    WIRE_0X2E_TANK_STATUS_SYNC = "wire_0x2E_tank_status_sync"
    WIRE_0X3D_MOVEMENT = "wire_0x3D_movement"
    WIRE_0X3E_TANK_STATUS = "wire_0x3E_tank_status"
    WIRE_0X41_DEACTIVATION = "wire_0x41_deactivation"
    WIRE_0X42_BUILD_PICKUP = "wire_0x42_build_pickup"
    WIRE_0X43_CACHE_UPDATE = "wire_0x43_cache_update"
    WIRE_0X44_FUEL_GAIN = "wire_0x44_fuel_gain"
    WIRE_0X47_MOVEMENT = "wire_0x47_movement"
    WIRE_0X48_ENEMY_DETECT = "wire_0x48_enemy_detect"
    WIRE_0X4A_TERRAIN_UPDATE = "wire_0x4A_terrain_update"
    WIRE_0X4B_MINE_PLACEMENT = "wire_0x4B_mine_placement"
    WIRE_0X4C_MAP_DATA = "wire_0x4C_map_data"
    WIRE_0X4F_RADAR_RESPONSE = "wire_0x4F_radar_response"
    WIRE_0X52_SUPERVISOR = "wire_0x52_supervisor"
    WIRE_0X53_SHOOT_EVENT = "wire_0x53_shoot_event"
    WIRE_0X5A_VIEWPORT_PATCH = "wire_0x5A_viewport_patch"
    WIRE_0X64_FUEL_TOTAL = "wire_0x64_fuel_total"
    DOM_REGISTRY_SCRAPE = "dom_registry_scrape"
    CLIENT_SIDE_INFERENCE = "client_side_inference"
    FLEET_REPORT = "fleet_report"


def is_observation_source(source: FactSource) -> bool:
    """Report whether ``source`` is a direct observation channel.

    ``CLIENT_SIDE_INFERENCE`` is the one derivation source; every other
    source is an observation. Deviation from the Phase 1 spec text
    ("non-derived Facts must have a wire-originating source"): the
    ``DOM_REGISTRY_SCRAPE`` channel counts as an observation origin here.
    The page DOM is a second wire the bot reads, not something it derives
    from prior beliefs -- so a DOM-scraped fact with an empty derivation
    list is rooted. Only ``CLIENT_SIDE_INFERENCE`` requires citations.

    Args:
        source: Fact source to classify.

    Returns:
        True for wire, DOM-scrape and fleet-report sources; False for
        inference.
    """
    return source is not FactSource.CLIENT_SIDE_INFERENCE


__all__ = [
    "FactSource",
    "is_observation_source",
]
