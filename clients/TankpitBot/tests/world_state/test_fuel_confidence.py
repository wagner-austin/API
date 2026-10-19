"""Enemy-refuel inference, correction authority, and own-fuel isolation."""

from __future__ import annotations

import pytest

from tankpit_bot.facts.source import FactSource
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_tiles import update_cache_tiles
from tankpit_bot.state.container_mutations import (
    apply_tile_cache_update,
    increment_container_failed_pickups,
    merge_container_sighting,
    pickup_container,
    update_container_from_radar,
)
from tankpit_bot.state.tank_mutations import apply_tank_observation
from tankpit_bot.state.types import TankObservation, WorldStateDict, make_tank_observation
from tankpit_bot.types.constants import ContainerRefreshKind, EntitySource, TankLiveness
from tests.bot.ai._support import make_container, make_world


def _refuel_pair() -> tuple[WorldStateDict, TankObservation]:
    """Build the incident's 580-fuel tile and enemy quartile 2-to-3 pair."""
    world, _ = make_world(
        fuel=740,
        containers={"148,249": make_container(148, 249, 580, is_fuel=True)},
    )
    world = apply_tank_observation(
        world,
        make_tank_observation(
            tank_id=814,
            timestamp_ms=100000,
            is_wire_sourced=True,
            storage_source=EntitySource.VIEWPORT,
            fact_source=FactSource.WIRE_0X3D_MOVEMENT,
            position=(148, 249),
            rank=6,
            team=2,
            damage_state=2,
        ),
    )
    return world, make_tank_observation(
        tank_id=814,
        timestamp_ms=100001,
        is_wire_sourced=True,
        storage_source=EntitySource.VIEWPORT,
        fact_source=FactSource.WIRE_0X2E_TANK_STATUS_SYNC,
        damage_state=3,
    )


def test_enemy_refuel_marks_suspect_without_claiming_a_transfer() -> None:
    """The incident lowers trust, preserves 580 and own fuel, and cites evidence."""
    world, obs = _refuel_pair()
    result = apply_tank_observation(world, obs)
    container = result["containers"]["148,249"]
    assert container["confidence"] == 0.5
    assert container["volume"] == 580
    assert container["timestamp_ms"] == world["containers"]["148,249"]["timestamp_ms"]
    assert container["provenance"]["origin"] is FactSource.CLIENT_SIDE_INFERENCE
    assert [ref["source"] for ref in container["provenance"]["derived_from"]][-2:] == [
        FactSource.WIRE_0X3D_MOVEMENT,
        FactSource.WIRE_0X2E_TANK_STATUS_SYNC,
    ]
    assert result["self_state"] == world["self_state"]
    assert world["containers"]["148,249"]["confidence"] == 1.0
    repeated = apply_tank_observation(result, obs)
    assert repeated["containers"] == result["containers"]


@pytest.mark.parametrize(
    "case",
    [
        "unknown",
        "no_self",
        "self",
        "ally",
        "dead",
        "dies",
        "rank",
        "same_tier",
    ],
)
def test_unproven_refuel_does_not_change_container(case: str) -> None:
    """Each missing premise leaves the container registry unchanged."""
    world, obs = _refuel_pair()
    previous = world["tanks"]["814"]
    own = world["self_state"]
    if own is None:
        pytest.fail("Seeded incident must include our tank")
    if case == "unknown":
        world["tanks"].clear()
    elif case == "no_self":
        world["self_state"] = None
    elif case == "self":
        own["tank_id"] = 814
    elif case == "ally":
        obs["team"] = own["team"]
    elif case == "dead":
        previous["liveness"] = TankLiveness.DEACTIVATED
    elif case == "dies":
        obs["direction"] = 32
    elif case == "rank":
        obs["rank"] = 7
    elif case == "same_tier":
        obs["damage_state"] = 2
    result = apply_tank_observation(world, obs)
    assert result["containers"] == world["containers"]


@pytest.mark.parametrize(
    "case",
    [
        "nonwire",
        "wrong_channel",
        "map",
        "stale",
        "future",
        "missing_container",
        "equipment",
        "empty",
    ],
)
def test_unproven_position_or_resource_does_not_change_container(case: str) -> None:
    """A quartile increase needs both fresh position evidence and known fuel."""
    world, obs = _refuel_pair()
    previous = world["tanks"]["814"]
    if case == "nonwire":
        obs["is_wire_sourced"] = False
    elif case == "wrong_channel":
        obs["fact_source"] = FactSource.DOM_REGISTRY_SCRAPE
    elif case == "map":
        previous["provenance"]["origin"] = FactSource.WIRE_0X4C_MAP_DATA
    elif case == "stale":
        obs["timestamp_ms"] = 102001
    elif case == "future":
        obs["timestamp_ms"] = 99999
    elif case == "missing_container":
        world["containers"].clear()
    elif case == "equipment":
        world["containers"]["148,249"]["is_fuel"] = False
    elif case == "empty":
        world["containers"]["148,249"]["volume"] = 0
    result = apply_tank_observation(world, obs)
    assert result["containers"] == world["containers"]


def test_refresh_authority_and_partial_consumption() -> None:
    """Advertisements retain suspicion; explicit quantity corrections clear it."""
    world, obs = _refuel_pair()
    suspect = apply_tank_observation(world, obs)
    viewport = apply_tile_cache_update(suspect, 148, 249, 580, 100010)
    fleet = merge_container_sighting(viewport, 148, 249, True, 580, 100020)
    cached = update_container_from_radar(
        fleet,
        148,
        249,
        580,
        100030,
        refresh_kind=ContainerRefreshKind.RADAR_KNOWN_RESOURCES,
    )
    failed = increment_container_failed_pickups(cached, 148, 249)
    for state in (viewport, fleet, cached, failed):
        assert state["containers"]["148,249"]["confidence"] == 0.5
        assert state["self_state"] == world["self_state"]
    radar = update_container_from_radar(cached, 148, 249, 200, 100040)
    partial = pickup_container(cached, 148, 249, 100040, 200)
    ws = WorldService()
    ws.world_state = cached
    update_cache_tiles(ws, [(148, 249, 200)])
    for state in (radar, partial, ws.world_state):
        assert state["containers"]["148,249"]["confidence"] == 1.0
        assert state["containers"]["148,249"]["volume"] == 200
        assert state["self_state"] == world["self_state"]
    assert pickup_container(cached, 148, 249, 100050)["containers"] == {}
    assert update_container_from_radar(cached, 148, 249, 0, 100050)["containers"] == {}
