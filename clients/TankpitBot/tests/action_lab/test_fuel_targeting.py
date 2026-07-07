"""Tests for visible fuel targeting helpers."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Protocol

import pytest
from tests.action_lab._replay_core import ReplayClock
from tests.action_lab.test_fuel_probe import _ProbeHarness, _terrain

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab.fuel_targeting import (
    FuelTargetingError,
    find_visible_fuel_landing_tile,
    find_visible_fuel_target,
    visible_fuel_requires_reposition,
)
from tankpit_bot.state import (
    ContainerStateDict,
    MineStateDict,
    SelfStateDict,
    WorldStateDict,
    make_container_state,
)


class _FindBestFuelProtocol(Protocol):
    """Callable protocol for best-fuel selection."""

    def __call__(
        self,
        world: WorldStateDict,
        self_state: SelfStateDict,
        terrain: TerrainMapProtocol,
        *,
        minimum_volume: int,
    ) -> ContainerStateDict | None: ...


class _ReachabilityProtocol(Protocol):
    """Callable protocol for viewport fuel reachability."""

    def __call__(
        self,
        world: WorldStateDict,
        terrain: TerrainMapProtocol,
        start_x: int,
        start_y: int,
        target_x: int,
        target_y: int,
        mines: dict[str, MineStateDict],
    ) -> bool: ...


class _FindLandingTileProtocol(Protocol):
    """Callable protocol for teleport landing-tile selection."""

    def __call__(
        self,
        terrain: TerrainMapProtocol,
        start_x: int,
        start_y: int,
        target_x: int,
        target_y: int,
        mines: dict[str, MineStateDict],
    ) -> tuple[int, int] | None: ...


class _FuelTargetingModuleProtocol(Protocol):
    """Typed access to patchable fuel-targeting globals."""

    get_terrain_map: Callable[[], TerrainMapProtocol | None]
    find_best_fuel: _FindBestFuelProtocol
    is_collection_reachable_in_viewport: _ReachabilityProtocol
    find_teleport_landing_tile: _FindLandingTileProtocol


_fuel_targeting_import = __import__(
    "tankpit_bot.action_lab.fuel_targeting",
    fromlist=["fuel_targeting"],
)
fuel_targeting_module: _FuelTargetingModuleProtocol = _fuel_targeting_import


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore patched fuel-targeting hooks after each test."""
    original_get_terrain_map = fuel_targeting_module.get_terrain_map
    original_find_best_fuel = fuel_targeting_module.find_best_fuel
    original_is_collection_reachable = fuel_targeting_module.is_collection_reachable_in_viewport
    original_find_teleport_landing_tile = fuel_targeting_module.find_teleport_landing_tile
    yield
    fuel_targeting_module.get_terrain_map = original_get_terrain_map
    fuel_targeting_module.find_best_fuel = original_find_best_fuel
    fuel_targeting_module.is_collection_reachable_in_viewport = original_is_collection_reachable
    fuel_targeting_module.find_teleport_landing_tile = original_find_teleport_landing_tile


def test_find_visible_fuel_target_uses_current_probe_state() -> None:
    """Visible fuel selection passes the live world state through unchanged."""
    probe = _ProbeHarness(ReplayClock(1000))
    probe.get_world_state()["timestamp_ms"] = 1500
    expected = make_container_state(101, 100, True, 350, timestamp_ms=1500)
    fuel_targeting_module.get_terrain_map = lambda: _terrain({(100, 100), (101, 100)})
    captured: dict[str, int | bool | WorldStateDict | SelfStateDict | TerrainMapProtocol] = {}

    def _find_best_fuel(
        world: WorldStateDict,
        self_state: SelfStateDict,
        terrain: TerrainMapProtocol,
        *,
        minimum_volume: int,
    ) -> ContainerStateDict | None:
        captured["world"] = world
        captured["self_state"] = self_state
        captured["terrain"] = terrain
        captured["minimum_volume"] = minimum_volume
        return expected

    fuel_targeting_module.find_best_fuel = _find_best_fuel

    result = find_visible_fuel_target(probe)

    assert result == expected
    assert captured["world"] == probe.get_world_state()
    assert captured["self_state"] == probe.get_self_state()
    assert captured["minimum_volume"] == 1


def test_find_visible_fuel_target_requires_terrain_and_self_state() -> None:
    """Visible fuel selection rejects missing terrain and self state."""
    probe = _ProbeHarness(ReplayClock(1000))
    fuel_targeting_module.get_terrain_map = lambda: None

    with pytest.raises(FuelTargetingError, match="terrain map is unavailable"):
        find_visible_fuel_target(probe)

    fuel_targeting_module.get_terrain_map = lambda: _terrain({(100, 100)})
    probe.get_world_state()["self_state"] = None

    with pytest.raises(FuelTargetingError, match="self state is unavailable"):
        find_visible_fuel_target(probe)


def test_visible_fuel_requires_reposition_uses_reachability_and_validates_state() -> None:
    """Reachability helper detects blocked visible fuel and validates prerequisites."""
    probe = _ProbeHarness(ReplayClock(1000))
    target = make_container_state(101, 100, True, 300, timestamp_ms=1000)
    fuel_targeting_module.get_terrain_map = lambda: _terrain({(100, 100), (101, 100)})
    captured: dict[str, int] = {}

    def _is_collection_reachable_in_viewport(
        world: WorldStateDict,
        terrain: TerrainMapProtocol,
        start_x: int,
        start_y: int,
        target_x: int,
        target_y: int,
        mines: dict[str, MineStateDict],
    ) -> bool:
        _ = (world, terrain, mines)
        captured["start_x"] = start_x
        captured["start_y"] = start_y
        captured["target_x"] = target_x
        captured["target_y"] = target_y
        return False

    fuel_targeting_module.is_collection_reachable_in_viewport = _is_collection_reachable_in_viewport

    assert visible_fuel_requires_reposition(probe, target) is True
    assert captured == {
        "start_x": 100,
        "start_y": 100,
        "target_x": 101,
        "target_y": 100,
    }

    fuel_targeting_module.get_terrain_map = lambda: None
    with pytest.raises(FuelTargetingError, match="terrain map is unavailable"):
        visible_fuel_requires_reposition(probe, target)

    fuel_targeting_module.get_terrain_map = lambda: _terrain({(100, 100), (101, 100)})
    probe.get_world_state()["self_state"] = None
    with pytest.raises(FuelTargetingError, match="self state is unavailable"):
        visible_fuel_requires_reposition(probe, target)


def test_find_visible_fuel_landing_tile_uses_current_state_and_validates_state() -> None:
    """Landing-tile selection uses current probe state and validates prerequisites."""
    probe = _ProbeHarness(ReplayClock(1000))
    target = make_container_state(101, 100, True, 300, timestamp_ms=1000)
    fuel_targeting_module.get_terrain_map = lambda: _terrain({(100, 100), (102, 100)})
    captured: dict[str, int] = {}

    def _find_teleport_landing_tile(
        terrain: TerrainMapProtocol,
        start_x: int,
        start_y: int,
        target_x: int,
        target_y: int,
        mines: dict[str, MineStateDict],
    ) -> tuple[int, int] | None:
        _ = (terrain, mines)
        captured["start_x"] = start_x
        captured["start_y"] = start_y
        captured["target_x"] = target_x
        captured["target_y"] = target_y
        return (102, 100)

    fuel_targeting_module.find_teleport_landing_tile = _find_teleport_landing_tile

    assert find_visible_fuel_landing_tile(probe, target) == (102, 100)
    assert captured == {
        "start_x": 100,
        "start_y": 100,
        "target_x": 101,
        "target_y": 100,
    }

    fuel_targeting_module.get_terrain_map = lambda: None
    with pytest.raises(FuelTargetingError, match="terrain map is unavailable"):
        find_visible_fuel_landing_tile(probe, target)

    fuel_targeting_module.get_terrain_map = lambda: _terrain({(100, 100), (102, 100)})
    probe.get_world_state()["self_state"] = None
    with pytest.raises(FuelTargetingError, match="self state is unavailable"):
        find_visible_fuel_landing_tile(probe, target)
