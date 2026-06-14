"""Tests for visible equipment targeting error guards."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Protocol

import pytest
from tests.fakes import InMemoryTerrainMap

from tankpit_bot._test_hooks import TerrainMapProtocol
from tankpit_bot.action_lab.equipment_targeting import (
    EquipmentTargetingError,
    find_visible_equipment_landing_tile,
    find_visible_equipment_target,
    visible_equipment_requires_reposition,
)
from tankpit_bot.state import (
    SelfStateDict,
    ViewportStateDict,
    WorldStateDict,
    make_container_state,
    make_empty_world_state,
    make_self_state,
)


class _EquipmentTargetingModuleProtocol(Protocol):
    """Typed access to patchable equipment-targeting globals."""

    get_terrain_map: Callable[[], TerrainMapProtocol | None]


_equipment_targeting_import = __import__(
    "tankpit_bot.action_lab.equipment_targeting",
    fromlist=["equipment_targeting"],
)
equipment_targeting_module: _EquipmentTargetingModuleProtocol = _equipment_targeting_import


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore patched equipment-targeting hooks after each test."""
    original_get_terrain_map = equipment_targeting_module.get_terrain_map
    yield
    equipment_targeting_module.get_terrain_map = original_get_terrain_map


def _terrain(passable: set[tuple[int, int]]) -> TerrainMapProtocol:
    return InMemoryTerrainMap.from_passable_set(passable)


def _make_world(timestamp_ms: int, x: int, y: int, fuel: int) -> WorldStateDict:
    """Build a world with one self tank."""
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1,
            x=x,
            y=y,
            team=2,
            rank=1,
            fuel=fuel,
            leaderboard_position=1,
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=ViewportStateDict(left=x - 8, top=y - 8, width=16, height=16),
        scanned_viewports=world["scanned_viewports"],
        map_fuel_dots={},
        timestamp_ms=timestamp_ms,
    )


class _SimpleProbe:
    """Minimal probe satisfying ``VisibleEquipmentTargetingProbeProtocol``."""

    def __init__(self, world: WorldStateDict) -> None:
        self._world = world

    def get_world_state(self) -> WorldStateDict:
        return self._world

    def get_self_state(self) -> SelfStateDict | None:
        return self._world["self_state"]


# ---------------------------------------------------------------------------
# find_visible_equipment_target error guards
# ---------------------------------------------------------------------------


def test_find_visible_equipment_target_raises_when_terrain_unavailable() -> None:
    """find_visible_equipment_target raises when terrain map is None."""
    world = _make_world(1000, 100, 100, 700)
    probe = _SimpleProbe(world)
    equipment_targeting_module.get_terrain_map = lambda: None

    with pytest.raises(EquipmentTargetingError, match="terrain map is unavailable"):
        find_visible_equipment_target(probe, allow_unreachable=False)


def test_find_visible_equipment_target_raises_when_self_state_unavailable() -> None:
    """find_visible_equipment_target raises when self state is None."""
    world = _make_world(1000, 100, 100, 700)
    probe = _SimpleProbe(world)
    equipment_targeting_module.get_terrain_map = lambda: _terrain({(100, 100)})
    probe._world["self_state"] = None

    with pytest.raises(EquipmentTargetingError, match="self state is unavailable"):
        find_visible_equipment_target(probe, allow_unreachable=False)


# ---------------------------------------------------------------------------
# visible_equipment_requires_reposition error guards
# ---------------------------------------------------------------------------


def test_visible_equipment_requires_reposition_raises_when_terrain_unavailable() -> None:
    """visible_equipment_requires_reposition raises when terrain map is None."""
    world = _make_world(1000, 100, 100, 700)
    probe = _SimpleProbe(world)
    container = make_container_state(105, 105, False, 100, timestamp_ms=1000)
    equipment_targeting_module.get_terrain_map = lambda: None

    with pytest.raises(EquipmentTargetingError, match="terrain map is unavailable"):
        visible_equipment_requires_reposition(probe, container)


def test_visible_equipment_requires_reposition_raises_when_self_state_unavailable() -> None:
    """visible_equipment_requires_reposition raises when self state is None."""
    world = _make_world(1000, 100, 100, 700)
    probe = _SimpleProbe(world)
    container = make_container_state(105, 105, False, 100, timestamp_ms=1000)
    equipment_targeting_module.get_terrain_map = lambda: _terrain({(100, 100)})
    probe._world["self_state"] = None

    with pytest.raises(EquipmentTargetingError, match="self state is unavailable"):
        visible_equipment_requires_reposition(probe, container)


# ---------------------------------------------------------------------------
# find_visible_equipment_landing_tile error guards
# ---------------------------------------------------------------------------


def test_find_visible_equipment_landing_tile_raises_when_terrain_unavailable() -> None:
    """find_visible_equipment_landing_tile raises when terrain map is None."""
    world = _make_world(1000, 100, 100, 700)
    probe = _SimpleProbe(world)
    container = make_container_state(105, 105, False, 100, timestamp_ms=1000)
    equipment_targeting_module.get_terrain_map = lambda: None

    with pytest.raises(EquipmentTargetingError, match="terrain map is unavailable"):
        find_visible_equipment_landing_tile(probe, container)


def test_find_visible_equipment_landing_tile_raises_when_self_state_unavailable() -> None:
    """find_visible_equipment_landing_tile raises when self state is None."""
    world = _make_world(1000, 100, 100, 700)
    probe = _SimpleProbe(world)
    container = make_container_state(105, 105, False, 100, timestamp_ms=1000)
    equipment_targeting_module.get_terrain_map = lambda: _terrain({(100, 100)})
    probe._world["self_state"] = None

    with pytest.raises(EquipmentTargetingError, match="self state is unavailable"):
        find_visible_equipment_landing_tile(probe, container)
