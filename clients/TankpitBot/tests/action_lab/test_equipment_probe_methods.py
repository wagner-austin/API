"""Tests for EquipmentProbe result-builder delegation methods."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Protocol

import pytest
from tests.action_lab.conftest import set_inventory_total
from tests.fakes import InMemoryTerrainMap

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol, TerrainMapProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.action_trace import ActionCycleTracker
from tankpit_bot.action_lab.equipment_probe import (
    EquipmentProbe,
    _make_reposition_target,
)
from tankpit_bot.action_lab.equipment_targeting import (
    find_visible_equipment_landing_tile,
    visible_equipment_requires_reposition,
)
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import SelfStateDict, make_container_state, make_self_state
from tankpit_bot.state.types import make_viewport_state
from tankpit_bot.types import CapturedMessage


class _EquipmentTargetingModuleProtocol(Protocol):
    """Typed access to patchable equipment-targeting globals."""

    get_terrain_map: Callable[[], TerrainMapProtocol | None]


_equipment_targeting_import = __import__(
    "tankpit_bot.action_lab.equipment_targeting",
    fromlist=["equipment_targeting"],
)
equipment_targeting_module: _EquipmentTargetingModuleProtocol = _equipment_targeting_import


@pytest.fixture(autouse=True)
def _restore_targeting_hooks() -> Generator[None, None, None]:
    """Restore patched equipment-targeting hooks after each test."""
    original_drain = action_hooks.drain_buffered_messages
    yield
    action_hooks.drain_buffered_messages = original_drain


class _Clock:
    """Mutable millisecond clock."""

    def __init__(self, start_ms: int) -> None:
        self._now_ms = start_ms

    def __call__(self) -> int:
        return self._now_ms


def _self_state() -> SelfStateDict:
    return make_self_state(
        tank_id=1, x=100, y=100, team=2, rank=1, fuel=700, leaderboard_position=1
    )


def _target() -> TeleportTargetDict:
    return TeleportTargetDict(label="t", x=10, y=20)


def _teleport_result() -> TeleportAttemptResultDict:
    return TeleportAttemptResultDict(
        target=_target(),
        teleport_cycle_id=1,
        status="landed_exact",
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        completion_timestamp_ms=1500,
        map_sync_elapsed_ms=100,
        teleport_elapsed_ms=300,
        fuel_before=700,
        fuel_after=690,
        world_timestamp_before=1100,
        world_timestamp_after=1450,
        landed_signal_received=True,
        landed_x=10,
        landed_y=20,
        message_start_index=0,
        message_end_index=0,
        page_snapshots=[],
    )


def _make_probe() -> EquipmentProbe:
    """Create a minimal EquipmentProbe bypassing __init__.

    Sets ``_messages`` so the inherited ``messages`` property works, and
    populates the world state's ``self_state`` so that the real
    ``_require_self_state`` returns a valid value. ``world`` is bound
    explicitly because ``__new__`` skips ``SessionBase.__init__``, which
    is where a real probe gets its service
    ([[session-state-deglobalisation]] step 8).
    """
    ws = WorldService()
    probe = EquipmentProbe.__new__(EquipmentProbe)
    probe.world = ws
    probe._cdp_service = CDPService()
    probe._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload="",
            ws_url="wss://test",
        )
    ]
    ws.get_world_state()["self_state"] = _self_state()
    return probe


def test_build_attempt_result_delegates() -> None:
    """_build_attempt_result delegates to build_attempt_result_for_probe."""
    probe = _make_probe()
    container = make_container_state(11, 20, False, 0, timestamp_ms=2000)

    result = probe._build_attempt_result(
        target=_target(),
        status="picked_up_equipment",
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        radar_started_ms=1300,
        radar_sync_timestamp_ms=1400,
        pickup_started_ms=1500,
        completion_timestamp_ms=1600,
        inventory_count_before=4,
        inventory_count_after=5,
        landed_signal_received=True,
        landed_x=10,
        landed_y=20,
        equipment_target=container,
        message_start_index=0,
        teleport_cycle_ids=[1],
    )

    assert result["status"] == "picked_up_equipment"
    assert result["message_end_index"] == 1


def test_build_map_sync_timeout_result_delegates() -> None:
    """_build_map_sync_timeout_result delegates to the operations builder."""
    action_hooks.get_current_time_ms = _Clock(2000)
    probe = _make_probe()
    set_inventory_total(probe.world, 4)

    result = probe._build_map_sync_timeout_result(
        target=_target(),
        map_open_started_ms=1000,
        inventory_count_before=4,
        message_start_index=0,
        teleport_cycle_ids=[1],
    )

    assert result["status"] == "map_sync_timeout"
    assert result["completion_timestamp_ms"] == 2000


def test_build_teleport_timeout_result_delegates() -> None:
    """_build_teleport_timeout_result delegates to the operations builder."""
    probe = _make_probe()
    set_inventory_total(probe.world, 2)

    result = probe._build_teleport_timeout_result(
        target=_target(),
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        inventory_count_before=2,
        teleport_result=_teleport_result(),
        message_start_index=0,
        teleport_cycle_ids=[1],
    )

    assert result["status"] == "teleport_timeout"


def test_build_reposition_map_sync_timeout_result_delegates() -> None:
    """_build_reposition_map_sync_timeout_result delegates correctly."""
    action_hooks.get_current_time_ms = _Clock(2500)
    probe = _make_probe()
    set_inventory_total(probe.world, 1)
    container = make_container_state(11, 20, False, 0, timestamp_ms=2000)

    result = probe._build_reposition_map_sync_timeout_result(
        target=_target(),
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        radar_started_ms=1300,
        radar_sync_timestamp_ms=1400,
        reposition_map_open_started_ms=1500,
        inventory_count_before=1,
        teleport_result=_teleport_result(),
        equipment_target=container,
        message_start_index=0,
        teleport_cycle_ids=[1, 2],
        radar_cycle_id=3,
        phase_overlaps=[],
    )

    assert result["status"] == "reposition_map_sync_timeout"
    assert result["completion_timestamp_ms"] == 2500


def test_build_reposition_teleport_timeout_result_delegates() -> None:
    """_build_reposition_teleport_timeout_result delegates correctly."""
    probe = _make_probe()
    set_inventory_total(probe.world, 1)
    container = make_container_state(11, 20, False, 0, timestamp_ms=2000)

    result = probe._build_reposition_teleport_timeout_result(
        target=_target(),
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        radar_started_ms=1300,
        radar_sync_timestamp_ms=1400,
        reposition_map_open_started_ms=1500,
        reposition_map_sync_timestamp_ms=1550,
        reposition_teleport_started_ms=1600,
        inventory_count_before=1,
        teleport_result=_teleport_result(),
        equipment_target=container,
        message_start_index=0,
        teleport_cycle_ids=[1, 2],
        radar_cycle_id=3,
        phase_overlaps=[],
    )

    assert result["status"] == "reposition_teleport_timeout"


def test_build_radar_timeout_result_delegates() -> None:
    """_build_radar_timeout_result delegates to the operations builder."""
    action_hooks.get_current_time_ms = _Clock(2200)
    probe = _make_probe()
    set_inventory_total(probe.world, 2)

    result = probe._build_radar_timeout_result(
        target=_target(),
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        radar_started_ms=1300,
        inventory_count_before=2,
        teleport_result=_teleport_result(),
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        phase_overlaps=[],
    )

    assert result["status"] == "radar_timeout"
    assert result["completion_timestamp_ms"] == 2200


def test_build_no_equipment_visible_result_delegates() -> None:
    """_build_no_equipment_visible_result delegates to the operations builder."""
    action_hooks.get_current_time_ms = _Clock(2500)
    probe = _make_probe()
    set_inventory_total(probe.world, 2)

    result = probe._build_no_equipment_visible_result(
        target=_target(),
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        radar_started_ms=1300,
        radar_sync_timestamp_ms=1400,
        inventory_count_before=2,
        teleport_result=_teleport_result(),
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
        phase_overlaps=[],
    )

    assert result["status"] == "no_equipment_visible"
    assert result["completion_timestamp_ms"] == 2500


def test_requires_reposition_phase_bridge_delegates_to_targeting() -> None:
    """The reposition phase bridge runs the shared walk-reachability check."""

    probe = _make_probe()
    probe.world.get_world_state()["viewport"] = make_viewport_state(
        left=92, top=92, width=16, height=16
    )
    container = make_container_state(102, 100, False, 0, timestamp_ms=2000)
    probe.world.terrain_map = InMemoryTerrainMap.from_passable_set(
        {(100, 100), (101, 100), (102, 100)}
    )

    assert visible_equipment_requires_reposition(probe, container) is False


def test_landing_tile_phase_bridge_delegates_to_targeting() -> None:
    """The landing-tile phase bridge returns the shared landing computation."""
    probe = _make_probe()
    container = make_container_state(105, 105, False, 0, timestamp_ms=2000)
    probe.world.terrain_map = InMemoryTerrainMap.from_passable_set({(100, 100), (105, 105)})

    assert find_visible_equipment_landing_tile(probe, container) == (105, 105)


def test_make_reposition_target_labels_coordinates() -> None:
    """Reposition targets carry a coordinate-stamped label."""
    target = _make_reposition_target(105, 107)

    assert target == TeleportTargetDict(label="equipment_reposition_105_107", x=105, y=107)


def test_run_pickup_attempt_delegates_on_fast_path() -> None:
    """_run_pickup_attempt delegates to the shared pickup operation.

    The drain hook grows the inventory total, so the shared operation
    takes the fast path (no move dispatch, no waiting) and builds the
    picked-up result through the probe's builder plumbing.
    """
    action_hooks.get_current_time_ms = _Clock(3000)
    probe = _make_probe()
    set_inventory_total(probe.world, 0)
    probe._action_cycle_tracker = ActionCycleTracker()
    probe._attempt_phase_overlaps = []
    container = make_container_state(101, 100, False, 0, timestamp_ms=2000)

    def _growing_drain(source: BufferedMessageSourceProtocol, ws: WorldService) -> int:
        _ = source
        set_inventory_total(ws, 1)
        return 1

    action_hooks.drain_buffered_messages = _growing_drain

    class _NoWaitPage:
        def wait_for_timeout(self, timeout: float) -> None:
            raise AssertionError("fast path must not wait")

    result = probe._run_pickup_attempt(
        page=_NoWaitPage(),
        target=_target(),
        map_open_started_ms=1000,
        map_sync_timestamp_ms=1100,
        teleport_started_ms=1200,
        radar_started_ms=1300,
        radar_sync_timestamp_ms=1400,
        reposition_map_open_started_ms=None,
        reposition_map_sync_timestamp_ms=None,
        reposition_teleport_started_ms=None,
        pickup_timeout_ms=3000,
        inventory_count_before=0,
        teleport_result=_teleport_result(),
        equipment_target=container,
        message_start_index=0,
        teleport_cycle_ids=[1],
        radar_cycle_id=2,
    )

    assert result["status"] == "picked_up_equipment"
    assert result["inventory_count_after"] == 1
