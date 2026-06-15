"""Tests for EquipmentProbe result-builder delegation methods."""

from __future__ import annotations

from tests.action_lab.conftest import set_inventory_total

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.equipment_probe import EquipmentProbe
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.sniffer.world_state import get_world_state
from tankpit_bot.state import SelfStateDict, make_container_state, make_self_state
from tankpit_bot.types import CapturedMessage


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
    populates the global world state's ``self_state`` so that the real
    ``_require_self_state`` (which reads the global world state via
    ``get_self_state``) returns a valid value.
    """
    probe = EquipmentProbe.__new__(EquipmentProbe)
    probe._messages = [
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload="",
            ws_url="wss://test",
        )
    ]
    get_world_state()["self_state"] = _self_state()
    return probe


def test_build_attempt_result_delegates(real_inventory: None) -> None:
    """_build_attempt_result delegates to build_attempt_result_for_probe."""
    _ = real_inventory
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


def test_build_map_sync_timeout_result_delegates(real_inventory: None) -> None:
    """_build_map_sync_timeout_result delegates to the operations builder."""
    _ = real_inventory
    set_inventory_total(4)
    action_hooks.get_current_time_ms = _Clock(2000)
    probe = _make_probe()

    result = probe._build_map_sync_timeout_result(
        target=_target(),
        map_open_started_ms=1000,
        inventory_count_before=4,
        message_start_index=0,
        teleport_cycle_ids=[1],
    )

    assert result["status"] == "map_sync_timeout"
    assert result["completion_timestamp_ms"] == 2000


def test_build_teleport_timeout_result_delegates(real_inventory: None) -> None:
    """_build_teleport_timeout_result delegates to the operations builder."""
    _ = real_inventory
    set_inventory_total(2)
    probe = _make_probe()

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


def test_build_reposition_map_sync_timeout_result_delegates(
    real_inventory: None,
) -> None:
    """_build_reposition_map_sync_timeout_result delegates correctly."""
    _ = real_inventory
    set_inventory_total(1)
    action_hooks.get_current_time_ms = _Clock(2500)
    probe = _make_probe()
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


def test_build_reposition_teleport_timeout_result_delegates(
    real_inventory: None,
) -> None:
    """_build_reposition_teleport_timeout_result delegates correctly."""
    _ = real_inventory
    set_inventory_total(1)
    probe = _make_probe()
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


def test_build_radar_timeout_result_delegates(real_inventory: None) -> None:
    """_build_radar_timeout_result delegates to the operations builder."""
    _ = real_inventory
    set_inventory_total(2)
    action_hooks.get_current_time_ms = _Clock(2200)
    probe = _make_probe()

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


def test_build_no_equipment_visible_result_delegates(
    real_inventory: None,
) -> None:
    """_build_no_equipment_visible_result delegates to the operations builder."""
    _ = real_inventory
    set_inventory_total(2)
    action_hooks.get_current_time_ms = _Clock(2500)
    probe = _make_probe()

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
