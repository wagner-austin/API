"""Tests for equipment-probe operations.

``test_equipment_probe_operations.py`` was 668 lines; the pickup and
radar phases are now a sibling.
"""

from __future__ import annotations

from tests.action_lab._equipment_operations_harness import (
    _BuilderProbe,
    _Clock,
    _target,
    _teleport_result,
)
from tests.action_lab.conftest import (
    set_inventory_total,
)

from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.equipment_probe_operations import (
    build_attempt_result_for_probe,
    build_map_sync_timeout_result_for_probe,
    build_no_equipment_visible_result_for_probe,
    build_radar_timeout_result_for_probe,
    build_reposition_map_sync_timeout_result_for_probe,
    build_reposition_teleport_timeout_result_for_probe,
    build_teleport_timeout_result_for_probe,
    effective_equipment_pickup_timeout_ms,
    finalize_attempt_delay,
)
from tankpit_bot.state import (
    make_container_state,
)
from tankpit_bot.types import CapturedMessage


def test_finalize_attempt_delay_skips_zero_delay() -> None:
    """A zero-or-negative delay never calls wait_for_timeout."""

    class _Page:
        def __init__(self) -> None:
            self.calls = 0

        def wait_for_timeout(self, timeout: float) -> None:
            self.calls += 1
            _ = timeout

        def set_content(self, html: str, *, timeout: float | None = None) -> None:
            _ = (html, timeout)

        def route(self, url: str, handler: RouteFulfillHandler) -> None:
            _ = (url, handler)

    page = _Page()
    finalize_attempt_delay(page, settle_delay_ms=0)
    finalize_attempt_delay(page, settle_delay_ms=200)

    assert page.calls == 1


def test_effective_pickup_timeout_scales_with_distance() -> None:
    """Pickup timeout grows with travel distance and never shrinks below base."""
    assert (
        effective_equipment_pickup_timeout_ms(
            current_x=100,
            current_y=100,
            target_x=101,
            target_y=100,
            base_timeout_ms=3000,
        )
        == 3000
    )
    assert (
        effective_equipment_pickup_timeout_ms(
            current_x=160,
            current_y=80,
            target_x=160,
            target_y=86,
            base_timeout_ms=3000,
        )
        == 4000
    )


def test_build_attempt_result_for_probe_uses_message_count() -> None:
    """Builders read message_end_index from the probe message log."""
    probe = _BuilderProbe(
        messages=[
            CapturedMessage(
                timestamp_ms=1000,
                direction="received",
                payload="",
                ws_url="wss://test",
            )
        ]
    )
    container = make_container_state(11, 20, False, 0, timestamp_ms=2000)

    result = build_attempt_result_for_probe(
        probe,
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

    assert result["message_end_index"] == 1


def test_build_map_sync_timeout_result_for_probe_uses_inventory_hook(
    real_inventory: None,
) -> None:
    """Map-sync-timeout reads the latest inventory total from the real tracker."""
    _ = real_inventory
    set_inventory_total(4)
    action_hooks.get_current_time_ms = _Clock(2000)
    probe = _BuilderProbe()

    result = build_map_sync_timeout_result_for_probe(
        probe,
        target=_target(),
        map_open_started_ms=1000,
        inventory_count_before=4,
        message_start_index=0,
        teleport_cycle_ids=[1],
    )

    assert result["status"] == "map_sync_timeout"
    assert result["inventory_count_after"] == 4
    assert result["completion_timestamp_ms"] == 2000


def test_build_teleport_timeout_result_for_probe_uses_inventory_hook(
    real_inventory: None,
) -> None:
    """Teleport-timeout reads the latest inventory total from the real tracker."""
    _ = real_inventory
    set_inventory_total(2)
    probe = _BuilderProbe()

    result = build_teleport_timeout_result_for_probe(
        probe,
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
    assert result["inventory_count_after"] == 2


def test_build_reposition_map_sync_timeout_result_for_probe_uses_inventory_hook(
    real_inventory: None,
) -> None:
    """Reposition map-sync-timeout reads the latest inventory total."""
    _ = real_inventory
    set_inventory_total(1)
    action_hooks.get_current_time_ms = _Clock(2500)
    probe = _BuilderProbe()
    container = make_container_state(11, 20, False, 0, timestamp_ms=2000)

    result = build_reposition_map_sync_timeout_result_for_probe(
        probe,
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


def test_build_reposition_teleport_timeout_result_for_probe_uses_inventory_hook(
    real_inventory: None,
) -> None:
    """Reposition teleport-timeout reads the latest inventory total."""
    _ = real_inventory
    set_inventory_total(1)
    probe = _BuilderProbe()
    container = make_container_state(11, 20, False, 0, timestamp_ms=2000)

    result = build_reposition_teleport_timeout_result_for_probe(
        probe,
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


def test_build_radar_timeout_result_for_probe_uses_inventory_hook(
    real_inventory: None,
) -> None:
    """Radar-timeout reads the latest inventory total."""
    _ = real_inventory
    set_inventory_total(2)
    action_hooks.get_current_time_ms = _Clock(2200)
    probe = _BuilderProbe()

    result = build_radar_timeout_result_for_probe(
        probe,
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


def test_build_no_equipment_visible_result_for_probe_uses_inventory_hook(
    real_inventory: None,
) -> None:
    """No-visible-equipment reads the latest inventory total."""
    _ = real_inventory
    set_inventory_total(2)
    action_hooks.get_current_time_ms = _Clock(2500)
    probe = _BuilderProbe()

    result = build_no_equipment_visible_result_for_probe(
        probe,
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
