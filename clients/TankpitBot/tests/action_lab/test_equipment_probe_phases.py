"""Tests for the equipment pickup and radar phases."""

from __future__ import annotations

from tests.action_lab._equipment_operations_harness import (
    _Clock,
    _PickupProbe,
    _target,
    _teleport_result,
)
from tests.action_lab.conftest import (
    INVENTORY_GROWTH_FRAME_INDEX,
    INVENTORY_TOTAL_AFTER_GROWTH,
    FailIfWaitedPage,
    ReplayPipeline,
)

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.equipment_pickup import total_inventory_count
from tankpit_bot.action_lab.equipment_probe_operations import (
    run_pickup_attempt_for_probe,
)
from tankpit_bot.sniffer.decoders import process_received_message
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.state import (
    make_container_state,
)


def test_run_pickup_attempt_takes_fast_path_against_real_inventory_frame(
    replay_pipeline: ReplayPipeline,
) -> None:
    """Real captured 0x49 frame, real decoders, real inventory tracker.

    Replays every received frame from fuel_probe.capture_session.json up to
    (not including) the first frame that grows inventory total. Asserts the
    real tracker shows total == 0, then injects a real drain that processes
    the next frame via process_received_message — the real XOR + decoder +
    world_state_inventory pipeline. Asserts run_pickup_attempt_for_probe
    takes the fast path: no move dispatched, no waiter entered, result
    inventory_count_after matches the real tracker's new total.
    """
    messages = replay_pipeline.messages
    xor_table = replay_pipeline.xor_table

    for msg in messages[:INVENTORY_GROWTH_FRAME_INDEX]:
        if msg["direction"] == "received":
            process_received_message(get_world_service(), str(msg["payload"]), xor_table)

    assert total_inventory_count(get_inventory_state(get_world_service())) == 0

    growth_frame = messages[INVENTORY_GROWTH_FRAME_INDEX]
    assert growth_frame["direction"] == "received"
    growth_payload = growth_frame["payload"]

    drain_calls: list[BufferedMessageSourceProtocol] = []

    def _real_drain(probe: BufferedMessageSourceProtocol) -> int:
        drain_calls.append(probe)
        process_received_message(get_world_service(), str(growth_payload), xor_table)
        return 1

    action_hooks.drain_buffered_messages = _real_drain
    action_hooks.get_current_time_ms = _Clock(10_000)

    probe = _PickupProbe(clock=_Clock(10_000), move_result=True)
    container = make_container_state(101, 100, False, 0, timestamp_ms=10_000)

    result = run_pickup_attempt_for_probe(
        probe,
        page=FailIfWaitedPage(),
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
        dispatch_failure_error=RuntimeError,
        dispatch_failure_message="dispatch failed",
    )

    assert (
        total_inventory_count(get_inventory_state(get_world_service()))
        == INVENTORY_TOTAL_AFTER_GROWTH
    )
    assert result["status"] == "picked_up_equipment"
    assert result["inventory_count_after"] == INVENTORY_TOTAL_AFTER_GROWTH
    assert probe.move_calls == []
    assert probe.reset_calls == 1
    assert len(drain_calls) == 1


def test_run_pickup_attempt_dispatches_move_and_polls_against_real_inventory_frame(
    replay_pipeline: ReplayPipeline,
) -> None:
    """Slow-path against real captured bytes.

    Schedule of drains across the function:
    1. Line 473 (pre-move drain) — no-op, inventory stays at 0.
    2. Wait-loop iter 1 — no-op, inventory still 0, page.wait_for_timeout(100).
    3. Wait-loop iter 2 — process the real growth frame, inventory 0 -> 112.
    """
    messages = replay_pipeline.messages
    xor_table = replay_pipeline.xor_table

    for msg in messages[:INVENTORY_GROWTH_FRAME_INDEX]:
        if msg["direction"] == "received":
            process_received_message(get_world_service(), str(msg["payload"]), xor_table)
    assert total_inventory_count(get_inventory_state(get_world_service())) == 0

    growth_payload = messages[INVENTORY_GROWTH_FRAME_INDEX]["payload"]
    drain_queue: list[str | None] = [None, None, str(growth_payload)]

    def _scheduled_drain(probe: BufferedMessageSourceProtocol) -> int:
        if not drain_queue:
            return 0
        payload = drain_queue.pop(0)
        if payload is None:
            return 0
        process_received_message(get_world_service(), payload, xor_table)
        return 1

    action_hooks.drain_buffered_messages = _scheduled_drain

    clock = _Clock(10_000)
    action_hooks.get_current_time_ms = clock

    probe = _PickupProbe(clock=clock, move_result=True)
    container = make_container_state(101, 100, False, 0, timestamp_ms=10_000)

    class _AdvancingPage:
        def __init__(self) -> None:
            self.waits: list[float] = []

        def wait_for_timeout(self, timeout: float) -> None:
            self.waits.append(timeout)
            clock.advance(int(timeout))

        def set_content(self, html: str, *, timeout: float | None = None) -> None:
            _ = (html, timeout)

        def route(self, url: str, handler: RouteFulfillHandler) -> None:
            _ = (url, handler)

    page = _AdvancingPage()
    result = run_pickup_attempt_for_probe(
        probe,
        page=page,
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
        dispatch_failure_error=RuntimeError,
        dispatch_failure_message="dispatch failed",
    )

    assert (
        total_inventory_count(get_inventory_state(get_world_service()))
        == INVENTORY_TOTAL_AFTER_GROWTH
    )
    assert result["status"] == "picked_up_equipment"
    assert result["inventory_count_after"] == INVENTORY_TOTAL_AFTER_GROWTH
    assert probe.move_calls == [(101, 100)]
    assert probe.reset_calls == 1
    assert page.waits == [100.0]


def test_run_pickup_attempt_for_probe_raises_when_move_dispatch_fails() -> None:
    """A failed move_to raises the configured dispatch error.

    Real inventory tracker is reset to empty, so the immediate-completion
    check sees 0 > inventory_count_before=4 as False, falls into the move
    dispatch, which the probe rejects.
    """
    clock = _Clock(2000)
    action_hooks.get_current_time_ms = clock
    probe = _PickupProbe(clock=clock, move_result=False)
    container = make_container_state(101, 100, False, 0, timestamp_ms=2000)

    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="dispatch failed"):
        run_pickup_attempt_for_probe(
            probe,
            page=FailIfWaitedPage(),
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
            inventory_count_before=4,
            teleport_result=_teleport_result(),
            equipment_target=container,
            message_start_index=0,
            teleport_cycle_ids=[1],
            radar_cycle_id=2,
            dispatch_failure_error=RuntimeError,
            dispatch_failure_message="dispatch failed",
        )
