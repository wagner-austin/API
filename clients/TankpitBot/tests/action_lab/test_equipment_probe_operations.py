"""Tests for equipment probe operations.

Uses real inventory tracker mutations (via the shared ``real_inventory`` and
``replay_pipeline`` fixtures in conftest) rather than patching
``get_inventory_state`` at module-level. The previous patching pattern was
fragile because the symbol is imported into multiple modules and patching one
binding leaves the others reading the real tracker — that mismatch caused
``test_run_pickup_attempt_for_probe_completes_immediately_when_inventory_grew``
to hang indefinitely in ``make check``.
"""

from __future__ import annotations

from platform_core.json_utils import JSONValue
from tests.action_lab.conftest import (
    INVENTORY_GROWTH_FRAME_INDEX,
    INVENTORY_TOTAL_AFTER_GROWTH,
    FailIfWaitedPage,
    set_inventory_total,
)

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict, ActionPhaseOverlapDict
from tankpit_bot.action_lab.equipment_pickup import total_inventory_count
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
    run_pickup_attempt_for_probe,
)
from tankpit_bot.action_lab.types import TeleportAttemptResultDict, TeleportTargetDict
from tankpit_bot.sniffer.decoders import process_received_message
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_inventory import get_inventory_state
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_container_state,
    make_empty_world_state,
    make_self_state,
)
from tankpit_bot.state.types import ContainerStateDict, make_viewport_state
from tankpit_bot.types import CapturedMessage


def _self_state() -> SelfStateDict:
    """Build a sample self state."""
    return make_self_state(
        tank_id=1, x=100, y=100, team=2, rank=1, fuel=700, leaderboard_position=1
    )


def _make_world() -> WorldStateDict:
    """Build a minimal world state."""
    base = make_empty_world_state()
    return WorldStateDict(
        self_state=_self_state(),
        tanks=base["tanks"],
        containers=base["containers"],
        mines=base["mines"],
        terrain=base["terrain"],
        viewport=make_viewport_state(left=92, top=92, width=16, height=16),
        scanned_tiles=base["scanned_tiles"],
        timestamp_ms=2000,
    )


def _target() -> TeleportTargetDict:
    """Build a sample teleport target."""
    return TeleportTargetDict(label="t", x=10, y=20)


def _teleport_result() -> TeleportAttemptResultDict:
    """Build a successful landed teleport result."""
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


class _Clock:
    """Mutable millisecond clock."""

    def __init__(self, start_ms: int) -> None:
        self._now_ms = start_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class _BuilderProbe:
    """Minimal probe satisfying the builder context protocol."""

    def __init__(
        self,
        *,
        messages: list[CapturedMessage] | None = None,
        self_state: SelfStateDict | None = None,
    ) -> None:
        self._messages = messages if messages is not None else []
        self._self_state = self_state if self_state is not None else _self_state()

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages

    def _require_self_state(self) -> SelfStateDict:
        return self._self_state


class _PickupProbe:
    """Minimal probe satisfying the equipment pickup context."""

    def __init__(
        self,
        *,
        clock: _Clock,
        move_result: bool,
    ) -> None:
        self._clock = clock
        self._messages: list[CapturedMessage] = []
        self._self_state = _self_state()
        self._world = _make_world()
        self._cycles: list[ActionPhaseCycleDict] = []
        self._cycle_id = 0
        self.move_result = move_result
        self.move_calls: list[tuple[int, int]] = []
        self.reset_calls = 0
        self._overlaps: list[ActionPhaseOverlapDict] = []
        self._cdp_message_buffer: list[str] = []

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._messages

    @property
    def magic(self) -> str | None:
        return None

    def get_world_state(self) -> WorldStateDict:
        return self._world

    def _require_self_state(self) -> SelfStateDict:
        return self._self_state

    def move_to(self, x: int, y: int) -> bool:
        self.move_calls.append((x, y))
        return self.move_result

    def _start_action_phase(
        self,
        phase: str,
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        _ = attempt_label
        self._cycle_id += 1
        if phase == "move":
            named: ActionPhaseCycleDict = ActionPhaseCycleDict(
                phase="move",
                cycle_id=self._cycle_id,
                started_ms=self._clock(),
            )
        else:
            named = ActionPhaseCycleDict(
                phase="pickup",
                cycle_id=self._cycle_id,
                started_ms=self._clock(),
            )
        self._cycles.append(named)
        return named

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        _ = cycle

    def _reset_probe_state_to_idle(self) -> None:
        self.reset_calls += 1

    def _get_attempt_phase_overlaps(self) -> list[ActionPhaseOverlapDict]:
        return list(self._overlaps)


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


def test_run_pickup_attempt_takes_fast_path_against_real_inventory_frame(
    replay_pipeline: list[dict[str, JSONValue]],
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
    messages = replay_pipeline

    for msg in messages[:INVENTORY_GROWTH_FRAME_INDEX]:
        if msg["direction"] == "received":
            process_received_message(str(msg["payload"]))

    assert total_inventory_count(get_inventory_state(get_world_service())) == 0

    growth_frame = messages[INVENTORY_GROWTH_FRAME_INDEX]
    assert growth_frame["direction"] == "received"
    growth_payload = growth_frame["payload"]

    drain_calls: list[BufferedMessageSourceProtocol] = []

    def _real_drain(probe: BufferedMessageSourceProtocol) -> int:
        drain_calls.append(probe)
        process_received_message(str(growth_payload))
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
    replay_pipeline: list[dict[str, JSONValue]],
) -> None:
    """Slow-path against real captured bytes.

    Schedule of drains across the function:
    1. Line 473 (pre-move drain) — no-op, inventory stays at 0.
    2. Wait-loop iter 1 — no-op, inventory still 0, page.wait_for_timeout(100).
    3. Wait-loop iter 2 — process the real growth frame, inventory 0 -> 112.
    """
    messages = replay_pipeline

    for msg in messages[:INVENTORY_GROWTH_FRAME_INDEX]:
        if msg["direction"] == "received":
            process_received_message(str(msg["payload"]))
    assert total_inventory_count(get_inventory_state(get_world_service())) == 0

    growth_payload = messages[INVENTORY_GROWTH_FRAME_INDEX]["payload"]
    drain_queue: list[str | None] = [None, None, str(growth_payload)]

    def _scheduled_drain(probe: BufferedMessageSourceProtocol) -> int:
        if not drain_queue:
            return 0
        payload = drain_queue.pop(0)
        if payload is None:
            return 0
        process_received_message(payload)
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


def test_run_pickup_attempt_for_probe_raises_when_move_dispatch_fails(
    real_inventory: None,
) -> None:
    """A failed move_to raises the configured dispatch error.

    Real inventory tracker is reset to empty, so the immediate-completion
    check sees 0 > inventory_count_before=4 as False, falls into the move
    dispatch, which the probe rejects.
    """
    _ = real_inventory
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


_ContainerAlias = ContainerStateDict
