"""Tests for equipment-pickup completion helpers.

Uses the real inventory tracker via ``set_inventory_total`` and
``update_inventory_from_gain`` (both routing through the real
``world_state_inventory`` mutators) rather than patching
``pickup_module.get_inventory_state``. The previous patching pattern only
worked here because this is the one module the helper happens to read from
directly — symmetric helpers in ``equipment_probe_operations.py`` import the
same name and were not patched, which caused
``test_run_pickup_attempt_for_probe_completes_immediately_when_inventory_grew``
to hang. Switching to real-tracker mutation removes the patching surface.
"""

from __future__ import annotations

from collections.abc import Callable

from tests.action_lab.conftest import set_inventory_total

from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.equipment_pickup import (
    EquipmentPickupError,
    get_completed_equipment_pickup_outcome,
    total_inventory_count,
    wait_for_equipment_pickup_outcome,
)
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_inventory import update_inventory_from_gain
from tankpit_bot.state import WorldStateDict, make_empty_world_state, make_self_state
from tankpit_bot.state.types import make_viewport_state
from tankpit_bot.types import CapturedMessage


def _inventory(
    *, armor: int = 0, dual: int = 0, missile: int = 0, homing: int = 0, radar: int = 0
) -> InventoryState:
    """Build a literal inventory state for total_inventory_count's pure unit test."""
    return InventoryState(
        armor_shields=InventoryItem(count=armor, enabled=True),
        dual_shots=InventoryItem(count=dual, enabled=True),
        missile_shots=InventoryItem(count=missile, enabled=True),
        homing_shots=InventoryItem(count=homing, enabled=True),
        extra_radars=InventoryItem(count=radar, enabled=True),
    )


class _Clock:
    """Mutable millisecond clock."""

    def __init__(self, start_ms: int) -> None:
        self._now_ms = start_ms

    def __call__(self) -> int:
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        self._now_ms += delta_ms


class _FakeProbe:
    """Minimal buffered-world-state provider used by pickup helpers."""

    def __init__(self, world: WorldStateDict) -> None:
        self.world = get_world_service()
        self._world = world
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None
        self._captured_messages: list[CapturedMessage] = []

    def get_world_state(self) -> WorldStateDict:
        return self._world

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._captured_messages

    @property
    def magic(self) -> str | None:
        return None


class _FakePage:
    """Page fake whose waits advance a synthetic clock and run a hook."""

    def __init__(self, clock: _Clock, hook: Callable[[], None]) -> None:
        self._clock = clock
        self._hook = hook
        self.waits: list[float] = []

    def wait_for_timeout(self, timeout: float) -> None:
        self.waits.append(timeout)
        self._clock.advance(int(timeout))
        self._hook()

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)


def _make_world(timestamp_ms: int) -> WorldStateDict:
    base = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=700,
            leaderboard_position=1,
        ),
        tanks=base["tanks"],
        containers=base["containers"],
        mines=base["mines"],
        terrain=base["terrain"],
        viewport=make_viewport_state(left=92, top=92, width=16, height=16),
        scanned_tiles=base["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


def test_total_inventory_count_sums_all_slots() -> None:
    """Pure unit test: total counts include every inventory slot."""
    state = _inventory(armor=1, dual=2, missile=3, homing=4, radar=5)

    assert total_inventory_count(state) == 15


def test_get_completed_outcome_returns_none_when_inventory_unchanged() -> None:
    """No completed outcome is reported while the inventory is unchanged."""
    set_inventory_total(2)
    probe = _FakeProbe(_make_world(1000))

    result = get_completed_equipment_pickup_outcome(
        probe,
        target_x=121,
        target_y=100,
        inventory_count_before=2,
    )

    assert result is None


def test_get_completed_outcome_returns_outcome_when_inventory_grows() -> None:
    """A completed outcome is reported once the real inventory total grows."""
    clock = _Clock(2000)
    action_hooks.get_current_time_ms = clock
    set_inventory_total(3)
    probe = _FakeProbe(_make_world(1000))

    result = get_completed_equipment_pickup_outcome(
        probe,
        target_x=121,
        target_y=100,
        inventory_count_before=2,
    )

    assert result == ("picked_up_equipment", 2000, 3)


def test_wait_returns_immediately_when_inventory_already_grown() -> None:
    """The waiter returns the outcome without sleeping when already grown."""
    clock = _Clock(2000)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda provider, ws: 0
    set_inventory_total(3)
    probe = _FakeProbe(_make_world(1000))
    page = _FakePage(clock, hook=lambda: None)

    status, completed_ms, total = wait_for_equipment_pickup_outcome(
        page,
        probe,
        target_x=121,
        target_y=100,
        pickup_started_ms=2000,
        inventory_count_before=2,
        timeout_ms=1000,
    )

    assert (status, completed_ms, total) == ("picked_up_equipment", 2000, 3)
    assert page.waits == []


def test_wait_polls_until_inventory_grows() -> None:
    """The waiter polls the real inventory until it observes growth.

    The page-wait hook simulates a 0x67 equip_gain frame arriving between the
    2nd and 3rd poll: it grows the real inventory via update_inventory_from_gain.
    """
    clock = _Clock(2000)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda provider, ws: 0
    set_inventory_total(2)

    wait_counter = {"count": 0}

    def _grow_inventory_on_second_wait() -> None:
        wait_counter["count"] += 1
        if wait_counter["count"] == 2:
            update_inventory_from_gain(get_world_service(), [0, 1, 0, 0, 0])

    probe = _FakeProbe(_make_world(1000))
    page = _FakePage(clock, hook=_grow_inventory_on_second_wait)

    status, completed_ms, total = wait_for_equipment_pickup_outcome(
        page,
        probe,
        target_x=121,
        target_y=100,
        pickup_started_ms=2000,
        inventory_count_before=2,
        timeout_ms=2000,
    )

    assert (status, completed_ms, total) == ("picked_up_equipment", 2200, 3)
    assert page.waits == [100.0, 100.0]


def test_wait_returns_pickup_timeout_when_budget_exhausts() -> None:
    """The waiter reports a pickup timeout when the budget elapses."""
    clock = _Clock(2000)
    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda provider, ws: 0
    set_inventory_total(2)
    probe = _FakeProbe(_make_world(1000))
    page = _FakePage(clock, hook=lambda: None)

    status, completed_ms, total = wait_for_equipment_pickup_outcome(
        page,
        probe,
        target_x=121,
        target_y=100,
        pickup_started_ms=2000,
        inventory_count_before=2,
        timeout_ms=300,
    )

    assert status == "pickup_timeout"
    assert completed_ms >= 2300
    assert total == 2


def test_equipment_pickup_error_is_exposed() -> None:
    """The custom pickup error class is constructible and stringifies its message."""
    err = EquipmentPickupError("boom")

    assert str(err) == "boom"
    assert type(err).__mro__[1] is Exception
