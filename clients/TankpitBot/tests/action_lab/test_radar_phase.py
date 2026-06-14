"""Tests for shared radar command-phase helpers."""

from __future__ import annotations

from collections.abc import Generator
from typing import Literal

import pytest
from tests.action_lab._replay_core import ReplayClock

from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import radar_phase
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.state import WorldStateDict
from tankpit_bot.types import CapturedMessage


class _Page:
    """Minimal page fake for radar sync waits."""

    def wait_for_timeout(self, timeout: float) -> None:
        """Ignore wait requests."""
        _ = timeout


class _Probe:
    """Minimal radar probe fake."""

    def __init__(self, *, radar_result: bool = True) -> None:
        """Initialize the probe."""
        self._cdp_message_buffer: list[str] = []
        self._messages: list[CapturedMessage] = []
        self._cycles: list[ActionPhaseCycleDict] = []
        self._ended: list[ActionPhaseCycleDict] = []
        self._radar_result = radar_result
        self.reset_count = 0

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return captured messages."""
        return self._messages

    @property
    def magic(self) -> str:
        """Return a stable fake magic key."""
        return "magic"

    def get_world_state(self) -> WorldStateDict:
        """Raise because radar helper should not read world state directly."""
        raise AssertionError("unexpected world-state access")

    def use_radar(self) -> bool:
        """Return the configured radar dispatch outcome."""
        return self._radar_result

    def _start_action_phase(self, phase: str, *, attempt_label: str) -> ActionPhaseCycleDict:
        """Start one typed phase cycle."""
        cycle = ActionPhaseCycleDict(
            phase=_require_radar_phase_name(phase),
            cycle_id=len(self._cycles) + 1,
            started_ms=1100,
        )
        assert attempt_label == "attempt-1"
        self._cycles.append(cycle)
        return cycle

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """Record one closed phase."""
        self._ended.append(cycle)

    def _reset_probe_state_to_idle(self) -> None:
        """Record one idle reset."""
        self.reset_count += 1


def _require_radar_phase_name(phase: str) -> Literal["radar"]:
    """Return the only valid radar phase name."""
    if phase != "radar":
        raise AssertionError(f"unexpected phase {phase}")
    return "radar"


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore patched action hooks after each test."""
    original_clock = action_hooks.get_current_time_ms
    original_drain = action_hooks.drain_buffered_messages
    original_check = action_hooks.check_and_clear_radar_scan_complete
    original_wait = action_session.wait_for_radar_sync
    wait_attr = "wait_for_radar_sync"
    yield
    action_hooks.get_current_time_ms = original_clock
    action_hooks.drain_buffered_messages = original_drain
    action_hooks.check_and_clear_radar_scan_complete = original_check
    setattr(action_session, wait_attr, original_wait)


def test_clear_stale_radar_completion_drains_all_pending_flags() -> None:
    """Radar helper drains every stale completion flag before a new scan."""
    values = [True, True, False]

    def _check_complete() -> bool:
        return values.pop(0)

    action_hooks.check_and_clear_radar_scan_complete = _check_complete

    radar_phase.clear_stale_radar_completion()

    assert values == []


def test_run_tracked_radar_phase_waits_for_sync() -> None:
    """Radar helper dispatches once, waits for sync, and closes the phase."""
    clock = ReplayClock(1200)
    probe = _Probe()
    page = _Page()
    drain_calls: list[str] = []
    wait_attr = "wait_for_radar_sync"

    def _drain(source: BufferedMessageSourceProtocol) -> int:
        assert source is probe
        drain_calls.append("drain")
        return 0

    def _wait_for_radar_sync(
        page_arg: action_session.WaitPageProtocol,
        provider_arg: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        assert page_arg is page
        assert provider_arg is probe
        assert started_ms == 1200
        assert timeout_ms == 4000
        return 1650

    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = _drain
    action_hooks.check_and_clear_radar_scan_complete = lambda: False
    setattr(action_session, wait_attr, _wait_for_radar_sync)

    radar_cycle, radar_started_ms, radar_sync_timestamp_ms = radar_phase.run_tracked_radar_phase(
        page,
        probe,
        attempt_label="attempt-1",
        timeout_ms=4000,
        dispatch_failure_error=RuntimeError,
    )

    assert drain_calls == ["drain"]
    assert radar_cycle == ActionPhaseCycleDict(phase="radar", cycle_id=1, started_ms=1100)
    assert radar_started_ms == 1200
    assert radar_sync_timestamp_ms == 1650
    assert probe._ended == [radar_cycle]
    assert probe.reset_count == 1


def test_run_tracked_radar_phase_raises_on_dispatch_failure() -> None:
    """Radar helper raises immediately when dispatch fails."""
    clock = ReplayClock(1400)
    probe = _Probe(radar_result=False)
    page = _Page()
    wait_attr = "wait_for_radar_sync"

    def _wait_for_radar_sync(
        page_arg: action_session.WaitPageProtocol,
        provider_arg: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page_arg, provider_arg, started_ms, timeout_ms)
        return 0

    action_hooks.get_current_time_ms = clock
    action_hooks.drain_buffered_messages = lambda source: 0
    action_hooks.check_and_clear_radar_scan_complete = lambda: False
    setattr(action_session, wait_attr, _wait_for_radar_sync)

    with pytest.raises(RuntimeError, match="radar command dispatch failed"):
        radar_phase.run_tracked_radar_phase(
            page,
            probe,
            attempt_label="attempt-1",
            timeout_ms=2000,
            dispatch_failure_error=RuntimeError,
        )

    assert probe._ended == [ActionPhaseCycleDict(phase="radar", cycle_id=1, started_ms=1100)]
    assert probe.reset_count == 0
