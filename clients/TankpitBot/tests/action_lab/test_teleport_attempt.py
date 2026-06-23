"""Tests for shared tracked teleport-attempt orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
from typing_extensions import Unpack

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab import teleport_attempt
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.teleport_phase import (
    TeleportOutcomeWaiterKwargs,
    TeleportOutcomeWaiterProtocol,
    TeleportPhaseProbeProtocol,
)
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)
from tankpit_bot.state import WorldStateDict, make_empty_world_state
from tankpit_bot.types import CapturedMessage


class _Probe:
    """Minimal probe fake for tracked teleport-attempt tests."""

    def __init__(self) -> None:
        """Initialize the fake probe."""
        self._messages = [
            CapturedMessage(
                direction="sent",
                payload="a",
                timestamp_ms=1,
                ws_url="wss://example.test/ws/",
            )
        ]
        self.started_cycles: list[tuple[str, str]] = []
        self.reset_idle_calls = 0
        self._cdp_message_buffer: list[str] = []

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return the captured message buffer."""
        return self._messages

    @property
    def magic(self) -> str:
        """Return a stable fake magic key."""
        return "magic"

    def get_world_state(self) -> WorldStateDict:
        """Return an empty world state."""
        return make_empty_world_state()

    def teleport_to(self, x: int, y: int) -> bool:
        """Reject direct teleport dispatch in this test layer."""
        _ = (x, y)
        raise AssertionError("teleport_to should not be called directly by this test")

    def _end_action_phase(self, cycle: ActionPhaseCycleDict) -> None:
        """Reject direct phase ending in this test layer."""
        _ = cycle
        raise AssertionError("_end_action_phase should not be called directly by this test")

    def _reset_probe_state_to_idle(self) -> None:
        """Record one idle reset."""
        self.reset_idle_calls += 1

    def _start_action_phase(
        self,
        phase: Literal["teleport"],
        *,
        attempt_label: str,
    ) -> ActionPhaseCycleDict:
        """Return one started teleport phase."""
        self.started_cycles.append((phase, attempt_label))
        return ActionPhaseCycleDict(phase="teleport", cycle_id=7, started_ms=1200)


class _Page:
    """Minimal page fake satisfying the wait protocol."""

    def wait_for_timeout(self, timeout: float) -> None:
        """Ignore wait requests."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)


def _target() -> TeleportTargetDict:
    """Build one sample teleport target."""
    return TeleportTargetDict(label="target", x=147, y=110)


def _snapshot(
    phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
) -> TeleportPageSnapshotDict:
    """Build one sample page snapshot."""
    return TeleportPageSnapshotDict(
        phase=phase,
        timestamp_ms=1000,
        client_present=True,
        map_visible=False,
        client_state=1,
        client_busy=False,
        pending_actions=0,
        heartbeat_age_ms=1,
        last_page_client_send_age_ms=2,
        last_bot_send_age_ms=3,
        ws_ready_state=1,
        current_send_label=None,
        sent_frame_meta_queue_length=0,
        self_fields={},
        world_fields={},
        world_collections={},
        map_fields={},
    )


def _result(target: TeleportTargetDict) -> TeleportAttemptResultDict:
    """Build one sample teleport result."""
    return TeleportAttemptResultDict(
        target=target,
        teleport_cycle_id=7,
        status="landed_exact",
        map_open_started_ms=1500,
        map_sync_timestamp_ms=1700,
        teleport_started_ms=1800,
        completion_timestamp_ms=2200,
        map_sync_elapsed_ms=200,
        teleport_elapsed_ms=400,
        fuel_before=1100,
        fuel_after=1004,
        world_timestamp_before=900,
        world_timestamp_after=2100,
        landed_signal_received=True,
        landed_x=147,
        landed_y=110,
        message_start_index=1,
        message_end_index=5,
        page_snapshots=[_snapshot("before_map_open"), _snapshot("landed")],
    )


class _WaitForOutcome(TeleportOutcomeWaiterProtocol):
    """Typed teleport-outcome waiter returning a stable sample result."""

    def __call__(
        self,
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        **kwargs: Unpack[TeleportOutcomeWaiterKwargs],
    ) -> TeleportAttemptResultDict:
        """Return one stable teleport result for typed helper calls."""
        _ = (page, provider, kwargs)
        return _result(target)


def test_run_tracked_teleport_attempt_runs_acquisition_then_teleport() -> None:
    """Shared helper returns full tracked state for a successful attempt."""
    original_acquisition = teleport_attempt.run_acquisition_phase
    original_teleport = teleport_attempt.run_teleport_phase
    expected_page = _Page()
    expected_probe = _Probe()
    expected_target = _target()
    dispatch_calls: list[str] = []
    teleport_calls: list[int] = []

    def _dispatch() -> bool:
        dispatch_calls.append("acquire")
        return True

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return _snapshot(phase)

    def _run_acquisition(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        *,
        cdp: CDPSessionProtocol | None,
        send_command: Callable[[], bool],
        command_name: str,
        capture_before_map_open: bool,
        wait_for_sync: bool,
        sync_timeout_ms: int,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
    ) -> tuple[
        int,
        int | None,
        list[TeleportPageSnapshotDict],
        Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ]:
        _ = (
            cdp,
            command_name,
            capture_before_map_open,
            wait_for_sync,
            sync_timeout_ms,
            dispatch_failure_error,
            dispatch_failure_message,
            unavailable_error,
            unavailable_message,
        )
        assert page is expected_page
        assert provider is expected_probe
        assert send_command()
        return (1500, 1700, [_snapshot("before_map_open")], _capture_page_snapshot)

    def _run_teleport(
        page: action_session.WaitPageProtocol,
        probe: TeleportPhaseProbeProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle: ActionPhaseCycleDict,
        message_start_index: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "",
    ) -> tuple[TeleportAttemptResultDict, int]:
        _ = (
            teleport_cycle,
            map_open_started_ms,
            map_sync_timestamp_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            capture_page_snapshot,
            wait_for_outcome,
            dispatch_failure_error,
            dispatch_failure_message,
        )
        assert page is expected_page
        assert probe is expected_probe
        assert target == expected_target
        assert message_start_index == 1
        assert [snapshot["phase"] for snapshot in page_snapshots] == ["before_map_open"]
        teleport_calls.append(message_start_index)
        return (_result(target), 1800)

    teleport_attempt.run_acquisition_phase = _run_acquisition
    teleport_attempt.run_teleport_phase = _run_teleport
    try:
        attempt = teleport_attempt.run_tracked_teleport_attempt(
            expected_page,
            expected_probe,
            expected_target,
            cdp=None,
            attempt_label=expected_target["label"],
            fuel_before=1100,
            world_timestamp_before=900,
            send_acquisition_command=_dispatch,
            acquisition_command_name="map_open",
            capture_before_map_open=True,
            wait_for_acquisition_sync=True,
            acquisition_timeout_ms=4000,
            teleport_timeout_ms=10000,
            wait_for_outcome=_WaitForOutcome(),
            dispatch_failure_error=RuntimeError,
            acquisition_dispatch_failure_message="acquisition failed",
            teleport_dispatch_failure_message="teleport failed",
            unavailable_error=RuntimeError,
            unavailable_message="missing",
            unexpected_result_error=RuntimeError,
            unexpected_result_message="impossible",
        )
    finally:
        teleport_attempt.run_acquisition_phase = original_acquisition
        teleport_attempt.run_teleport_phase = original_teleport

    assert expected_probe.reset_idle_calls == 1
    assert expected_probe.started_cycles == [("teleport", "target")]
    assert dispatch_calls == ["acquire"]
    assert teleport_calls == [1]
    assert attempt.message_start_index == 1
    assert attempt.teleport_cycle["cycle_id"] == 7
    assert attempt.acquisition_started_ms == 1500
    assert attempt.acquisition_sync_timestamp_ms == 1700
    assert attempt.teleport_started_ms == 1800
    assert attempt.teleport_result == _result(expected_target)


def test_run_tracked_teleport_attempt_returns_early_on_acquisition_timeout() -> None:
    """Shared helper returns without teleport dispatch when acquisition times out."""
    original_acquisition = teleport_attempt.run_acquisition_phase
    original_teleport = teleport_attempt.run_teleport_phase
    expected_page = _Page()
    expected_probe = _Probe()
    expected_target = _target()

    def _dispatch() -> bool:
        return True

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return _snapshot(phase)

    def _run_acquisition(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        *,
        cdp: CDPSessionProtocol | None,
        send_command: Callable[[], bool],
        command_name: str,
        capture_before_map_open: bool,
        wait_for_sync: bool,
        sync_timeout_ms: int,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
    ) -> tuple[
        int,
        int | None,
        list[TeleportPageSnapshotDict],
        Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ]:
        _ = (
            cdp,
            command_name,
            capture_before_map_open,
            wait_for_sync,
            sync_timeout_ms,
            dispatch_failure_error,
            dispatch_failure_message,
            unavailable_error,
            unavailable_message,
        )
        assert page is expected_page
        assert provider is expected_probe
        assert send_command()
        return (1500, None, [_snapshot("before_map_open")], _capture_page_snapshot)

    def _run_teleport(
        page: action_session.WaitPageProtocol,
        probe: TeleportPhaseProbeProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle: ActionPhaseCycleDict,
        message_start_index: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "",
    ) -> tuple[TeleportAttemptResultDict, int]:
        _ = (
            page,
            probe,
            target,
            teleport_cycle,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
            wait_for_outcome,
            dispatch_failure_error,
            dispatch_failure_message,
        )
        raise AssertionError("teleport dispatch should not run after acquisition timeout")

    teleport_attempt.run_acquisition_phase = _run_acquisition
    teleport_attempt.run_teleport_phase = _run_teleport
    try:
        attempt = teleport_attempt.run_tracked_teleport_attempt(
            expected_page,
            expected_probe,
            expected_target,
            cdp=None,
            attempt_label=expected_target["label"],
            fuel_before=1100,
            world_timestamp_before=900,
            send_acquisition_command=_dispatch,
            acquisition_command_name="map_open",
            capture_before_map_open=True,
            wait_for_acquisition_sync=True,
            acquisition_timeout_ms=4000,
            teleport_timeout_ms=10000,
            wait_for_outcome=_WaitForOutcome(),
            dispatch_failure_error=RuntimeError,
            acquisition_dispatch_failure_message="acquisition failed",
            teleport_dispatch_failure_message="teleport failed",
            unavailable_error=RuntimeError,
            unavailable_message="missing",
            unexpected_result_error=RuntimeError,
            unexpected_result_message="impossible",
        )
    finally:
        teleport_attempt.run_acquisition_phase = original_acquisition
        teleport_attempt.run_teleport_phase = original_teleport

    assert expected_probe.reset_idle_calls == 1
    assert attempt.message_start_index == 1
    assert attempt.acquisition_started_ms == 1500
    assert attempt.acquisition_sync_timestamp_ms is None
    assert attempt.teleport_result is None
    assert attempt.teleport_started_ms is None


def test_run_tracked_teleport_attempt_rejects_impossible_map_sync_timeout() -> None:
    """Shared helper raises when teleport returns an impossible map-sync timeout."""
    original_acquisition = teleport_attempt.run_acquisition_phase
    original_teleport = teleport_attempt.run_teleport_phase
    expected_page = _Page()
    expected_probe = _Probe()
    expected_target = _target()

    def _dispatch() -> bool:
        return True

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return _snapshot(phase)

    def _run_acquisition(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        *,
        cdp: CDPSessionProtocol | None,
        send_command: Callable[[], bool],
        command_name: str,
        capture_before_map_open: bool,
        wait_for_sync: bool,
        sync_timeout_ms: int,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
    ) -> tuple[
        int,
        int | None,
        list[TeleportPageSnapshotDict],
        Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ]:
        _ = (
            cdp,
            command_name,
            capture_before_map_open,
            wait_for_sync,
            sync_timeout_ms,
            dispatch_failure_error,
            dispatch_failure_message,
            unavailable_error,
            unavailable_message,
        )
        assert page is expected_page
        assert provider is expected_probe
        assert send_command()
        return (1500, 1700, [_snapshot("before_map_open")], _capture_page_snapshot)

    def _run_teleport(
        page: action_session.WaitPageProtocol,
        probe: TeleportPhaseProbeProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle: ActionPhaseCycleDict,
        message_start_index: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "",
    ) -> tuple[TeleportAttemptResultDict, int]:
        _ = (
            page,
            probe,
            target,
            teleport_cycle,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
            wait_for_outcome,
            dispatch_failure_error,
            dispatch_failure_message,
        )
        impossible_result = _result(target)
        impossible_result["status"] = "map_sync_timeout"
        return (impossible_result, 1800)

    teleport_attempt.run_acquisition_phase = _run_acquisition
    teleport_attempt.run_teleport_phase = _run_teleport
    try:
        with pytest.raises(RuntimeError, match="impossible"):
            teleport_attempt.run_tracked_teleport_attempt(
                expected_page,
                expected_probe,
                expected_target,
                cdp=None,
                attempt_label=expected_target["label"],
                fuel_before=1100,
                world_timestamp_before=900,
                send_acquisition_command=_dispatch,
                acquisition_command_name="map_open",
                capture_before_map_open=True,
                wait_for_acquisition_sync=True,
                acquisition_timeout_ms=4000,
                teleport_timeout_ms=10000,
                wait_for_outcome=_WaitForOutcome(),
                dispatch_failure_error=RuntimeError,
                acquisition_dispatch_failure_message="acquisition failed",
                teleport_dispatch_failure_message="teleport failed",
                unavailable_error=RuntimeError,
                unavailable_message="missing",
                unexpected_result_error=RuntimeError,
                unexpected_result_message="impossible",
            )
    finally:
        teleport_attempt.run_acquisition_phase = original_acquisition
        teleport_attempt.run_teleport_phase = original_teleport


def test_run_tracked_teleport_attempt_skips_idle_reset_when_disabled() -> None:
    """Shared helper leaves probe state untouched when reset is disabled."""
    original_acquisition = teleport_attempt.run_acquisition_phase
    original_teleport = teleport_attempt.run_teleport_phase
    expected_page = _Page()
    expected_probe = _Probe()
    expected_target = _target()

    def _dispatch() -> bool:
        return True

    def _capture_page_snapshot(
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        return _snapshot(phase)

    def _run_acquisition(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        *,
        cdp: CDPSessionProtocol | None,
        send_command: Callable[[], bool],
        command_name: str,
        capture_before_map_open: bool,
        wait_for_sync: bool,
        sync_timeout_ms: int,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
    ) -> tuple[
        int,
        int | None,
        list[TeleportPageSnapshotDict],
        Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ]:
        _ = (
            cdp,
            command_name,
            capture_before_map_open,
            wait_for_sync,
            sync_timeout_ms,
            dispatch_failure_error,
            dispatch_failure_message,
            unavailable_error,
            unavailable_message,
        )
        assert page is expected_page
        assert provider is expected_probe
        assert send_command()
        return (1500, None, [_snapshot("before_map_open")], _capture_page_snapshot)

    def _run_teleport(
        page: action_session.WaitPageProtocol,
        probe: TeleportPhaseProbeProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle: ActionPhaseCycleDict,
        message_start_index: int,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        dispatch_failure_message: str = "",
    ) -> tuple[TeleportAttemptResultDict, int]:
        _ = (
            page,
            probe,
            target,
            teleport_cycle,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
            wait_for_outcome,
            dispatch_failure_error,
            dispatch_failure_message,
        )
        raise AssertionError("teleport phase should not run after acquisition timeout")

    teleport_attempt.run_acquisition_phase = _run_acquisition
    teleport_attempt.run_teleport_phase = _run_teleport
    try:
        attempt = teleport_attempt.run_tracked_teleport_attempt(
            expected_page,
            expected_probe,
            expected_target,
            cdp=None,
            attempt_label=expected_target["label"],
            fuel_before=1100,
            world_timestamp_before=900,
            send_acquisition_command=_dispatch,
            acquisition_command_name="map_open",
            capture_before_map_open=True,
            wait_for_acquisition_sync=True,
            acquisition_timeout_ms=4000,
            teleport_timeout_ms=10000,
            wait_for_outcome=_WaitForOutcome(),
            dispatch_failure_error=RuntimeError,
            acquisition_dispatch_failure_message="acquisition failed",
            teleport_dispatch_failure_message="teleport failed",
            unavailable_error=RuntimeError,
            unavailable_message="missing",
            unexpected_result_error=RuntimeError,
            unexpected_result_message="impossible",
            reset_to_idle_before_start=False,
        )
    finally:
        teleport_attempt.run_acquisition_phase = original_acquisition
        teleport_attempt.run_teleport_phase = original_teleport

    assert expected_probe.reset_idle_calls == 0
    assert attempt.teleport_result is None
