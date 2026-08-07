"""Tests for shared pre-teleport acquisition helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
from platform_core.json_utils import JSONObject
from tests.action_lab._replay_core import ReplayClock

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab import teleport_acquisition
from tankpit_bot.action_lab.types import TeleportPageSnapshotDict
from tankpit_bot.state import WorldStateDict
from tankpit_bot.types import CapturedMessage


class _Page:
    """Minimal page fake for sync waits."""

    def wait_for_timeout(self, timeout: float) -> None:
        """Ignore wait requests."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)


class _Provider:
    """Minimal provider satisfying buffered world-state protocol."""

    def __init__(self) -> None:
        """Initialize the provider."""
        self._cdp_message_buffer: list[str] = []
        self.xor_table: bytes | None = None
        self._messages: list[CapturedMessage] = []

    @property
    def messages(self) -> list[CapturedMessage]:
        """Return captured messages."""
        return self._messages

    @property
    def magic(self) -> str:
        """Return a stable fake magic key."""
        return "magic"

    def get_world_state(self) -> WorldStateDict:
        """Raise when world state access is unexpected."""
        raise AssertionError("unexpected world state read")


class _CDP:
    """Minimal CDP fake satisfying the session protocol."""

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Reject unexpected CDP sends."""
        _ = (method, params)
        raise AssertionError("unexpected CDP send")

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Ignore event handler registration."""
        _ = (event, handler)

    def detach(self) -> None:
        """Ignore detach calls."""


def _snapshot(
    phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
) -> TeleportPageSnapshotDict:
    """Build a sample page snapshot."""
    return TeleportPageSnapshotDict(
        phase=phase,
        timestamp_ms=3000,
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


def test_start_teleport_page_snapshots_captures_initial_snapshot() -> None:
    """Snapshot helper captures the initial map-open phase when requested."""
    original_capture = action_hooks.capture_teleport_page_snapshot
    expected_cdp = _CDP()

    def _capture(
        cdp: CDPSessionProtocol,
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        assert cdp is expected_cdp
        return _snapshot(phase)

    action_hooks.capture_teleport_page_snapshot = _capture
    try:
        snapshots, capture_page_snapshot = teleport_acquisition.start_teleport_page_snapshots(
            cdp=expected_cdp,
            capture_before_map_open=True,
            unavailable_error=RuntimeError,
            unavailable_message="missing",
        )
        assert capture_page_snapshot("landed")["phase"] == "landed"
    finally:
        action_hooks.capture_teleport_page_snapshot = original_capture

    assert [snapshot["phase"] for snapshot in snapshots] == ["before_map_open"]


def test_run_tracked_acquisition_phase_short_circuits_when_map_already_open() -> None:
    """Acquisition skips the map_open dispatch and sync wait when the map is showing.

    The wire ``map_open`` only opens; re-sending against an already-open
    map produces no fresh map-sync response, so the probe's
    ``wait_for_world_sync`` would either time out or return a stale sync
    and break the rest of the attempt. The live snapshot's
    ``map_visible`` flag short-circuits this case.
    """
    clock = ReplayClock(2200)
    original_clock = action_hooks.get_current_time_ms
    original_wait = action_hooks.wait_for_world_sync
    original_capture = action_hooks.capture_teleport_page_snapshot
    expected_provider = _Provider()
    expected_page = _Page()
    expected_cdp = _CDP()
    dispatch_calls: list[str] = []
    wait_calls: list[str] = []

    def _dispatch() -> bool:
        dispatch_calls.append("sent")
        return True

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        wait_calls.append("waited")
        return None

    def _capture(
        cdp: CDPSessionProtocol,
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        assert cdp is expected_cdp
        base = _snapshot(phase)
        base["map_visible"] = True
        return base

    action_hooks.get_current_time_ms = clock
    action_hooks.wait_for_world_sync = _wait_for_world_sync
    action_hooks.capture_teleport_page_snapshot = _capture
    try:
        started_ms, sync_timestamp_ms, snapshots, _capture_callback = (
            teleport_acquisition.run_tracked_acquisition_phase(
                expected_page,
                expected_provider,
                cdp=expected_cdp,
                send_command=_dispatch,
                command_name="map_open",
                capture_before_map_open=True,
                wait_for_sync=True,
                sync_timeout_ms=4000,
                dispatch_failure_error=RuntimeError,
                dispatch_failure_message="dispatch failed",
                unavailable_error=RuntimeError,
                unavailable_message="missing",
            )
        )
    finally:
        action_hooks.get_current_time_ms = original_clock
        action_hooks.wait_for_world_sync = original_wait
        action_hooks.capture_teleport_page_snapshot = original_capture

    assert started_ms == 2200
    assert sync_timestamp_ms == 2200
    assert dispatch_calls == []
    assert wait_calls == []
    assert [snapshot["map_visible"] for snapshot in snapshots] == [True]


def test_run_tracked_acquisition_phase_waits_for_sync() -> None:
    """Acquisition helper dispatches once and waits for world sync when requested."""
    clock = ReplayClock(1500)
    original_clock = action_hooks.get_current_time_ms
    original_wait = action_hooks.wait_for_world_sync
    original_capture = action_hooks.capture_teleport_page_snapshot
    expected_provider = _Provider()
    expected_page = _Page()
    expected_cdp = _CDP()
    dispatch_calls: list[str] = []

    def _dispatch() -> bool:
        dispatch_calls.append("sent")
        return True

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        assert page is expected_page
        assert provider is expected_provider
        assert started_ms == 1500
        assert timeout_ms == 4000
        return 1700

    def _capture(
        cdp: CDPSessionProtocol,
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        assert cdp is expected_cdp
        return _snapshot(phase)

    action_hooks.get_current_time_ms = clock
    action_hooks.wait_for_world_sync = _wait_for_world_sync
    action_hooks.capture_teleport_page_snapshot = _capture
    try:
        started_ms, sync_timestamp_ms, snapshots, capture_page_snapshot = (
            teleport_acquisition.run_tracked_acquisition_phase(
                expected_page,
                expected_provider,
                cdp=expected_cdp,
                send_command=_dispatch,
                command_name="map_open",
                capture_before_map_open=True,
                wait_for_sync=True,
                sync_timeout_ms=4000,
                dispatch_failure_error=RuntimeError,
                dispatch_failure_message="dispatch failed",
                unavailable_error=RuntimeError,
                unavailable_message="missing",
            )
        )
        assert capture_page_snapshot("timeout")["phase"] == "timeout"
    finally:
        action_hooks.get_current_time_ms = original_clock
        action_hooks.wait_for_world_sync = original_wait
        action_hooks.capture_teleport_page_snapshot = original_capture

    assert started_ms == 1500
    assert sync_timestamp_ms == 1700
    assert dispatch_calls == ["sent"]
    assert [snapshot["phase"] for snapshot in snapshots] == ["before_map_open"]


def test_run_tracked_acquisition_phase_skips_sync_when_disabled() -> None:
    """Acquisition helper skips world sync when the caller disables it."""
    clock = ReplayClock(2500)
    original_clock = action_hooks.get_current_time_ms
    original_wait = action_hooks.wait_for_world_sync
    original_capture = action_hooks.capture_teleport_page_snapshot
    expected_provider = _Provider()
    expected_page = _Page()
    expected_cdp = _CDP()

    def _dispatch() -> bool:
        return True

    def _wait_for_world_sync(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        raise AssertionError("wait_for_world_sync should not run when wait_for_sync=False")

    def _capture(
        cdp: CDPSessionProtocol,
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        assert cdp is expected_cdp
        return _snapshot(phase)

    action_hooks.get_current_time_ms = clock
    action_hooks.wait_for_world_sync = _wait_for_world_sync
    action_hooks.capture_teleport_page_snapshot = _capture
    try:
        (
            started_ms,
            sync_timestamp_ms,
            snapshots,
            _,
        ) = teleport_acquisition.run_tracked_acquisition_phase(
            expected_page,
            expected_provider,
            cdp=expected_cdp,
            send_command=_dispatch,
            command_name="map_open",
            capture_before_map_open=False,
            wait_for_sync=False,
            sync_timeout_ms=4000,
            dispatch_failure_error=RuntimeError,
            dispatch_failure_message="dispatch failed",
            unavailable_error=RuntimeError,
            unavailable_message="missing",
        )
    finally:
        action_hooks.get_current_time_ms = original_clock
        action_hooks.wait_for_world_sync = original_wait
        action_hooks.capture_teleport_page_snapshot = original_capture

    assert started_ms == 2500
    assert sync_timestamp_ms is None
    assert snapshots == []


def test_run_tracked_acquisition_phase_raises_on_dispatch_failure() -> None:
    """Acquisition helper raises immediately on dispatch failure."""
    clock = ReplayClock(3500)
    original_clock = action_hooks.get_current_time_ms
    original_capture = action_hooks.capture_teleport_page_snapshot
    expected_provider = _Provider()
    expected_page = _Page()
    expected_cdp = _CDP()

    def _dispatch() -> bool:
        return False

    def _capture(
        cdp: CDPSessionProtocol,
        phase: Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"],
    ) -> TeleportPageSnapshotDict:
        assert cdp is expected_cdp
        return _snapshot(phase)

    action_hooks.get_current_time_ms = clock
    action_hooks.capture_teleport_page_snapshot = _capture
    try:
        with pytest.raises(RuntimeError, match="dispatch failed"):
            teleport_acquisition.run_tracked_acquisition_phase(
                expected_page,
                expected_provider,
                cdp=expected_cdp,
                send_command=_dispatch,
                command_name="enemy_acquisition",
                capture_before_map_open=True,
                wait_for_sync=True,
                sync_timeout_ms=4000,
                dispatch_failure_error=RuntimeError,
                dispatch_failure_message="dispatch failed",
                unavailable_error=RuntimeError,
                unavailable_message="missing",
            )
    finally:
        action_hooks.get_current_time_ms = original_clock
        action_hooks.capture_teleport_page_snapshot = original_capture
