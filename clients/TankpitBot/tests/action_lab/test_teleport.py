"""Tests for ``probe_single_target`` and the probe base guards.

``test_teleport.py`` was 1,506 lines; targets, formatting, outcome
waiting, and execution are now siblings.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import Literal

import pytest
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from tests.action_lab._teleport_harness import (
    _make_attempt,
    _make_page_snapshot,
    _make_world,
    _ProbeMethodHarness,
    _ProbeMissingPageHarness,
    _SequencedProvider,
)

from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.action_trace_types import ActionPhaseCycleDict
from tankpit_bot.action_lab.teleport import (
    TeleportProbe,
)
from tankpit_bot.action_lab.teleport_attempt import (
    TeleportAttemptProbeProtocol,
    TrackedTeleportAttempt,
)
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
)
from tankpit_bot.action_lab.teleport_phase import TeleportOutcomeWaiterProtocol
from tankpit_bot.action_lab.types import (
    TeleportAttemptResultDict,
    TeleportPageSnapshotDict,
    TeleportTargetDict,
)


def test_probe_helpers_cover_guards_and_clear_action() -> None:
    probe = _ProbeMethodHarness()
    assert probe._require_self_state()["x"] == 158
    assert probe._require_page() is probe._fake_page

    probe._clear_in_flight_action()
    probe._reset_probe_state_to_idle()
    assert probe.get_state() == "IDLE"

    probe._self_state = None
    from tankpit_bot.action_lab.probe_base import ProbeError

    with pytest.raises(ProbeError, match="self state is unavailable"):
        probe._require_self_state()

    probe = _ProbeMissingPageHarness()
    with pytest.raises(ProbeError, match="page is unavailable"):
        probe._require_page()


def test_base_require_page_returns_page_and_raises_when_missing() -> None:
    from tankpit_bot.action_lab.probe_base import ProbeError

    probe = TeleportProbe("https://tankpit.com/play", headless=True, prefer_account=False)
    with pytest.raises(ProbeError, match="page is unavailable"):
        probe._require_page()
    fake_page = ClockAdvancingPage(
        ReplayClock(1000),
        on_wait=_SequencedProvider([_make_world(900, 158, 132, 900)]).advance,
    )
    probe._page = fake_page
    assert probe._require_page() is fake_page


def test_probe_single_target_returns_map_sync_timeout_and_settles() -> None:
    probe = _ProbeMethodHarness()
    page = probe._fake_page
    original_wait = action_hooks.wait_for_world_sync

    def _wait_sync_timeout(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return None

    wait_sync_name = "wait_for_world_sync"
    setattr(action_hooks, wait_sync_name, _wait_sync_timeout)
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="sync_before_teleport",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=250,
        )
    finally:
        setattr(action_hooks, wait_sync_name, original_wait)
    assert result["status"] == "map_sync_timeout"
    assert probe.teleport_calls == []
    assert result["message_start_index"] == 0
    assert result["message_end_index"] == 0
    assert page.waits[-1] == 250.0


def test_probe_single_target_returns_map_sync_timeout_without_settle() -> None:
    probe = _ProbeMethodHarness()
    page = probe._fake_page
    original_wait = action_hooks.wait_for_world_sync

    def _wait_sync_timeout(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return None

    wait_sync_name = "wait_for_world_sync"
    setattr(action_hooks, wait_sync_name, _wait_sync_timeout)
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="sync_before_teleport",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=0,
        )
    finally:
        setattr(action_hooks, wait_sync_name, original_wait)
    assert result["status"] == "map_sync_timeout"
    assert page.waits == []


def test_probe_single_target_returns_wait_result_without_settle() -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    probe = _ProbeMethodHarness()
    page = probe._fake_page
    expected = _make_attempt("landed_exact")
    original_wait_sync = action_hooks.wait_for_world_sync
    original_wait_outcome = teleport_module._wait_for_teleport_outcome

    def _wait_sync_success(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return 1200

    def _wait_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            target,
            teleport_cycle_id,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return expected

    action_hooks.wait_for_world_sync = _wait_sync_success
    teleport_module._wait_for_teleport_outcome = _wait_outcome
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="sync_before_teleport",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=0,
        )
    finally:
        action_hooks.wait_for_world_sync = original_wait_sync
        teleport_module._wait_for_teleport_outcome = original_wait_outcome
    assert result == expected
    assert probe.teleport_calls == [(150, 171)]
    assert result["message_start_index"] == 10
    assert result["message_end_index"] == 14
    assert page.waits == []


def test_probe_single_target_rejects_missing_teleport_result_after_acquisition() -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    probe = _ProbeMethodHarness()
    original_attempt_runner = teleport_module.run_tracked_teleport_attempt

    def _run_attempt(
        page: action_session.WaitPageProtocol,
        probe: TeleportAttemptProbeProtocol,
        target: TeleportTargetDict,
        *,
        cdp: CDPSessionProtocol | None,
        attempt_label: str,
        fuel_before: int,
        world_timestamp_before: int,
        send_acquisition_command: Callable[[], bool],
        acquisition_command_name: str,
        capture_before_map_open: bool,
        wait_for_acquisition_sync: bool,
        acquisition_timeout_ms: int,
        teleport_timeout_ms: int,
        wait_for_outcome: TeleportOutcomeWaiterProtocol,
        dispatch_failure_error: type[Exception],
        acquisition_dispatch_failure_message: str,
        teleport_dispatch_failure_message: str,
        unavailable_error: type[Exception],
        unavailable_message: str,
        unexpected_result_error: type[Exception],
        unexpected_result_message: str,
        reset_to_idle_before_start: bool = True,
    ) -> TrackedTeleportAttempt:
        _ = (
            page,
            probe,
            target,
            cdp,
            attempt_label,
            fuel_before,
            world_timestamp_before,
            send_acquisition_command,
            acquisition_command_name,
            capture_before_map_open,
            wait_for_acquisition_sync,
            acquisition_timeout_ms,
            teleport_timeout_ms,
            wait_for_outcome,
            dispatch_failure_error,
            acquisition_dispatch_failure_message,
            teleport_dispatch_failure_message,
            unavailable_error,
            unavailable_message,
            unexpected_result_error,
            unexpected_result_message,
            reset_to_idle_before_start,
        )
        return TrackedTeleportAttempt(
            message_start_index=0,
            teleport_cycle=ActionPhaseCycleDict(phase="teleport", cycle_id=1, started_ms=1000),
            acquisition_started_ms=1000,
            acquisition_sync_timestamp_ms=1200,
            page_snapshots=[],
            capture_page_snapshot=lambda phase: _make_page_snapshot(phase),
            teleport_result=None,
            teleport_started_ms=None,
        )

    teleport_module.run_tracked_teleport_attempt = _run_attempt
    try:
        with pytest.raises(
            TeleportProbeError,
            match="teleport attempt ended before teleport dispatch",
        ):
            probe._probe_single_target(
                TeleportTargetDict(label="target_0", x=150, y=171),
                teleport_strategy="sync_before_teleport",
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                settle_delay_ms=0,
            )
    finally:
        teleport_module.run_tracked_teleport_attempt = original_attempt_runner


def test_probe_single_target_returns_wait_result_with_settle() -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    probe = _ProbeMethodHarness()
    page = probe._fake_page
    expected = _make_attempt("landed_exact")
    original_wait_sync = action_hooks.wait_for_world_sync
    original_wait_outcome = teleport_module._wait_for_teleport_outcome

    def _wait_sync_success(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        return 1200

    def _wait_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            target,
            teleport_cycle_id,
            message_start_index,
            map_open_started_ms,
            map_sync_timestamp_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        return expected

    action_hooks.wait_for_world_sync = _wait_sync_success
    teleport_module._wait_for_teleport_outcome = _wait_outcome
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="sync_before_teleport",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=250,
        )
    finally:
        action_hooks.wait_for_world_sync = original_wait_sync
        teleport_module._wait_for_teleport_outcome = original_wait_outcome
    assert result == expected
    assert result["message_start_index"] == 10
    assert result["message_end_index"] == 14
    assert page.waits[-1] == 250.0


def test_probe_single_target_immediate_strategy_skips_map_sync_wait() -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    probe = _ProbeMethodHarness()
    expected = _make_attempt("landed_exact")
    original_wait_sync = action_hooks.wait_for_world_sync
    original_wait_outcome = teleport_module._wait_for_teleport_outcome
    wait_sync_calls: list[int] = []

    def _wait_sync_unexpected(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> int | None:
        _ = (page, provider, started_ms, timeout_ms)
        wait_sync_calls.append(1)
        return 1200

    def _wait_outcome(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        target: TeleportTargetDict,
        *,
        teleport_cycle_id: int,
        message_start_index: int = 0,
        map_open_started_ms: int,
        map_sync_timestamp_ms: int | None,
        teleport_started_ms: int,
        fuel_before: int,
        world_timestamp_before: int,
        timeout_ms: int,
        page_snapshots: list[TeleportPageSnapshotDict],
        capture_page_snapshot: Callable[
            [Literal["before_map_open", "before_teleport", "after_map_data", "landed", "timeout"]],
            TeleportPageSnapshotDict,
        ],
    ) -> TeleportAttemptResultDict:
        _ = (
            page,
            provider,
            target,
            teleport_cycle_id,
            message_start_index,
            map_open_started_ms,
            teleport_started_ms,
            fuel_before,
            world_timestamp_before,
            timeout_ms,
            page_snapshots,
            capture_page_snapshot,
        )
        assert map_sync_timestamp_ms is None
        return expected

    action_hooks.wait_for_world_sync = _wait_sync_unexpected
    teleport_module._wait_for_teleport_outcome = _wait_outcome
    try:
        result = probe._probe_single_target(
            TeleportTargetDict(label="target_0", x=150, y=171),
            teleport_strategy="immediate_after_map_open",
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=0,
        )
    finally:
        action_hooks.wait_for_world_sync = original_wait_sync
        teleport_module._wait_for_teleport_outcome = original_wait_outcome
    assert result == expected
    assert wait_sync_calls == []
    assert probe.teleport_calls == [(150, 171)]
