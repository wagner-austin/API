"""The teardown escalation ladder: close, engine kill, forced exit.

Rung 1 is ``browser.close()``; rung 2 kills the browser engine while
sparing the driver so the close can resolve; rung 3 force-exits with
the SESSION's outcome code. A close that returns disarms rungs 2 and
3 through the shared event. The CLI's separate post-session exit
deadline (``arm_session_exit_deadline``) bounds the shutdown tail
after ``bot.run`` returns.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.browser.lifecycle import (
    _SESSION_EXIT_DEADLINE_SECONDS,
    _TEARDOWN_HANG_EXIT_CODE,
    _TEARDOWN_REMEDY_SECONDS,
    _TEARDOWN_WATCHDOG_SECONDS,
    _handle_session_exit_wedge,
    _handle_teardown_hang,
    _remedy_close_hang,
    _session_outcome_exit_code,
    _thread_stacks_snapshot,
    arm_session_exit_deadline,
    cleanup_browser,
)
from tests.fakes import FakeBrowser


class _RecordingWatchdogs:
    """Recording ``start_watchdog`` fake: captures every armed timer."""

    def __init__(self) -> None:
        self.armed: list[tuple[float, Callable[[], None]]] = []

    def __call__(self, seconds: float, on_fire: Callable[[], None]) -> None:
        self.armed.append((seconds, on_fire))


class _FailCloseBrowser(FakeBrowser):
    def close(self, *, reason: str | None = None) -> None:
        _ = reason
        raise OSError("browser already closed")


class _RuntimeErrorCloseBrowser(FakeBrowser):
    def close(self, *, reason: str | None = None) -> None:
        _ = reason
        raise RuntimeError("browser teardown failed")


class _DisconnectCloseBrowser(FakeBrowser):
    """A close that resolves by driver disconnect — rung 2's designed outcome."""

    def close(self, *, reason: str | None = None) -> None:
        _ = reason
        from playwright._impl._errors import Error as PlaywrightError

        raise PlaywrightError("Browser closed unexpectedly")


class TestCleanupBrowser:
    def test_closes_browser_and_arms_the_ladder(self) -> None:
        """Both rungs are armed at their documented delays, then disarmed.

        After a clean close, firing BOTH captured callbacks must do
        nothing: the conftest defaults for ``force_exit`` and
        ``kill_browser_processes`` raise on any call, so silence here
        is proof the set event disarmed the ladder.
        """
        watchdogs = _RecordingWatchdogs()
        original = _test_hooks.start_watchdog
        _test_hooks.start_watchdog = watchdogs
        try:
            cleanup_browser(FakeBrowser())
        finally:
            _test_hooks.start_watchdog = original

        assert [seconds for seconds, _ in watchdogs.armed] == [
            _TEARDOWN_REMEDY_SECONDS,
            _TEARDOWN_WATCHDOG_SECONDS,
        ]
        for _, on_fire in watchdogs.armed:
            on_fire()

    def test_handles_os_error(self) -> None:
        cleanup_browser(_FailCloseBrowser())

    def test_handles_runtime_error(self) -> None:
        cleanup_browser(_RuntimeErrorCloseBrowser())

    def test_close_resolved_by_disconnect_is_a_completed_teardown(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rung 2's kill makes close raise a Playwright error, not return.

        That exception IS the ladder working — it must disarm the
        terminal rung exactly like a clean return, and say so.
        """
        watchdogs = _RecordingWatchdogs()
        original = _test_hooks.start_watchdog
        _test_hooks.start_watchdog = watchdogs
        try:
            with caplog.at_level(logging.ERROR):
                cleanup_browser(_DisconnectCloseBrowser())
        finally:
            _test_hooks.start_watchdog = original

        assert any("resolved by disconnect" in r.message for r in caplog.records)
        for _, on_fire in watchdogs.armed:
            on_fire()


class TestSessionOutcomeExitCode:
    def test_completed_session_exits_zero(self) -> None:
        """No exception in flight: the teardown inherits success."""
        assert _session_outcome_exit_code() == 0

    def test_crashing_session_keeps_the_failure_code(self) -> None:
        """An exception propagating through the finally keeps 75."""
        try:
            raise ValueError("session crashed")
        except ValueError as exc:
            assert str(exc) == "session crashed"
            assert _session_outcome_exit_code() == _TEARDOWN_HANG_EXIT_CODE


class TestRemedyCloseHang:
    def test_kills_the_engine_while_the_close_is_stuck(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unset event means the close is wedged: kill and log pids."""
        kills: list[int] = []

        def fake_kill() -> list[int]:
            kills.append(1)
            return [4242, 4243]

        original = _test_hooks.kill_browser_processes
        _test_hooks.kill_browser_processes = fake_kill
        try:
            with caplog.at_level(logging.ERROR):
                _remedy_close_hang(threading.Event())
        finally:
            _test_hooks.kill_browser_processes = original

        assert kills == [1]
        assert any("[4242, 4243]" in r.message for r in caplog.records)

    def test_a_completed_close_is_left_alone(self) -> None:
        """A set event disarms the rung: the raising conftest fake proves
        no kill is attempted."""
        closed = threading.Event()
        closed.set()
        _remedy_close_hang(closed)


class TestHandleTeardownHang:
    def test_calls_force_exit_with_the_session_outcome(self) -> None:
        calls: list[int] = []
        original = _test_hooks.force_exit
        _test_hooks.force_exit = lambda code: calls.append(code)
        try:
            _handle_teardown_hang(threading.Event(), _TEARDOWN_HANG_EXIT_CODE)
            assert calls == [75]
        finally:
            _test_hooks.force_exit = original

    def test_a_completed_session_hang_exits_zero(self) -> None:
        """The forced exit carries the session's outcome, not a blanket 75."""
        calls: list[int] = []
        original = _test_hooks.force_exit
        _test_hooks.force_exit = lambda code: calls.append(code)
        try:
            _handle_teardown_hang(threading.Event(), 0)
            assert calls == [0]
        finally:
            _test_hooks.force_exit = original

    def test_a_completed_teardown_is_never_shot(self) -> None:
        """A set event means teardown finished: the process must live.

        Before this guard the uncancelled watchdog force-exited a
        HEALTHY long-running service 30 s after every clean session
        teardown (run bot-20260729-010551: "browser closed" logged at
        :28:35, forced exit at :29:04). The raising conftest fake
        proves force_exit is never reached.
        """
        closed = threading.Event()
        closed.set()
        _handle_teardown_hang(closed, _TEARDOWN_HANG_EXIT_CODE)

    def test_logs_thread_stacks_before_exit(self, caplog: pytest.LogCaptureFixture) -> None:
        """The hang autopsy lands in the log before the forced exit.

        The dumped stack of the calling thread must contain this very
        test's frame — real frames, not a placeholder — so the next
        live hang shows exactly which call ``browser.close()`` wedged
        in.
        """
        calls: list[int] = []
        original = _test_hooks.force_exit
        _test_hooks.force_exit = lambda code: calls.append(code)
        try:
            with caplog.at_level(logging.ERROR):
                _handle_teardown_hang(threading.Event(), _TEARDOWN_HANG_EXIT_CODE)
        finally:
            _test_hooks.force_exit = original

        assert calls == [75]
        dumps = [
            record.message
            for record in caplog.records
            if "Thread stacks at the moment of the hang" in record.message
        ]
        assert len(dumps) == 1
        assert "test_logs_thread_stacks_before_exit" in dumps[0]


class TestSessionExitDeadline:
    def test_arms_the_documented_deadline(self) -> None:
        watchdogs = _RecordingWatchdogs()
        original = _test_hooks.start_watchdog
        _test_hooks.start_watchdog = watchdogs
        try:
            arm_session_exit_deadline()
        finally:
            _test_hooks.start_watchdog = original

        assert [seconds for seconds, _ in watchdogs.armed] == [_SESSION_EXIT_DEADLINE_SECONDS]
        assert watchdogs.armed[0][1] is _handle_session_exit_wedge

    def test_the_wedge_handler_exits_zero_with_stacks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A finished session's shutdown wedge is forced out as SUCCESS."""
        calls: list[int] = []
        original = _test_hooks.force_exit
        _test_hooks.force_exit = lambda code: calls.append(code)
        try:
            with caplog.at_level(logging.ERROR):
                _handle_session_exit_wedge()
        finally:
            _test_hooks.force_exit = original

        assert calls == [0]
        assert any(
            "Thread stacks at the moment of the hang" in record.message for record in caplog.records
        )


class TestThreadStacksSnapshot:
    def test_captures_a_parked_sibling_thread(self) -> None:
        """The snapshot crosses threads — the watchdog's whole reason to exist.

        A live teardown hang has the MAIN thread stuck inside
        ``browser.close()`` while the watchdog timer thread takes the
        snapshot, so the snapshot must show OTHER threads' frames, not
        just its caller's. A named sibling parked on an ``Event`` stands
        in for the wedged closer: its name and its parked function must
        both appear.
        """
        parked = threading.Event()
        release = threading.Event()

        def _wedged_close_stand_in() -> None:
            parked.set()
            release.wait()

        sibling = threading.Thread(
            target=_wedged_close_stand_in,
            name="teardown-stand-in",
            daemon=True,
        )
        sibling.start()
        assert parked.wait(timeout=5.0)
        try:
            snapshot = _thread_stacks_snapshot()
        finally:
            release.set()
            sibling.join(timeout=5.0)

        assert "teardown-stand-in" in snapshot
        assert "_wedged_close_stand_in" in snapshot
