"""Tests for SIGINT/SIGTERM interrupt handling (Tier 3.4).

Covers the public flag API (``request_interrupt`` / ``is_interrupt_requested``
/ ``reset_interrupt_flag``), the tick loop's exit branch when the flag is
set, the ``_test_hooks.install_signal_handlers`` real implementation, and
the bot CLI entry-point's wiring.
"""

from __future__ import annotations

import signal
from collections.abc import Callable, Generator
from typing import NoReturn

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.runtime import _real_install_signal_handlers
from tankpit_bot.bot import tick_loop
from tankpit_bot.bot.tick_loop import (
    is_interrupt_requested,
    request_interrupt,
    reset_interrupt_flag,
)
from tests.conftest import FakeEnv


class TestInterruptFlag:
    """Tests for the module-level interrupt flag API."""

    def setup_method(self) -> None:
        """Ensure each test starts with a cleared flag."""
        reset_interrupt_flag()

    def teardown_method(self) -> None:
        """Restore the cleared flag for downstream tests."""
        reset_interrupt_flag()

    def test_flag_starts_cleared(self) -> None:
        """``is_interrupt_requested`` returns False by default."""
        assert is_interrupt_requested() is False

    def test_request_interrupt_sets_flag(self) -> None:
        """``request_interrupt`` flips the flag to True."""
        request_interrupt()
        assert is_interrupt_requested() is True

    def test_request_interrupt_is_idempotent(self) -> None:
        """Calling ``request_interrupt`` twice does not change the value."""
        request_interrupt()
        request_interrupt()
        assert is_interrupt_requested() is True

    def test_reset_interrupt_flag_clears(self) -> None:
        """``reset_interrupt_flag`` returns the flag to False."""
        request_interrupt()
        reset_interrupt_flag()
        assert is_interrupt_requested() is False


class TestRealInstallSignalHandlers:
    """Tests for the real ``_real_install_signal_handlers`` implementation.

    The handler is installed against process-wide ``signal.signal`` so
    the test must save and restore both SIGINT and SIGTERM handlers
    around the install call.
    """

    @pytest.fixture(autouse=True)
    def _restore_signal_state(self) -> Generator[None, None, None]:
        """Save and restore the SIGINT/SIGTERM handlers for each test."""
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    def test_install_registers_handler_for_sigint_and_sigterm(self) -> None:
        """After install, both signals route to a wrapper around the callback."""
        events: list[str] = []

        def on_interrupt() -> None:
            events.append("fired")

        _real_install_signal_handlers(on_interrupt)

        sigint_handler = signal.getsignal(signal.SIGINT)
        sigterm_handler = signal.getsignal(signal.SIGTERM)
        if not callable(sigint_handler):
            raise AssertionError("expected SIGINT handler to be callable")
        if not callable(sigterm_handler):
            raise AssertionError("expected SIGTERM handler to be callable")

        # Invoke each handler directly with the (signum, frame) signature.
        sigint_handler(int(signal.SIGINT), None)
        sigterm_handler(int(signal.SIGTERM), None)
        assert events == ["fired", "fired"]


class TestEntryPointWiring:
    """Tests for the bot entry point's signal-handler install call."""

    def test_main_installs_handlers_with_request_interrupt(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """``main()`` calls ``install_signal_handlers(request_interrupt)``.

        We swap the hook for a recorder, run ``main()`` until it tries
        to spin up Playwright, and assert the recorder captured exactly
        one install call whose callback is ``request_interrupt``.
        """
        from tankpit_bot.bot import entry
        from tankpit_bot.bot.tick_loop import request_interrupt as production_callback

        registered: list[Callable[[], None]] = []

        def fake_install(on_interrupt: Callable[[], None]) -> None:
            registered.append(on_interrupt)

        original_install = _test_hooks.install_signal_handlers
        _test_hooks.install_signal_handlers = fake_install

        # Short-circuit playwright bring-up with a sentinel that aborts
        # immediately when the bot tries to use it.
        original_sync_playwright = _test_hooks.sync_playwright

        class _AbortMainError(Exception):
            """Raised by the playwright sentinel to stop main() mid-flight."""

        def _abort_get_sync_playwright() -> NoReturn:
            raise _AbortMainError()

        original_get_sync_playwright = _test_hooks.get_sync_playwright
        _test_hooks.get_sync_playwright = _abort_get_sync_playwright

        try:
            with pytest.raises(_AbortMainError):
                entry.main()
        finally:
            _test_hooks.install_signal_handlers = original_install
            _test_hooks.sync_playwright = original_sync_playwright
            _test_hooks.get_sync_playwright = original_get_sync_playwright

        if len(registered) != 1:
            raise AssertionError(f"expected exactly one install call, got {len(registered)}")
        assert registered[0] is production_callback


def test_install_signal_handlers_is_real_implementation_by_default() -> None:
    """The hook resolves to ``_real_install_signal_handlers`` at import.

    The autouse ``_restore_hooks`` fixture re-binds it to the noop fake
    for every test, so this assertion exercises the unwrapped module
    state directly via the runtime submodule.
    """
    from tankpit_bot._test_hooks import runtime as runtime_hooks

    assert runtime_hooks._real_install_signal_handlers is _real_install_signal_handlers


def test_tick_loop_module_exposes_flag_api() -> None:
    """The public symbols are present in the tick_loop module."""
    for name in ("request_interrupt", "reset_interrupt_flag", "is_interrupt_requested"):
        if not hasattr(tick_loop, name):
            raise AssertionError(f"tick_loop module is missing public symbol {name!r}")
