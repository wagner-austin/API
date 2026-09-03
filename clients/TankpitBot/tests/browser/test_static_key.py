"""Tests for static key functionality.

The brute-force discovery pair (``extract_xor_first_bytes`` /
``find_best_static_byte``) and its tests were removed 2026-08-17 —
the functions' only callers were these tests, and the hook slot that
supposedly let tests inject the discovery was read by nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.browser import load_static_key
from tests.no_op_keyboard import NoOpKeyboard

# =============================================================================
# Static Key Load Tests
# =============================================================================


def test_load_static_key_success() -> None:
    """Test load_static_key loads 1000-character key."""
    original = _test_hooks.read_text
    key_content = "a" * 1000

    def fake_read_text(path: Path) -> str:
        _ = path
        return key_content + "\n"

    _test_hooks.read_text = fake_read_text
    try:
        result = load_static_key()
        assert result == key_content
    finally:
        _test_hooks.read_text = original


def test_load_static_key_wrong_length_raises() -> None:
    """Test load_static_key raises ValueError for wrong key length."""
    original = _test_hooks.read_text
    key_content = "a" * 500  # Too short

    def fake_read_text(path: Path) -> str:
        _ = path
        return key_content + "\n"

    _test_hooks.read_text = fake_read_text
    try:
        with pytest.raises(ValueError, match="expected 1000"):
            load_static_key()
    finally:
        _test_hooks.read_text = original


# =============================================================================
# Fake Page Classes for Static Key Tests
# =============================================================================


class FakePageWithStaticKey:
    """Fake page that can find and fetch tpclient script for testing."""

    def __init__(self) -> None:
        """Initialize with eval count tracker."""
        self._eval_count = 0
        self._url = "https://tankpit.com/play"
        self._keyboard = NoOpKeyboard()

    @property
    def url(self) -> str:
        """Return test URL."""
        return self._url

    @property
    def keyboard(self) -> NoOpKeyboard:
        """Return keyboard interface."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for timeout."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for function - always succeeds."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Return script URL or JS content based on the expression.

        First call looks for tpclient URL -> return string URL
        Second call fetches content -> return JS with 1000-char key
        """
        self._eval_count += 1
        if "fetch" in expression:
            # Return JS content with a 1000-char key
            key = "A" * 1000
            return f'var x = "{key}";'
        # First call - looking for tpclient script URL
        return "https://tankpit.com/js/tpclient.min.js"


class FakePageNoKey:
    """Fake page that returns JS without a 1000-char key."""

    def __init__(self) -> None:
        """Initialize."""
        self._url = "https://tankpit.com/play"
        self._keyboard = NoOpKeyboard()

    @property
    def url(self) -> str:
        """Return test URL."""
        return self._url

    @property
    def keyboard(self) -> NoOpKeyboard:
        """Return keyboard interface."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for timeout."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for function - always succeeds."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Return script URL or JS content without 1000-char key."""
        if "fetch" in expression:
            # Return JS without a 1000-char key
            return 'var x = "short_key";'
        return "https://tankpit.com/js/tpclient.min.js"


class FakePageFetchFails:
    """Fake page where fetch returns non-string."""

    def __init__(self) -> None:
        """Initialize."""
        self._url = "https://tankpit.com/play"
        self._keyboard = NoOpKeyboard()

    @property
    def url(self) -> str:
        """Return test URL."""
        return self._url

    @property
    def keyboard(self) -> NoOpKeyboard:
        """Return keyboard interface."""
        return self._keyboard

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for timeout."""
        _ = timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for function - always succeeds."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Return script URL or None for fetch."""
        if "fetch" in expression:
            # Return None (simulates failed fetch)
            return None
        return "https://tankpit.com/js/tpclient.min.js"


# =============================================================================
# BrowserSession Static Key Tests
# =============================================================================
