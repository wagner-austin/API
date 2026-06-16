"""Tests for handle_account_login function and edge cases."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject, JSONValue

from tankpit_bot._test_hooks import KeyboardProtocol, ResponseProtocol
from tankpit_bot._test_hooks.cdp import RouteFulfillHandler
from tankpit_bot.browser.login import handle_account_login
from tests.login.conftest import FakeCDPLogin, FakePageLogin
from tests.no_op_keyboard import NoOpKeyboard

# =============================================================================
# Basic Account Login Tests
# =============================================================================


def test_handle_account_login_success() -> None:
    """Account login succeeds with valid credentials."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    result = handle_account_login(page, cdp, "testuser", "testpass")

    assert result["success"] is True
    assert result["error_message"] == ""


def test_handle_account_login_failure() -> None:
    """Account login fails with invalid credentials."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin(login_error="Invalid username or password")

    result = handle_account_login(page, cdp, "baduser", "badpass")

    assert result["success"] is False
    assert "Invalid" in result["error_message"]


# =============================================================================
# Edge Case Fake Classes
# =============================================================================


class FakeCDPLoginFormNeverReady:
    """Fake CDP where login form is never ready (returns 'waiting' always)."""

    def __init__(self) -> None:
        """Initialize fake CDP session."""
        self._account_login_started = False

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command."""
        if method == "Runtime.evaluate":
            expression = str(params.get("expression", "")) if params else ""

            # Open login overlay
            if "#login" in expression:
                self._account_login_started = True

            # Login form visibility check - ALWAYS return 'waiting'
            if "login-username" in expression and "offsetParent" in expression:
                return {"result": {"value": "waiting"}}

            return {"result": {"value": "success"}}
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach CDP session."""


class FakePageLoginTimeout:
    """Fake page that stays on before-playing forever (for timeout testing)."""

    def __init__(self) -> None:
        """Initialize fake page."""
        self._url = "https://tankpit.com/before-playing"
        self._wait_count = 0

    @property
    def url(self) -> str:
        """Get current URL - always stays on before-playing."""
        return self._url

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url
        return None

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for timeout."""
        _ = timeout
        self._wait_count += 1
        # Never transition to /play - causes timeout

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for a JavaScript function."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression."""
        _ = expression
        return []

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get keyboard for typing."""
        return NoOpKeyboard()


class FakeCDPLoginNoErrors:
    """Fake CDP that returns no errors during account login (for timeout testing)."""

    def __init__(self) -> None:
        """Initialize fake CDP session."""
        self._account_login_started = False

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command."""
        if method == "Runtime.evaluate":
            expression = str(params.get("expression", "")) if params else ""

            # Open login overlay
            if "#login" in expression:
                self._account_login_started = True

            # Form visibility check - return ready
            if "login-username" in expression and "offsetParent" in expression:
                return {"result": {"value": "ready"}}

            # Error check - return empty (no errors)
            if "errors" in expression or "error" in expression:
                return {"result": {"value": ""}}

            return {"result": {"value": "success"}}
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach CDP session."""


class FakePageLoginIntermediateUrl:
    """Fake page that goes through intermediate URL during login."""

    def __init__(self) -> None:
        """Initialize fake page."""
        self._url = "https://tankpit.com/before-playing"
        self._wait_count = 0

    @property
    def url(self) -> str:
        """Get current URL - returns intermediate URL then timeout."""
        # After a few waits, switch to an intermediate URL (not /play, not before-playing)
        if self._wait_count >= 5:
            return "https://tankpit.com/loading"  # Intermediate URL
        return self._url

    def goto(
        self,
        url: str,
        *,
        referer: str | None = None,
        timeout: float | None = None,
        wait_until: str | None = None,
    ) -> ResponseProtocol | None:
        """Navigate to URL."""
        _ = (referer, timeout, wait_until)
        self._url = url
        return None

    def wait_for_timeout(self, timeout: float) -> None:
        """Wait for timeout."""
        _ = timeout
        self._wait_count += 1

    def set_content(self, html: str, *, timeout: float | None = None) -> None:
        _ = (html, timeout)

    def route(self, url: str, handler: RouteFulfillHandler) -> None:
        _ = (url, handler)

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for a JavaScript function."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression."""
        _ = expression
        return []

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get keyboard for typing."""
        return NoOpKeyboard()


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_handle_account_login_form_never_ready() -> None:
    """Account login warns when form is never ready after 10 retries."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLoginFormNeverReady()

    # This exercises lines 228-230 (for/else: form not ready after waiting)
    result = handle_account_login(page, cdp, "testuser", "testpass")

    # Login still succeeds because fake page transitions to /play after 2 waits
    assert result["success"] is True


def test_handle_account_login_timeout() -> None:
    """Account login times out when login never completes."""
    page = FakePageLoginTimeout()
    cdp = FakeCDPLoginNoErrors()

    # This exercises lines 313-314 (login timeout)
    result = handle_account_login(page, cdp, "testuser", "testpass")

    assert result["success"] is False
    assert "timeout" in result["error_message"].lower()


def test_handle_account_login_intermediate_url_then_timeout() -> None:
    """Account login goes through intermediate URL before timing out.

    This exercises the branch partial at line 289->279 where URL is
    neither /play nor before-playing.
    """
    page = FakePageLoginIntermediateUrl()
    cdp = FakeCDPLoginNoErrors()

    result = handle_account_login(page, cdp, "testuser", "testpass")

    assert result["success"] is False
    assert "timeout" in result["error_message"].lower()
