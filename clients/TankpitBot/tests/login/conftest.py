"""Shared fixtures and fake classes for login testing."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject, JSONValue

from tankpit_bot._test_hooks import KeyboardProtocol, ResponseProtocol
from tests.fakes import FakeKeyboard


class FakeCDPLogin:
    """Fake CDP session for login testing."""

    def __init__(
        self,
        *,
        rate_limited: bool = False,
        login_error: str = "",
        map_click_result: str = "clicked field-image at center",
        troop_click_result: str = "clicked pick-troop-blue",
    ) -> None:
        """Initialize fake CDP session."""
        self._eval_count = 0
        self._rate_limited = rate_limited
        self._login_error = login_error
        self._account_login_started = False
        self._map_click_result = map_click_result
        self._troop_click_result = troop_click_result
        self.join_room_called = False
        self.troop_click_called = False

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command."""
        if method == "Runtime.evaluate":
            self._eval_count += 1
            expression = str(params.get("expression", "")) if params else ""

            # Detect map click (field-image)
            if "field-image" in expression:
                self.join_room_called = True
                return {"result": {"value": self._map_click_result}}

            # Detect troop color click (pick-troop)
            if "pick-troop" in expression:
                self.troop_click_called = True
                return {"result": {"value": self._troop_click_result}}

            # Detect error checks by looking at the expression
            if "errors" in expression or "error" in expression:
                # Guest login error check
                if self._rate_limited and not self._account_login_started:
                    return {"result": {"value": "There are too many tanks"}}
                # Account login error check
                if self._account_login_started:
                    return {"result": {"value": self._login_error}}
                return {"result": {"value": ""}}

            # Detect account login by login overlay open
            if "#login" in expression:
                self._account_login_started = True

            # Login form visibility check (looking for 'ready'/'waiting')
            if "login-username" in expression and "offsetParent" in expression:
                return {"result": {"value": "ready"}}

            return {"result": {"value": "success"}}
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach CDP session."""


class FakeCDPLoginNonDictResult:
    """Fake CDP session that returns non-dict result for map click.

    This tests the defensive branch where result_obj is not a dict.
    """

    def __init__(self) -> None:
        """Initialize fake CDP session."""
        self.join_room_called = False

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command returning non-dict result for map click."""
        if method == "Runtime.evaluate":
            expression = str(params.get("expression", "")) if params else ""
            if "field-image" in expression:
                self.join_room_called = True
                # Return result that is not a dict (a list instead)
                return {"result": ["not", "a", "dict"]}
            return {"result": {"value": "success"}}
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach CDP session."""


class FakePageLogin:
    """Fake page for login testing."""

    def __init__(
        self,
        *,
        start_url: str = "https://tankpit.com/play",
        stays_on_before_playing: bool = False,
    ) -> None:
        """Initialize fake page."""
        self._url = start_url
        self._stays_on_before_playing = stays_on_before_playing
        self._wait_count = 0

    @property
    def url(self) -> str:
        """Get current URL."""
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
        # After 2nd wait (post-submit), update URL unless staying on before-playing
        if (
            self._wait_count == 2
            and not self._stays_on_before_playing
            and "before-playing" in self._url
        ):
            self._url = "https://tankpit.com/play"

    def wait_for_event(self, event: str, *, timeout: float | None = None) -> None:
        """Wait for an event - returns immediately in tests."""
        _ = (event, timeout)

    def wait_for_function(self, expression: str, *, timeout: float | None = None) -> None:
        """Wait for a JavaScript function - returns immediately in tests."""
        _ = (expression, timeout)

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression - returns empty list in tests."""
        _ = expression
        return []

    @property
    def keyboard(self) -> KeyboardProtocol:
        """Get keyboard for typing."""
        return FakeKeyboard()


class FakeCDPNonDictResult:
    """Fake CDP that returns non-dict results."""

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command returning non-dict result."""
        _ = params
        if method == "Runtime.evaluate":
            # Return result as a list instead of dict
            return {"result": ["not", "a", "dict"]}
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        """Register event handler."""
        _ = (event, handler)

    def detach(self) -> None:
        """Detach CDP session."""
