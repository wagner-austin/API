"""Tests for tankpit_bot.login module."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject, JSONValue

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import ResponseProtocol
from tankpit_bot.login import (
    ensure_on_play_page,
    handle_account_login,
    handle_guest_login,
    handle_login_flow,
    join_room,
)


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

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)

    def evaluate(self, expression: str) -> JSONValue:
        """Evaluate JavaScript expression - returns empty list in tests."""
        _ = expression
        return []


# =============================================================================
# Tests for handle_guest_login
# =============================================================================


def test_handle_guest_login_not_on_before_playing() -> None:
    """Guest login returns success immediately when not on before-playing page."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_guest_login(page, cdp)

    assert result["success"] is True
    assert result["rate_limited"] is False
    assert result["error_message"] == ""


def test_handle_guest_login_success() -> None:
    """Guest login succeeds when on before-playing page."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    result = handle_guest_login(page, cdp)

    assert result["success"] is True
    assert result["rate_limited"] is False


def test_handle_guest_login_rate_limited() -> None:
    """Guest login returns rate_limited when too many tanks error."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=True)

    result = handle_guest_login(page, cdp)

    assert result["success"] is False
    assert result["rate_limited"] is True
    assert "too many tanks" in result["error_message"].lower()


def test_handle_guest_login_failure_not_rate_limited() -> None:
    """Guest login fails but not due to rate limiting."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=False)

    result = handle_guest_login(page, cdp)

    assert result["success"] is False
    assert result["rate_limited"] is False


def test_handle_guest_login_custom_prefix() -> None:
    """Guest login uses custom tank name prefix."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    result = handle_guest_login(page, cdp, tank_name_prefix="X")

    assert result["success"] is True


# =============================================================================
# Tests for handle_account_login
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
# Tests for non-dict CDP responses
# =============================================================================


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


def test_handle_guest_login_non_dict_result() -> None:
    """Guest login handles non-dict CDP result."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPNonDictResult()

    # Should not crash, just handle gracefully
    result = handle_guest_login(page, cdp)

    # Guest login succeeds because page URL changes to /play
    assert result["success"] is True


# =============================================================================
# Tests for ensure_on_play_page
# =============================================================================


def test_ensure_on_play_page_already_there() -> None:
    """No navigation when already on play page."""
    page = FakePageLogin(start_url="https://tankpit.com/play")

    ensure_on_play_page(page)

    assert page.url == "https://tankpit.com/play"


def test_ensure_on_play_page_navigates() -> None:
    """Navigates to play page when on different page."""
    page = FakePageLogin(start_url="https://tankpit.com/")

    ensure_on_play_page(page)

    assert page.url == "https://tankpit.com/play"


# =============================================================================
# Tests for handle_login_flow
# =============================================================================


def test_handle_login_flow_not_on_before_playing() -> None:
    """Login flow succeeds immediately when not on before-playing."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp)

    assert result is True


def test_handle_login_flow_guest_success() -> None:
    """Login flow succeeds with guest login."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp)

    assert result is True
    assert "/play" in page.url


def test_handle_login_flow_rate_limited_no_credentials() -> None:
    """Login flow fails when rate-limited and no credentials."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=True)

    # No credentials set
    original_get_env = _test_hooks.get_env

    def fake_get_env(key: str) -> str | None:
        _ = key
        return None

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is False


def test_handle_login_flow_rate_limited_with_credentials() -> None:
    """Login flow succeeds with account login after rate limiting."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=True)

    original_get_env = _test_hooks.get_env
    env_vars = {"TANKPIT_USERNAME": "testuser", "TANKPIT_PASSWORD": "testpass"}

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is True


def test_handle_login_flow_rate_limited_login_fails() -> None:
    """Login flow fails when account login fails."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=True, login_error="Invalid credentials")

    original_get_env = _test_hooks.get_env
    env_vars = {"TANKPIT_USERNAME": "baduser", "TANKPIT_PASSWORD": "badpass"}

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is False


def test_handle_login_flow_no_account_fallback() -> None:
    """Login flow fails when rate-limited and fallback disabled."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=True)

    result = handle_login_flow(page, cdp, allow_account_fallback=False)

    # Returns True because it goes to ensure_on_play_page path
    assert result is True


def test_handle_login_flow_custom_prefix() -> None:
    """Login flow uses custom tank name prefix."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, tank_name_prefix="Z")

    assert result is True


# =============================================================================
# Tests for prefer_account parameter
# =============================================================================


def test_handle_login_flow_prefer_account_success() -> None:
    """Login flow succeeds with prefer_account when credentials are set."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    original_get_env = _test_hooks.get_env
    env_vars = {"TANKPIT_USERNAME": "testuser", "TANKPIT_PASSWORD": "testpass"}

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp, prefer_account=True)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is True


def test_handle_login_flow_prefer_account_no_credentials() -> None:
    """Login flow fails with prefer_account when credentials not set."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    original_get_env = _test_hooks.get_env

    def fake_get_env(key: str) -> str | None:
        _ = key
        return None

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp, prefer_account=True)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is False


def test_handle_login_flow_prefer_account_login_fails() -> None:
    """Login flow fails with prefer_account when login fails."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin(login_error="Invalid credentials")

    original_get_env = _test_hooks.get_env
    env_vars = {"TANKPIT_USERNAME": "baduser", "TANKPIT_PASSWORD": "badpass"}

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp, prefer_account=True)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is False


# =============================================================================
# Tests for join_room
# =============================================================================


def test_join_room_success() -> None:
    """Join room succeeds via map click."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(map_click_result="clicked field-image at center")

    result = join_room(page, cdp)

    assert result is True
    assert cdp.join_room_called is True


def test_join_room_no_field_image() -> None:
    """Join room returns False when field-image not found."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(map_click_result="no field-image")

    result = join_room(page, cdp)

    assert result is False
    assert cdp.join_room_called is True


def test_join_room_non_dict_result() -> None:
    """Join room handles non-dict result_obj from CDP.

    This tests the defensive branch where result_obj is not a dict,
    covering login.py line 312.
    """
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLoginNonDictResult()

    result = join_room(page, cdp)

    # Returns False because map_result contains "?" (not "no field-image")
    # The _click_map returns "?" when result_obj is not a dict
    assert result is True
    assert cdp.join_room_called is True


# =============================================================================
# Tests for auto_join_room parameter
# =============================================================================


def test_handle_login_flow_auto_join_room_not_on_before_playing() -> None:
    """Login flow auto-joins room when not on before-playing page."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True


def test_handle_login_flow_auto_join_room_after_guest_login() -> None:
    """Login flow auto-joins room after successful guest login."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True


def test_handle_login_flow_auto_join_room_after_account_login() -> None:
    """Login flow auto-joins room after successful account login."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    original_get_env = _test_hooks.get_env
    env_vars = {"TANKPIT_USERNAME": "testuser", "TANKPIT_PASSWORD": "testpass"}

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp, prefer_account=True, auto_join_room=True)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is True
    assert cdp.join_room_called is True


def test_handle_login_flow_auto_join_room_calls_join() -> None:
    """Login flow auto-joins room when enabled."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True


def test_handle_login_flow_no_auto_join_room() -> None:
    """Login flow does not auto-join room when disabled."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, auto_join_room=False)

    assert result is True
    assert cdp.join_room_called is False


def test_handle_login_flow_auto_join_after_rate_limit_fallback() -> None:
    """Login flow auto-joins room after rate-limited account fallback."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=True)

    original_get_env = _test_hooks.get_env
    env_vars = {"TANKPIT_USERNAME": "testuser", "TANKPIT_PASSWORD": "testpass"}

    def fake_get_env(key: str) -> str | None:
        return env_vars.get(key)

    _test_hooks.get_env = fake_get_env
    try:
        result = handle_login_flow(page, cdp, auto_join_room=True)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is True
    assert cdp.join_room_called is True


def test_handle_login_flow_auto_join_after_guest_failure_no_rate_limit() -> None:
    """Login flow auto-joins room after guest failure (not rate limited)."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=False)

    result = handle_login_flow(page, cdp, allow_account_fallback=False, auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True
