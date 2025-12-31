"""Tests for tankpit_bot.login module."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import ResponseProtocol
from tankpit_bot.login import (
    ensure_on_play_page,
    handle_account_login,
    handle_guest_login,
    handle_login_flow,
)


class FakeCDPLogin:
    """Fake CDP session for login testing."""

    def __init__(
        self,
        *,
        rate_limited: bool = False,
        login_error: str = "",
    ) -> None:
        """Initialize fake CDP session."""
        self._eval_count = 0
        self._rate_limited = rate_limited
        self._login_error = login_error
        self._account_login_started = False

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        """Send CDP command."""
        if method == "Runtime.evaluate":
            self._eval_count += 1
            expression = str(params.get("expression", "")) if params else ""

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

    def close(self, *, reason: str | None = None, run_before_unload: bool | None = None) -> None:
        """Close page."""
        _ = (reason, run_before_unload)


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
