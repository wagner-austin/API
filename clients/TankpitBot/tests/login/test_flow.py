"""Tests for handle_login_flow function."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.browser.login import handle_login_flow
from tests.login.conftest import FakeCDPLogin, FakePageLogin

# =============================================================================
# Basic Login Flow Tests
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
