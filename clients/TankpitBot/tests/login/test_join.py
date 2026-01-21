"""Tests for join_room, auto_join_room, and ensure_on_play_page."""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.browser.login import ensure_on_play_page, handle_login_flow, join_room
from tests.login.conftest import FakeCDPLogin, FakeCDPLoginNonDictResult, FakePageLogin

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
