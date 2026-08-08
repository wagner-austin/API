"""Tests for handle_login_flow's auto_join_room orchestration.

Split from ``test_join.py`` (2026-08-07) at the 600-line ceiling: the
room-join primitives and the login-flow orchestration that drives them
are separate concerns and now live in separate modules.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.browser.login import handle_login_flow
from tankpit_bot.sniffer.world_service import WorldService
from tests.login.conftest import (
    FakeCDPLogin,
    FakePageLogin,
)


def test_handle_login_flow_auto_join_room_not_on_before_playing() -> None:
    """Login flow auto-joins room when not on before-playing page."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, WorldService(), auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


def test_handle_login_flow_auto_join_room_after_guest_login() -> None:
    """Login flow auto-joins room after successful guest login."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, WorldService(), auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


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
        result = handle_login_flow(
            page, cdp, WorldService(), prefer_account=True, auto_join_room=True
        )
    finally:
        _test_hooks.get_env = original_get_env

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


def test_handle_login_flow_auto_join_room_calls_join() -> None:
    """Login flow auto-joins room when enabled."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, WorldService(), auto_join_room=True)

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


def test_handle_login_flow_no_auto_join_room() -> None:
    """Login flow does not auto-join room when disabled."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin()

    result = handle_login_flow(page, cdp, WorldService(), auto_join_room=False)

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
        result = handle_login_flow(page, cdp, WorldService(), auto_join_room=True)
    finally:
        _test_hooks.get_env = original_get_env

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.selected_room_id == "1"
    assert cdp.enter_room_called is True


def test_handle_login_flow_auto_join_after_guest_failure_no_rate_limit() -> None:
    """Login flow auto-joins room after guest failure (not rate limited)."""
    page = FakePageLogin(
        start_url="https://tankpit.com/before-playing",
        stays_on_before_playing=True,
    )
    cdp = FakeCDPLogin(rate_limited=False)

    result = handle_login_flow(
        page, cdp, WorldService(), allow_account_fallback=False, auto_join_room=True
    )

    assert result is True
    assert cdp.join_room_called is True
    assert cdp.enter_room_called is True


def test_handle_login_flow_returns_false_when_auto_join_fails() -> None:
    """Login flow propagates room-join failure instead of continuing."""
    page = FakePageLogin(start_url="https://tankpit.com/play")
    cdp = FakeCDPLogin(include_practice_room=False)

    result = handle_login_flow(page, cdp, WorldService(), auto_join_room=True)

    assert result is False
    assert cdp.join_room_called is False
    assert cdp.enter_room_called is False
