"""Tests for handle_guest_login function."""

from __future__ import annotations

from tankpit_bot.browser.login import handle_guest_login
from tests.login.conftest import FakeCDPLogin, FakeCDPNonDictResult, FakePageLogin


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


def test_handle_guest_login_non_dict_result() -> None:
    """Guest login handles non-dict CDP result."""
    page = FakePageLogin(start_url="https://tankpit.com/before-playing")
    cdp = FakeCDPNonDictResult()

    # Should not crash, just handle gracefully
    result = handle_guest_login(page, cdp)

    # Guest login succeeds because page URL changes to /play
    assert result["success"] is True
