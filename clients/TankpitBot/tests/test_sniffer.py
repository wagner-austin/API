"""Tests for tankpit_bot.sniffer module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
from tankpit_bot.sniffer import (
    PlaywrightNotInstalledError,
    SnifferError,
    WebSocketSniffer,
    _cdp_timestamp_to_ms,
    _get_current_time_ms,
    main,
    run_sniffer,
)
from tankpit_bot.types import decode_capture_session
from tests.conftest import FakeEnv, FakeFileSystem
from tests.fakes import fake_sync_playwright, fake_sync_playwright_no_messages

# =============================================================================
# Helper Function Tests
# =============================================================================


def test_get_current_time_ms_returns_int() -> None:
    """Test _get_current_time_ms returns an integer."""
    result = _get_current_time_ms()
    assert type(result) is int
    assert result > 0


def test_cdp_timestamp_to_ms() -> None:
    """Test _cdp_timestamp_to_ms converts seconds to milliseconds."""
    result = _cdp_timestamp_to_ms(12345.678)
    assert result == 12345678


# =============================================================================
# WebSocketSniffer Tests
# =============================================================================


def test_websocket_sniffer_init() -> None:
    """Test WebSocketSniffer initialization."""
    sniffer = WebSocketSniffer("https://example.com", headless=True)
    assert sniffer._target_url == "https://example.com"
    assert sniffer._headless is True


def test_websocket_sniffer_run_without_playwright() -> None:
    """Test WebSocketSniffer.run raises error when Playwright not installed."""
    _test_hooks.sync_playwright = None
    sniffer = WebSocketSniffer("https://example.com")
    with pytest.raises(PlaywrightNotInstalledError, match="Playwright is not installed"):
        sniffer.run(1000)


def test_websocket_sniffer_run_captures_messages(fake_fs: FakeFileSystem) -> None:
    """Test WebSocketSniffer.run captures WebSocket messages."""
    _test_hooks.sync_playwright = fake_sync_playwright

    sniffer = WebSocketSniffer("https://tankpit.com", headless=True)
    session = sniffer.run(5000)

    assert session["base_url"] == "https://tankpit.com"
    assert len(session["messages"]) == 2
    assert session["messages"][0]["direction"] == "sent"
    assert session["messages"][0]["payload"] == "sent message"
    assert session["messages"][1]["direction"] == "received"
    assert session["messages"][1]["payload"] == "received message"


def test_websocket_sniffer_records_websocket_urls(fake_fs: FakeFileSystem) -> None:
    """Test WebSocketSniffer records WebSocket URLs from created events."""
    _test_hooks.sync_playwright = fake_sync_playwright

    sniffer = WebSocketSniffer("https://tankpit.com")
    session = sniffer.run(1000)

    for msg in session["messages"]:
        assert msg["ws_url"] == "wss://example.com/ws"


# =============================================================================
# run_sniffer Tests
# =============================================================================


def test_run_sniffer_saves_to_file(fake_fs: FakeFileSystem) -> None:
    """Test run_sniffer saves capture session to file."""
    _test_hooks.sync_playwright = fake_sync_playwright

    session = run_sniffer(
        "https://tankpit.com",
        "output.json",
        headless=True,
        capture_duration_ms=1000,
    )

    written_files = fake_fs.get_written_files()
    content = written_files["output.json"]
    parsed = load_json_str(content)
    parsed_dict = narrow_json_to_dict(parsed)
    decoded = decode_capture_session(parsed_dict)
    assert decoded["session_id"] == session["session_id"]


# =============================================================================
# main() Tests
# =============================================================================


def test_main_with_defaults(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() uses default values when env vars not set."""
    _test_hooks.sync_playwright = fake_sync_playwright

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Captured 2 WebSocket messages in" in output
    assert "Saved to: capture_session.json" in output


def test_main_with_custom_env(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() reads custom values from environment."""
    _test_hooks.sync_playwright = fake_sync_playwright
    fake_env.set("TANKPIT_URL", "https://custom.tankpit.com")
    fake_env.set("TANKPIT_OUTPUT", "custom_output.json")
    fake_env.set("TANKPIT_HEADLESS", "true")
    fake_env.set("TANKPIT_DURATION_MS", "5000")

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Saved to: custom_output.json" in output


def test_main_headless_variations(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test main() parses various headless env values."""
    _test_hooks.sync_playwright = fake_sync_playwright

    fake_env.set("TANKPIT_HEADLESS", "1")
    main()

    fake_env.set("TANKPIT_HEADLESS", "yes")
    main()

    fake_env.set("TANKPIT_HEADLESS", "TRUE")
    main()


def test_main_prints_discovered_urls(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints discovered WebSocket URLs."""
    _test_hooks.sync_playwright = fake_sync_playwright

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Discovered WebSocket URLs (1):" in output
    assert "wss://example.com/ws" in output


def test_main_installs_playwright_when_none(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test main() installs playwright via get_sync_playwright when None."""

    def get_fake_factory() -> SyncPlaywrightFactoryProtocol:
        """Return the fake sync_playwright factory function."""
        return fake_sync_playwright

    _test_hooks.sync_playwright = None
    _test_hooks.get_sync_playwright = get_fake_factory

    main()

    assert _test_hooks.sync_playwright == fake_sync_playwright


def test_main_guest_login_on_before_playing(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() attempts guest login when on before-playing page."""
    _test_hooks.sync_playwright = fake_sync_playwright
    fake_env.set("TANKPIT_URL", "https://tankpit.com/before-playing")

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Attempting guest login" in output
    assert "Fill result:" in output
    assert "Submit result:" in output


def test_main_rate_limit_no_credentials(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() shows warning when rate-limited without credentials."""
    from tests.fakes import fake_sync_playwright_rate_limited

    _test_hooks.sync_playwright = fake_sync_playwright_rate_limited
    fake_env.set("TANKPIT_URL", "https://tankpit.com/before-playing")

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Rate limited. Set TANKPIT_USERNAME" in output


def test_main_rate_limit_with_credentials(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() attempts login when rate-limited with credentials."""
    from tests.fakes import fake_sync_playwright_rate_limited

    _test_hooks.sync_playwright = fake_sync_playwright_rate_limited
    fake_env.set("TANKPIT_URL", "https://tankpit.com/before-playing")
    fake_env.set("TANKPIT_USERNAME", "testuser")
    fake_env.set("TANKPIT_PASSWORD", "testpass")

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Rate limited - logging in as testuser" in output
    assert "Login:" in output
    assert "After login, URL:" in output
    assert "Login successful" in output


def test_main_rate_limit_login_fails(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() shows warning when login fails."""
    from tests.fakes import fake_sync_playwright_login_fails

    _test_hooks.sync_playwright = fake_sync_playwright_login_fails
    fake_env.set("TANKPIT_URL", "https://tankpit.com/before-playing")
    fake_env.set("TANKPIT_USERNAME", "testuser")
    fake_env.set("TANKPIT_PASSWORD", "wrongpass")

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Rate limited - logging in as testuser" in output
    assert "Login errors:" in output


def test_main_with_no_websocket_urls(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() when no WebSocket URLs are discovered."""
    _test_hooks.sync_playwright = fake_sync_playwright_no_messages

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Captured 0 WebSocket messages in" in output
    assert "Discovered WebSocket URLs" not in output


# =============================================================================
# Error Class Tests
# =============================================================================


def test_sniffer_error_is_exception() -> None:
    """Test SnifferError is an Exception."""
    assert issubclass(SnifferError, Exception)
    err = SnifferError("test error")
    assert str(err) == "test error"


def test_playwright_not_installed_error_is_sniffer_error() -> None:
    """Test PlaywrightNotInstalledError is a SnifferError."""
    assert issubclass(PlaywrightNotInstalledError, SnifferError)
