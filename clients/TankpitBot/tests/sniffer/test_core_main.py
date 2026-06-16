"""Tests for run_sniffer function and main() entry point."""

from __future__ import annotations

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
from tankpit_bot.sniffer.core import main, run_sniffer
from tankpit_bot.types import decode_capture_session
from tests.conftest import FakeEnv, FakeFileSystem
from tests.fakes import (
    fake_sync_playwright,
    fake_sync_playwright_no_messages,
    fake_sync_playwright_with_mixed_scripts,
    fake_sync_playwright_with_scripts,
)

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


def test_run_sniffer_with_live_decode(fake_fs: FakeFileSystem) -> None:
    """Test run_sniffer with live_decode enabled."""
    _test_hooks.sync_playwright = fake_sync_playwright

    session = run_sniffer(
        "https://tankpit.com",
        "output.json",
        live_decode=True,
    )

    assert len(session["messages"]) == 4


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
    assert "Captured 4 WebSocket messages in" in output
    assert "Saved to: runs\\sniff\\latest.capture_session.json" in output
    assert "Sniffer latest capture:" in output
    assert "runs\\sniff\\latest.capture_session.json" in output


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
    assert "Latest capture mirror:" in output
    assert "runs\\sniff\\latest.capture_session.json" in output


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


def test_main_live_decode_disabled(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test main() can disable live decode via env var."""
    _test_hooks.sync_playwright = fake_sync_playwright

    fake_env.set("TANKPIT_LIVE_DECODE", "false")
    main()

    fake_env.set("TANKPIT_LIVE_DECODE", "0")
    main()

    fake_env.set("TANKPIT_LIVE_DECODE", "no")
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


def test_main_logs_script_urls(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() logs script URLs discovered on the page."""
    _test_hooks.sync_playwright = fake_sync_playwright_with_scripts

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Loaded scripts (2):" in output
    assert "- https://tankpit.com/js/game.js" in output
    assert "- https://tankpit.com/js/protocol.js" in output


def test_main_logs_only_string_script_urls(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() only logs script URLs that are strings, skipping non-strings."""
    _test_hooks.sync_playwright = fake_sync_playwright_with_mixed_scripts

    main()

    captured = capsys.readouterr()
    output = captured.out
    # Should log the valid string URLs
    assert "Loaded scripts (4):" in output
    assert "- https://tankpit.com/js/valid.js" in output
    assert "- https://tankpit.com/js/another.js" in output
    # Should NOT log the non-string values (123, None)
    assert "- 123" not in output
    assert "- None" not in output


# =============================================================================
# MainPreferAccount Tests
# =============================================================================


class TestMainPreferAccount:
    """Tests for main() with prefer_account option."""

    def test_main_prefer_account_enabled(
        self,
        fake_env: FakeEnv,
        fake_fs: FakeFileSystem,
    ) -> None:
        """Test main() enables prefer_account via env var."""
        _test_hooks.sync_playwright = fake_sync_playwright

        fake_env.set("TANKPIT_PREFER_ACCOUNT", "true")
        main()  # Should not raise

        fake_env.set("TANKPIT_PREFER_ACCOUNT", "1")
        main()  # Should not raise

        fake_env.set("TANKPIT_PREFER_ACCOUNT", "yes")
        main()  # Should not raise
