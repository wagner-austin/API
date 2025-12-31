"""Tests for tankpit_bot.probe module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
from tankpit_bot.probe import (
    DEFAULT_MOUSE_POSITIONS,
    DEFAULT_PROBE_KEYS,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
    ProbeError,
    ProtocolProbe,
    _cdp_timestamp_to_ms,
    _get_current_time_ms,
    _log_discovered_commands,
    main,
    run_probe,
)
from tankpit_bot.types import ProbeInput, ProbeResult, decode_probe_session
from tests.conftest import FakeEnv, FakeFileSystem
from tests.fakes import (
    fake_sync_playwright_probe,
    fake_sync_playwright_probe_no_messages,
)

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
# Constants Tests
# =============================================================================


def test_default_probe_keys_contains_wasd() -> None:
    """Test DEFAULT_PROBE_KEYS contains basic movement keys."""
    assert "w" in DEFAULT_PROBE_KEYS
    assert "a" in DEFAULT_PROBE_KEYS
    assert "s" in DEFAULT_PROBE_KEYS
    assert "d" in DEFAULT_PROBE_KEYS


def test_default_probe_keys_contains_arrows() -> None:
    """Test DEFAULT_PROBE_KEYS contains arrow keys."""
    assert "ArrowUp" in DEFAULT_PROBE_KEYS
    assert "ArrowDown" in DEFAULT_PROBE_KEYS
    assert "ArrowLeft" in DEFAULT_PROBE_KEYS
    assert "ArrowRight" in DEFAULT_PROBE_KEYS


def test_default_mouse_positions_contains_center() -> None:
    """Test DEFAULT_MOUSE_POSITIONS contains center position."""
    assert (0.5, 0.5) in DEFAULT_MOUSE_POSITIONS


def test_default_mouse_positions_has_five_entries() -> None:
    """Test DEFAULT_MOUSE_POSITIONS has 5 positions."""
    assert len(DEFAULT_MOUSE_POSITIONS) == 5


# =============================================================================
# ProtocolProbe Tests
# =============================================================================


def test_protocol_probe_init() -> None:
    """Test ProtocolProbe initialization."""
    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    assert probe._target_url == "https://tankpit.com/play"
    assert probe._headless is True


def test_protocol_probe_run_without_playwright() -> None:
    """Test ProtocolProbe.run raises error when Playwright not installed."""
    _test_hooks.sync_playwright = None
    probe = ProtocolProbe("https://tankpit.com/play")
    with pytest.raises(PlaywrightNotInstalledError, match="Playwright is not installed"):
        probe.run(
            probe_keys=["w"],
            probe_mouse_positions=[],
            wait_after_join_ms=1000,
            wait_after_input_ms=100,
        )


def test_protocol_probe_run_game_not_joined(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run raises error when no messages captured."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe_no_messages

    probe = ProtocolProbe("https://tankpit.com/play")
    with pytest.raises(GameNotJoinedError, match="No WebSocket messages captured"):
        probe.run(
            probe_keys=["w"],
            probe_mouse_positions=[],
            wait_after_join_ms=1000,
            wait_after_input_ms=100,
        )


def test_protocol_probe_run_captures_key_inputs(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run captures messages from key inputs."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["w", "a"],
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    assert session["base_url"] == "https://tankpit.com/play"
    assert len(session["results"]) == 2

    # First result should be for key 'w'
    first_result = session["results"][0]
    assert first_result["input"]["input_type"] == "key"
    key_input = first_result["input"]["key_input"]
    assert type(key_input) is dict
    assert key_input["key"] == "w"
    # Should have captured a sent message
    assert len(first_result["messages_after"]) == 1


def test_protocol_probe_run_captures_mouse_inputs(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run captures messages from mouse inputs."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    probe = ProtocolProbe("https://tankpit.com/play")
    session = probe.run(
        probe_keys=[],
        probe_mouse_positions=[(0.5, 0.5)],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    assert len(session["results"]) == 1

    # Result should be for mouse input
    result = session["results"][0]
    assert result["input"]["input_type"] == "mouse"
    mouse_input = result["input"]["mouse_input"]
    assert type(mouse_input) is dict
    # Viewport is 800x600, so 50% = 400, 300
    assert mouse_input["x"] == 400
    assert mouse_input["y"] == 300


# =============================================================================
# run_probe Tests
# =============================================================================


def test_run_probe_saves_to_file(fake_fs: FakeFileSystem) -> None:
    """Test run_probe saves probe session to file."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    session = run_probe(
        "https://tankpit.com/play",
        "probe_output.json",
        headless=True,
        probe_keys=["w"],
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    written_files = fake_fs.get_written_files()
    content = written_files["probe_output.json"]
    parsed = load_json_str(content)
    parsed_dict = narrow_json_to_dict(parsed)
    decoded = decode_probe_session(parsed_dict)
    assert decoded["session_id"] == session["session_id"]


def test_run_probe_uses_defaults(fake_fs: FakeFileSystem) -> None:
    """Test run_probe uses default keys and positions when not specified."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    session = run_probe(
        "https://tankpit.com/play",
        "probe_output.json",
        headless=True,
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # Should have results for all default keys + mouse positions
    expected_count = len(DEFAULT_PROBE_KEYS) + len(DEFAULT_MOUSE_POSITIONS)
    assert len(session["results"]) == expected_count


# =============================================================================
# main() Tests
# =============================================================================


def test_main_with_defaults(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() uses default values when env vars not set."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Probe complete:" in output
    assert "Saved to: probe_session.json" in output


def test_main_with_custom_env(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() reads custom values from environment."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe
    fake_env.set("TANKPIT_URL", "https://custom.tankpit.com/play")
    fake_env.set("TANKPIT_PROBE_OUTPUT", "custom_probe.json")
    fake_env.set("TANKPIT_HEADLESS", "true")
    fake_env.set("TANKPIT_WAIT_JOIN_MS", "2000")
    fake_env.set("TANKPIT_WAIT_INPUT_MS", "100")

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Saved to: custom_probe.json" in output


def test_main_headless_variations(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test main() parses various headless env values."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    fake_env.set("TANKPIT_HEADLESS", "1")
    main()

    fake_env.set("TANKPIT_HEADLESS", "yes")
    main()

    fake_env.set("TANKPIT_HEADLESS", "TRUE")
    main()


def test_main_installs_playwright_when_none(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
) -> None:
    """Test main() installs playwright via get_sync_playwright when None."""

    def get_fake_factory() -> SyncPlaywrightFactoryProtocol:
        """Return the fake sync_playwright factory function."""
        return fake_sync_playwright_probe

    _test_hooks.sync_playwright = None
    _test_hooks.get_sync_playwright = get_fake_factory

    main()

    assert _test_hooks.sync_playwright == fake_sync_playwright_probe


def test_main_prints_discovered_commands(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints discovered commands that generated messages."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Discovered: Key 'w'" in output


def test_protocol_probe_run_mouse_emits_messages(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run captures messages from mouse inputs that emit."""
    from tests.fakes import fake_sync_playwright_probe_mouse_emits

    _test_hooks.sync_playwright = fake_sync_playwright_probe_mouse_emits

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=[],
        probe_mouse_positions=[(0.5, 0.5)],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    assert len(session["results"]) == 1
    result = session["results"][0]
    assert result["input"]["input_type"] == "mouse"
    # Should have captured a sent message from mouse click
    assert len(result["messages_after"]) == 1


def test_main_prints_mouse_discovered_commands(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints discovered mouse commands that generated messages."""
    from tests.fakes import fake_sync_playwright_probe_mouse_emits

    _test_hooks.sync_playwright = fake_sync_playwright_probe_mouse_emits

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Discovered: Mouse" in output


def test_run_probe_multiple_mouse_positions_with_messages(
    fake_fs: FakeFileSystem,
) -> None:
    """Test run_probe with multiple mouse positions that emit messages.

    This covers the branch 573->567 (loop continuation after mouse result).
    """
    from tests.fakes import fake_sync_playwright_probe_mouse_emits

    _test_hooks.sync_playwright = fake_sync_playwright_probe_mouse_emits

    session = run_probe(
        "https://tankpit.com/play",
        "probe_output.json",
        headless=True,
        probe_keys=[],
        probe_mouse_positions=[(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # All 3 mouse positions should have generated messages
    results_with_messages = [r for r in session["results"] if len(r["messages_after"]) > 0]
    assert len(results_with_messages) == 3
    for r in results_with_messages:
        assert r["input"]["input_type"] == "mouse"


def test_main_prints_both_key_and_mouse_commands(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints both key and mouse discovered commands.

    This test uses a probe that emits messages on both key and mouse inputs
    to cover the loop iteration from mouse (573) back to for (567).
    """
    from tests.fakes import fake_sync_playwright_probe_both_emit

    _test_hooks.sync_playwright = fake_sync_playwright_probe_both_emit

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Discovered: Key 'w'" in output
    assert "Discovered: Mouse" in output


def test_protocol_probe_run_no_key_emit(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run when keys don't generate messages."""
    from tests.fakes import fake_sync_playwright_probe_no_key_emits

    _test_hooks.sync_playwright = fake_sync_playwright_probe_no_key_emits

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["w"],
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    assert len(session["results"]) == 1
    result = session["results"][0]
    assert result["input"]["input_type"] == "key"
    # No messages should be captured since emit_on_key is False
    assert len(result["messages_after"]) == 0


def test_main_no_discovered_commands(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() when no inputs generate messages."""
    from tests.fakes import fake_sync_playwright_probe_no_key_emits

    _test_hooks.sync_playwright = fake_sync_playwright_probe_no_key_emits

    main()

    captured = capsys.readouterr()
    output = captured.out
    assert "Probe complete:" in output
    # Should NOT print discovered commands since none generated messages
    assert "Discovered:" not in output


def test_protocol_probe_run_invalid_viewport(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run handles invalid viewport result gracefully."""
    from tests.fakes import fake_sync_playwright_probe_invalid_viewport

    _test_hooks.sync_playwright = fake_sync_playwright_probe_invalid_viewport

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["w"],
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # Should complete successfully with default viewport
    assert len(session["results"]) == 1


def test_protocol_probe_run_non_dict_viewport(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run handles non-dict viewport result gracefully."""
    from tests.fakes import fake_sync_playwright_probe_non_dict_viewport

    _test_hooks.sync_playwright = fake_sync_playwright_probe_non_dict_viewport

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["w"],
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # Should complete successfully with default viewport
    assert len(session["results"]) == 1


# =============================================================================
# Error Class Tests
# =============================================================================


def test_probe_error_is_exception() -> None:
    """Test ProbeError is an Exception."""
    assert issubclass(ProbeError, Exception)
    err = ProbeError("test error")
    assert str(err) == "test error"


def test_playwright_not_installed_error_is_probe_error() -> None:
    """Test PlaywrightNotInstalledError is a ProbeError."""
    assert issubclass(PlaywrightNotInstalledError, ProbeError)


def test_game_not_joined_error_is_probe_error() -> None:
    """Test GameNotJoinedError is a ProbeError."""
    assert issubclass(GameNotJoinedError, ProbeError)


# =============================================================================
# Defensive Branch Tests
# =============================================================================


def test_log_discovered_commands_key_with_none_input() -> None:
    """Test _log_discovered_commands handles key result with None key_input."""
    result = ProbeResult(
        input=ProbeInput(input_type="key", key_input=None, mouse_input=None),
        timestamp_ms=12345,
        messages_before_count=0,
        messages_after=[],
    )
    # Should not raise, just skip logging
    _log_discovered_commands([result])


def test_log_discovered_commands_mouse_with_none_input() -> None:
    """Test _log_discovered_commands handles mouse result with None mouse_input."""
    result = ProbeResult(
        input=ProbeInput(input_type="mouse", key_input=None, mouse_input=None),
        timestamp_ms=12345,
        messages_before_count=0,
        messages_after=[],
    )
    # Should not raise, just skip logging
    _log_discovered_commands([result])
