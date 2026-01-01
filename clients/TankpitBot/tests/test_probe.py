"""Tests for tankpit_bot.probe module."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
from tankpit_bot.browser import BrowserError, cdp_timestamp_to_ms, get_current_time_ms
from tankpit_bot.probe import (
    DEFAULT_MOUSE_POSITIONS,
    DEFAULT_PROBE_KEYS,
    GameNotJoinedError,
    PlaywrightNotInstalledError,
    ProbeError,
    ProtocolProbe,
    _log_discovered_commands,
    extract_cdp_evaluate_value,
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


def test_extract_cdp_evaluate_value_success() -> None:
    """Test extract_cdp_evaluate_value extracts value from valid result."""
    result: JSONObject = {"result": {"value": "test_value"}}
    assert extract_cdp_evaluate_value(result) == "test_value"


def test_extract_cdp_evaluate_value_converts_to_string() -> None:
    """Test extract_cdp_evaluate_value converts non-string values to string."""
    result: JSONObject = {"result": {"value": 123}}
    assert extract_cdp_evaluate_value(result) == "123"


def test_extract_cdp_evaluate_value_raises_on_invalid_result() -> None:
    """Test extract_cdp_evaluate_value raises ProbeError when result is not dict."""
    result: JSONObject = {"error": "simulated error"}
    with pytest.raises(ProbeError, match=r"CDP Runtime\.evaluate returned invalid result"):
        extract_cdp_evaluate_value(result)


def test_extract_cdp_evaluate_value_raises_on_missing_value() -> None:
    """Test extract_cdp_evaluate_value raises ProbeError when value is missing."""
    result: JSONObject = {"result": {}}
    with pytest.raises(ProbeError, match=r"CDP Runtime\.evaluate result missing value"):
        extract_cdp_evaluate_value(result)


def test_extract_cdp_evaluate_value_raises_on_none_value() -> None:
    """Test extract_cdp_evaluate_value raises ProbeError when value is None."""
    result: JSONObject = {"result": {"value": None}}
    with pytest.raises(ProbeError, match=r"CDP Runtime\.evaluate result missing value"):
        extract_cdp_evaluate_value(result)


def test_get_current_time_ms_returns_int() -> None:
    """Test get_current_time_ms returns an integer."""
    result = get_current_time_ms()
    assert type(result) is int
    assert result > 0


def test_cdp_timestamp_to_ms() -> None:
    """Test cdp_timestamp_to_ms converts seconds to milliseconds."""
    result = cdp_timestamp_to_ms(12345.678)
    assert result == 12345678


# =============================================================================
# Constants Tests
# =============================================================================


def test_default_probe_keys_contains_known_commands() -> None:
    """Test DEFAULT_PROBE_KEYS contains keys with known command mappings."""
    # Only keys with known command IDs are included
    assert "s" in DEFAULT_PROBE_KEYS  # Radar (CMD_RADAR = 102)
    assert "d" in DEFAULT_PROBE_KEYS  # Mine (CMD_MINE = 107)
    assert "f" in DEFAULT_PROBE_KEYS  # Map open (CMD_MAP_OPEN = 108)
    assert "q" in DEFAULT_PROBE_KEYS  # Quit (plain command '-')
    # Keys without known commands are NOT included
    assert "w" not in DEFAULT_PROBE_KEYS  # Unknown
    assert " " not in DEFAULT_PROBE_KEYS  # Unknown
    # Exact expected list
    assert DEFAULT_PROBE_KEYS == ["s", "d", "f", "f", "q"]


def test_default_mouse_positions_is_empty() -> None:
    """Test DEFAULT_MOUSE_POSITIONS is empty (no mouse probing by default)."""
    assert len(DEFAULT_MOUSE_POSITIONS) == 0


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
            probe_keys=["s"],  # Use key with known command mapping
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
            probe_keys=["s"],  # Use key with known command mapping
            probe_mouse_positions=[],
            wait_after_join_ms=1000,
            wait_after_input_ms=100,
        )


def test_protocol_probe_run_captures_key_inputs(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run captures messages from key inputs."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["s", "d"],  # Use keys with known command mappings
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    assert session["base_url"] == "https://tankpit.com/play"
    assert len(session["results"]) == 2

    # First result should be for key 's' (radar)
    first_result = session["results"][0]
    assert first_result["input"]["input_type"] == "key"
    key_input = first_result["input"]["key_input"]
    assert type(key_input) is dict
    assert key_input["key"] == "s"
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
        probe_keys=["s"],  # Use key with known command mapping
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
    # s is the first key (Radar) in DEFAULT_PROBE_KEYS
    assert "Discovered: Key 's'" in output


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


def test_main_prints_key_discovered_commands(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints discovered key commands that generated messages."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    main()

    captured = capsys.readouterr()
    output = captured.out
    # Keys with known command mappings should generate messages
    assert "Discovered: Key 's'" in output
    assert "Discovered: Key 'd'" in output


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


def test_main_prints_all_default_key_commands(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() prints all discovered key commands from defaults."""
    from tests.fakes import fake_sync_playwright_probe_both_emit

    _test_hooks.sync_playwright = fake_sync_playwright_probe_both_emit

    main()

    captured = capsys.readouterr()
    output = captured.out
    # All default keys with known command mappings should generate discovered messages
    assert "Discovered: Key 's'" in output
    assert "Discovered: Key 'd'" in output
    assert "Discovered: Key 'f'" in output
    assert "Discovered: Key 'q'" in output


def test_protocol_probe_run_no_key_emit(fake_fs: FakeFileSystem) -> None:
    """Test ProtocolProbe.run when keys don't generate messages."""
    from tests.fakes import fake_sync_playwright_probe_no_key_emits

    _test_hooks.sync_playwright = fake_sync_playwright_probe_no_key_emits

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["s"],  # Use key with known command mapping
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
        probe_keys=["s"],  # Use key with known command mapping
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
        probe_keys=["s"],  # Use key with known command mapping
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


def test_playwright_not_installed_error_is_browser_error() -> None:
    """Test PlaywrightNotInstalledError is a BrowserError."""
    assert issubclass(PlaywrightNotInstalledError, BrowserError)


def test_game_not_joined_error_is_browser_error() -> None:
    """Test GameNotJoinedError is a BrowserError."""
    assert issubclass(GameNotJoinedError, BrowserError)


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


def test_log_discovered_commands_mouse_with_messages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test _log_discovered_commands logs mouse results with messages."""
    from tankpit_bot.types import CapturedMessage, MouseInput

    result = ProbeResult(
        input=ProbeInput(
            input_type="mouse",
            key_input=None,
            mouse_input=MouseInput(x=100, y=200, button="left"),
        ),
        timestamp_ms=12345,
        messages_before_count=0,
        messages_after=[
            CapturedMessage(
                timestamp_ms=12346,
                direction="sent",
                payload="test_payload",
                ws_url="wss://test.com/ws",
            )
        ],
    )
    _log_discovered_commands([result])

    captured = capsys.readouterr()
    output = captured.out
    assert "Discovered: Mouse (100,200) -> 1 msg(s)" in output


def test_protocol_probe_run_stabilization_reset(fake_fs: FakeFileSystem) -> None:
    """Test probe stabilization loop resets when new messages arrive.

    This covers the branch where stable_checks is reset to 0 when
    messages continue to arrive during the stabilization wait.
    """
    from tests.fakes import fake_sync_playwright_probe_delayed_messages

    _test_hooks.sync_playwright = fake_sync_playwright_probe_delayed_messages

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    session = probe.run(
        probe_keys=["s"],  # Use key with known command mapping
        probe_mouse_positions=[],
        wait_after_join_ms=100,
        wait_after_input_ms=50,
    )

    # Should still complete successfully with extra message from stabilization
    assert len(session["results"]) == 1
    # Should have captured the extra message during stabilization
    assert len(probe._messages) > 2  # More than just initial auth + room_list


def test_protocol_probe_build_xor_table_raises_without_magic(fake_fs: FakeFileSystem) -> None:
    """Test _build_xor_table raises ProbeError when magic is not captured."""
    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    # Magic is None by default
    with pytest.raises(ProbeError, match="Cannot build XOR table: magic key not captured"):
        probe._build_xor_table()


def test_protocol_probe_encode_xor_command_raises_without_table(fake_fs: FakeFileSystem) -> None:
    """Test _encode_xor_command raises ProbeError when XOR table is not built."""
    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    # XOR table is None by default
    with pytest.raises(ProbeError, match="XOR table not initialized"):
        probe._encode_xor_command(102)  # CMD_RADAR


def test_protocol_probe_send_key_command_unknown_key(
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test _send_key_command returns False for unknown keys."""
    from tests.fakes import FakeCDPSessionProbe

    probe = ProtocolProbe("https://tankpit.com/play", headless=True)
    cdp = FakeCDPSessionProbe()

    # 'w' is not in KEY_TO_COMMAND or KEY_TO_PLAIN_COMMAND
    result = probe._send_key_command(cdp, "w")

    assert result is False
    captured = capsys.readouterr()
    output = captured.out
    assert "Unknown key: w" in output


def test_main_with_keys_cli_arg(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() parses --keys CLI argument."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    # Set argv hook to return args with --keys
    _test_hooks.get_argv = lambda: ["probe", "--keys", "s,d,f"]

    main()

    captured = capsys.readouterr()
    output = captured.out
    # Should have overridden the default keys
    assert "Overriding probe keys" in output
    assert "Probe complete:" in output


def test_main_with_keys_cli_arg_missing_value(
    fake_env: FakeEnv,
    fake_fs: FakeFileSystem,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test main() ignores --keys when no value follows."""
    _test_hooks.sync_playwright = fake_sync_playwright_probe

    # Set argv hook to return args with --keys at the end (no value)
    _test_hooks.get_argv = lambda: ["probe", "--keys"]

    main()

    captured = capsys.readouterr()
    output = captured.out
    # Should NOT print "Overriding probe keys" since no value provided
    assert "Overriding probe keys" not in output
    # Should still complete with defaults
    assert "Probe complete:" in output


def test_send_websocket_bytes_returns_false_on_non_dict_result(
    fake_fs: FakeFileSystem,
) -> None:
    """Test _send_websocket_bytes returns False when result is not a dict."""

    from tankpit_bot.browser import BrowserSession

    class FakeCDPNonDictResult:
        """Fake CDP session that returns non-dict result for ws.send."""

        def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
            """Return non-dict result for ws.send evaluation."""
            _ = method
            _ = params
            # Return result where "result" is a string, not a dict
            return {"result": "NOT_A_DICT"}

        def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
            """Stub handler registration."""
            _ = event
            _ = handler

        def detach(self) -> None:
            """Stub detach."""

    session = BrowserSession("https://tankpit.com/play", headless=True)
    cdp = FakeCDPNonDictResult()

    # Call _send_websocket_bytes with the fake CDP
    result = session._send_websocket_bytes(cdp, b"test_data")

    # Should return False since result["result"] is not a dict
    assert result is False
