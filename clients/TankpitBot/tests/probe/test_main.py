"""Tests for probe main() function."""

from __future__ import annotations

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import SyncPlaywrightFactoryProtocol
from tankpit_bot.probe import main
from tests.conftest import FakeEnv, FakeFileSystem
from tests.fakes import fake_sync_playwright_probe


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
