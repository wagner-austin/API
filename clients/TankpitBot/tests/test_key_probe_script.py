"""Tests for scripts.key_probe."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Generator

import pytest

from scripts import _test_hooks as script_hooks
from scripts import key_probe as key_probe_script
from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab.key_probe import DEFAULT_KEYS, KeyProbeSessionDict
from tests.test_enemy_teleport_probe_script import _FakeSyncPlaywrightFactory


@pytest.fixture()
def _restore_script_hooks() -> Generator[None, None, None]:
    original_logging = script_hooks.setup_rich_logging
    original_get_env = core_hooks.get_env
    original_sync_playwright = core_hooks.sync_playwright
    original_get_sync_playwright = core_hooks.get_sync_playwright
    original_run = key_probe_script.run_key_probe
    yield
    script_hooks.setup_rich_logging = original_logging
    core_hooks.get_env = original_get_env
    core_hooks.sync_playwright = original_sync_playwright
    core_hooks.get_sync_playwright = original_get_sync_playwright
    key_probe_script.run_key_probe = original_run


def _session() -> KeyProbeSessionDict:
    return KeyProbeSessionDict(
        session_id="key-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="key_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
            "game_ready_timestamp_ms": 300,
            "intel_ready_timestamp_ms": 350,
            "initial_sync_started_ms": 400,
            "initial_world_timestamp_ms": 450,
            "command_ready_timestamp_ms": 460,
            "first_attempt_started_ms": 500,
            "game_ready_to_intel_ready_ms": 50,
            "intel_ready_to_initial_world_ms": 100,
            "initial_world_to_command_ready_ms": 10,
            "command_ready_to_first_attempt_ms": 40,
        },
        inter_key_delay_ms=1500,
        presses=[],
    )


def test_parse_keys_env_defaults_and_custom() -> None:
    assert key_probe_script._parse_keys_env(None) == DEFAULT_KEYS
    assert key_probe_script._parse_keys_env("  ") == DEFAULT_KEYS
    assert key_probe_script._parse_keys_env("r, s ,t") == ("r", "s", "t")


def test_parse_bool_env() -> None:
    assert key_probe_script._parse_bool_env("yes") is True
    assert key_probe_script._parse_bool_env(None) is False


def test_format_saved_path() -> None:
    assert key_probe_script._format_saved_path("key_probe.json") == "Saved to: key_probe.json"


def test_main_uses_defaults_and_initializes_sync_playwright(_restore_script_hooks: None) -> None:
    captured: list[str | bool | int | tuple[str, ...]] = []
    factory_calls: list[str] = []

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.sync_playwright = None

    def _get_sync_playwright() -> _FakeSyncPlaywrightFactory:
        factory_calls.append("factory")
        return _FakeSyncPlaywrightFactory()

    core_hooks.get_sync_playwright = _get_sync_playwright

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        keys: tuple[str, ...] = DEFAULT_KEYS,
        initial_sync_timeout_ms: int = 10000,
        inter_key_delay_ms: int = 1500,
    ) -> KeyProbeSessionDict:
        captured.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                keys,
                initial_sync_timeout_ms,
                inter_key_delay_ms,
            ]
        )
        return _session()

    key_probe_script.run_key_probe = _fake_run

    assert key_probe_script.main() == 0
    assert factory_calls == ["factory"]
    assert captured[0] == "https://tankpit.com/play"
    output = str(captured[1])
    assert output.startswith("runs/probe/key-")
    assert output.endswith(".json")
    assert captured[2:] == [False, False, DEFAULT_KEYS, 10000, 1500]
    assert callable(core_hooks.sync_playwright)


def test_main_uses_env_overrides(_restore_script_hooks: None) -> None:
    recorded: list[str | bool | int | tuple[str, ...]] = []

    script_hooks.setup_rich_logging = lambda level: None
    env = {
        "TANKPIT_URL": "https://tankpit.com/custom",
        "TANKPIT_KEY_PROBE_OUTPUT": "custom_keys.json",
        "TANKPIT_HEADLESS": "true",
        "TANKPIT_PREFER_ACCOUNT": "yes",
        "TANKPIT_KEY_PROBE_KEYS": "r,s,t",
        "TANKPIT_KEY_PROBE_INITIAL_SYNC_TIMEOUT_MS": "9000",
        "TANKPIT_KEY_PROBE_DELAY_MS": "800",
    }
    core_hooks.get_env = lambda key: env.get(key)
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()

    def _fake_run(
        target_url: str,
        output_path: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        keys: tuple[str, ...] = DEFAULT_KEYS,
        initial_sync_timeout_ms: int = 10000,
        inter_key_delay_ms: int = 1500,
    ) -> KeyProbeSessionDict:
        recorded.extend(
            [
                target_url,
                output_path,
                headless,
                prefer_account,
                keys,
                initial_sync_timeout_ms,
                inter_key_delay_ms,
            ]
        )
        return _session()

    key_probe_script.run_key_probe = _fake_run

    assert key_probe_script.main() == 0
    assert recorded == [
        "https://tankpit.com/custom",
        "custom_keys.json",
        True,
        True,
        ("r", "s", "t"),
        9000,
        800,
    ]


def test_module_entrypoint_runs_main(_restore_script_hooks: None) -> None:
    import tankpit_bot.action_lab.key_probe as key_module

    script_hooks.setup_rich_logging = lambda level: None
    core_hooks.get_env = lambda key: None
    core_hooks.sync_playwright = _FakeSyncPlaywrightFactory()
    original_run = key_module.run_key_probe
    key_module.run_key_probe = lambda target_url, output_path, **kwargs: _session()

    old_argv = sys.argv
    sys.argv = ["scripts.key_probe"]
    try:
        sys.modules.pop("scripts.key_probe", None)
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("scripts.key_probe", run_name="__main__")
    finally:
        key_module.run_key_probe = original_run
        sys.argv = old_argv

    assert exc.value.code == 0
