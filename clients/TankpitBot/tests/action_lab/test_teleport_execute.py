"""Tests for ``execute`` and the teleport probe's session plumbing."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_cdp import StubSnapshotCDPSession
from tests.action_lab._replay_page import ReplayClock
from tests.action_lab._teleport_harness import (
    _FUEL_CAPTURE_PATH,
    _ExecuteHarness,
    _FakeTeleportProbe,
    _make_attempt,
    _ProbeMethodHarness,
)
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
)
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.teleport import (
    TeleportProbe,
    run_teleport_probe,
)
from tankpit_bot.action_lab.teleport_helpers import (
    TeleportProbeError,
)
from tankpit_bot.action_lab.types import (
    TeleportTargetDict,
)
from tankpit_bot.action_lab.types_codecs import decode_teleport_probe_session
from tankpit_bot.browser import PlaywrightNotInstalledError
from tankpit_bot.state import (
    SelfStateDict,
    make_self_state,
)
from tankpit_bot.types import (
    CapturedMessage,
    decode_capture_session,
)


def test_execute_raises_when_playwright_is_missing() -> None:
    from tankpit_bot import _test_hooks as core_hooks

    probe = _ProbeMethodHarness()
    original_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = None
    try:
        with pytest.raises(PlaywrightNotInstalledError):
            probe.execute(
                explicit_targets=[],
                box_step_x=8,
                box_step_y=8,
                max_targets=None,
                teleport_strategy="sync_before_teleport",
                initial_sync_timeout_ms=10000,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_playwright


def test_execute_rejects_empty_explicit_targets_and_cleans_up() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    original_sync = core_hooks.sync_playwright
    original_wait_initial = action_hooks.wait_for_initial_self_state
    core_hooks.sync_playwright = recorded.sync_playwright_factory

    def _wait_initial(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return 1200, make_self_state(
            tank_id=1,
            x=158,
            y=132,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )

    wait_initial_name = "wait_for_initial_self_state"
    setattr(action_hooks, wait_initial_name, _wait_initial)
    try:
        with pytest.raises(TeleportProbeError, match="requires at least one target"):
            probe.execute(
                explicit_targets=[],
                box_step_x=8,
                box_step_y=8,
                max_targets=None,
                teleport_strategy="sync_before_teleport",
                initial_sync_timeout_ms=10000,
                map_sync_timeout_ms=3000,
                teleport_timeout_ms=10000,
                settle_delay_ms=500,
            )
    finally:
        core_hooks.sync_playwright = original_sync
        setattr(action_hooks, wait_initial_name, original_wait_initial)
    assert recorded.browser_type.launches == [False]


def test_execute_builds_default_targets_and_collects_attempts() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
    probe.result_attempts = [_make_attempt("landed_exact") for _ in range(10)]
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    original_sync = core_hooks.sync_playwright
    original_wait_initial = action_hooks.wait_for_initial_self_state
    core_hooks.sync_playwright = recorded.sync_playwright_factory

    def _wait_initial(
        page_arg: action_session.WaitPageProtocol,
        provider: action_session.WorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page_arg, provider, started_ms, timeout_ms)
        return 1200, make_self_state(
            tank_id=1,
            x=158,
            y=132,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )

    wait_initial_name = "wait_for_initial_self_state"
    setattr(action_hooks, wait_initial_name, _wait_initial)
    try:
        session = probe.execute(
            explicit_targets=None,
            box_step_x=8,
            box_step_y=8,
            max_targets=3,
            teleport_strategy="sync_before_teleport",
            initial_sync_timeout_ms=10000,
            map_sync_timeout_ms=3000,
            teleport_timeout_ms=10000,
            settle_delay_ms=500,
        )
    finally:
        core_hooks.sync_playwright = original_sync
        setattr(action_hooks, wait_initial_name, original_wait_initial)
    assert len(session["targets"]) == 3
    assert len(session["attempts"]) == 3
    assert len(probe.probed_targets) == 3
    assert recorded.browser_type.launches == [False]
    assert session["teleport_strategy"] == "sync_before_teleport"
    assert session["max_targets"] == 3
    assert session["initial_sync_timeout_ms"] == 10000
    assert session["startup_timing"]["initial_world_timestamp_ms"] == 1200
    assert session["startup_timing"]["first_attempt_started_ms"] == 1000


def test_run_teleport_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    from tankpit_bot.action_lab import teleport as teleport_module

    original_factory = teleport_module.build_teleport_probe

    def _build_fake_probe(
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
    ) -> TeleportProbe:
        return _FakeTeleportProbe(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
        )

    teleport_module.build_teleport_probe = _build_fake_probe
    try:
        session = run_teleport_probe(
            "https://tankpit.com/play",
            "teleport_probe.json",
            explicit_targets=[TeleportTargetDict(label="target_0", x=150, y=171)],
        )
    finally:
        teleport_module.build_teleport_probe = original_factory

    written = fake_fs.read_text(Path("teleport_probe.json"))
    decoded = decode_teleport_probe_session(narrow_json_to_dict(load_json_str(written)))
    capture_written = fake_fs.read_text(Path("teleport_probe.capture_session.json"))
    capture_decoded = decode_capture_session(narrow_json_to_dict(load_json_str(capture_written)))
    assert session == decoded
    assert session["capture_session_path"] == "teleport_probe.capture_session.json"
    assert session["targets"] == [TeleportTargetDict(label="target_0", x=150, y=171)]
    assert capture_decoded["session_id"] == "fake-session"


def test_send_bytes_delegates_to_command_service() -> None:
    """_send_bytes syncs CDP and delegates to CommandService."""

    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    sent: list[str] = []

    def _fake_send(cdp: CDPSessionProtocol, data: bytes, label: str) -> str:
        sent.append(label)
        return ""

    probe._commands._send_ws_bytes = _fake_send
    probe._cdp = StubSnapshotCDPSession()
    probe._commands.xor_table = b"\x00" * 256
    assert probe._send_bytes(b"\x04\x00!test", "test_cmd") is True
    assert sent == ["test_cmd"]


def test_send_bytes_returns_false_when_cdp_none() -> None:
    """_send_bytes returns False when no CDP session is attached."""
    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    assert probe._send_bytes(b"\x04\x00!test", "test_cmd") is False


def test_teleport_to_returns_false_when_cdp_none() -> None:
    """teleport_to returns False early when no CDP session."""
    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    assert probe.teleport_to(100, 200) is False


def test_on_message_captured_buffers_received() -> None:
    """_on_message_captured appends received payloads to buffer."""

    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    msg = CapturedMessage(
        direction="received",
        payload="dGVzdA==",
        timestamp_ms=1000,
        ws_url="wss://tankpit.com/ws/",
    )
    probe._on_message_captured(msg)
    assert probe._cdp_message_buffer == ["dGVzdA=="]


def test_on_message_captured_ignores_sent() -> None:
    """_on_message_captured does not buffer sent messages."""

    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    msg = CapturedMessage(
        direction="sent",
        payload="dGVzdA==",
        timestamp_ms=1000,
        ws_url="wss://tankpit.com/ws/",
    )
    probe._on_message_captured(msg)
    assert probe._cdp_message_buffer == []


def test_on_magic_captured_builds_xor_table(fake_fs: FakeFileSystem) -> None:
    """_on_magic_captured builds XOR table on CommandService."""
    probe = TeleportProbe("https://tankpit.com/play", headless=True)
    assert probe._commands.xor_table is None
    probe._on_magic_captured("TESTMAGIC" * 5)
    xor_table = probe._commands.xor_table
    if xor_table is None:
        raise AssertionError("xor_table should be set after magic capture")
    assert len(xor_table) == 1000


def test_build_teleport_probe_default_factory_constructs_real_probe() -> None:
    """Default ``build_teleport_probe`` hook constructs a wired TeleportProbe."""
    from tankpit_bot.action_lab import teleport as teleport_module

    probe = teleport_module._create_teleport_probe(
        "https://tankpit.com/play",
        headless=True,
        prefer_account=False,
    )
    assert probe._target_url == "https://tankpit.com/play"
    assert probe._headless is True
    assert probe._prefer_account is False
    assert probe._commands.xor_table is None
    assert probe._cdp_service.messages == []
    assert probe._cdp_service.magic is None
