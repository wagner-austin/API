"""Tests for the keyboard probe."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import BufferedMessageSourceProtocol, KeyboardProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.key_probe import (
    DEFAULT_KEYS,
    KeyPressWindowDict,
    KeyProbe,
    KeyProbeSessionDict,
    encode_key_probe_session,
    format_key_probe_summary,
    run_key_probe,
)
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.action_lab.types import TeleportStartupTimingDict
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state
from tankpit_bot.state.types import make_self_state, make_viewport_state
from tankpit_bot.types import CapturedMessage

_FUEL_CAPTURE_PATH = Path(__file__).resolve().parents[2] / "fuel_probe.capture_session.json"


def _make_world(timestamp_ms: int, x: int, y: int, fuel: int) -> WorldStateDict:
    world = make_empty_world_state()
    return WorldStateDict(
        self_state=make_self_state(
            tank_id=1,
            x=x,
            y=y,
            team=2,
            rank=1,
            fuel=fuel,
            leaderboard_position=5,
        ),
        tanks=world["tanks"],
        containers=world["containers"],
        mines=world["mines"],
        terrain=world["terrain"],
        viewport=make_viewport_state(left=0, top=0, width=16, height=16),
        scanned_tiles=world["scanned_tiles"],
        timestamp_ms=timestamp_ms,
    )


class _RecordingKeyboard:
    def __init__(self) -> None:
        self.pressed: list[str] = []

    def press(self, key: str, *, delay: float | None = None) -> None:
        _ = delay
        self.pressed.append(key)

    def type(self, text: str, *, delay: float | None = None) -> None:
        _ = (text, delay)


class _KeyPage(ClockAdvancingPage):
    def __init__(self, clock: ReplayClock) -> None:
        super().__init__(clock)
        self._recording_keyboard = _RecordingKeyboard()

    @property
    def keyboard(self) -> KeyboardProtocol:
        return self._recording_keyboard


class _KeyHarness(KeyProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=False)
        self._clock = ReplayClock(1000)
        self._key_page = _KeyPage(self._clock)
        self._page = self._key_page
        self._fake_messages: list[CapturedMessage] = []

    @property
    def messages(self) -> list[CapturedMessage]:
        return self._fake_messages

    def grow_messages(self) -> None:
        self._fake_messages.append(
            CapturedMessage(
                timestamp_ms=self._clock.now_ms,
                direction="received",
                payload="",
                ws_url="wss://test",
            )
        )


def _startup_timing() -> TeleportStartupTimingDict:
    return {
        "game_ready_timestamp_ms": 100,
        "intel_ready_timestamp_ms": 150,
        "initial_sync_started_ms": 200,
        "initial_world_timestamp_ms": 400,
        "command_ready_timestamp_ms": 450,
        "first_attempt_started_ms": 500,
        "game_ready_to_intel_ready_ms": 50,
        "intel_ready_to_initial_world_ms": 250,
        "initial_world_to_command_ready_ms": 50,
        "command_ready_to_first_attempt_ms": 50,
    }


def _session(presses_keys: list[str]) -> KeyProbeSessionDict:
    return KeyProbeSessionDict(
        session_id="key-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="key_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing=_startup_timing(),
        inter_key_delay_ms=1500,
        presses=[
            {
                "key": key,
                "pressed_at_ms": 1000 + index,
                "message_start_index": index,
                "message_end_index": index + 1,
            }
            for index, key in enumerate(presses_keys)
        ],
    )


def test_press_keys_records_ordered_windows_and_drains() -> None:
    probe = _KeyHarness()
    action_hooks.get_current_time_ms = probe._clock
    drains = 0

    def _drain(provider: BufferedMessageSourceProtocol, ws: WorldService) -> int:
        nonlocal drains
        del provider
        drains += 1
        probe.grow_messages()
        return 1

    action_hooks.drain_buffered_messages = _drain

    presses = probe._press_keys(("r", "s", "t"), 1500)
    assert probe._key_page._recording_keyboard.pressed == ["r", "s", "t"]
    assert drains == 3
    assert [press["key"] for press in presses] == ["r", "s", "t"]
    assert [press["message_start_index"] for press in presses] == [0, 1, 2]
    assert [press["message_end_index"] for press in presses] == [1, 2, 3]
    assert [press["pressed_at_ms"] for press in presses] == [1000, 2500, 4000]


def test_press_keys_requires_a_page() -> None:
    probe = _KeyHarness()
    probe._page = None
    with pytest.raises(ProbeError, match="page is unavailable"):
        probe._press_keys(("r",), 100)


def test_execute_probe_rejects_empty_keys() -> None:
    probe = _KeyHarness()
    with pytest.raises(ProbeError, match="keys must not be empty"):
        probe.execute_probe(
            keys=(),
            initial_sync_timeout_ms=1000,
            inter_key_delay_ms=100,
        )


def test_encode_round_trips_press_windows() -> None:
    session = _session(["r", "s"])
    encoded = encode_key_probe_session(session)
    assert encoded["inter_key_delay_ms"] == 1500
    assert encoded["presses"] == [
        {
            "key": "r",
            "pressed_at_ms": 1000,
            "message_start_index": 0,
            "message_end_index": 1,
        },
        {
            "key": "s",
            "pressed_at_ms": 1001,
            "message_start_index": 1,
            "message_end_index": 2,
        },
    ]


def test_format_summary_lists_keys() -> None:
    summary = format_key_probe_summary(_session(["r", "s"]))
    assert summary == "Key probe complete: presses=2 delay_ms=1500 keys=r,s"


class _KeyProbeModuleProtocol(Protocol):
    KeyProbe: type[KeyProbe]


_key_module_import = __import__(
    "tankpit_bot.action_lab.key_probe",
    fromlist=["key_probe"],
)
key_module: _KeyProbeModuleProtocol = _key_module_import


class _FakeKeyProbe(KeyProbe):
    def __init__(
        self,
        target_url: str,
        *,
        headless: bool,
        prefer_account: bool,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
    ) -> None:
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            cdp_service=cdp_service,
            command_service=command_service,
        )

    def execute_probe(
        self,
        *,
        keys: tuple[str, ...],
        initial_sync_timeout_ms: int,
        inter_key_delay_ms: int,
    ) -> KeyProbeSessionDict:
        session = _session(list(keys))
        session["initial_sync_timeout_ms"] = initial_sync_timeout_ms
        session["inter_key_delay_ms"] = inter_key_delay_ms
        session["capture_session_path"] = ""
        return session


def test_run_key_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    original_class = key_module.KeyProbe
    key_module.KeyProbe = _FakeKeyProbe
    try:
        session = run_key_probe(
            "https://tankpit.com/play",
            "key_probe.json",
            keys=("r", "s"),
            inter_key_delay_ms=800,
        )
    finally:
        key_module.KeyProbe = original_class

    written = fake_fs.read_text(Path("key_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["capture_session_path"] == "key_probe.capture_session.json"
    assert decoded["inter_key_delay_ms"] == 800
    assert session["presses"][0]["key"] == "r"
    assert session["presses"][1]["key"] == "s"


def test_default_keys_press_map_keys_last_and_skip_dangerous_ones() -> None:
    assert DEFAULT_KEYS[-2:] == ("f", "m")
    for banned in ("q", " ", "d", "1", "2", "3", "4", "5"):
        assert banned not in DEFAULT_KEYS


class _KeyExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, KeyProbe):
    def __init__(self) -> None:
        KeyProbe.__init__(self, "https://tankpit.com/play", headless=False, prefer_account=False)
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.pressed_specs: list[tuple[tuple[str, ...], int]] = []

    def _press_keys(
        self,
        keys: tuple[str, ...],
        inter_key_delay_ms: int,
    ) -> list[KeyPressWindowDict]:
        self.pressed_specs.append((keys, inter_key_delay_ms))
        return [
            KeyPressWindowDict(
                key=key,
                pressed_at_ms=1000 + index,
                message_start_index=index,
                message_end_index=index + 1,
            )
            for index, key in enumerate(keys)
        ]


def test_execute_probe_builds_session_envelope() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _KeyExecuteHarness()
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
    original_sync_playwright = core_hooks.sync_playwright
    core_hooks.sync_playwright = recorded.sync_playwright_factory

    def _wait_initial(
        page: action_session.WaitPageProtocol,
        provider: action_session.BufferedWorldStateProviderProtocol,
        started_ms: int,
        timeout_ms: int,
    ) -> tuple[int, SelfStateDict]:
        _ = (page, provider, started_ms, timeout_ms)
        return (
            1200,
            make_self_state(
                tank_id=1,
                x=100,
                y=100,
                team=2,
                rank=1,
                fuel=900,
                leaderboard_position=1,
            ),
        )

    action_hooks.wait_for_initial_self_state = _wait_initial
    try:
        session = probe.execute_probe(
            keys=("r", "s"),
            initial_sync_timeout_ms=10000,
            inter_key_delay_ms=700,
        )
    finally:
        core_hooks.sync_playwright = original_sync_playwright

    assert probe.pressed_specs == [(("r", "s"), 700)]
    assert session["inter_key_delay_ms"] == 700
    assert session["initial_sync_timeout_ms"] == 10000
    assert [press["key"] for press in session["presses"]] == ["r", "s"]
    assert session["capture_session_path"] == ""
    assert session["base_url"] == "https://tankpit.com/play"
