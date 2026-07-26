"""Tests for the viewport/autoscroll probe."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Protocol

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    ClockAdvancingPage,
    ReplayClock,
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.action_lab.viewport_probe import (
    ViewportProbe,
    ViewportProbeSessionDict,
    encode_viewport_probe_session,
    format_viewport_probe_summary,
    run_viewport_probe,
)
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


def _ack_message(enabled: bool) -> CapturedMessage:
    """One received frame batch holding the short 0x41 autoscroll ack."""
    frame = bytes([0x02, 0x00, 0x41, 0x01 if enabled else 0x00])
    return CapturedMessage(
        timestamp_ms=1000,
        direction="received",
        payload=base64.b64encode(frame).decode("ascii"),
        ws_url="wss://test",
    )


def _noise_messages() -> list[CapturedMessage]:
    """Frames the ack reader must skip: sent, empty, non-0x41, truncated."""
    return [
        CapturedMessage(
            timestamp_ms=1000,
            direction="sent",
            payload=base64.b64encode(bytes([0x02, 0x00, 0x41, 0x01])).decode("ascii"),
            ws_url="wss://test",
        ),
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload="",
            ws_url="wss://test",
        ),
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload=base64.b64encode(bytes([0x03, 0x00, 0x2E, 0x01, 0x02])).decode("ascii"),
            ws_url="wss://test",
        ),
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload=base64.b64encode(bytes([0xFF, 0x00, 0x41])).decode("ascii"),
            ws_url="wss://test",
        ),
        CapturedMessage(
            timestamp_ms=1000,
            direction="received",
            payload=base64.b64encode(
                bytes([0x07, 0x00, 0x41, 0x00, 0x0A, 0x02, 0x00, 0x24, 0x0A])
            ).decode("ascii"),
            ws_url="wss://test",
        ),
    ]


class _FakeKeyboard:
    """Keyboard recorder that appends the server's ack on each press."""

    def __init__(self, harness: _ViewportHarness) -> None:
        self._harness = harness
        self.pressed: list[str] = []

    def press(self, key: str, *, delay: float | None = None) -> None:
        _ = delay
        self.pressed.append(key)
        if self._harness.ack_script:
            self._harness.message_log.append(_ack_message(self._harness.ack_script.pop(0)))

    def type(self, text: str, *, delay: float | None = None) -> None:
        _ = (text, delay)


class _KeyedPage(ClockAdvancingPage):
    """Clock page with a recording keyboard, like the real page."""

    def __init__(self, clock: ReplayClock, harness: _ViewportHarness) -> None:
        super().__init__(clock)
        self.fake_keyboard = _FakeKeyboard(harness)

    @property
    def keyboard(self) -> _FakeKeyboard:
        """Return the recording keyboard."""
        return self.fake_keyboard


class _ViewportHarness(ViewportProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=True)
        self._clock = ReplayClock(1000)
        self.message_log: list[CapturedMessage] = []
        self.ack_script: list[bool] = []
        self._keyed_page = _KeyedPage(self._clock, self)
        self._page = self._keyed_page
        self.map_calls = 0
        self.teleports: list[tuple[int, int]] = []
        self.moves: list[tuple[int, int]] = []
        self.quits = 0
        self.fuel = 1000
        self.position = (100, 100)

    @property
    def messages(self) -> list[CapturedMessage]:
        return self.message_log

    def open_map(self) -> bool:
        self.map_calls += 1
        return True

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleports.append((x, y))
        self.position = (x, y)
        return True

    def move_to(self, x: int, y: int) -> bool:
        self.moves.append((x, y))
        self.position = (x, y)
        return True

    def quit_to_lobby(self) -> bool:
        self.quits += 1
        return True

    def get_self_state(self) -> SelfStateDict | None:
        return make_self_state(
            tank_id=1,
            x=self.position[0],
            y=self.position[1],
            team=2,
            rank=1,
            fuel=self.fuel,
            leaderboard_position=1,
        )


def _install_noop_drain() -> None:
    def _drain(provider: BufferedMessageSourceProtocol) -> int:
        del provider
        return 0

    action_hooks.drain_buffered_messages = _drain


def test_current_fuel_raises_without_self_state() -> None:
    class _Blind(_ViewportHarness):
        def get_self_state(self) -> SelfStateDict | None:
            return None

    probe = _Blind()
    with pytest.raises(ProbeError, match="self state unavailable"):
        probe._current_fuel()


def test_read_autoscroll_ack_finds_the_flag_and_skips_noise() -> None:
    probe = _ViewportHarness()
    probe.message_log.extend(_noise_messages())
    probe.message_log.append(_ack_message(True))
    assert probe._read_autoscroll_ack(0) is True

    probe.message_log.append(_ack_message(False))
    assert probe._read_autoscroll_ack(len(probe.message_log) - 1) is False


def test_read_autoscroll_ack_raises_when_absent() -> None:
    probe = _ViewportHarness()
    probe.message_log.extend(_noise_messages())
    with pytest.raises(ProbeError, match="no autoscroll ack"):
        probe._read_autoscroll_ack(0)


def test_toggle_autoscroll_presses_a_and_reads_the_ack() -> None:
    probe = _ViewportHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.ack_script = [True]

    assert probe._toggle_autoscroll() is True
    keyed_page: _KeyedPage = probe._keyed_page
    assert keyed_page.fake_keyboard.pressed == ["a"]


def test_toggle_autoscroll_requires_a_page() -> None:
    probe = _ViewportHarness()
    probe._page = None
    with pytest.raises(ProbeError, match="page is unavailable"):
        probe._toggle_autoscroll()


def test_anchor_opens_map_then_hops_east() -> None:
    probe = _ViewportHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    assert probe._anchor() is True
    assert probe.map_calls == 1
    assert probe.teleports == [(106, 100)]


def test_anchor_skips_below_the_fuel_floor() -> None:
    probe = _ViewportHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 50

    assert probe._anchor() is False
    assert probe.map_calls == 0
    assert probe.teleports == []


def test_walk_east_steps_one_tile_at_a_time() -> None:
    probe = _ViewportHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    steps = probe._walk_east()
    assert steps == 16
    assert probe.moves[0] == (101, 100)
    assert probe.moves[-1] == (116, 100)


def test_walk_east_stops_at_the_fuel_floor() -> None:
    class _Draining(_ViewportHarness):
        def move_to(self, x: int, y: int) -> bool:
            self.fuel = 50
            return super().move_to(x, y)

    probe = _Draining()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    assert probe._walk_east() == 1


def test_long_moves_fire_each_offset_from_the_current_tile() -> None:
    probe = _ViewportHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    sent = probe._long_moves()
    assert sent == 5
    assert probe.moves[0] == (106, 100)


def test_long_moves_stop_at_the_fuel_floor() -> None:
    class _Draining(_ViewportHarness):
        def move_to(self, x: int, y: int) -> bool:
            self.fuel = 50
            return super().move_to(x, y)

    probe = _Draining()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    assert probe._long_moves() == 1


def test_run_phase_walks_then_probes_after_a_good_anchor() -> None:
    probe = _ViewportHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    assert probe._run_phase() == (16, 5)
    assert probe.teleports == [(106, 100)]
    assert len(probe.moves) == 21


def test_run_phase_returns_zeroes_when_the_anchor_fails() -> None:
    probe = _ViewportHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 50

    assert probe._run_phase() == (0, 0)


def _session() -> ViewportProbeSessionDict:
    return ViewportProbeSessionDict(
        session_id="viewport-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="viewport_probe.capture_session.json",
        initial_sync_timeout_ms=10000,
        startup_timing={
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
        },
        walk_steps_per_phase=16,
        long_offsets=[6, 10, 14, 18, 24],
        walks_sent_off=16,
        longs_sent_off=5,
        walks_sent_on=16,
        longs_sent_on=5,
        toggles_sent=2,
        ack_states=[True, False],
        fuel_before=1000,
        fuel_after=520,
    )


def test_encode_and_summary() -> None:
    session = _session()
    encoded = encode_viewport_probe_session(session)
    assert encoded["walks_sent_off"] == 16
    assert encoded["ack_states"] == [True, False]
    assert format_viewport_probe_summary(session) == (
        "Viewport probe complete: walks off/on=16/16 longs off/on=5/5 "
        "toggles=2 acks=[True, False] fuel 1000->520"
    )


class _FakeViewportProbe(ViewportProbe):
    def execute_probe(
        self,
        *,
        initial_sync_timeout_ms: int,
    ) -> ViewportProbeSessionDict:
        session = _session()
        session["initial_sync_timeout_ms"] = initial_sync_timeout_ms
        session["capture_session_path"] = ""
        return session


class _ViewportModuleProtocol(Protocol):
    ViewportProbe: type[ViewportProbe]


_viewport_module_import = __import__(
    "tankpit_bot.action_lab.viewport_probe",
    fromlist=["viewport_probe"],
)
viewport_module: _ViewportModuleProtocol = _viewport_module_import


def test_run_viewport_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    original_class = viewport_module.ViewportProbe
    viewport_module.ViewportProbe = _FakeViewportProbe
    try:
        session = run_viewport_probe(
            "https://tankpit.com/play",
            "viewport_probe.json",
            initial_sync_timeout_ms=9000,
        )
    finally:
        viewport_module.ViewportProbe = original_class

    written = fake_fs.read_text(Path("viewport_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["capture_session_path"] == "viewport_probe.capture_session.json"
    assert decoded["initial_sync_timeout_ms"] == 9000
    assert session["walks_sent_on"] == 16


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, ViewportProbe):
    def __init__(self, ack_script: list[bool]) -> None:
        ViewportProbe.__init__(
            self, "https://tankpit.com/play", headless=False, prefer_account=True
        )
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.phases: list[str] = []
        self.ack_script = ack_script
        self.quits = 0

    def _current_fuel(self) -> tuple[int, int, int]:
        self.phases.append("fuel")
        return 900, 100, 100

    def _run_phase(self) -> tuple[int, int]:
        self.phases.append("phase")
        return 16, 5

    def _toggle_autoscroll(self) -> bool:
        self.phases.append("toggle")
        return self.ack_script.pop(0)

    def quit_to_lobby(self) -> bool:
        self.quits += 1
        return True


def _boot_recorded(probe: _ExecuteHarness) -> None:
    recorded = RecordedChromiumSession.from_capture_path(probe, _FUEL_CAPTURE_PATH)
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


def test_execute_probe_runs_both_phases_and_restores_off() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness(ack_script=[True, False])
    original_sync_playwright = core_hooks.sync_playwright
    _boot_recorded(probe)
    try:
        session = probe.execute_probe(initial_sync_timeout_ms=10000)
    finally:
        core_hooks.sync_playwright = original_sync_playwright

    assert probe.phases == ["fuel", "phase", "toggle", "phase", "toggle", "fuel"]
    assert probe.quits == 1
    assert session["walks_sent_off"] == 16
    assert session["walks_sent_on"] == 16
    assert session["ack_states"] == [True, False]
    assert session["toggles_sent"] == 2


def test_execute_probe_refuses_an_unexpected_initial_state() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness(ack_script=[False])
    original_sync_playwright = core_hooks.sync_playwright
    _boot_recorded(probe)
    try:
        with pytest.raises(ProbeError, match="was not in the expected OFF state"):
            probe.execute_probe(initial_sync_timeout_ms=10000)
    finally:
        core_hooks.sync_playwright = original_sync_playwright


def test_execute_probe_refuses_a_failed_restore() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness(ack_script=[True, True])
    original_sync_playwright = core_hooks.sync_playwright
    _boot_recorded(probe)
    try:
        with pytest.raises(ProbeError, match="still enabled after the restore"):
            probe.execute_probe(initial_sync_timeout_ms=10000)
    finally:
        core_hooks.sync_playwright = original_sync_playwright
