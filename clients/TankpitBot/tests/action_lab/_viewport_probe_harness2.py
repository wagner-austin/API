"""Shared builders and probe doubles for the viewport-probe tests."""

from __future__ import annotations

import base64
from pathlib import Path

from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_core import (
    StubbedBootstrapMixin,
    WorldStateOverrideMixin,
)
from tests.action_lab._replay_page import (
    ClockAdvancingPage,
    ReplayClock,
)

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks import BufferedMessageSourceProtocol
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.viewport_probe import (
    ViewportProbe,
    ViewportProbeSessionDict,
)
from tankpit_bot.state import (
    SelfStateDict,
    WorldStateDict,
    make_empty_world_state,
)
from tankpit_bot.state.types import (
    make_self_state,
    make_viewport_state,
)
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
    """One received frame batch holding the plaintext autoscroll ack.

    The real ack is the server's un-XORed echo of the toggle command:
    raw ``"A1"``/``"A0"`` (key-probe capture 2026-07-24).
    """
    frame = bytes([0x02, 0x00]) + (b"A1" if enabled else b"A0")
    return CapturedMessage(
        timestamp_ms=1000,
        direction="received",
        payload=base64.b64encode(frame).decode("ascii"),
        ws_url="wss://test",
    )


def _truncated_message() -> CapturedMessage:
    """A frame whose length prefix claims 255 bytes and carries one.

    Kept out of :func:`_noise_messages` deliberately. It is corruption,
    not noise, and the reader now says so instead of skipping it
    ([[session-state-deglobalisation]]).
    """
    return CapturedMessage(
        timestamp_ms=1000,
        direction="received",
        payload=base64.b64encode(bytes([0xFF, 0x00, 0x41])).decode("ascii"),
        ws_url="wss://test",
    )


def _noise_messages() -> list[CapturedMessage]:
    """Frames the ack reader must skip: sent, empty, non-0x41.

    Every entry is well-formed; each is simply not an autoscroll ack.
    A torn frame used to sit in this list too — see
    :func:`_truncated_message`.
    """
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
        self.window = (95, 92, 16, 16)

    @property
    def messages(self) -> list[CapturedMessage]:
        return self.message_log

    def get_world_state(self) -> WorldStateDict:
        world = _make_world(1000, self.position[0], self.position[1], self.fuel)
        return WorldStateDict(
            self_state=world["self_state"],
            tanks=world["tanks"],
            containers=world["containers"],
            mines=world["mines"],
            terrain=world["terrain"],
            viewport=make_viewport_state(
                left=self.window[0],
                top=self.window[1],
                width=self.window[2],
                height=self.window[3],
            ),
            scanned_tiles=world["scanned_tiles"],
            timestamp_ms=world["timestamp_ms"],
        )

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
