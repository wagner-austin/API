"""Tests for the radar-watch probe."""

from __future__ import annotations

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
from tankpit_bot.action_lab.radar_watch import (
    RadarWatchProbe,
    RadarWatchSessionDict,
    encode_radar_watch_session,
    format_radar_watch_summary,
    run_radar_watch_probe,
)
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state
from tankpit_bot.state.types import make_self_state, make_viewport_state

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


def _inventory(*, radar_count: int, radar_enabled: bool) -> InventoryState:
    item = InventoryItem(count=25, enabled=True)
    return InventoryState(
        armor_shields=item,
        dual_shots=item,
        missile_shots=item,
        homing_shots=item,
        extra_radars=InventoryItem(count=radar_count, enabled=radar_enabled),
    )


class _WatchHarness(RadarWatchProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=True)
        self._clock = ReplayClock(1000)
        self._page = ClockAdvancingPage(self._clock)
        self.inventory_calls = 0
        self.radar_calls = 0
        self.map_calls = 0
        self.sent_toggles: list[int] = []
        self.inventory_script: list[InventoryState] = []
        self.move_calls: list[tuple[int, int]] = []
        self._self_state: SelfStateDict | None = make_self_state(
            tank_id=1,
            x=100,
            y=100,
            team=2,
            rank=1,
            fuel=900,
            leaderboard_position=1,
        )

    def request_inventory(self) -> bool:
        if self.inventory_script:
            get_world_service().inventory_state = self.inventory_script.pop(0)
        self.inventory_calls += 1
        return True

    def use_radar(self) -> bool:
        self.radar_calls += 1
        return True

    def open_map(self) -> bool:
        self.map_calls += 1
        return True

    def toggle_equipment_slot(self, slot: int) -> bool:
        self.sent_toggles.append(slot)
        return True

    def get_self_state(self) -> SelfStateDict | None:
        return self._self_state

    def move_to(self, x: int, y: int) -> bool:
        self.move_calls.append((x, y))
        return True


def _install_noop_drain() -> None:
    def _drain(provider: BufferedMessageSourceProtocol) -> int:
        del provider
        return 0

    action_hooks.drain_buffered_messages = _drain


def test_ensure_extras_disabled_toggles_once_and_verifies() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.inventory_script = [
        _inventory(radar_count=22, radar_enabled=True),
        _inventory(radar_count=22, radar_enabled=False),
    ]

    count, was_enabled, toggles = probe._ensure_extras_disabled()
    assert (count, was_enabled, toggles) == (22, True, 1)
    assert probe.sent_toggles == [5]


def test_ensure_extras_disabled_skips_when_already_off() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=7, radar_enabled=False)

    count, was_enabled, toggles = probe._ensure_extras_disabled()
    assert (count, was_enabled, toggles) == (7, False, 0)
    assert probe.sent_toggles == []


def test_ensure_extras_disabled_raises_when_toggle_fails() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=22, radar_enabled=True)

    with pytest.raises(ProbeError, match="still enabled after toggle"):
        probe._ensure_extras_disabled()
    assert probe.sent_toggles == [5]


def test_watch_loop_scans_and_map_polls_on_schedule() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    scans, map_polls, walks = probe._watch_loop(60000, 15000, 30000)
    assert scans == 4
    assert probe.radar_calls == 4
    assert map_polls == 2
    assert probe.map_calls == 2
    assert walks == 4
    assert probe.move_calls == [(101, 100), (99, 100), (101, 100), (99, 100)]


def test_watch_loop_skips_walks_without_self_state() -> None:
    probe = _WatchHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe._self_state = None

    scans, map_polls, walks = probe._watch_loop(30000, 15000, 30000)
    assert scans == 2
    assert map_polls == 1
    assert walks == 0
    assert probe.move_calls == []


def test_execute_probe_rejects_bad_intervals() -> None:
    probe = _WatchHarness()
    with pytest.raises(ProbeError, match="duration_ms must be positive"):
        probe.execute_probe(
            duration_ms=0,
            scan_interval_ms=1000,
            map_poll_interval_ms=1000,
            initial_sync_timeout_ms=1000,
        )
    with pytest.raises(ProbeError, match="scan_interval_ms must be positive"):
        probe.execute_probe(
            duration_ms=1000,
            scan_interval_ms=0,
            map_poll_interval_ms=1000,
            initial_sync_timeout_ms=1000,
        )


def _session() -> RadarWatchSessionDict:
    return RadarWatchSessionDict(
        session_id="radar-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="radar_watch_probe.capture_session.json",
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
        duration_ms=1800000,
        scan_interval_ms=15000,
        map_poll_interval_ms=30000,
        walks_sent=118,
        extras_before=22,
        extras_enabled_at_start=True,
        toggles_sent=1,
        scans_sent=120,
        map_polls_sent=60,
        extras_after=22,
    )


def test_encode_and_summary() -> None:
    session = _session()
    encoded = encode_radar_watch_session(session)
    assert encoded["extras_before"] == 22
    assert encoded["extras_after"] == 22
    assert encoded["toggles_sent"] == 1
    assert format_radar_watch_summary(session) == (
        "Radar watch complete: scans=120 map_polls=60 walks=118 toggles=1 "
        "extras 22->22 duration_ms=1800000"
    )


class _FakeRadarWatchProbe(RadarWatchProbe):
    def execute_probe(
        self,
        *,
        duration_ms: int,
        scan_interval_ms: int,
        map_poll_interval_ms: int,
        initial_sync_timeout_ms: int,
    ) -> RadarWatchSessionDict:
        session = _session()
        session["duration_ms"] = duration_ms
        session["scan_interval_ms"] = scan_interval_ms
        session["map_poll_interval_ms"] = map_poll_interval_ms
        session["initial_sync_timeout_ms"] = initial_sync_timeout_ms
        session["capture_session_path"] = ""
        return session


class _RadarModuleProtocol(Protocol):
    RadarWatchProbe: type[RadarWatchProbe]


_radar_module_import = __import__(
    "tankpit_bot.action_lab.radar_watch",
    fromlist=["radar_watch"],
)
radar_module: _RadarModuleProtocol = _radar_module_import


def test_run_radar_watch_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    original_class = radar_module.RadarWatchProbe
    radar_module.RadarWatchProbe = _FakeRadarWatchProbe
    try:
        session = run_radar_watch_probe(
            "https://tankpit.com/play",
            "radar_watch_probe.json",
            duration_ms=600000,
            scan_interval_ms=10000,
            map_poll_interval_ms=20000,
        )
    finally:
        radar_module.RadarWatchProbe = original_class

    written = fake_fs.read_text(Path("radar_watch_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["capture_session_path"] == "radar_watch_probe.capture_session.json"
    assert decoded["duration_ms"] == 600000
    assert session["scan_interval_ms"] == 10000
    assert session["map_poll_interval_ms"] == 20000


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, RadarWatchProbe):
    def __init__(self) -> None:
        RadarWatchProbe.__init__(
            self, "https://tankpit.com/play", headless=False, prefer_account=True
        )
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.phases: list[str] = []

    def _ensure_extras_disabled(self) -> tuple[int, bool, int]:
        self.phases.append("disable")
        return 22, True, 1

    def _watch_loop(
        self,
        duration_ms: int,
        scan_interval_ms: int,
        map_poll_interval_ms: int,
    ) -> tuple[int, int, int]:
        self.phases.append(f"watch:{duration_ms}:{scan_interval_ms}:{map_poll_interval_ms}")
        return 4, 2, 4

    def _read_extras(self) -> tuple[int, bool]:
        self.phases.append("read")
        return 22, False


def test_execute_probe_builds_session_envelope() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness()
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
            duration_ms=60000,
            scan_interval_ms=15000,
            map_poll_interval_ms=30000,
            initial_sync_timeout_ms=10000,
        )
    finally:
        core_hooks.sync_playwright = original_sync_playwright

    assert probe.phases == ["disable", "watch:60000:15000:30000", "read"]
    assert session["extras_before"] == 22
    assert session["extras_enabled_at_start"] is True
    assert session["toggles_sent"] == 1
    assert session["scans_sent"] == 4
    assert session["map_polls_sent"] == 2
    assert session["walks_sent"] == 4
    assert session["extras_after"] == 22
    assert session["capture_session_path"] == ""


class _ToggleRecorder(RadarWatchProbe):
    def __init__(self) -> None:
        self._dispatched: list[tuple[bytes, str]] = []
        self._commands_xor_table = None

    def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
        self._dispatched.append((data, cmd_name))
        return True


def test_toggle_equipment_slot_dispatches_the_hotkey_command() -> None:
    """Slot 5 frames as ``[len]['!'][3]['r']['5']`` — the 0x72 hotkey."""
    probe = _ToggleRecorder()
    assert probe.toggle_equipment_slot(5) is True
    data, label = probe._dispatched[0]
    assert label == "toggle_equipment(5)"
    assert data.endswith(b"\x03r5")
