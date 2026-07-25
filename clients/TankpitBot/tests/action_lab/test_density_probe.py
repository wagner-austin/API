"""Tests for the container-density probe."""

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
from tankpit_bot.action_lab.density_probe import (
    DENSITY_SITES,
    DensityProbe,
    DensityProbeSessionDict,
    encode_density_probe_session,
    format_density_probe_summary,
    run_density_probe,
)
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.inventory import InventoryItem, InventoryState
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state import SelfStateDict, WorldStateDict, make_empty_world_state
from tankpit_bot.state.types import (
    ContainerStateDict,
    make_container_state,
    make_self_state,
    make_viewport_state,
)

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


class _DensityHarness(DensityProbe):
    def __init__(self) -> None:
        super().__init__("https://tankpit.com/play", headless=True, prefer_account=True)
        self._clock = ReplayClock(1000)
        self._page = ClockAdvancingPage(self._clock)
        self.inventory_calls = 0
        self.radar_calls = 0
        self.map_calls = 0
        self.sent_toggles: list[int] = []
        self.inventory_script: list[InventoryState] = []
        self.teleports: list[tuple[int, int]] = []
        self.pickups: list[tuple[int, int]] = []
        self.fuel = 1100
        self.position = (100, 100)
        self.visible_containers: dict[str, ContainerStateDict] = {}

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

    def teleport_to(self, x: int, y: int) -> bool:
        self.teleports.append((x, y))
        self.position = (x, y)
        return True

    def pickup_fuel(self, x: int, y: int) -> bool:
        self.pickups.append((x, y))
        return True

    def get_world_state(self) -> WorldStateDict:
        world = _make_world(1000, self.position[0], self.position[1], self.fuel)
        return WorldStateDict(
            self_state=world["self_state"],
            tanks=world["tanks"],
            containers=self.visible_containers,
            mines=world["mines"],
            terrain=world["terrain"],
            viewport=world["viewport"],
            scanned_tiles=world["scanned_tiles"],
            timestamp_ms=world["timestamp_ms"],
        )

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


def test_density_sites_are_a_map_spread_grid() -> None:
    """Sixteen interior grid sites, none in the unencodable atlas edge."""
    assert len(DENSITY_SITES) == 16
    assert len(set(DENSITY_SITES)) == 16
    assert all(40 <= x <= 208 and 40 <= y <= 208 for x, y in DENSITY_SITES)


def test_ensure_extras_enabled_toggles_once_and_verifies() -> None:
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.inventory_script = [
        _inventory(radar_count=22, radar_enabled=False),
        _inventory(radar_count=22, radar_enabled=True),
    ]

    count, was_enabled, toggles = probe._ensure_extras_enabled()
    assert (count, was_enabled, toggles) == (22, False, 1)
    assert probe.sent_toggles == [5]


def test_ensure_extras_enabled_skips_when_already_on() -> None:
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=7, radar_enabled=True)

    count, was_enabled, toggles = probe._ensure_extras_enabled()
    assert (count, was_enabled, toggles) == (7, True, 0)
    assert probe.sent_toggles == []


def test_ensure_extras_enabled_refuses_empty_stock() -> None:
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=0, radar_enabled=False)

    with pytest.raises(ProbeError, match="no extra radars in stock"):
        probe._ensure_extras_enabled()


def test_ensure_extras_enabled_raises_when_toggle_fails() -> None:
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=22, radar_enabled=False)

    with pytest.raises(ProbeError, match="still disabled after toggle"):
        probe._ensure_extras_enabled()
    assert probe.sent_toggles == [5]


def test_restore_extras_state_toggles_back_off_and_verifies() -> None:
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=10, radar_enabled=False)

    assert probe._restore_extras_state(True) == 0
    assert probe.sent_toggles == []
    assert probe._restore_extras_state(False) == 1
    assert probe.sent_toggles == [5]


def test_restore_extras_state_raises_when_still_enabled() -> None:
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=10, radar_enabled=True)

    with pytest.raises(ProbeError, match="still enabled after restore"):
        probe._restore_extras_state(False)


class _PayingDotHarness(_DensityHarness):
    """A harness whose refuel landings always pay out."""

    def teleport_to(self, x: int, y: int) -> bool:
        self.fuel = 900
        return super().teleport_to(x, y)


def test_refuel_toward_hops_nearest_dots_until_funded() -> None:
    """Below the funding line the probe hops the NEAREST unvisited dot."""
    probe = _PayingDotHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 100
    get_world_service().map_fuel_dots = ((105, 100), (200, 200))

    hops = probe._refuel_toward(40, 40)
    assert hops == 1
    assert probe.teleports == [(105, 100)]


def test_refuel_toward_returns_when_already_funded() -> None:
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 1100
    assert probe._refuel_toward(40, 40) == 0
    assert probe.teleports == []
    assert probe.map_calls == 0


def test_refuel_toward_tolerates_a_dotless_map() -> None:
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 50
    get_world_service().map_fuel_dots = ()
    assert probe._refuel_toward(40, 40) == 0
    assert probe.map_calls == 1


def test_refuel_toward_stops_after_a_dry_streak() -> None:
    """Dry dots never pay; the hop budget caps the streak."""
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 50
    get_world_service().map_fuel_dots = ((105, 100), (110, 100), (115, 100), (120, 100))
    hops = probe._refuel_toward(40, 40)
    assert hops == 3
    assert len(probe.teleports) == 3


def test_refuel_toward_picks_the_cheaper_later_dot() -> None:
    """A nearer dot listed second still wins the hop."""
    probe = _PayingDotHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 100
    get_world_service().map_fuel_dots = ((115, 100), (105, 100), (116, 100))

    assert probe._refuel_toward(40, 40) == 1
    assert probe.teleports == [(105, 100)]


class _PayingPickupHarness(_DensityHarness):
    """A harness whose fuel pickups always pay out."""

    def pickup_fuel(self, x: int, y: int) -> bool:
        self.fuel += 400
        return super().pickup_fuel(x, y)


def test_bootstrap_fuel_walks_nearest_visible_fuel_first() -> None:
    """At fuel 0 the probe funds itself from viewport pickups."""
    probe = _PayingPickupHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 0
    probe.visible_containers = {
        "103,100": make_container_state(
            x=103, y=100, is_fuel=True, volume=500, timestamp_ms=1000, failed_pickups=0
        ),
        "120,100": make_container_state(
            x=120, y=100, is_fuel=True, volume=500, timestamp_ms=1000, failed_pickups=0
        ),
    }

    attempts = probe._bootstrap_fuel(700)
    assert attempts == 2
    assert probe.pickups == [(103, 100), (120, 100)]
    assert probe.fuel == 800


def test_bootstrap_fuel_returns_when_nothing_anywhere() -> None:
    """No visible fuel and a dotless map: no attempts, loud log."""
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 0
    get_world_service().map_fuel_dots = ()

    assert probe._bootstrap_fuel(700) == 0
    assert probe.pickups == []
    assert probe.map_calls == 1


class _PayingBlindWalkHarness(_DensityHarness):
    """Blind dot-walk pickups pay out."""

    def pickup_fuel(self, x: int, y: int) -> bool:
        self.fuel += 600
        return super().pickup_fuel(x, y)

    def move_to(self, x: int, y: int) -> bool:
        self.position = (x, y)
        return True


def test_bootstrap_fuel_blind_walks_to_the_nearest_map_dot() -> None:
    """Marooned at fuel 0 with a dry viewport, the probe walks the
    nearest atlas dot (free and instant at 0) and picks up there —
    the recovery the second live run needed."""
    probe = _PayingBlindWalkHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 0
    get_world_service().map_fuel_dots = ((180, 180), (110, 126))

    assert probe._bootstrap_fuel(500) == 1
    assert probe.pickups == [(110, 126)]
    assert probe.position == (110, 126)


def test_bootstrap_fuel_gives_up_after_dry_attempts() -> None:
    """Pickups that never pay stop at the attempt budget's dry set."""
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 0
    probe.visible_containers = {
        "103,100": make_container_state(
            x=103, y=100, is_fuel=True, volume=500, timestamp_ms=1000, failed_pickups=0
        ),
        "104,100": make_container_state(
            x=104, y=100, is_fuel=False, volume=0, timestamp_ms=1000, failed_pickups=0
        ),
    }

    assert probe._bootstrap_fuel(700) == 1
    assert probe.pickups == [(103, 100)]


def test_bootstrap_fuel_exhausts_the_attempt_budget() -> None:
    """Plenty of visible fuel that never pays stops at the budget."""
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 0
    probe.visible_containers = {
        f"{110 + i},100": make_container_state(
            x=110 + i, y=100, is_fuel=True, volume=500, timestamp_ms=1000, failed_pickups=0
        )
        for i in range(14)
    }

    assert probe._bootstrap_fuel(700) == 12
    assert len(probe.pickups) == 12


def test_reach_site_verifies_the_landing() -> None:
    """A landed teleport reports True; a rejected one preserves the extra."""
    landed_probe = _DensityHarness()
    action_hooks.get_current_time_ms = landed_probe._clock
    _install_noop_drain()
    ok, hops, picks = landed_probe._reach_site(40, 40)
    assert (ok, hops, picks) == (True, 0, 0)

    class _Rejecting(_DensityHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            return True

    stuck = _Rejecting()
    action_hooks.get_current_time_ms = stuck._clock
    ok, _, _ = stuck._reach_site(40, 40)
    assert ok is False


def test_sweep_sites_skips_unreached_sites_without_spending_extras() -> None:
    """The first live run's failure mode, pinned: rejected teleports
    must never burn extras on re-scans of the same viewport."""

    class _Stuck(_DensityHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            return True

    probe = _Stuck()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=22, radar_enabled=True)
    get_world_service().map_fuel_dots = ()

    scanned, _, _, skipped = probe._sweep_sites(12)
    # (96, 96) is within the landing tolerance of the (100, 100)
    # spawn, so that one site legitimately scans without moving;
    # every other rejected teleport preserves its extra.
    assert scanned == 1
    assert skipped == 15
    assert probe.radar_calls == 1


def test_current_fuel_raises_without_self_state() -> None:
    class _Blind(_DensityHarness):
        def get_self_state(self) -> SelfStateDict | None:
            return None

    probe = _Blind()
    with pytest.raises(ProbeError, match="self state unavailable"):
        probe._current_fuel()


class _ToggleRecorder(DensityProbe):
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


def test_sweep_sites_scans_within_budget_and_stock() -> None:
    """One teleport + one scan per site, capped by the extras budget."""
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=22, radar_enabled=True)

    scanned, refuels, _pickups, _skipped = probe._sweep_sites(3)
    assert scanned == 3
    assert refuels == 0
    assert probe.radar_calls == 3
    assert probe.teleports == list(DENSITY_SITES[:3])


def test_sweep_sites_exhausts_the_whole_grid_under_a_big_budget() -> None:
    """A budget above the site count sweeps all sixteen sites."""
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    get_world_service().inventory_state = _inventory(radar_count=30, radar_enabled=True)

    scanned, refuels, _pickups, _skipped = probe._sweep_sites(20)
    assert scanned == 16
    assert refuels == 0
    assert probe.teleports == list(DENSITY_SITES)


def test_sweep_sites_stops_when_stock_runs_out() -> None:
    probe = _DensityHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.inventory_script = [
        _inventory(radar_count=1, radar_enabled=True),
        _inventory(radar_count=0, radar_enabled=True),
    ]

    scanned, _, _, _ = probe._sweep_sites(12)
    assert scanned == 1
    assert probe.radar_calls == 1


def test_execute_probe_rejects_bad_budget() -> None:
    probe = _DensityHarness()
    with pytest.raises(ProbeError, match="max_extras must be positive"):
        probe.execute_probe(max_extras=0, initial_sync_timeout_ms=1000)


def _session() -> DensityProbeSessionDict:
    return DensityProbeSessionDict(
        session_id="density-session",
        start_timestamp_ms=1,
        end_timestamp_ms=2,
        base_url="https://tankpit.com/play",
        spawn_x=100,
        spawn_y=100,
        capture_session_path="density_probe.capture_session.json",
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
        max_extras=12,
        sites_planned=16,
        sites_scanned=12,
        sites_skipped=1,
        refuel_hops=4,
        bootstrap_pickups=2,
        extras_before=22,
        extras_enabled_at_start=False,
        toggles_sent=2,
        extras_after=10,
        fuel_before=1100,
        fuel_after=430,
    )


def test_encode_and_summary() -> None:
    session = _session()
    encoded = encode_density_probe_session(session)
    assert encoded["max_extras"] == 12
    assert encoded["extras_after"] == 10
    assert encoded["refuel_hops"] == 4
    assert format_density_probe_summary(session) == (
        "Density probe complete: sites=12/16 skipped=1 refuels=4 pickups=2 "
        "toggles=2 extras 22->10 fuel 1100->430"
    )


class _FakeDensityProbe(DensityProbe):
    def execute_probe(
        self,
        *,
        max_extras: int,
        initial_sync_timeout_ms: int,
    ) -> DensityProbeSessionDict:
        session = _session()
        session["max_extras"] = max_extras
        session["initial_sync_timeout_ms"] = initial_sync_timeout_ms
        session["capture_session_path"] = ""
        return session


class _DensityModuleProtocol(Protocol):
    DensityProbe: type[DensityProbe]


_density_module_import = __import__(
    "tankpit_bot.action_lab.density_probe",
    fromlist=["density_probe"],
)
density_module: _DensityModuleProtocol = _density_module_import


def test_run_density_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    original_class = density_module.DensityProbe
    density_module.DensityProbe = _FakeDensityProbe
    try:
        session = run_density_probe(
            "https://tankpit.com/play",
            "density_probe.json",
            max_extras=8,
        )
    finally:
        density_module.DensityProbe = original_class

    written = fake_fs.read_text(Path("density_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["capture_session_path"] == "density_probe.capture_session.json"
    assert decoded["max_extras"] == 8
    assert session["sites_scanned"] == 12


class _ExecuteHarness(StubbedBootstrapMixin, WorldStateOverrideMixin, DensityProbe):
    def __init__(self) -> None:
        DensityProbe.__init__(self, "https://tankpit.com/play", headless=False, prefer_account=True)
        self._init_bootstrap_stubs()
        self._world_state = _make_world(900, 100, 100, 900)
        self.phases: list[str] = []

    def _current_fuel(self) -> tuple[int, int, int]:
        self.phases.append("fuel")
        return 900, 100, 100

    def _ensure_extras_enabled(self) -> tuple[int, bool, int]:
        self.phases.append("enable")
        return 22, False, 1

    def _sweep_sites(self, max_extras: int) -> tuple[int, int, int, int]:
        self.phases.append(f"sweep:{max_extras}")
        return 12, 3, 2, 1

    def _restore_extras_state(self, was_enabled: bool) -> int:
        self.phases.append(f"restore:{was_enabled}")
        return 1

    def _read_extras(self) -> tuple[int, bool]:
        self.phases.append("read")
        return 10, False


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
        session = probe.execute_probe(max_extras=12, initial_sync_timeout_ms=10000)
    finally:
        core_hooks.sync_playwright = original_sync_playwright

    assert probe.phases == ["fuel", "enable", "sweep:12", "restore:False", "read", "fuel"]
    assert session["max_extras"] == 12
    assert session["sites_scanned"] == 12
    assert session["refuel_hops"] == 3
    assert session["bootstrap_pickups"] == 2
    assert session["sites_skipped"] == 1
    assert session["extras_before"] == 22
    assert session["extras_enabled_at_start"] is False
    assert session["toggles_sent"] == 2
    assert session["extras_after"] == 10
    assert session["fuel_before"] == 900
    assert session["fuel_after"] == 900
    assert session["capture_session_path"] == ""
