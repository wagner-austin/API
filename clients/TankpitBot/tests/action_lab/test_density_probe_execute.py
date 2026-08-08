"""Tests for density-probe execution and the run summary."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)
from tests.action_lab._density_probe_harness import (
    _FUEL_CAPTURE_PATH,
    _DensityHarness,
    _ExecuteHarness,
    _FakeDensityProbe,
    _install_noop_drain,
    _inventory,
    _session,
    _ToggleRecorder,
    density_module,
)
from tests.action_lab._replay_browser import RecordedChromiumSession
from tests.action_lab._replay_page import (
    ReplayClock,
)
from tests.conftest import FakeFileSystem

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab import session as action_session
from tankpit_bot.action_lab.density_probe import (
    DENSITY_SITES,
    encode_density_probe_session,
    format_density_probe_summary,
    run_density_probe,
)
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.state import (
    SelfStateDict,
)
from tankpit_bot.state.types import (
    make_self_state,
)


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
    probe.world.inventory_state = _inventory(radar_count=22, radar_enabled=True)
    probe.world.map_fuel_dots = ()

    scanned, _, _, skipped = probe._sweep_sites(12)
    # (96, 96) is within the landing tolerance of the (100, 100)
    # spawn, so that one site legitimately scans without moving;
    # every other rejected teleport preserves its extra.
    assert scanned == 1
    assert skipped == 15
    assert probe.radar_calls == 1


def test_sweep_aborts_when_marooned_and_broke() -> None:
    """An unreachable site with an empty tank ends the sweep at once —
    the 2026-07-25 sitting-duck rule."""

    class _StuckBroke(_DensityHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            return True

    probe = _StuckBroke()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 0
    probe.world.inventory_state = _inventory(radar_count=10, radar_enabled=True)
    probe.world.map_fuel_dots = ()

    scanned, _, _, skipped = probe._sweep_sites(8)
    assert scanned == 0
    assert skipped == 1
    assert probe.radar_calls == 0


def test_quit_to_lobby_sends_the_graceful_quit() -> None:
    """The quit frame is the plain '-' with its length header."""

    class _QuitRecorder(_DensityHarness):
        def __init__(self) -> None:
            super().__init__()
            self.sent_frames: list[tuple[bytes, str]] = []

        def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
            self.sent_frames.append((data, cmd_name))
            return True

    probe = _QuitRecorder()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe._quit_to_lobby()
    data, label = probe.sent_frames[0]
    assert label == "quit_game"
    assert data.endswith(b"-")


def test_current_fuel_raises_without_self_state() -> None:
    class _Blind(_DensityHarness):
        def get_self_state(self) -> SelfStateDict | None:
            return None

    probe = _Blind()
    with pytest.raises(ProbeError, match="self state unavailable"):
        probe._current_fuel()


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
    probe.world.inventory_state = _inventory(radar_count=22, radar_enabled=True)

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
    probe.world.inventory_state = _inventory(radar_count=30, radar_enabled=True)

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

    assert probe.phases == ["fuel", "enable", "sweep:12", "restore:False", "read", "fuel", "quit"]
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
