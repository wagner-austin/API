"""Tests for viewport-probe execution and the run summary."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
)
from tests.action_lab._replay_page import (
    ReplayClock,
)
from tests.action_lab._viewport_probe_harness import viewport_module
from tests.action_lab._viewport_probe_harness2 import (
    _boot_recorded,
    _ExecuteHarness,
    _FakeViewportProbe,
    _install_noop_drain,
    _session,
    _ViewportHarness,
)
from tests.conftest import FakeFileSystem
from tests.fakes import InMemoryTerrainMap

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.action_lab.viewport_probe import (
    encode_viewport_probe_session,
    format_viewport_probe_summary,
    run_viewport_probe,
)


def test_cross_edge_scans_rows_for_a_passable_landing() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.position = (110, 100)
    probe.world.terrain_map = InMemoryTerrainMap.from_passable_set({(111, 102)})
    probe._cross_edge()
    assert probe.moves == [(111, 102)]


def test_cross_edge_skips_when_nothing_past_the_edge_is_passable() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.position = (110, 100)
    probe.world.terrain_map = InMemoryTerrainMap.from_passable_set(set())
    probe._cross_edge()
    assert probe.moves == []


def test_long_moves_fire_each_offset_from_the_current_tile() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap()
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


def test_run_phase_walks_crosses_then_probes_after_a_good_anchor() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    assert probe._run_phase() == (4, 5)
    assert probe.teleports == [(106, 100)]
    # 4 walk steps (107..110), the edge crossing to 111, then 5 longs.
    assert len(probe.moves) == 10
    assert probe.moves[4] == (111, 100)


def test_run_phase_returns_zeroes_when_the_anchor_fails() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.fuel = 50

    assert probe._run_phase() == (0, 0)


def test_encode_and_summary() -> None:
    session = _session()
    encoded = encode_viewport_probe_session(session)
    assert encoded["walks_sent_off"] == 16
    assert encoded["ack_states"] == [True, False]
    assert format_viewport_probe_summary(session) == (
        "Viewport probe complete: walks off/on=16/16 longs off/on=5/5 "
        "toggles=2 acks=[True, False] fuel 1000->520"
    )


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


def test_execute_probe_normalizes_from_on_then_runs_both_phases() -> None:
    """The first live run's lesson: a fresh browser session can start
    ON regardless of the user's own client — the first press reveals
    and flips the state, and an ON start costs one extra press."""
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness(ack_script=[True, False, True, False])
    original_sync_playwright = core_hooks.sync_playwright
    _boot_recorded(probe)
    try:
        session = probe.execute_probe(initial_sync_timeout_ms=10000)
    finally:
        core_hooks.sync_playwright = original_sync_playwright

    assert probe.phases == [
        "fuel",
        "toggle",
        "toggle",
        "phase",
        "toggle",
        "phase",
        "toggle",
        "fuel",
    ]
    assert probe.quits == 1
    assert session["ack_states"] == [True, False, True, False]
    assert session["toggles_sent"] == 4


def test_execute_probe_runs_directly_when_the_first_press_lands_off() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness(ack_script=[False, True, False])
    original_sync_playwright = core_hooks.sync_playwright
    _boot_recorded(probe)
    try:
        session = probe.execute_probe(initial_sync_timeout_ms=10000)
    finally:
        core_hooks.sync_playwright = original_sync_playwright

    assert probe.phases == ["fuel", "toggle", "phase", "toggle", "phase", "toggle", "fuel"]
    assert session["ack_states"] == [False, True, False]
    assert session["toggles_sent"] == 3


def test_execute_probe_refuses_a_stuck_normalization() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness(ack_script=[True, True])
    original_sync_playwright = core_hooks.sync_playwright
    _boot_recorded(probe)
    try:
        with pytest.raises(ProbeError, match="after the normalization press"):
            probe.execute_probe(initial_sync_timeout_ms=10000)
    finally:
        core_hooks.sync_playwright = original_sync_playwright
    assert probe.quits == 1


def test_execute_probe_refuses_a_failed_on_switch() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness(ack_script=[False, False])
    original_sync_playwright = core_hooks.sync_playwright
    _boot_recorded(probe)
    try:
        with pytest.raises(ProbeError, match="when switching to the ON phase"):
            probe.execute_probe(initial_sync_timeout_ms=10000)
    finally:
        core_hooks.sync_playwright = original_sync_playwright
    assert probe.quits == 1


def test_execute_probe_refuses_a_failed_restore() -> None:
    clock = ReplayClock(1000)
    action_hooks.get_current_time_ms = clock
    probe = _ExecuteHarness(ack_script=[False, True, True])
    original_sync_playwright = core_hooks.sync_playwright
    _boot_recorded(probe)
    try:
        with pytest.raises(ProbeError, match="still enabled after the restore"):
            probe.execute_probe(initial_sync_timeout_ms=10000)
    finally:
        core_hooks.sync_playwright = original_sync_playwright
    assert probe.quits == 1
