"""Tests for larder-probe execution and the run summary."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)
from tests.action_lab._larder_probe_harness import (
    _equipment,
    _ExecuteHarness,
    _FakeLarderProbe,
    _install_noop_drain,
    _LarderHarness,
    _run_execute_harness,
    _session,
    larder_module,
)
from tests.conftest import FakeFileSystem

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.larder_probe import (
    encode_larder_probe_session,
    format_larder_probe_summary,
    run_larder_probe,
)
from tankpit_bot.action_lab.probe_base import ProbeError


def test_attempt_displaced_landing_walks_onto_the_tile() -> None:
    """A displaced teleport walks onto the container before the own trial."""

    class _DisplacingHarness(_LarderHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            self.position = (x + 3, y)
            return True

    probe = _DisplacingHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.move_script = [True, False, False, False, False]
    attempt = probe._attempt_container(_equipment(104, 100))
    assert attempt["status"] == "no_pickup"
    assert attempt["landed_on_container"] is False
    assert attempt["walked_onto_container"] is True
    assert attempt["own_tile_sent"] is True
    assert attempt["own_tile_picked"] is False
    assert attempt["stepped_off"] is False
    assert attempt["adjacent_sent"] is False


def test_attempt_never_stood_still_runs_the_adjacent_control() -> None:
    """When the tile is unreachable the attempt still proves the container."""

    class _AdjacentHarness(_LarderHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            self.position = (x + 1, y)
            return True

    probe = _AdjacentHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.pays_adjacent = True
    probe.move_script = [False]
    attempt = probe._attempt_container(_equipment(104, 100))
    assert attempt["status"] == "adjacent_pickup"
    assert attempt["stood_on_container"] is False
    assert attempt["own_tile_sent"] is False
    assert attempt["stepped_off"] is True
    assert attempt["adjacent_picked"] is True


def test_execute_larder_probe_rejects_bad_budgets() -> None:
    probe = _LarderHarness()
    with pytest.raises(ProbeError, match="max_attempts must be positive"):
        probe.execute_larder_probe(max_attempts=0, max_extras=6, initial_sync_timeout_ms=1000)
    with pytest.raises(ProbeError, match="max_extras must be positive"):
        probe.execute_larder_probe(max_attempts=3, max_extras=0, initial_sync_timeout_ms=1000)


def test_encode_and_summary() -> None:
    session = _session()
    encoded = encode_larder_probe_session(session)
    assert encoded["max_attempts"] == 3
    assert encoded["own_tile_successes"] == 1
    attempts = narrow_json_to_list(encoded["attempts"])
    assert len(attempts) == 2
    first = narrow_json_to_dict(attempts[0])
    assert first["status"] == "own_tile_pickup"
    assert first["container_x"] == 104
    assert format_larder_probe_summary(session) == (
        "Larder probe complete: attempts=2/3 own-tile 1/2 adjacent=1 "
        "scans=2 hops=3 extras 20->18 fuel 5000->4400"
    )


def test_run_larder_probe_writes_session_json(fake_fs: FakeFileSystem) -> None:
    original_class = larder_module.LarderProbe
    larder_module.LarderProbe = _FakeLarderProbe
    try:
        session = run_larder_probe(
            "https://tankpit.com/play",
            "larder_probe.json",
            max_attempts=2,
            max_extras=4,
        )
    finally:
        larder_module.LarderProbe = original_class

    written = fake_fs.read_text(Path("larder_probe.json"))
    decoded = narrow_json_to_dict(load_json_str(written))
    assert decoded["capture_session_path"] == "larder_probe.capture_session.json"
    assert decoded["max_attempts"] == 2
    assert decoded["max_extras"] == 4
    assert session["own_tile_successes"] == 1


def test_execute_probe_fills_the_attempt_budget() -> None:
    """The loop ends by attempt count and books searches plus tallies."""
    probe = _ExecuteHarness(containers_to_serve=5)
    session = _run_execute_harness(probe, max_attempts=2)

    assert probe.phases == [
        "fuel",
        "enable",
        "search:6",
        "attempt:104",
        "search:5",
        "attempt:105",
        "restore:False",
        "read",
        "fuel",
        "quit",
    ]
    assert len(session["attempts"]) == 2
    assert session["search_scans"] == 2
    assert session["search_hops"] == 2
    assert session["own_tile_successes"] == 0
    assert session["own_tile_failures"] == 2
    assert session["adjacent_successes"] == 2
    assert session["extras_before"] == 20
    assert session["extras_after"] == 18
    assert session["toggles_sent"] == 2
    assert session["fuel_before"] == 5000
    assert session["fuel_after"] == 5000
    assert session["capture_session_path"] == ""


def test_execute_probe_stops_when_no_equipment_is_found() -> None:
    """A dry search breaks the loop and still restores the slot state."""
    probe = _ExecuteHarness(containers_to_serve=1)
    session = _run_execute_harness(probe, max_attempts=3)

    assert probe.phases == [
        "fuel",
        "enable",
        "search:6",
        "attempt:104",
        "search:5",
        "restore:False",
        "read",
        "fuel",
        "quit",
    ]
    assert len(session["attempts"]) == 1
    assert session["search_scans"] == 2
    assert session["search_hops"] == 3
    assert session["max_attempts"] == 3
    assert session["max_extras"] == 6
