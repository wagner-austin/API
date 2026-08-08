"""Tests for the larder probe: harvest selection.

``test_larder_probe.py`` was 668 lines; execution is now a sibling.
"""

from __future__ import annotations

import pytest
from tests.action_lab._larder_probe_harness import (
    _equipment,
    _harness,
    _install_noop_drain,
    _LarderHarness,
    _slots,
)
from tests.in_memory_terrain_map import InMemoryTerrainMap

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.state.types import (
    make_container_state,
)


def test_inventory_total_reads_the_wire_state() -> None:
    """The total is the live sum of all five slot counts."""
    probe = _harness()
    probe.world.inventory_state = _slots(3)
    assert probe._inventory_total() == 15


def test_nearest_equipment_skips_fuel_failed_and_tried() -> None:
    """Fuel, blacklisted, and already-attempted containers never win."""
    probe = _harness()
    probe.visible_containers = {
        "90,100": make_container_state(
            x=90, y=100, is_fuel=True, volume=500, timestamp_ms=1000, failed_pickups=0
        ),
        "99,100": _equipment(99, 100, failed_pickups=2),
        "98,100": _equipment(98, 100),
        "140,140": _equipment(140, 140),
        "104,100": _equipment(104, 100),
        "150,150": _equipment(150, 150),
    }
    found = probe._nearest_equipment({(98, 100)})
    assert found == _equipment(104, 100)


def test_nearest_equipment_skips_water_sitting_containers() -> None:
    """The first live run's failure mode, pinned: shore containers ON
    water can never host the own-tile trial and are never candidates."""
    probe = _harness()
    probe.world.terrain_map = InMemoryTerrainMap({(101, 100): "W"})
    probe.visible_containers = {
        "101,100": _equipment(101, 100),
        "110,100": _equipment(110, 100),
    }
    assert probe._nearest_equipment(set()) == _equipment(110, 100)


def test_nearest_equipment_none_when_no_candidates() -> None:
    probe = _harness()
    assert probe._nearest_equipment(set()) is None


def test_nearest_equipment_requires_the_terrain_map() -> None:
    probe = _harness()
    probe.world.terrain_map = None
    probe.world.selected_room = None
    with pytest.raises(ProbeError, match="terrain map is unavailable"):
        probe._nearest_equipment(set())


def test_search_equipment_returns_visible_without_spending() -> None:
    """An already-believed container costs zero scans and zero hops."""
    probe = _harness()
    probe.visible_containers = {"104,100": _equipment(104, 100)}
    found, scans, hops = probe._search_equipment(set(), 6)
    assert found == _equipment(104, 100)
    assert (scans, hops) == (0, 0)
    assert probe.radar_calls == 0


def test_search_equipment_hops_and_scans_until_found() -> None:
    """The nearest-first site sweep reveals equipment via one extra scan."""

    class _RevealingHarness(_LarderHarness):
        def use_radar(self) -> bool:
            self.visible_containers = {"97,96": _equipment(97, 96)}
            return super().use_radar()

    probe = _RevealingHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    found, scans, hops = probe._search_equipment(set(), 6)
    assert found == _equipment(97, 96)
    assert (scans, hops) == (1, 1)
    assert probe.teleports == [(96, 96)]


def test_search_equipment_stops_at_the_scan_budget() -> None:
    """A dry sweep never exceeds the extras budget."""
    probe = _harness()
    found, scans, hops = probe._search_equipment(set(), 2)
    assert found is None
    assert (scans, hops) == (2, 2)
    assert probe.radar_calls == 2


def test_search_equipment_skips_unlanded_sites_without_scanning() -> None:
    """A rejected site teleport preserves its extra, like the density sweep."""

    class _StuckHarness(_LarderHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            return True

    probe = _StuckHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.world.map_fuel_dots = ()
    found, scans, hops = probe._search_equipment(set(), 2)
    # (96, 96) is within landing tolerance of the (100, 100) start, so
    # exactly one site scans; every other rejected teleport is skipped.
    assert found is None
    assert scans == 1
    assert hops == 16


def test_step_off_returns_immediately_when_already_adjacent() -> None:
    probe = _harness()
    probe.position = (105, 100)
    assert probe._step_off(104, 100) is True
    assert probe.moves == []


def test_step_off_walks_to_the_first_cardinal_neighbor() -> None:
    probe = _harness()
    probe.position = (104, 100)
    assert probe._step_off(104, 100) is True
    assert probe.moves == [(105, 100)]


def test_step_off_fails_when_no_neighbor_is_reachable() -> None:
    probe = _harness()
    probe.position = (104, 100)
    probe.move_script = [False, False, False, False]
    assert probe._step_off(104, 100) is False
    assert len(probe.moves) == 4


def test_attempt_landing_on_tile_and_own_pickup_pays() -> None:
    """The gate's YES case: land ON the container, own-tile pickup credits."""
    probe = _harness()
    probe.pays_on_tile = True
    attempt = probe._attempt_container(_equipment(104, 100))
    assert attempt["status"] == "own_tile_pickup"
    assert attempt["landed_on_container"] is True
    assert attempt["walked_onto_container"] is False
    assert attempt["stood_on_container"] is True
    assert attempt["own_tile_sent"] is True
    assert attempt["own_tile_picked"] is True
    assert attempt["stepped_off"] is False
    assert attempt["adjacent_sent"] is False
    assert attempt["inventory_after"] == attempt["inventory_before"] + 5
    assert probe.pickups == [(104, 100)]


def test_attempt_own_tile_fails_then_adjacent_control_pays() -> None:
    """The gate's NO case: own-tile silent, the adjacent control credits."""
    probe = _harness()
    probe.pays_adjacent = True
    attempt = probe._attempt_container(_equipment(104, 100))
    assert attempt["status"] == "adjacent_pickup"
    assert attempt["own_tile_sent"] is True
    assert attempt["own_tile_picked"] is False
    assert attempt["stepped_off"] is True
    assert attempt["adjacent_sent"] is True
    assert attempt["adjacent_picked"] is True
    assert probe.pickups == [(104, 100), (104, 100)]
    assert probe.moves == [(105, 100)]
