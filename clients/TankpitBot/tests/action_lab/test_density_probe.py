"""Tests for the density probe: site selection and budgeting.

``test_density_probe.py`` was 733 lines; execution is now a sibling.
"""

from __future__ import annotations

import pytest
from tests.action_lab._density_probe_harness import (
    _DensityHarness,
    _install_noop_drain,
    _inventory,
    _PayingBlindWalkHarness,
    _PayingDotHarness,
    _PayingPickupHarness,
)

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.density_probe import (
    DENSITY_SITES,
)
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.state.types import (
    make_container_state,
)


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
