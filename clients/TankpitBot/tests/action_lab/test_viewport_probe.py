"""Tests for the viewport probe: edge walks and boundary moves.

``test_viewport_probe.py`` was 758 lines; execution and the summary
are now a sibling.
"""

from __future__ import annotations

import pytest
from tests.action_lab._viewport_probe_harness2 import (
    _ack_message,
    _install_noop_drain,
    _KeyedPage,
    _noise_messages,
    _truncated_message,
    _ViewportHarness,
)
from tests.fakes import InMemoryTerrainMap

from tankpit_bot.action_lab import _test_hooks as action_hooks
from tankpit_bot.action_lab.probe_base import ProbeError
from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.state import (
    SelfStateDict,
)


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


def test_read_autoscroll_ack_refuses_a_truncated_frame() -> None:
    """Corruption is reported, not skipped on the way to the ack.

    The inline frame walk dropped a torn tail and read on, so a probe
    could report an unverified toggle as verified
    ([[session-state-deglobalisation]]).
    """
    probe = _ViewportHarness()
    probe.message_log.append(_truncated_message())
    probe.message_log.append(_ack_message(True))
    with pytest.raises(FramingError, match="Incomplete frame"):
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


def test_anchor_skips_when_the_teleport_never_echoes() -> None:
    class _Rejected(_ViewportHarness):
        def teleport_to(self, x: int, y: int) -> bool:
            self.teleports.append((x, y))
            return True

    probe = _Rejected()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    assert probe._anchor() is False
    assert probe.teleports == [(106, 100)]


def test_walk_to_edge_steps_east_to_the_edge_column() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    steps = probe._walk_to_edge()
    assert steps == 10
    assert probe.moves[0] == (101, 100)
    assert probe.moves[-1] == (110, 100)


def test_walk_to_edge_stops_at_the_fuel_floor() -> None:
    class _Draining(_ViewportHarness):
        def move_to(self, x: int, y: int) -> bool:
            self.fuel = 50
            return super().move_to(x, y)

    probe = _Draining()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    assert probe._walk_to_edge() == 1


def test_walk_to_edge_stops_when_no_walkable_step_remains() -> None:
    probe = _ViewportHarness()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.world.terrain_map = InMemoryTerrainMap.from_passable_set({(100, 100)})
    assert probe._walk_to_edge() == 0
    assert probe.moves == []


def test_walk_to_edge_stops_when_a_step_never_echoes() -> None:
    class _Frozen(_ViewportHarness):
        def move_to(self, x: int, y: int) -> bool:
            self.moves.append((x, y))
            return True

    probe = _Frozen()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    assert probe._walk_to_edge() == 1
    assert probe.moves == [(101, 100)]


def test_terrain_map_raises_when_no_map_is_loaded() -> None:
    probe = _ViewportHarness()
    with pytest.raises(ProbeError, match="terrain map unavailable"):
        probe._terrain_map()


def test_pick_step_prefers_east_then_routes_around_water() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap.from_passable_set({(101, 99), (100, 101)})
    assert probe._pick_step(100, 100, 92, 16, set()) == (101, 99)
    assert probe._pick_step(100, 100, 92, 16, {(101, 99)}) == (100, 101)
    assert probe._pick_step(100, 100, 92, 16, {(101, 99), (100, 101)}) is None


def test_pick_step_skips_rows_outside_the_window() -> None:
    probe = _ViewportHarness()
    # Standing on the window's top row with everything east under
    # water: the north candidates fall outside the window and must be
    # skipped, leaving the south sidestep.
    probe.world.terrain_map = InMemoryTerrainMap.from_passable_set({(100, 93)})
    assert probe._pick_step(100, 92, 92, 16, set()) == (100, 93)


def test_walk_to_edge_spends_all_steps_short_of_a_wide_window() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.window = (95, 92, 32, 16)

    assert probe._walk_to_edge() == 16
    assert probe.moves[-1] == (116, 100)


def test_cross_edge_skips_before_the_edge_column() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()

    probe._cross_edge()
    assert probe.moves == []


def test_cross_edge_steps_past_the_edge() -> None:
    probe = _ViewportHarness()
    probe.world.terrain_map = InMemoryTerrainMap()
    action_hooks.get_current_time_ms = probe._clock
    _install_noop_drain()
    probe.position = (110, 100)

    probe._cross_edge()
    assert probe.moves == [(111, 100)]
