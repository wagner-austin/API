"""Tests for the in-flight action stall timeouts.

The per-kind stall clears, the started_ms guard, and the fuel-0 move
budget (the measured slow-service law). The blocked-state clears and
the map_open/scope completion holds are
:mod:`tests.bot.test_cdp_inflight`.
"""

from __future__ import annotations

from tankpit_bot.bot.states import (
    make_in_flight_action,
)
from tankpit_bot.browser import get_current_time_ms
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_total as _update_fuel_total,
)
from tankpit_bot.sniffer.world_state_radar import (
    update_world_state_from_radar as _update_radar,
)
from tests.bot._cdp_harness import _sba
from tests.conftest import FakeEnv


class TestStallTimeouts:
    """Stall clears per action kind, and the fuel-0 move budget."""

    def test_has_in_flight_action_clears_stalled_move(self, fake_env: FakeEnv) -> None:
        """Stalled movement times out so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        ws = WorldService()
        ws.update_world_state_from_position(10, 10)
        _update_fuel_total(ws, 800)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_fuel_zero_move_holds_through_the_slow_service_window(self, fake_env: FakeEnv) -> None:
        """A fuel-0 move 12 s in is NOT a stall — the wire is just slow.

        The measured law (578 paired echoes across the two paralysis
        artifacts): fuel-0 moves answer at median 3.8 s with a tail to
        15.75 s. The standard 10 s budget produced 114 false stalls,
        each writing a false failed-move mark; the fuel-0 budget holds
        to 20 s.
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        ws = WorldService()
        ws.update_world_state_from_position(10, 10)
        _update_fuel_total(ws, 0)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)
        bot._state_data["in_flight_action"]["started_ms"] = get_current_time_ms() - 12_000

        waiting = has_in_flight_action(bot)

        assert waiting is True
        assert bot.get_state() == "MOVING"
        assert not ws.is_move_target_failed(15, 10, get_current_time_ms())

    def test_fuel_zero_move_still_stalls_past_the_extended_budget(self, fake_env: FakeEnv) -> None:
        """21 s of silence at fuel 0 is a real stall: clear and mark."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        ws = WorldService()
        ws.update_world_state_from_position(10, 10)
        _update_fuel_total(ws, 0)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)
        bot._state_data["in_flight_action"]["started_ms"] = get_current_time_ms() - 21_000

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"
        assert ws.is_move_target_failed(15, 10, get_current_time_ms())

    def test_fuel_zero_extension_is_move_only(self, fake_env: FakeEnv) -> None:
        """A fuel-0 TELEPORT at 12 s keeps the standard stall budget.

        The slow-service measurement covers walks; nothing entitles
        other kinds to the extension (and a fuel-0 teleport is already
        unaffordable upstream — a stalled one is a real fault).
        """
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        ws = WorldService()
        ws.update_world_state_from_position(64, 64)
        _update_fuel_total(ws, 0)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "TELEPORTING", "teleport", 128, 128)
        bot._state_data["in_flight_action"]["started_ms"] = get_current_time_ms() - 12_000

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_fuel_zero_move_helper_needs_a_self_state(self, fake_env: FakeEnv) -> None:
        """No self_state yet: the helper stays False (standard budget)."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _fuel_zero_move

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"

        assert _fuel_zero_move(bot, "move") is False

    def test_has_in_flight_action_clears_stalled_collection(self, fake_env: FakeEnv) -> None:
        """Stalled collection times out so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict

        ws = WorldService()
        ws.update_world_state_from_position(64, 64)
        _update_fuel_total(ws, 800)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=72, y=63, volume=-1)]
        mines: list[RadarMineDict] = []
        _update_radar(ws, containers, mines, [])

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 72, 63)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_stalled_teleport(self, fake_env: FakeEnv) -> None:
        """Stalled teleport times out so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        ws = WorldService()
        ws.update_world_state_from_position(64, 64)
        _update_fuel_total(ws, 800)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "TELEPORTING", "teleport", 128, 128)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_stalled_map_open_clears_via_timeout(self, fake_env: FakeEnv) -> None:
        """A map_open that stalls past timeout clears so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        _update_fuel_total(ws, 1400)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "IDLE", "map_open", 0, 0)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_stall_guard_prevents_clear_when_started_ms_is_zero(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A zero started_ms prevents the stall timeout from firing."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.states import InFlightActionDict
        from tankpit_bot.bot.tick_loop_actions import _clear_stalled_action

        bot = Bot("https://test.tankpit.com/", headless=True)
        action: InFlightActionDict = make_in_flight_action(
            "move",
            15,
            10,
            0,
        )

        result = _clear_stalled_action(bot, action)

        assert result is False

    def test_fresh_scan_does_not_trigger_stall_timeout(self, fake_env: FakeEnv) -> None:
        """A recently started scan does not trigger stall timeout."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import _clear_stalled_action

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = _sba(bot._state_data, "SCANNING", "scan", 0, 0)
        action = bot._state_data["in_flight_action"]

        assert _clear_stalled_action(bot, action) is False
        assert bot.get_state() == "SCANNING"

    def test_stalled_scan_clears_via_timeout(self, fake_env: FakeEnv) -> None:
        """A scan that stalls past timeout clears so the bot can replan."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        _update_fuel_total(ws, 1400)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "SCANNING", "scan", 0, 0)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"
        assert ws.is_scan_viewport_failed(0, 0, get_current_time_ms()) is True

    def test_stalled_move_marks_failed_move_target(self, fake_env: FakeEnv) -> None:
        """Stalled move records the destination as a failed move target."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.bot.tick_loop_actions import has_in_flight_action
        from tankpit_bot.browser import get_current_time_ms

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        _update_fuel_total(ws, 1400)

        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 73, 158)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        has_in_flight_action(bot)

        now = get_current_time_ms()
        assert ws.is_move_target_failed(73, 158, now) is True
        assert ws.is_move_target_failed(74, 158, now) is False
