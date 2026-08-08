"""Tests for bot state-update detail."""

from __future__ import annotations

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_combat import (
    mark_teleport_landed,
)
from tankpit_bot.sniffer.world_state_containers import (
    update_world_state_from_fuel_total as _sm_update_fuel,
)
from tankpit_bot.sniffer.world_state_radar import (
    update_world_state_from_radar,
)
from tests.bot._state_machine_fixtures import _set_bot_action
from tests.conftest import FakeEnv


class TestBotStateUpdateDetail:
    """Tests for bot state-update detail."""

    def test_update_state_teleporting_to_idle_on_landed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """TELEPORTING completes when the server confirms landing."""
        from tankpit_bot.bot.base import Bot

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        ws.update_world_state_from_position(196, 85)
        _sm_update_fuel(ws, 582)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "TELEPORTING", "teleport", 196, 86)
        mark_teleport_landed(ws)
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"

    def test_update_state_teleport_landing_mismatch_marks_failed_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Mismatched teleport landing blacklists the requested destination."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.browser import get_current_time_ms

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        ws.update_world_state_from_position(196, 85)
        _sm_update_fuel(ws, 582)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "TELEPORTING", "teleport", 196, 86)

        mark_teleport_landed(ws)
        bot._update_state_from_world()

        now = get_current_time_ms()
        assert bot.get_state() == "IDLE"
        assert ws.is_move_target_failed(196, 86, now) is True

    def test_low_fuel_does_not_stomp_teleporting(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """LOW_FUEL does not override an in-flight TELEPORTING state."""
        from tankpit_bot.bot.base import Bot

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        ws.update_world_state_from_position(50, 50)
        _sm_update_fuel(ws, 100)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "TELEPORTING", "teleport", 60, 70)
        bot._state_data["fuel_threshold"] = 200
        bot._update_state_from_world()
        assert bot.get_state() == "TELEPORTING"

    def test_low_fuel_does_not_stomp_collecting(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """LOW_FUEL does not override an in-flight COLLECTING state."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.protocol import RadarContainerDict

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        ws.update_world_state_from_position(50, 50)
        _sm_update_fuel(ws, 100)
        update_world_state_from_radar(ws, [RadarContainerDict(x=55, y=55, volume=500)], [], [])
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "COLLECTING", "collect", 55, 55)
        bot._state_data["fuel_threshold"] = 200
        bot._update_state_from_world()
        assert bot.get_state() == "COLLECTING"

    def test_low_fuel_does_not_stomp_scanning(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """LOW_FUEL does not override an in-flight SCANNING state."""
        from tankpit_bot.bot.base import Bot

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        ws.update_world_state_from_position(50, 50)
        _sm_update_fuel(ws, 100)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "SCANNING", "scan", 0, 0)
        bot._state_data["fuel_threshold"] = 200
        bot._update_state_from_world()
        assert bot.get_state() == "SCANNING"

    def test_teleport_completes_before_low_fuel_checked(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """TELEPORTING completes to IDLE even when fuel is below threshold."""
        from tankpit_bot.bot.base import Bot

        ws = WorldService()
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        ws.update_world_state_from_position(50, 50)
        _sm_update_fuel(ws, 100)
        bot._update_state_from_world()
        bot._state_data = _set_bot_action(bot._state_data, "TELEPORTING", "teleport", 50, 50)
        bot._state_data["fuel_threshold"] = 200
        mark_teleport_landed(ws)
        bot._update_state_from_world()
        assert bot.get_state() == "IDLE"
