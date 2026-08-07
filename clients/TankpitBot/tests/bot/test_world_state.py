"""Tests for Bot methods that work with populated world state."""

from __future__ import annotations

from tankpit_bot.sniffer.world_state import get_world_service
from tankpit_bot.sniffer.world_state_containers import update_world_state_from_fuel_total
from tankpit_bot.sniffer.world_state_radar import update_world_state_from_radar
from tests.conftest import FakeEnv


class TestBotWithWorldState:
    """Tests for Bot methods that work with populated world state."""

    def test_get_position_with_self_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_position returns position when tracked."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )

        update_world_state_from_position(100, 150)
        bot = Bot("https://test.tankpit.com/", headless=True)
        pos = bot.get_position()
        assert pos == (100, 150)

    def test_get_self_state_with_position(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_self_state returns state when tracked."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )

        update_world_state_from_position(50, 75)
        bot = Bot("https://test.tankpit.com/", headless=True)
        self_state = bot.get_self_state()
        # Type guard: fail test if None
        if self_state is None:
            raise AssertionError("Expected self_state to be populated")
        assert self_state["x"] == 50
        assert self_state["y"] == 75

    def test_get_fuel_with_self_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_fuel returns fuel when tracked."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )

        update_world_state_from_position(50, 75)
        update_world_state_from_fuel_total(get_world_service(), 500)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fuel = bot.get_fuel()
        # fuel_total sets absolute value
        assert fuel == 500

    def test_get_fuel_containers_with_containers(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_fuel_containers returns fuel containers."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.protocol import RadarContainerDict

        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=10, y=20, volume=100),
            RadarContainerDict(x=30, y=40, volume=200),
        ]
        update_world_state_from_radar(get_world_service(), containers, [], [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        fuel_containers = bot.get_fuel_containers()
        assert len(fuel_containers) == 2

    def test_get_nearest_fuel_container_with_containers(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_nearest_fuel_container returns nearest container."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.protocol import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )

        update_world_state_from_position(50, 50)
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=10, y=10, volume=100),  # Distance: 80
            RadarContainerDict(x=60, y=60, volume=200),  # Distance: 20
            RadarContainerDict(x=100, y=100, volume=300),  # Distance: 100
        ]
        update_world_state_from_radar(get_world_service(), containers, [], [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        nearest = bot.get_nearest_fuel_container()
        # Type guard: fail test if None
        if nearest is None:
            raise AssertionError("Expected nearest container to be found")
        assert nearest["x"] == 60
        assert nearest["y"] == 60

    def test_get_nearest_fuel_container_no_fuel_containers(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_nearest_fuel_container returns None when no fuel containers."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.sniffer.world_state import (
            update_world_state_from_position,
        )

        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        nearest = bot.get_nearest_fuel_container()
        assert nearest is None
