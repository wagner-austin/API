"""Tests for Bot vision methods."""

from __future__ import annotations

from tests.conftest import FakeEnv


class TestBotVisionMethods:
    """Tests for Bot vision methods."""

    def test_get_vision_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_vision_state returns the vision state."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        vision_state = bot.get_vision_state()

        assert vision_state["self_fuel"] == 1000
        assert vision_state["self_tank_id"] == -1
        assert vision_state["tank_registry"] == {}

    def test_get_all_fuel_containers_empty(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_all_fuel_containers returns empty list when none tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        containers = bot.get_all_fuel_containers()
        assert containers == []

    def test_get_all_fuel(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_all_fuel returns vision state fuel when no world state."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fuel = bot.get_all_fuel()
        # Falls back to vision state default fuel
        assert fuel == 1000

    def test_render_ascii(self, fake_env: FakeEnv) -> None:
        """Test Bot.render_ascii returns ASCII viewport or None."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.render_ascii()
        # Either None or string with viewport info
        if result is not None:
            assert "Viewport" in result

    def test_render_debug(self, fake_env: FakeEnv) -> None:
        """Test Bot.render_debug returns debug info string."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        debug = bot.render_debug()

        assert "Vision Cache Debug" in debug
        assert "Self tank ID" in debug
        assert "Self fuel" in debug

    def test_get_nearest_all_fuel_container_no_position(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_nearest_all_fuel_container returns None when no position."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        container = bot.get_nearest_all_fuel_container()
        assert container is None

    def test_get_nearest_all_fuel_container_no_containers(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_nearest_all_fuel_container returns None when no containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(100, 100)

        bot = Bot("https://test.tankpit.com/", headless=True)
        container = bot.get_nearest_all_fuel_container()
        assert container is None

    def test_get_nearest_all_fuel_container_with_container(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_nearest_all_fuel_container returns nearest container."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.vision import update_container
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(100, 100)

        bot = Bot("https://test.tankpit.com/", headless=True)
        # Add a container to vision cache
        bot._vision_state = update_container(bot._vision_state, x=110, y=100, volume=200)

        container = bot.get_nearest_all_fuel_container()
        # Verify we got the container we added
        assert container == {"x": 110, "y": 100, "volume": 200, "is_fuel": True}
