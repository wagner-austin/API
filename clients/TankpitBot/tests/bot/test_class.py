"""Tests for Bot class initialization and basic methods."""

from __future__ import annotations

from tests.conftest import FakeEnv


class TestBotClass:
    """Tests for Bot class methods."""

    def test_bot_init(self, fake_env: FakeEnv) -> None:
        """Test Bot.__init__ sets up state correctly."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.get_state() == "INITIALIZING"
        assert bot._cdp is None
        assert bot._page is None

    def test_bot_get_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_state returns current state name."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        state = bot.get_state()
        assert state == "INITIALIZING"

    def test_bot_get_state_data(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_state_data returns full state dict."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        state_data = bot.get_state_data()
        assert state_data["state"] == "INITIALIZING"
        assert state_data["fuel_threshold"] == 200
        assert state_data["in_flight_action"]["kind"] == "none"

    def test_bot_get_world_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_world_state returns world state from module."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        world = bot.get_world_state()
        assert world["self_state"] is None
        assert world["containers"] == {}

    def test_bot_get_self_state_none(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_self_state returns None when not tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        self_state = bot.get_self_state()
        assert self_state is None

    def test_bot_get_fuel_when_no_self_state(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_fuel returns 0 when self_state not tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fuel = bot.get_fuel()
        assert fuel == 0

    def test_bot_get_position_none(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_position returns None when not tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        pos = bot.get_position()
        assert pos is None

    def test_bot_get_containers_empty(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_containers returns empty dict when none tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        containers = bot.get_containers()
        assert containers == {}

    def test_bot_get_fuel_containers_empty(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_fuel_containers returns empty list when none tracked."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fuel_containers = bot.get_fuel_containers()
        assert fuel_containers == []

    def test_bot_get_nearest_fuel_container_no_position(self, fake_env: FakeEnv) -> None:
        """Test Bot.get_nearest_fuel_container returns None when no position."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        container = bot.get_nearest_fuel_container()
        assert container is None


class TestBotCommandsWithoutCDP:
    """Tests for Bot command methods when CDP session is not available."""

    def test_move_to_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.move_to returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.move_to(100, 100)
        assert result is False

    def test_pickup_move_to_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.pickup_move_to returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.pickup_move_to(100, 100)
        assert result is False

    def test_teleport_to_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.teleport_to returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.teleport_to(100, 100)
        assert result is False

    def test_shoot_at_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.shoot_at returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.shoot_at(100, 100)
        assert result is False

    def test_use_radar_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.use_radar returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.use_radar()
        assert result is False

    def test_toggle_equipment_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.toggle_equipment returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.toggle_equipment(1)
        assert result is False

    def test_toggle_equipment_invalid_slot(self, fake_env: FakeEnv) -> None:
        """Test Bot.toggle_equipment returns False for invalid slot."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.toggle_equipment(0)
        assert result is False
        result = bot.toggle_equipment(6)
        assert result is False

    def test_enable_equipment_invalid_slot(self, fake_env: FakeEnv) -> None:
        """Test Bot.enable_equipment returns False for invalid slot."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.enable_equipment(0)
        assert result is False
        result = bot.enable_equipment(6)
        assert result is False

    def test_request_nearest_enemy_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.request_nearest_enemy returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.request_nearest_enemy()
        assert result is False

    def test_open_map_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.open_map returns False when CDP not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.open_map()
        assert result is False

    def test_close_map_returns_true_when_already_closed(self, fake_env: FakeEnv) -> None:
        """Test Bot.close_map returns False without CDP."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.close_map()
        assert result is False


class TestBotEquipmentState:
    """Tests for Bot equipment state management using server inventory."""

    def test_is_equipment_enabled_false_by_default(self, fake_env: FakeEnv) -> None:
        """Test Bot.is_equipment_enabled returns False by default (inventory all disabled)."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        # Default inventory starts with enabled=False for all slots
        for slot in range(1, 6):
            assert bot.is_equipment_enabled(slot) is False

    def test_is_equipment_enabled_invalid_slot(self, fake_env: FakeEnv) -> None:
        """Test Bot.is_equipment_enabled returns False for invalid slot."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.is_equipment_enabled(0) is False
        assert bot.is_equipment_enabled(6) is False

    def test_is_equipment_enabled_reads_from_inventory(self, fake_env: FakeEnv) -> None:
        """Test Bot.is_equipment_enabled reads from server inventory state."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state, update_inventory_from_toggle

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        # Toggle some off via inventory update
        update_inventory_from_toggle([False, True, False, True, False])
        assert bot.is_equipment_enabled(1) is False
        assert bot.is_equipment_enabled(2) is True
        assert bot.is_equipment_enabled(3) is False
        assert bot.is_equipment_enabled(4) is True
        assert bot.is_equipment_enabled(5) is False

    def test_enable_equipment_already_enabled(self, fake_env: FakeEnv) -> None:
        """Test Bot.enable_equipment returns True if already enabled."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_toggle,
        )

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        # Set slot 1 to enabled via protocol
        update_inventory_from_toggle([True, False, False, False, False])
        result = bot.enable_equipment(1)
        assert result is True


class TestBotMapState:
    """Tests for Bot map toggle helpers."""

    def test_open_map_ignores_legacy_map_flag_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.open_map does not trust the legacy local map flag."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._map_is_open = True
        result = bot.open_map()
        assert result is False

    def test_close_map_returns_false_without_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.close_map returns False without CDP even if flag says open."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._map_is_open = True
        result = bot.close_map()
        assert result is False
