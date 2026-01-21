"""Tests for Bot command methods with mocked CDP session."""

from __future__ import annotations

from tests.conftest import FakeEnv


class TestBotWithCDP:
    """Tests for Bot command methods with mocked CDP session."""

    def test_send_bytes_success(self, fake_env: FakeEnv) -> None:
        """Test Bot._send_bytes succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot._send_bytes(b"test", "test_cmd")
        assert result is True

    def test_move_to_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.move_to succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.move_to(100, 100)
        assert result is True
        assert bot.get_state() == "MOVING"

    def test_pickup_move_to_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.pickup_move_to succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.pickup_move_to(100, 100)
        assert result is True
        assert bot.get_state() == "COLLECTING"

    def test_teleport_to_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.teleport_to succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession, FakePage

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = FakePage(fake_cdp)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.teleport_to(200, 200)
        assert result is True
        assert bot.get_state() == "MOVING"

    def test_shoot_at_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.shoot_at succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.shoot_at(100, 100)
        assert result is True
        assert bot.get_state() == "COMBAT"

    def test_shoot_at_already_combat(self, fake_env: FakeEnv) -> None:
        """Test Bot.shoot_at stays in COMBAT if already in COMBAT."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "COMBAT"
        result = bot.shoot_at(100, 100)
        assert result is True
        assert bot.get_state() == "COMBAT"

    def test_use_radar_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.use_radar succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.use_radar()
        assert result is True
        assert bot.get_state() == "SCANNING"

    def test_open_map_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.open_map succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.open_map()
        assert result is True
        assert bot._map_is_open is True

    def test_close_map_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.close_map succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._map_is_open = True
        result = bot.close_map()
        assert result is True
        assert bot._map_is_open is False

    def test_toggle_equipment_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.toggle_equipment succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.toggle_equipment(1)
        assert result is True

    def test_teleport_fails_if_send_fails(self, fake_env: FakeEnv) -> None:
        """Test teleport_to returns False if _send_bytes fails."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession, FakePage

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = FakePage(fake_cdp)
        bot._map_is_open = True  # Skip open_map
        # Remove CDP to make _send_bytes fail
        bot._cdp = None
        result = bot.teleport_to(100, 100)
        assert result is False


class TestBotTeleportBranches:
    """Tests for Bot.teleport_to branch coverage."""

    def test_teleport_without_page(self, fake_env: FakeEnv) -> None:
        """Test teleport_to works when _page is None (skips waits)."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = None  # No page - skips wait_for_timeout calls
        bot._map_is_open = False
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.teleport_to(100, 100)
        assert result is True
        assert bot.get_state() == "MOVING"


class TestBotIdleAndLowFuelHandlers:
    """Tests for _handle_idle_state and _handle_low_fuel_state."""

    def test_handle_idle_state_no_containers(self, fake_env: FakeEnv) -> None:
        """Test _handle_idle_state scans when no fuel containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state
        from tests.fakes import FakeCDPSession

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        bot._handle_idle_state()
        # Should have called use_radar (transition to SCANNING)
        assert bot.get_state() == "SCANNING"

    def test_handle_idle_state_with_containers(self, fake_env: FakeEnv) -> None:
        """Test _handle_idle_state moves when fuel containers exist."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
            update_world_state_from_radar,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=100, y=100, volume=50),
        ]
        update_world_state_from_radar(containers, [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        bot._handle_idle_state()
        # Should have called go_to_nearest_fuel (transition to COLLECTING)
        assert bot.get_state() == "COLLECTING"

    def test_handle_low_fuel_state_no_containers(self, fake_env: FakeEnv) -> None:
        """Test _handle_low_fuel_state scans when no fuel containers."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state
        from tests.fakes import FakeCDPSession

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "LOW_FUEL"
        bot._handle_low_fuel_state()
        # Should have called use_radar (transition to SCANNING)
        assert bot.get_state() == "SCANNING"

    def test_handle_low_fuel_state_with_containers(self, fake_env: FakeEnv) -> None:
        """Test _handle_low_fuel_state moves when fuel containers exist."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.container import RadarContainerDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_position,
            update_world_state_from_radar,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        containers: list[RadarContainerDict] = [
            RadarContainerDict(x=100, y=100, volume=50),
        ]
        update_world_state_from_radar(containers, [])
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "LOW_FUEL"
        bot._handle_low_fuel_state()
        # Should have called go_to_nearest_fuel (transition to COLLECTING)
        assert bot.get_state() == "COLLECTING"
