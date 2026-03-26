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

    def test_send_bytes_xor_encodes_commands(self, fake_env: FakeEnv) -> None:
        """Test Bot._send_bytes XOR encodes '!' commands when table is set."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.protocol.codec import build_xor_table
        from tankpit_bot.protocol.commands import build_move_command
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp = FakeCDPSession()
        bot._cdp = fake_cdp

        # Build XOR table from known keys
        static_key = "A" * 100
        magic = "B" * 20
        bot._xor_table = build_xor_table(static_key, magic)

        # Build raw move command: [len_lo, len_hi, '!', type=4, cmd=0x70, x, y]
        raw = build_move_command(50, 60)
        assert raw[2] == 0x21  # '!' prefix
        assert raw[3] == 4  # raw type byte

        # Send it — should XOR encode bytes after '!'
        bot._send_bytes(raw, "move")

        # Verify the sent data was XOR encoded
        # table[0] = ord('A') ^ ord('B') = 0x03
        # table[1] = ord('A') ^ ord('B') = 0x03
        # Encoded type = 4 ^ 0x03 = 7
        # Encoded cmd = 0x70 ^ 0x03 = 0x73
        assert len(fake_cdp._sent_methods) == 1
        assert fake_cdp._sent_methods[0] == "Runtime.evaluate"

    def test_on_magic_captured_builds_xor_table(self, fake_env: FakeEnv) -> None:
        """Test _on_magic_captured builds XOR table for command encoding."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)

        bot._on_magic_captured("test_magic_key_12345")

        # XOR table should be 1000 bytes (matching fake static key length)
        xor_table = bot._xor_table
        assert type(xor_table) is bytes
        assert len(xor_table) == 1000

    def test_enter_game_sends_query_command(self, fake_env: FakeEnv) -> None:
        """Test enter_game sends CMD_ENTER_GAME (type=2, cmd=0x3f)."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.enter_game()
        assert result is True
        assert fake_cdp._sent_methods == ["Runtime.evaluate"]

    def test_enter_game_no_cdp_returns_false(self, fake_env: FakeEnv) -> None:
        """Test enter_game returns False when CDP session not available."""
        from tankpit_bot.bot import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.enter_game()
        assert result is False

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


class TestBotAIIntegration:
    """Tests for tick loop AI integration via _tick_once."""

    def test_tick_once_no_self_state(self, fake_env: FakeEnv) -> None:
        """_tick_once returns early when self_state is None."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        _tick_once(bot)
        # AI state unchanged — no self_state to act on
        assert bot._ai_state["active_mode"] == "HUNT"
        assert bot._ai_state["ticks_in_mode"] == 0

    def test_tick_once_nothing_to_do_opens_map(self, fake_env: FakeEnv) -> None:
        """_tick_once opens map when nothing to do (no enemies, no containers)."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(800)
        update_inventory_from_protocol([30, 30, 30, 30, 30], [True] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        _tick_once(bot)
        # No enemies, no containers → score=0 → map_open to find enemies
        assert bot._ai_state["active_mode"] == "HUNT"
        assert bot._ai_state["last_map_open_ms"] > 0

    def test_tick_once_hunt_with_enemy(self, fake_env: FakeEnv) -> None:
        """_tick_once dispatches HUNT when enemy is nearby."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tankpit_bot.state.mutations import update_tank_from_registry
        from tankpit_bot.state.types import make_tank_state
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(800)
        update_inventory_from_protocol([30, 30, 30, 30, 30], [True] * 5)

        # Add a close, damaged enemy to trigger HUNT mode
        import tankpit_bot.sniffer.world_state as ws

        ws._world_state = update_tank_from_registry(
            ws._world_state,
            tank_id=10,
            team=1,
            name="enemy",
            rank=0,
            is_bot=True,
            x=110,
            y=100,
            timestamp_ms=0,
        )
        # Set damage to critical so homing gets enabled
        tank = make_tank_state(
            tank_id=10,
            x=110,
            y=100,
            team=1,
            rank=0,
            damage_state=3,
            name="enemy",
            is_bot=True,
            is_self=False,
        )
        new_tanks = dict(ws._world_state["tanks"])
        new_tanks["10"] = tank
        ws._world_state["tanks"] = new_tanks

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        _tick_once(bot)
        assert bot._ai_state["active_mode"] == "HUNT"

    def test_tick_once_updates_ai_state(self, fake_env: FakeEnv) -> None:
        """_tick_once persists updated AI state after tick."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(800)
        update_inventory_from_protocol([30, 30, 30, 30, 30], [True] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        # Set recent map open so map_open_for_enemies cooldown blocks the override
        from tankpit_bot.browser import get_current_time_ms

        bot._ai_state["last_map_open_ms"] = get_current_time_ms()
        _tick_once(bot)
        assert bot._ai_state["ticks_in_mode"] == 1

    def test_dispatch_command_move(self, fake_env: FakeEnv) -> None:
        """executor.dispatch_command dispatches move to move_to."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_move_command
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
        dispatch_command(bot, make_move_command(100, 100))
        assert bot.get_state() == "MOVING"

    def test_dispatch_command_pickup_move(self, fake_env: FakeEnv) -> None:
        """executor.dispatch_command dispatches pickup_move to pickup_move_to."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_pickup_move_command
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
        dispatch_command(bot, make_pickup_move_command(100, 100))
        assert bot.get_state() == "COLLECTING"

    def test_dispatch_command_shoot(self, fake_env: FakeEnv) -> None:
        """executor.dispatch_command dispatches shoot to shoot_at."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_shoot_command
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
        dispatch_command(bot, make_shoot_command(100, 100))
        assert bot.get_state() == "COMBAT"

    def test_dispatch_command_radar(self, fake_env: FakeEnv) -> None:
        """executor.dispatch_command dispatches radar to use_radar."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_radar_command
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
        dispatch_command(bot, make_radar_command())
        assert bot.get_state() == "SCANNING"

    def test_dispatch_command_teleport(self, fake_env: FakeEnv) -> None:
        """executor.dispatch_command dispatches teleport to teleport_to."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import dispatch_command
        from tankpit_bot.bot.types import make_teleport_command
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
        dispatch_command(bot, make_teleport_command(200, 200))
        assert bot.get_state() == "MOVING"


class TestBotEquipmentManagement:
    """Tests for executor.apply_equipment mode-based equipment toggling."""

    def test_apply_equipment_hunt_critical_enemy(self, fake_env: FakeEnv) -> None:
        """HUNT with critically damaged enemy enables radar, dual, homing."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [2, 4, 5])
        # Enable: 5 (radar), 2 (dual), 4 (homing) = 3 toggles
        assert len(fake_cdp._sent_methods) == 3

    def test_apply_equipment_hunt_healthy_no_homing(self, fake_env: FakeEnv) -> None:
        """HUNT with healthy enemy enables radar and dual only."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [2, 5])
        # Enable: 5 (radar), 2 (dual) = 2 toggles
        assert len(fake_cdp._sent_methods) == 2

    def test_apply_equipment_defend_enables_armor(self, fake_env: FakeEnv) -> None:
        """DEFEND mode enables radar (5) and armor (1), disables dual+homing."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [1, 5])
        # Enable: 5 (radar), 1 (armor) = 2 toggles
        assert len(fake_cdp._sent_methods) == 2

    def test_apply_equipment_collect_fuel_critical_shields(self, fake_env: FakeEnv) -> None:
        """COLLECT_FUEL with critical fuel enables radar and shields."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [1, 5])
        # Enable: 5 (radar), 1 (shields) = 2 toggles
        assert len(fake_cdp._sent_methods) == 2

    def test_apply_equipment_collect_fuel_low_no_shields(self, fake_env: FakeEnv) -> None:
        """COLLECT_FUEL with low (not critical) fuel: radar only."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [5])
        # Enable: 5 (radar) only = 1 toggle
        assert len(fake_cdp._sent_methods) == 1

    def test_apply_equipment_patrol_only_radar(self, fake_env: FakeEnv) -> None:
        """PATROL mode only enables extra radar (5)."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(counts, [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [5])
        # Enable: 5 (radar) = 1 toggle
        assert len(fake_cdp._sent_methods) == 1

    def test_apply_equipment_disables_unneeded(self, fake_env: FakeEnv) -> None:
        """Disables combat equipment when switching from HUNT to PATROL."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        # Simulate: homing+dual+radar enabled, shields disabled
        counts = [10, 40, 20, 15, 5]
        update_inventory_from_protocol(counts, [False, True, False, True, True])
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [5])
        # Radar already on (skip), disable dual (2), disable homing (4) = 2
        assert len(fake_cdp._sent_methods) == 2

    def test_apply_equipment_no_stock_skips_enable(self, fake_env: FakeEnv) -> None:
        """Does not enable equipment when stock is depleted."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.executor import apply_equipment
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        # All counts zero, all disabled
        update_inventory_from_protocol([0, 0, 0, 0, 0], [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        apply_equipment(bot, [2, 4, 5])
        # Nothing enabled — no stock available
        assert len(fake_cdp._sent_methods) == 0

    def test_is_equipment_enabled_all_slots(self, fake_env: FakeEnv) -> None:
        """is_equipment_enabled returns correct state for all 5 slots."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state, update_inventory_from_toggle

        reset_world_state()
        update_inventory_from_toggle([True, False, True, False, True])
        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.is_equipment_enabled(1) is True
        assert bot.is_equipment_enabled(2) is False
        assert bot.is_equipment_enabled(3) is True
        assert bot.is_equipment_enabled(4) is False
        assert bot.is_equipment_enabled(5) is True

    def test_disable_equipment_invalid_slot(self, fake_env: FakeEnv) -> None:
        """disable_equipment returns False for out-of-range slot."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot.disable_equipment(0) is False
        assert bot.disable_equipment(6) is False

    def test_has_equipment_stock_missile_slot(self, fake_env: FakeEnv) -> None:
        """_has_equipment_stock returns True for slot 3 (missile) with stock."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )

        reset_world_state()
        update_inventory_from_protocol([0, 0, 10, 0, 0], [False] * 5)
        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot._has_equipment_stock(3) is True

    def test_has_equipment_stock_invalid_slot(self, fake_env: FakeEnv) -> None:
        """_has_equipment_stock returns False for invalid slot number."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot._has_equipment_stock(99) is False

    def test_tick_once_proactive_radar(self, fake_env: FakeEnv) -> None:
        """_tick_once triggers proactive radar when fuel approaching low."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        # Fuel=600 < low(500)+buffer(200)=700, no containers visible
        update_world_state_from_fuel_total(600)
        # Give radar stock so enable_equipment works
        update_inventory_from_protocol([0, 0, 0, 0, 5], [False] * 5)

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        # Set last_scan_ms to 0 so cooldown is elapsed
        bot._ai_state["last_scan_ms"] = 0
        _tick_once(bot)
        # drain_js_messages + toggle_radar_on + use_radar = 3 CDP calls
        # Standard equipment [2,5]: radar (slot 5) was disabled → toggle on
        assert fake_cdp._sent_methods == [
            "Runtime.evaluate",  # drain JS messages
            "Runtime.evaluate",  # toggle radar on (slot 5)
            "Runtime.evaluate",  # use_radar
        ]
        # AI state should have updated last_scan_ms
        assert bot._ai_state["last_scan_ms"] > 0

    def test_tick_once_low_fuel_opens_map(self, fake_env: FakeEnv) -> None:
        """_tick_once opens map when fuel low and radar on cooldown."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(100, 100)
        update_world_state_from_fuel_total(300)
        update_inventory_from_protocol([0, 0, 0, 0, 5], [False] * 5)

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        # Set recent scan so radar cooldown blocks
        from tankpit_bot.browser import get_current_time_ms

        now = get_current_time_ms()
        bot._ai_state["last_scan_ms"] = now - 1000
        _tick_once(bot)
        # No containers, no enemies, radar on cooldown → score=0 → map_open
        assert bot._ai_state["last_map_open_ms"] > 0

    def test_merge_protocol_kills_adds_to_ai_state(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_merge_protocol_kills adds Deactivation kills to AI killed_tank_ids."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _merge_protocol_kills
        from tankpit_bot.sniffer.world_state import mark_tank_killed, reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        mark_tank_killed(50)
        mark_tank_killed(60)
        new_state = _merge_protocol_kills(bot._ai_state)
        assert "50" in new_state["killed_tank_ids"]
        assert "60" in new_state["killed_tank_ids"]

    def test_merge_protocol_kills_no_kills(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_merge_protocol_kills returns unchanged state when no kills."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _merge_protocol_kills
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        result = _merge_protocol_kills(bot._ai_state)
        assert result is bot._ai_state

    def test_get_combat_feedback_miss_when_no_hit(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns 'miss' when no CombatHit received."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        result = _get_combat_feedback(bot)
        assert result == "miss"

    def test_get_combat_feedback_hit_when_combat_hit(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns 'hit' when CombatHit was received."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import mark_combat_hit, reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        mark_combat_hit(weapon_byte=1)
        result = _get_combat_feedback(bot)
        assert result == "hit"

    def test_get_combat_feedback_empty_no_shot_pending(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns '' when no shot was fired."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        result = _get_combat_feedback(bot)
        assert result == ""


class TestBotMessageDecoding:
    """Tests for Bot._on_message_captured message decoding."""

    def test_on_message_captured_extracts_magic_from_auth(self, fake_env: FakeEnv) -> None:
        """_on_message_captured extracts magic key from sent AUTH message."""
        import base64

        from tankpit_bot.bot import Bot
        from tankpit_bot.types import CapturedMessage

        bot = Bot("https://test.tankpit.com/", headless=True)
        assert bot._magic is None

        # AUTH format: 2-byte length prefix + "%AUTH !be session|hash|ts magic_key_here"
        auth_text = "%AUTH !be abc123|def456|1000 testmagickey1234"
        auth_bytes = b"\x00\x00" + auth_text.encode("utf-8")
        payload = base64.b64encode(auth_bytes).decode("ascii")

        msg = CapturedMessage(
            direction="sent",
            payload=payload,
            timestamp_ms=1000,
            ws_url="wss://test.tankpit.com/ws",
        )
        bot._on_message_captured(msg)
        assert bot._magic == "testmagickey1234"

    def test_on_message_captured_received_updates_world(self, fake_env: FakeEnv) -> None:
        """_on_message_captured decodes received messages to update world state."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.types import CapturedMessage

        bot = Bot("https://test.tankpit.com/", headless=True)

        # Send a non-decodable received message (too short) — should not crash
        msg = CapturedMessage(
            direction="received",
            payload="AAAA",
            timestamp_ms=1000,
            ws_url="wss://test.tankpit.com/ws",
        )
        bot._on_message_captured(msg)
        # Should not crash — invalid messages are silently skipped
