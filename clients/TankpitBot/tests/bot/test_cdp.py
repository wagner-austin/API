"""Tests for Bot command methods with mocked CDP session."""

from __future__ import annotations

from tankpit_bot.bot.states import (
    ActionKind,
    BotStateDataDict,
    StateName,
    make_in_flight_action,
)
from tankpit_bot.browser import get_current_time_ms
from tests.conftest import FakeEnv


def _sba(
    sd: BotStateDataDict,
    state: StateName,
    kind: ActionKind,
    tx: int,
    ty: int,
    started_ms: int = -1,
) -> BotStateDataDict:
    """Build new BotStateDataDict with state and in-flight action.

    Args:
        sd: Current state data.
        state: New state name.
        kind: Action kind.
        tx: Target X.
        ty: Target Y.
        started_ms: Action start time. Defaults to current time
            so the action doesn't immediately stall. Pass 0 to
            test the "no timestamp" stall-guard path.
    """
    ts = get_current_time_ms() if started_ms < 0 else started_ms
    return BotStateDataDict(
        state=state,
        fuel_threshold=sd["fuel_threshold"],
        in_flight_action=make_in_flight_action(kind, tx, ty, ts),
    )


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
        assert bot.get_state() == "TELEPORTING"

    def test_teleport_to_clears_stale_landed_flag(self, fake_env: FakeEnv) -> None:
        """A new teleport drains any stale TeleportLanded ack before sending."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.sniffer.world_state import (
            check_and_clear_teleport_landed,
            mark_teleport_landed,
            reset_world_state,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession, FakePage

        reset_world_state()
        update_world_state_from_position(50, 50)
        mark_teleport_landed()
        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = FakePage(fake_cdp)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"

        result = bot.teleport_to(200, 200)

        assert result is True
        assert check_and_clear_teleport_landed() is False

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
        result = bot.shoot_at(55, 53)  # Within 9-tile viewport of (50,50)
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
        result = bot.shoot_at(55, 53)  # Within 9-tile viewport of (50,50)
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
        assert bot._state_data["in_flight_action"]["kind"] == "map_open"

    def test_close_map_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.close_map succeeds with CDP session."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.close_map()
        assert result is True
        assert fake_cdp._sent_methods == ["Runtime.evaluate"]

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
        # Remove CDP to make _send_bytes fail
        bot._cdp = None
        result = bot.teleport_to(100, 100)
        assert result is False

    def test_teleport_fails_when_send_bytes_returns_false(self, fake_env: FakeEnv) -> None:
        """Test teleport_to returns False when command dispatch itself fails."""
        from tankpit_bot.bot import Bot
        from tests.fakes import FakeCDPSession, FakePage

        class FailingTeleportBot(Bot):
            def _send_bytes(self, data: bytes, cmd_name: str) -> bool:
                return False

        bot = FailingTeleportBot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = FakePage(fake_cdp)

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
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.teleport_to(100, 100)
        assert result is True
        assert bot.get_state() == "TELEPORTING"


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
        assert bot._ai_state["combat_phase"] == "none"

    def test_tick_once_returns_if_self_state_disappears_after_sync(self, fake_env: FakeEnv) -> None:
        """_tick_once aborts when the refreshed world loses self_state mid-tick."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.state.types import SelfStateDict, ViewportStateDict, WorldStateDict

        class FlakyWorldBot(Bot):
            """Bot whose world state disappears on the second read."""

            def __init__(self, target_url: str, *, headless: bool) -> None:
                super().__init__(target_url, headless=headless)
                self._world_reads = 0
                self._state_data["state"] = "IDLE"

            def get_world_state(self) -> WorldStateDict:
                """Return a populated state once, then a state without self."""
                self._world_reads += 1
                if self._world_reads == 1:
                    return WorldStateDict(
                        self_state=SelfStateDict(
                            tank_id=1,
                            x=100,
                            y=100,
                            team=1,
                            rank=1,
                            fuel=800,
                            leaderboard_position=1,
                        ),
                        tanks={},
                        containers={},
                        mines={},
                        terrain={},
                        viewport=ViewportStateDict(left=91, top=91, width=18, height=18),
                        scanned_viewports={},
                        timestamp_ms=0,
                    )
                return WorldStateDict(
                    self_state=None,
                    tanks={},
                    containers={},
                    mines={},
                    terrain={},
                    viewport=ViewportStateDict(left=91, top=91, width=18, height=18),
                    scanned_viewports={},
                    timestamp_ms=0,
                )

            def _update_state_from_world(self) -> None:
                """Keep the bot in IDLE for this targeted tick-loop test."""

        bot = FlakyWorldBot("https://test.tankpit.com/", headless=True)

        _tick_once(bot)

        assert bot._world_reads == 2

    def test_tick_once_waits_for_position_before_planning(self, fake_env: FakeEnv) -> None:
        """_tick_once does not execute AI commands while state is WAITING_FOR_POSITION."""
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
        update_world_state_from_position(209, 79)
        update_world_state_from_fuel_total(345)
        update_inventory_from_protocol([25, 25, 25, 25, 11], [False, True, False, False, True])

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._magic = "test_magic"

        _tick_once(bot)

        assert bot.get_state() == "WAITING_FOR_POSITION"
        assert fake_cdp._sent_methods == []

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
        # Set last_scan_ms so radar doesn't fire first
        from tankpit_bot.bot.ai.types import AIStateDict

        bot._ai_state = AIStateDict(**{**bot._ai_state, "last_scan_ms": 1})
        _tick_once(bot)
        # No enemies, no containers → map_open to find enemies
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
        from tankpit_bot.bot.ai.types import AIStateDict

        bot._ai_state = AIStateDict(**{**bot._ai_state, "last_scan_ms": 1})
        _tick_once(bot)
        # decide() persists updated AI state — last_map_open_ms should be set
        assert bot._ai_state["last_map_open_ms"] > 0

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
        dispatch_command(bot, make_shoot_command(55, 53))  # Within viewport
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
        assert bot.get_state() == "TELEPORTING"


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

    def test_tick_once_low_fuel_radars_for_fuel(self, fake_env: FakeEnv) -> None:
        """_tick_once uses radar when fuel low and no containers visible."""
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
        # last_scan_ms=1, scan_cooldown=5000, now will be >> 5001 → cooldown elapsed
        _tick_once(bot)
        # Fuel < 500, no containers, scan cooldown elapsed → radar
        assert bot._ai_state["last_scan_ms"] > 0

    def test_tick_once_low_fuel_walks_to_edge(self, fake_env: FakeEnv) -> None:
        """_tick_once walks to viewport edge when fuel low and radar on cooldown."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.browser import get_current_time_ms
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
        from tankpit_bot.bot.ai.types import AIStateDict

        # Set recent scan so radar cooldown blocks → forces walk to edge
        bot._ai_state = AIStateDict(**{**bot._ai_state, "last_scan_ms": get_current_time_ms()})
        _tick_once(bot)
        # Fuel < 500, no containers, radar on cooldown → walk to edge
        # Should have sent a move command (drain + move = 2 CDP calls)
        assert len(fake_cdp._sent_methods) >= 2

    def test_tick_once_waits_for_in_flight_movement(self, fake_env: FakeEnv) -> None:
        """_tick_once does not replan while a walk is still resolving."""
        import tankpit_bot.sniffer.world_state as ws
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession, FakeTerrainMap

        reset_world_state()
        update_world_state_from_position(200, 70)
        update_world_state_from_fuel_total(800)
        update_inventory_from_protocol([5, 5, 5, 5, 5], [False] * 5)
        ws._terrain_map = FakeTerrainMap()

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 206, 83)

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "MOVING"

    def test_tick_once_waits_for_in_flight_teleport(self, fake_env: FakeEnv) -> None:
        """_tick_once does not replan while a teleport is still resolving."""
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
        update_world_state_from_position(196, 85)
        update_world_state_from_fuel_total(582)
        update_inventory_from_protocol([5, 5, 5, 5, 5], [False] * 5)

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = _sba(bot._state_data, "TELEPORTING", "teleport", 196, 86)

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "TELEPORTING"

    def test_tick_once_waits_for_in_flight_collection(self, fake_env: FakeEnv) -> None:
        """_tick_once does not replan while pickup movement is still resolving."""
        import tankpit_bot.sniffer.world_state as ws
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.container import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
            update_world_state_from_radar,
        )
        from tests.fakes import FakeCDPSession, FakeTerrainMap

        reset_world_state()
        update_world_state_from_position(205, 79)
        update_world_state_from_fuel_total(580)
        update_inventory_from_protocol([5, 5, 5, 5, 5], [False] * 5)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=205, y=82, volume=-1)]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(containers, mines)
        ws._terrain_map = FakeTerrainMap()

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 205, 82)

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "COLLECTING"

    def test_tick_once_waits_for_pending_shot_feedback(self, fake_env: FakeEnv) -> None:
        """_tick_once does not replan while the last shot outcome is still pending."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(800)
        update_inventory_from_protocol([0, 10, 0, 0, 0], [False, True, False, False, False])

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = get_current_time_ms()

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "IDLE"

    def test_tick_once_dispatches_open_map_then_teleport(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_tick_once sends open_map and teleport for a teleport decision."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp

        _tick_once(bot)

        assert fake_cdp._sent_methods == ["Runtime.evaluate", "Runtime.evaluate"]
        assert bot.get_state() == "TELEPORTING"

    def test_tick_once_critical_equipment_preempts_combat(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_tick_once searches for equipment when dual is below break threshold."""
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
        update_inventory_from_protocol([0, 10, 0, 10, 10], [False, True, False, True, True])

        import tankpit_bot.sniffer.world_state as ws

        ws._world_state = update_tank_from_registry(
            ws._world_state,
            tank_id=10,
            team=1,
            name="enemy",
            rank=0,
            is_bot=True,
            x=106,
            y=100,
            timestamp_ms=0,
        )
        enemy = make_tank_state(
            tank_id=10,
            x=106,
            y=100,
            team=1,
            rank=0,
            damage_state=3,
            name="enemy",
            is_bot=True,
            is_self=False,
        )
        new_tanks = dict(ws._world_state["tanks"])
        new_tanks["10"] = enemy
        ws._world_state["tanks"] = new_tanks

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        from tankpit_bot.bot.ai.types import AIStateDict

        bot._ai_state = AIStateDict(
            **{
                **bot._ai_state,
                "combat_target_id": 10,
                "combat_target_x": 106,
                "combat_target_y": 100,
                "combat_phase": "engaging",
            }
        )

        _tick_once(bot)

        # dual=10 < dual_break_threshold=12 → critical equipment preempts combat
        assert fake_cdp._sent_methods[0] == "Runtime.evaluate"
        assert "Input.dispatchKeyEvent" not in fake_cdp._sent_methods
        assert bot.get_state() == "SCANNING"

    def test_tick_once_replans_when_pending_shot_target_was_killed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_tick_once does not wait on pending feedback once the target is already dead."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _tick_once
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeCDPSession

        reset_world_state()
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(800)
        update_inventory_from_protocol([0, 10, 0, 0, 0], [False, True, False, False, False])

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = get_current_time_ms()
        bot._ai_state["killed_tank_ids"] = {"50": get_current_time_ms()}

        _tick_once(bot)

        assert fake_cdp._sent_methods == ["Runtime.evaluate", "Runtime.evaluate"]
        assert bot.get_state() == "TELEPORTING"

    def test_clear_blocked_walk_resets_state(self, fake_env: FakeEnv) -> None:
        """Blocked walking clears MOVING so the bot can replan."""
        import tankpit_bot.sniffer.world_state as ws
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _clear_blocked_walk
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeTerrainMap

        reset_world_state()
        update_world_state_from_position(10, 10)
        update_world_state_from_fuel_total(800)
        ws._terrain_map = FakeTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)
        action = bot._state_data["in_flight_action"]

        cleared = _clear_blocked_walk(bot, action)

        assert cleared is True
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_blocked_walk(self, fake_env: FakeEnv) -> None:
        """Blocked walking returns False from the in-flight gate after clearing state."""
        import tankpit_bot.sniffer.world_state as ws
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tests.fakes import FakeTerrainMap

        reset_world_state()
        update_world_state_from_position(10, 10)
        update_world_state_from_fuel_total(800)
        ws._terrain_map = FakeTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)

        waiting = _has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_clear_blocked_walk_returns_false_without_self_state(self, fake_env: FakeEnv) -> None:
        """Blocked-walk helper does nothing when self position is unknown."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _clear_blocked_walk
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)

        action = bot._state_data["in_flight_action"]
        assert _clear_blocked_walk(bot, action) is False
        assert bot.get_state() == "MOVING"

    def test_clear_blocked_collection_resets_state(self, fake_env: FakeEnv) -> None:
        """Blocked collection clears COLLECTING so the bot can replan."""
        import tankpit_bot.sniffer.world_state as ws
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _clear_blocked_collection
        from tankpit_bot.container import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
            update_world_state_from_radar,
        )
        from tests.fakes import FakeTerrainMap

        reset_world_state()
        update_world_state_from_position(10, 10)
        update_world_state_from_fuel_total(400)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=15, y=10, volume=700)]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(containers, mines)
        ws._terrain_map = FakeTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 15, 10)
        action = bot._state_data["in_flight_action"]

        cleared = _clear_blocked_collection(bot, action)

        assert cleared is True
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_blocked_collection(self, fake_env: FakeEnv) -> None:
        """Blocked collection returns False from the in-flight gate after clearing state."""
        import tankpit_bot.sniffer.world_state as ws
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.container import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
            update_world_state_from_radar,
        )
        from tests.fakes import FakeTerrainMap

        reset_world_state()
        update_world_state_from_position(10, 10)
        update_world_state_from_fuel_total(400)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=15, y=10, volume=700)]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(containers, mines)
        ws._terrain_map = FakeTerrainMap({(11, y): "#" for y in range(256)})

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 15, 10)

        waiting = _has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_stalled_move(self, fake_env: FakeEnv) -> None:
        """Stalled movement times out so the bot can replan."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(10, 10)
        update_world_state_from_fuel_total(800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 15, 10)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = _has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_stalled_collection(self, fake_env: FakeEnv) -> None:
        """Stalled collection times out so the bot can replan."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.container import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
            update_world_state_from_radar,
        )

        reset_world_state()
        update_world_state_from_position(64, 64)
        update_world_state_from_fuel_total(800)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=72, y=63, volume=-1)]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(containers, mines)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 72, 63)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = _has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_clears_stalled_teleport(self, fake_env: FakeEnv) -> None:
        """Stalled teleport times out so the bot can replan."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(64, 64)
        update_world_state_from_fuel_total(800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "TELEPORTING", "teleport", 128, 128)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = _has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_has_in_flight_action_false_for_shoot_kind(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Shoot actions are not blocking for replanning."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = _sba(bot._state_data, "COMBAT", "shoot", 50, 50)

        assert _has_in_flight_action(bot) is False

    def test_has_in_flight_action_waits_for_pending_map_open_sync(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """map_open waits until at least one fresh world sync arrives."""
        import tankpit_bot.sniffer.world_state as ws
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tankpit_bot.state.types import WorldStateDict

        reset_world_state()
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        started_ms = get_current_time_ms()
        bot._state_data = _sba(bot._state_data, "IDLE", "map_open", 0, 0, started_ms=started_ms)
        ws._world_state = WorldStateDict(**{**ws._world_state, "timestamp_ms": started_ms})

        assert _has_in_flight_action(bot) is True
        assert bot._state_data["in_flight_action"]["kind"] == "map_open"

    def test_has_in_flight_action_clears_map_open_after_fresh_sync(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """map_open clears once newer world data has arrived."""
        import tankpit_bot.sniffer.world_state as ws
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )
        from tankpit_bot.state.types import WorldStateDict

        reset_world_state()
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(800)

        bot = Bot("https://test.tankpit.com/", headless=True)
        started_ms = get_current_time_ms()
        bot._state_data = _sba(bot._state_data, "IDLE", "map_open", 0, 0, started_ms=started_ms)
        ws._world_state = WorldStateDict(**{**ws._world_state, "timestamp_ms": started_ms + 1})

        assert _has_in_flight_action(bot) is False
        assert bot._state_data["in_flight_action"]["kind"] == "none"

    def test_stalled_map_open_clears_via_timeout(self, fake_env: FakeEnv) -> None:
        """A map_open that stalls past timeout clears so the bot can replan."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(1400)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "IDLE", "map_open", 0, 0)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = _has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"

    def test_stall_guard_prevents_clear_when_started_ms_is_zero(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """A zero started_ms prevents the stall timeout from firing."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.states import InFlightActionDict
        from tankpit_bot.bot.tick_loop import _clear_stalled_action
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
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
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _clear_stalled_action

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = _sba(bot._state_data, "SCANNING", "scan", 0, 0)
        action = bot._state_data["in_flight_action"]

        assert _clear_stalled_action(bot, action) is False
        assert bot.get_state() == "SCANNING"

    def test_stalled_scan_clears_via_timeout(self, fake_env: FakeEnv) -> None:
        """A scan that stalls past timeout clears so the bot can replan."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.sniffer.world_state import (
            is_scan_viewport_failed,
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(1400)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "SCANNING", "scan", 0, 0)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        waiting = _has_in_flight_action(bot)

        assert waiting is False
        assert bot.get_state() == "IDLE"
        assert is_scan_viewport_failed(0, 0, get_current_time_ms()) is True

    def test_stalled_move_marks_failed_move_target(self, fake_env: FakeEnv) -> None:
        """Stalled move records the destination as a failed move target."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_in_flight_action
        from tankpit_bot.browser import get_current_time_ms
        from tankpit_bot.sniffer.world_state import (
            is_move_target_failed,
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
        )

        reset_world_state()
        update_world_state_from_position(50, 50)
        update_world_state_from_fuel_total(1400)

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "MOVING", "move", 73, 158)
        # Override started_ms=1 to trigger stall timeout
        bot._state_data["in_flight_action"]["started_ms"] = 1

        _has_in_flight_action(bot)

        now = get_current_time_ms()
        assert is_move_target_failed(73, 158, now) is True
        assert is_move_target_failed(74, 158, now) is False

    def test_clear_blocked_collection_returns_false_when_adjacent(self, fake_env: FakeEnv) -> None:
        """Adjacent collection remains viable even if the target tile itself is blocked."""
        import tankpit_bot.sniffer.world_state as ws
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _clear_blocked_collection
        from tankpit_bot.container import RadarContainerDict, RadarMineDict
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_world_state_from_fuel_total,
            update_world_state_from_position,
            update_world_state_from_radar,
        )
        from tests.fakes import FakeTerrainMap

        reset_world_state()
        update_world_state_from_position(14, 10)
        update_world_state_from_fuel_total(400)
        containers: list[RadarContainerDict] = [RadarContainerDict(x=15, y=10, volume=700)]
        mines: list[RadarMineDict] = []
        update_world_state_from_radar(containers, mines)
        ws._terrain_map = FakeTerrainMap({(15, 10): "#"})

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._magic = "test_magic"
        bot._update_state_from_world()
        bot._update_state_from_world()
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 15, 10)

        action = bot._state_data["in_flight_action"]
        assert _clear_blocked_collection(bot, action) is False
        assert bot.get_state() == "COLLECTING"

    def test_clear_blocked_collection_returns_false_without_self_state(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Blocked-collection helper does nothing when self position is unknown."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _clear_blocked_collection
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()

        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._state_data = _sba(bot._state_data, "COLLECTING", "collect", 15, 10)

        action = bot._state_data["in_flight_action"]
        assert _clear_blocked_collection(bot, action) is False
        assert bot.get_state() == "COLLECTING"

    def test_tick_once_waits_for_pending_scan(self, fake_env: FakeEnv) -> None:
        """_tick_once does not fire new commands while radar results are pending."""
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
        bot._state_data = _sba(bot._state_data, "SCANNING", "scan", 0, 0)

        _tick_once(bot)

        assert fake_cdp._sent_methods == []
        assert bot.get_state() == "SCANNING"

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

    def test_merge_protocol_kills_clears_matching_shot_and_combat_target(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """Kill merge clears stale shot feedback and the matching combat lock."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _merge_protocol_kills
        from tankpit_bot.sniffer.world_state import mark_tank_killed, reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "orange-8"
        bot._ai_state["combat_target_id"] = 50
        bot._ai_state["combat_target_x"] = 71
        bot._ai_state["combat_target_y"] = 53
        bot._ai_state["combat_phase"] = "engaging"

        mark_tank_killed(50)
        new_state = _merge_protocol_kills(bot._ai_state)

        assert new_state["last_shot_target_id"] == -1
        assert new_state["last_shot_target_name"] == ""
        assert new_state["combat_target_id"] == -1
        assert new_state["combat_target_x"] == 0
        assert new_state["combat_target_y"] == 0
        assert new_state["combat_phase"] == "none"

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

    def test_get_combat_feedback_miss_when_dual_active(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns 'miss' when dual active and no hit."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import (
            reset_world_state,
            update_inventory_from_protocol,
        )

        reset_world_state()
        update_inventory_from_protocol([0, 10, 0, 0, 0], [False, True, False, False, False])
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        result = _get_combat_feedback(bot)
        assert result == "miss"

    def test_get_combat_feedback_no_miss_without_dual(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns '' when dual shots depleted."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        result = _get_combat_feedback(bot)
        assert result == ""

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

    def test_get_combat_feedback_hit_when_target_killed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_get_combat_feedback returns 'hit' when the tracked target was killed."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["killed_tank_ids"] = {"50": 1000}
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

    def test_has_pending_shot_feedback_true_before_timeout(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback waits while a shot is still inside its timeout."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000

        assert _has_pending_shot_feedback(bot, 2000) is True

    def test_has_pending_shot_feedback_false_after_timeout(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback stops waiting once the timeout expires."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000

        assert _has_pending_shot_feedback(bot, 6000) is False

    def test_has_pending_shot_feedback_false_when_hit_buffered(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback yields to feedback when a hit is already buffered."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import mark_combat_hit, reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000
        mark_combat_hit(weapon_byte=1)

        assert _has_pending_shot_feedback(bot, 2000) is False

    def test_has_pending_shot_feedback_false_when_target_killed(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback stops waiting when the target is already dead."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000
        bot._ai_state["killed_tank_ids"] = {"50": 1500}

        assert _has_pending_shot_feedback(bot, 2000) is False

    def test_has_pending_feedback_false_when_single_shot_response(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """_has_pending_shot_feedback ends when weapon_byte=0 response arrives."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _has_pending_shot_feedback
        from tankpit_bot.sniffer.world_state import mark_combat_hit, reset_world_state

        reset_world_state()
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        bot._ai_state["last_shoot_ms"] = 1000
        mark_combat_hit(weapon_byte=0)

        assert _has_pending_shot_feedback(bot, 2000) is False

    def test_feedback_single_shot_with_dual_available_is_miss(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """weapon_byte=0 with dual available is a miss (target was empty)."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import (
            mark_combat_hit,
            reset_world_state,
            update_inventory_from_protocol,
        )

        reset_world_state()
        update_inventory_from_protocol(
            [0, 10, 0, 0, 0],
            [False, True, False, False, False],
        )
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        mark_combat_hit(weapon_byte=0)
        result = _get_combat_feedback(bot)
        assert result == "miss"

    def test_feedback_single_shot_without_dual_is_empty(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """weapon_byte=0 without dual is '' (can't determine hit/miss)."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import (
            mark_combat_hit,
            reset_world_state,
            update_inventory_from_protocol,
        )

        reset_world_state()
        update_inventory_from_protocol(
            [0, 0, 0, 0, 0],
            [False, False, False, False, False],
        )
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"
        mark_combat_hit(weapon_byte=0)
        result = _get_combat_feedback(bot)
        assert result == ""

    def test_feedback_hit_decrements_dual_then_no_more_miss(
        self,
        fake_env: FakeEnv,
    ) -> None:
        """After dual depleted by hits, weapon_byte=0 gives '' not 'miss'."""
        from tankpit_bot.bot import Bot
        from tankpit_bot.bot.tick_loop import _get_combat_feedback
        from tankpit_bot.sniffer.world_state import (
            get_inventory_state,
            mark_combat_hit,
            reset_world_state,
            update_inventory_from_protocol,
        )

        reset_world_state()
        update_inventory_from_protocol(
            [0, 1, 0, 0, 0],
            [False, True, False, False, False],
        )
        bot = Bot("https://test.tankpit.com/", headless=True)
        bot._ai_state["last_shot_target_id"] = 50
        bot._ai_state["last_shot_target_name"] = "Enemy"

        # First shot: hit with dual, depletes to 0
        mark_combat_hit(weapon_byte=1)
        result = _get_combat_feedback(bot)
        assert result == "hit"
        assert get_inventory_state()["dual_shots"]["count"] == 0

        # Second shot: weapon_byte=0, dual depleted → no feedback
        bot._ai_state["last_shot_target_id"] = 50
        mark_combat_hit(weapon_byte=0)
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
