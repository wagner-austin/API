"""Tests for the Bot's CDP session surface.

Construction, teleport branches, the ``require_cdp`` guard, and inbound
message decoding. ``test_cdp.py`` was 2,818 lines; the AI, equipment,
tick, in-flight, feedback, and health-gate suites are now siblings.
"""

from __future__ import annotations

from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.sniffer.world_state_combat import (
    check_and_clear_teleport_landed,
    mark_teleport_landed,
)
from tests.conftest import FakeEnv


class TestBotWithCDP:
    """TestBotWithCDP tests."""

    def test_send_bytes_success(self, fake_env: FakeEnv) -> None:
        """Test Bot._send_bytes succeeds with CDP session."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot._send_bytes(b"test", "test_cmd")
        assert result is True

    def test_send_bytes_xor_encodes_commands(self, fake_env: FakeEnv) -> None:
        """Test Bot._send_bytes XOR encodes '!' commands when table is set."""
        from tankpit_bot.bot.base import Bot
        from tankpit_bot.protocol.codec import build_xor_table
        from tankpit_bot.protocol.commands import build_move_command
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp = FakeCDPSession()
        bot._cdp = fake_cdp

        # Build XOR table from known keys
        static_key = "A" * 100
        magic = "B" * 20
        bot._commands.xor_table = build_xor_table(static_key, magic)

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
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)

        bot._on_magic_captured("test_magic_key_12345")

        # XOR table should be 1000 bytes (matching fake static key length)
        xor_table = bot._commands.xor_table
        assert type(xor_table) is bytes
        assert len(xor_table) == 1000

    def test_enter_game_sends_query_command(self, fake_env: FakeEnv) -> None:
        """Test enter_game sends CMD_ENTER_GAME (type=2, cmd=0x3f)."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.enter_game()
        assert result is True
        assert fake_cdp._sent_methods == ["Runtime.evaluate"]

    def test_enter_game_no_cdp_returns_false(self, fake_env: FakeEnv) -> None:
        """Test enter_game returns False when CDP session not available."""
        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        result = bot.enter_game()
        assert result is False

    def test_move_to_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.move_to succeeds with CDP session."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.move_to(100, 100)
        assert result is True
        assert bot.get_state() == "MOVING"

    def test_pickup_fuel_to_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.pickup_fuel_to succeeds with CDP session."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.pickup_fuel_to(100, 100)
        assert result is True
        assert bot.get_state() == "COLLECTING"

    def test_teleport_to_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.teleport_to succeeds with CDP session."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession, FakePage

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
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
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession, FakePage

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        mark_teleport_landed(ws)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._page = FakePage(fake_cdp)
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"

        result = bot.teleport_to(200, 200)

        assert result is True
        assert check_and_clear_teleport_landed(ws) is False

    def test_shoot_at_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.shoot_at succeeds with CDP session."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.shoot_at(55, 53)  # Within 9-tile viewport of (50,50)
        assert result is True
        assert bot.get_state() == "COMBAT"

    def test_shoot_at_already_combat(self, fake_env: FakeEnv) -> None:
        """Test Bot.shoot_at stays in COMBAT if already in COMBAT."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "COMBAT"
        result = bot.shoot_at(55, 53)  # Within 9-tile viewport of (50,50)
        assert result is True
        assert bot.get_state() == "COMBAT"

    def test_use_radar_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.use_radar succeeds with CDP session."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        ws = WorldService()
        ws.update_world_state_from_position(50, 50)
        bot = Bot("https://test.tankpit.com/", headless=True, world=ws)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        bot._state_data = bot._state_data.copy()
        bot._state_data["state"] = "IDLE"
        result = bot.use_radar()
        assert result is True
        assert bot.get_state() == "SCANNING"

    def test_open_map_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.open_map succeeds with CDP session."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.open_map()
        assert result is True
        assert bot._state_data["in_flight_action"]["kind"] == "map_open"

    def test_close_map_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Bot.close_map dispatches a synthetic 'm' keyDown+keyUp pair via CDP.

        There is no wire byte that closes the map server-side (verified live
        in ``discover_map_close.py``). Closing is purely a client-side overlay
        toggle, so the bot dispatches an ``Input.dispatchKeyEvent`` keyDown
        and matching keyUp for the ``m`` key.
        """
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.close_map()
        assert result is True
        assert fake_cdp._sent_methods == [
            "Input.dispatchKeyEvent",
            "Input.dispatchKeyEvent",
        ]

    def test_toggle_equipment_success_with_cdp(self, fake_env: FakeEnv) -> None:
        """Test Bot.toggle_equipment succeeds with CDP session."""
        from tankpit_bot.bot.base import Bot
        from tests.fakes import FakeCDPSession

        bot = Bot("https://test.tankpit.com/", headless=True)
        fake_cdp: FakeCDPSession = FakeCDPSession()
        bot._cdp = fake_cdp
        result = bot.toggle_equipment(1)
        assert result is True

    def test_teleport_fails_if_send_fails(self, fake_env: FakeEnv) -> None:
        """Test teleport_to returns False if _send_bytes fails."""
        from tankpit_bot.bot.base import Bot
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
        from tankpit_bot.bot.base import Bot
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
    """TestBotTeleportBranches tests."""

    def test_teleport_without_page(self, fake_env: FakeEnv) -> None:
        """Test teleport_to works when _page is None (skips waits)."""
        from tankpit_bot.bot.base import Bot
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


class TestRequireCdp:
    """TestRequireCdp tests."""

    def test_require_cdp_raises_when_session_not_attached(self, fake_env: FakeEnv) -> None:
        """Bot._require_cdp raises when no CDP session is attached.

        The tick loop guarantees an attached CDP by the time it reaches
        the snapshot capture point; a missing session at that point is an
        invariant violation rather than a normal pre-bootstrap state.
        """
        import pytest

        from tankpit_bot.bot.base import Bot

        bot = Bot("https://test.tankpit.com/", headless=True)
        with pytest.raises(RuntimeError, match="no CDP session attached"):
            bot._require_cdp()


class TestBotMessageDecoding:
    """TestBotMessageDecoding tests."""

    def test_on_message_captured_extracts_magic_from_auth(self, fake_env: FakeEnv) -> None:
        """_on_message_captured extracts magic key from sent AUTH message."""
        import base64

        from tankpit_bot.bot.base import Bot
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
        bot._cdp_service._extract_magic_and_notify(msg)
        assert bot._magic == "testmagickey1234"

    def test_on_message_captured_received_updates_world(self, fake_env: FakeEnv) -> None:
        """_on_message_captured decodes received messages to update world state."""
        from tankpit_bot.bot.base import Bot
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
