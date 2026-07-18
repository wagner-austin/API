"""Tests for bot.command_service.CommandService."""

from __future__ import annotations

from collections.abc import Callable

from platform_core.json_utils import JSONObject

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.bot.command_service import CommandService


class _FakeCDP:
    """Minimal CDP session for testing."""

    def send(self, method: str, params: JSONObject | None = None) -> JSONObject:
        return {}

    def on(self, event: str, handler: Callable[[JSONObject], None]) -> None:
        pass

    def detach(self) -> None:
        pass


class TestCommandService:
    """Tests for all CommandService methods."""

    def setup_method(self) -> None:
        self._sent: list[tuple[str, bytes]] = []

        def _send(cdp: CDPSessionProtocol, data: bytes, label: str) -> str:
            self._sent.append((label, data))
            return ""

        self.svc = CommandService(send_ws_bytes=_send)
        self.svc.cdp = _FakeCDP()
        self.svc.xor_table = b"\x00" * 256

    def test_enter_game(self) -> None:
        assert self.svc.enter_game() is True
        assert len(self._sent) == 1
        assert self._sent[0][0] == "enter_game"

    def test_quit_game_sends_plain_unencoded_frame(self) -> None:
        """Graceful quit sends the plain q-key frame, never XOR-encoded."""
        assert self.svc.quit_game() is True
        label, data = self._sent[0]
        assert label == "quit_game"
        assert data == b"\x01\x00-"

    def test_move(self) -> None:
        assert self.svc.move(50, 60) is True
        assert self._sent[0][0] == "move"

    def test_pickup_fuel(self) -> None:
        assert self.svc.pickup_fuel(10, 20) is True
        assert self._sent[0][0] == "pickup_fuel"

    def test_pickup_equipment(self) -> None:
        assert self.svc.pickup_equipment(10, 20) is True
        assert self._sent[0][0] == "pickup_equipment"

    def test_teleport(self) -> None:
        assert self.svc.teleport(100, 200) is True
        assert "teleport" in self._sent[0][0]

    def test_teleport_no_cdp(self) -> None:
        self.svc.cdp = None
        assert self.svc.teleport(100, 200) is False
        assert len(self._sent) == 0

    def test_shoot(self) -> None:
        assert self.svc.shoot(30, 40, target_id=99) is True
        assert "shoot" in self._sent[0][0]

    def test_radar(self) -> None:
        assert self.svc.radar() is True
        assert self._sent[0][0] == "radar"

    def test_open_map(self) -> None:
        assert self.svc.open_map() is True
        assert self._sent[0][0] == "map_open"

    def test_request_nearest_enemy(self) -> None:
        assert self.svc.request_nearest_enemy() is True
        assert self._sent[0][0] == "nearest_enemy"

    def test_toggle_equipment_valid(self) -> None:
        assert self.svc.toggle_equipment(2) is True
        assert "toggle_dual" in self._sent[0][0]

    def test_toggle_equipment_slot_1(self) -> None:
        assert self.svc.toggle_equipment(1) is True
        assert "toggle_armor" in self._sent[0][0]

    def test_toggle_equipment_slot_5(self) -> None:
        assert self.svc.toggle_equipment(5) is True
        assert "toggle_radar" in self._sent[0][0]

    def test_toggle_equipment_invalid_low(self) -> None:
        assert self.svc.toggle_equipment(0) is False
        assert len(self._sent) == 0

    def test_toggle_equipment_invalid_high(self) -> None:
        assert self.svc.toggle_equipment(6) is False
        assert len(self._sent) == 0

    def test_send_bytes_no_cdp(self) -> None:
        self.svc.cdp = None
        assert self.svc.send_bytes(b"\x00\x00", "test") is False


class TestCommandServiceNoCDP:
    """All command methods return False when CDP is None."""

    def setup_method(self) -> None:
        self._sent: list[tuple[str, bytes]] = []

        def _send(cdp: CDPSessionProtocol, data: bytes, label: str) -> str:
            self._sent.append((label, data))
            return ""

        self.svc = CommandService(send_ws_bytes=_send)

    def test_enter_game_no_cdp(self) -> None:
        assert self.svc.enter_game() is False

    def test_move_no_cdp(self) -> None:
        assert self.svc.move(50, 60) is False

    def test_shoot_no_cdp(self) -> None:
        assert self.svc.shoot(30, 40) is False

    def test_radar_no_cdp(self) -> None:
        assert self.svc.radar() is False
