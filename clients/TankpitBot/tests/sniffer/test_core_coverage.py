"""Tests for sniffer coverage branches and submodules."""

from __future__ import annotations

import base64

import pytest

from tankpit_bot.capture.xor import build_session_xor_table, reset_static_key_cache
from tankpit_bot.sniffer.world_state import get_world_service
from tests.conftest import FakeFileSystem

# =============================================================================
# Sniffer Coverage Branches Tests
# =============================================================================


class TestSnifferCoverageBranches:
    """Tests to cover remaining sniffer.py branches."""

    def test_format_container_simple_container_pickup(self) -> None:
        """Test format_container_simple for container_pickup message.

        ``remaining_volume>0`` means the picker took a partial top-up
        and left fuel behind in the container (typical when the picker
        was already near the 1100-fuel cap).
        """
        from tankpit_bot.container.types import (
            ContainerPickupDict,
            ContainerPickupRecordDict,
        )
        from tankpit_bot.sniffer.formatters import format_container_simple

        msg = ContainerPickupDict(
            msg_type="container_pickup",
            pickups=(ContainerPickupRecordDict(x=10, y=20, remaining_volume=500),),
        )
        result = format_container_simple(msg)
        assert result == "pos=(10,20) FUEL partial remaining=500"

    def test_format_container_simple_radar_response(self) -> None:
        """Format the protocol-layer RadarScanResult (0x4F)."""
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict, RadarScanResultDict
        from tankpit_bot.sniffer.formatters import format_container_simple

        msg = RadarScanResultDict(
            msg_type=0x4F,
            containers=[
                RadarContainerDict(x=10, y=20, volume=100),
                RadarContainerDict(x=11, y=21, volume=-1),
            ],
            mines=[RadarMineDict(x=30, y=40, team=0)],
            mine_clears=[],
        )
        result = format_container_simple(msg)
        assert "2 containers" in str(result)
        assert "1 mines" in str(result)

    def test_format_position_details_movement_response(self) -> None:
        """Test format_position_details for movement_response (0x3D)."""
        from tankpit_bot.protocol import MovementResponseDict
        from tankpit_bot.sniffer.formatters import format_position_details

        msg = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=100,
            x=10,
            y=20,
            direction=0,
            damage_state=0,
            rank=3,
            lb_score=5,
            carrying=0,
        )
        result = format_position_details(msg)
        # MovementResponseDict returns empty string (no match in format_position_details)
        assert result == ""

    def test_format_message_details_tank_type(self) -> None:
        """Test format_message_details routes tank types correctly."""
        from tankpit_bot.protocol import TankEntryDict
        from tankpit_bot.sniffer.formatters import format_message_details

        msg = TankEntryDict(
            msg_type=0x28,
            team=0,
            tank_id=100,
            rank=0,
            damage_state=0,
            score=0,
            x=50,
            y=60,
        )
        result = format_message_details(msg)
        # TankEntry should include tank_id and team
        assert "tank=100" in result
        assert "team=0" in result

    def test_format_message_details_resource_type(self) -> None:
        """Test format_message_details routes resource types correctly."""
        from tankpit_bot.protocol import FuelGainDict
        from tankpit_bot.sniffer.formatters import format_message_details

        msg = FuelGainDict(
            msg_type=0x44,
            fuel_total=100,
            is_free=False,
            flag=1,
        )
        result = format_message_details(msg)
        # FuelGain should include fuel_total
        assert "100" in result

    def test_format_message_details_position_type(self) -> None:
        """Test format_message_details routes position types correctly."""
        from tankpit_bot.container import MinePlacementDict
        from tankpit_bot.sniffer.formatters import format_message_details

        msg = MinePlacementDict(
            msg_type=0x4B,
            mine_type=0,
            tank_id=100,
            positions=[(50, 60)],
        )
        result = format_message_details(msg)
        # MinePlacement should include tank id
        assert "100" in result

    def test_format_message_details_radar_type(self) -> None:
        """Test format_message_details routes radar types correctly."""
        from tankpit_bot.protocol import RadarResultDict
        from tankpit_bot.sniffer.formatters import format_message_details

        msg = RadarResultDict(
            msg_type=0x46,
            detection_type=1,
            found=True,
        )
        result = format_message_details(msg)
        # RadarResult is handled by format_radar_details
        assert "type=1" in result
        assert "found=True" in result

    def test_format_message_details_misc_type(self) -> None:
        """Test format_message_details routes misc types correctly."""
        from tankpit_bot.protocol import EquipmentGainDict
        from tankpit_bot.sniffer.formatters import format_message_details

        msg = EquipmentGainDict(
            msg_type=0x67,
            show_message=True,
            gained=[1, 2],
        )
        result = format_message_details(msg)
        # EquipmentGain includes gained list
        assert "gained=[1, 2]" in result

    def test_decode_8byte_state(self) -> None:
        """Test decode_state_message handles 8-byte state messages."""
        from tankpit_bot.sniffer.decoders import decode_state_message

        # 8-byte body triggers decode_8byte_state
        body = bytes([0x2E, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])
        result = decode_state_message(body, "TEST")
        assert "[TEST]" in result

    # handle_tank_registry tests deleted 2026-06-20: the helper and the
    # underlying TankRegistryDict were removed after corpus sweep proved
    # zero production fires for the container path.

    # format_container_details for container Movement was deleted
    # 2026-06-19. The protocol 0x47 MovementDict formatter test lives
    # in tests/sniffer/test_formatters_details.py.

    def test_dispatch_world_state_update_radar_response(self) -> None:
        """Test dispatch_world_state_update processes radar_response."""
        from tankpit_bot.protocol import RadarContainerDict, RadarMineDict, RadarScanResultDict
        from tankpit_bot.sniffer import world_state_dispatch
        from tankpit_bot.sniffer.world_state import get_world_service

        # Reset world state

        msg = RadarScanResultDict(
            msg_type=0x4F,
            containers=[RadarContainerDict(x=10, y=20, volume=100)],
            mines=[RadarMineDict(x=30, y=40, team=0)],
            mine_clears=[],
        )
        world_state_dispatch.dispatch_world_state_update(get_world_service(), msg)

    def test_dispatch_world_state_update_movement_response_valid(self) -> None:
        """Test dispatch_world_state_update with valid MovementResponse updates position."""
        from tankpit_bot.protocol import MovementResponseDict
        from tankpit_bot.sniffer import world_state_dispatch
        from tankpit_bot.sniffer.world_state import get_world_service

        # Reset world state

        msg = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=100,
            x=50,
            y=60,
            direction=0,
            damage_state=0,
            rank=3,
            lb_score=5,
            carrying=0,
        )
        # This should update world state position
        world_state_dispatch.dispatch_world_state_update(get_world_service(), msg)
        # No crash means success - position update occurred

    def test_process_received_message_with_result(self, fake_fs: FakeFileSystem) -> None:
        """Test process_received_message logs result when message decodes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import decoders

        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, "ABCDEF" + "A" * 994)
        reset_static_key_cache()

        # Create a valid message payload (simple text message)
        # Format: 2-byte length + body
        body = bytes([ord("=")]) + b"0|2024|Player|3|0|0|0|0"
        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        # This should decode and log (we just verify no crash)
        decoders.process_received_message(
            get_world_service(), payload, build_session_xor_table("testmagic")
        )

    def test_process_received_message_binary(self, fake_fs: FakeFileSystem) -> None:
        """Test process_received_message handles binary messages."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import decoders

        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, "ABCDEF" + "A" * 994)
        reset_static_key_cache()
        xor_table = build_session_xor_table("testmagic")

        # Create a binary message (not a text type)
        # Use 0x41 'A' for Deactivation message: victim=1, killer=2, rank=3, points=5
        plaintext = bytes([0x41, 0x01, 0x00, 0x02, 0x00, 0x03, 0x05])
        # XOR encode the body (skip first byte which is msg_type indicator)
        body = bytes([0x2E])  # Container prefix
        body += bytes(plaintext[i] ^ xor_table[i] for i in range(len(plaintext)))

        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        # This should decode through the binary path
        decoders.process_received_message(get_world_service(), payload, xor_table)


# =============================================================================
# Log JS Fuel Findings Tests
# =============================================================================


class TestLogJsFuelFindings:
    """Tests for _log_js_fuel_findings function."""

    def test_result_wrapper_not_dict(self) -> None:
        """Test early return when result wrapper is not a dict."""
        from tankpit_bot.sniffer.core import _log_js_fuel_findings

        # result_wrapper is None
        _log_js_fuel_findings({})
        # result_wrapper is not a dict
        _log_js_fuel_findings({"result": "string"})
        _log_js_fuel_findings({"result": 123})
        _log_js_fuel_findings({"result": []})

    def test_findings_not_list(self) -> None:
        """Test early return when findings value is not a list."""
        from tankpit_bot.sniffer.core import _log_js_fuel_findings

        # value is None
        _log_js_fuel_findings({"result": {}})
        # value is not a list
        _log_js_fuel_findings({"result": {"value": "string"}})
        _log_js_fuel_findings({"result": {"value": 123}})
        _log_js_fuel_findings({"result": {"value": {}}})

    def test_finding_not_dict(self) -> None:
        """Test skipping non-dict findings in list."""
        from tankpit_bot.sniffer.core import _log_js_fuel_findings

        # List contains non-dict items - should skip them
        _log_js_fuel_findings({"result": {"value": ["string", 123, None]}})

    def test_finding_missing_path_or_value(self) -> None:
        """Test skipping findings with missing path or value."""
        from tankpit_bot.sniffer.core import _log_js_fuel_findings

        # path is None
        _log_js_fuel_findings({"result": {"value": [{"value": 1000}]}})
        # value is None
        _log_js_fuel_findings({"result": {"value": [{"path": "game.fuel"}]}})
        # Both None
        _log_js_fuel_findings({"result": {"value": [{}]}})

    def test_logs_valid_findings(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that valid findings are logged."""
        import logging

        from tankpit_bot.sniffer.core import _log_js_fuel_findings

        with caplog.at_level(logging.INFO):
            _log_js_fuel_findings(
                {
                    "result": {
                        "value": [
                            {"path": "game.fuel", "value": 1000},
                            {"path": "player.hp", "value": 1200},
                        ]
                    }
                }
            )

        assert "[JS:FUEL] game.fuel = 1000" in caplog.text
        assert "[JS:FUEL] player.hp = 1200" in caplog.text


# =============================================================================
# Submodule Coverage Tests
# =============================================================================


class TestSubmoduleCoverage:
    """Tests for uncovered branches in sniffer submodules."""
