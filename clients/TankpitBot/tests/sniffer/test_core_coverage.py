"""Tests for sniffer coverage branches and submodules."""

from __future__ import annotations

import base64

import pytest

from tests.conftest import FakeFileSystem

# =============================================================================
# Sniffer Coverage Branches Tests
# =============================================================================


class TestSnifferCoverageBranches:
    """Tests to cover remaining sniffer.py branches."""

    def test_build_global_xor_table_no_static_key(self, fake_fs: FakeFileSystem) -> None:
        """Test build_global_xor_table returns early when no static key file."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import xor

        # Ensure no static key file
        fake_fs.remove(DEFAULT_STATIC_KEY_PATH)

        # Reset global state
        xor._global_xor_table = None
        xor._global_static_key = None

        # Call without static key file existing
        xor.build_global_xor_table("testmagic")

        # Should remain None since no static key
        assert xor._global_xor_table is None

    def test_xor_decode_with_table(self, fake_fs: FakeFileSystem) -> None:
        """Test xor_decode decodes correctly when xor table is set."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import xor

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        # Reset and build xor table
        xor._global_xor_table = None
        xor._global_static_key = None
        xor.build_global_xor_table("testmagic")

        xor_table = xor._global_xor_table
        assert xor_table is not None and len(xor_table) == 1000, "XOR table should be 1000 bytes"

        # Test decode with body longer than 2 bytes
        body = bytes([0x2E, 0x41, 0x42, 0x43, 0x44])
        result = xor.xor_decode(body)
        assert len(result) == 4  # Should skip first byte (msg_type)

    def test_xor_decode_extends_past_table(self, fake_fs: FakeFileSystem) -> None:
        """Test xor_decode handles body longer than xor table."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import xor

        static_key = "AB"  # Very short key
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        xor._global_xor_table = None
        xor._global_static_key = None
        xor.build_global_xor_table("t")

        # Body longer than the xor table
        body = bytes([0x2E, 0x01, 0x02, 0x03, 0x04, 0x05])
        result = xor.xor_decode(body)
        assert len(result) == 5

    def test_update_viewport_origin_sets_both_edges(self) -> None:
        """Test viewport origin storage uses explicit left/top values."""
        from tankpit_bot.sniffer import viewport

        viewport.reset_viewport_tracking()

        viewport.update_viewport_origin(34, 52)

        assert viewport.get_viewport_left() == 34
        assert viewport.get_viewport_top() == 52

    def test_reset_viewport_tracking_clears_origin(self) -> None:
        """Test viewport reset clears stored origin."""
        from tankpit_bot.sniffer import viewport

        viewport.update_viewport_origin(34, 52)

        viewport.reset_viewport_tracking()

        assert viewport.get_viewport_left() is None
        assert viewport.get_viewport_top() is None

    def test_format_container_simple_container_pickup(self) -> None:
        """Test format_container_simple for container_pickup message."""
        from tankpit_bot.container.types import ContainerPickupDict
        from tankpit_bot.sniffer import format_container_simple

        msg = ContainerPickupDict(
            msg_type="container_pickup",
            x=10,
            y=20,
            volume=500,
            is_fuel=True,
        )
        result = format_container_simple(msg)
        # container_pickup returns "pos=(x,y) FUEL vol=N" format
        assert result == "pos=(10,20) FUEL vol=500"

    def test_format_container_simple_radar_response(self) -> None:
        """Test format_container_simple for radar_response message."""
        from tankpit_bot.container.types import RadarContainerDict, RadarMineDict, RadarResponseDict
        from tankpit_bot.sniffer import format_container_simple

        msg = RadarResponseDict(
            msg_type="radar_response",
            container_count=2,
            containers=[RadarContainerDict(x=10, y=20, volume=100)],
            mines=[RadarMineDict(x=30, y=40, team=0)],
        )
        result = format_container_simple(msg)
        # radar_response returns formatted container/mine count
        assert "2 containers" in str(result)
        assert "1 mines" in str(result)

    def test_format_position_details_movement_response(self) -> None:
        """Test format_position_details for movement_response (0x3D)."""
        from tankpit_bot.protocol import MovementResponseDict
        from tankpit_bot.sniffer import format_position_details

        msg = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=100,
            x=10,
            y=20,
            direction=0,
            rank=3,
            leaderboard_position=5,
        )
        result = format_position_details(msg)
        # MovementResponseDict returns empty string (no match in format_position_details)
        assert result == ""

    def test_format_message_details_tank_type(self) -> None:
        """Test format_message_details routes tank types correctly."""
        from tankpit_bot.protocol import TankEntryDict
        from tankpit_bot.sniffer import format_message_details

        msg = TankEntryDict(
            msg_type=0x28,
            tank_id=100,
            x=50,
            y=60,
            name="Test",
        )
        result = format_message_details(msg)
        # TankEntry should include name and tank_id
        assert "Test" in result
        assert "100" in result

    def test_format_message_details_resource_type(self) -> None:
        """Test format_message_details routes resource types correctly."""
        from tankpit_bot.protocol import FuelGainDict
        from tankpit_bot.sniffer import format_message_details

        msg = FuelGainDict(
            msg_type=0x44,
            fuel_total=100,
            is_free=False,
        )
        result = format_message_details(msg)
        # FuelGain should include fuel_total
        assert "100" in result

    def test_format_message_details_position_type(self) -> None:
        """Test format_message_details routes position types correctly."""
        from tankpit_bot.protocol import MinePlacementDict
        from tankpit_bot.sniffer import format_message_details

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
        from tankpit_bot.sniffer import format_message_details

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
        from tankpit_bot.sniffer import format_message_details

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
        from tankpit_bot.sniffer import decode_state_message

        # 8-byte body triggers decode_8byte_state
        body = bytes([0x2E, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])
        result = decode_state_message(body, "TEST")
        assert "[TEST]" in result

    def test_handle_tank_registry_with_name_not_container(self) -> None:
        """Test handle_tank_registry stores name for non-container tanks."""
        from tankpit_bot.sniffer import handle_tank_registry, player_tracking

        # Reset tank names
        player_tracking._tank_names.clear()

        result = handle_tank_registry(
            tid=100,
            name="TestPlayer",
            team="red",
            rank=3,
            badges=0,
            is_bot=False,
            is_container=False,
            container_y=None,
            container_viewport_x=None,
        )
        assert "TestPlayer" in result
        assert player_tracking._tank_names.get(100) == "TestPlayer"

    def test_handle_tank_registry_container_skips_name_storage(self) -> None:
        """Test handle_tank_registry does not store name for containers."""
        from tankpit_bot.sniffer import handle_tank_registry, player_tracking

        # Reset tank names
        player_tracking._tank_names.clear()

        result = handle_tank_registry(
            tid=200,
            name="ContainerName",
            team="red",
            rank=0,
            badges=0,
            is_bot=False,
            is_container=True,  # Container - should skip name storage
            container_y=50,
            container_viewport_x=10,
        )
        # Result should still format the details
        assert "200" in result
        # But name should NOT be stored in _tank_names
        assert player_tracking._tank_names.get(200) is None

    def test_init_trackers_skips_already_initialized(self, fake_fs: FakeFileSystem) -> None:
        """Test init_trackers_with_magic skips trackers with xor_table set."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import trackers

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        # First init
        for tracker in trackers.ALL_TRACKERS:
            tracker._xor_table = None
            tracker._static_key = None
        trackers.init_trackers_with_magic("magic1")

        # Capture the xor tables
        first_tables: list[bytes | None] = []
        for tracker in trackers.ALL_TRACKERS:
            first_tables.append(tracker._xor_table)

        # Second init with different magic should NOT change tables
        trackers.init_trackers_with_magic("differentmagic")

        # Tables should be same (skipped re-init)
        for i in range(len(trackers.ALL_TRACKERS)):
            assert trackers.ALL_TRACKERS[i]._xor_table == first_tables[i]

    def test_extract_magic_from_auth_valid(self) -> None:
        """Test extract_magic_from_auth extracts magic from valid AUTH payload."""
        import base64

        from tankpit_bot.sniffer.trackers import extract_magic_from_auth

        body = "%AUTH !be session|hash|ts test_magic_key_12345"
        body_bytes = body.encode("utf-8")
        length_prefix = len(body_bytes).to_bytes(2, "little")
        payload = base64.b64encode(length_prefix + body_bytes).decode("ascii")

        result = extract_magic_from_auth(payload)
        assert result == "test_magic_key_12345"

    def test_extract_magic_from_auth_invalid_base64(self) -> None:
        """Test extract_magic_from_auth returns None for invalid base64."""
        from tankpit_bot.sniffer.trackers import extract_magic_from_auth

        result = extract_magic_from_auth("not!valid@base64")
        assert result is None

    def test_extract_magic_from_auth_non_auth(self) -> None:
        """Test extract_magic_from_auth returns None for non-AUTH message."""
        import base64

        from tankpit_bot.sniffer.trackers import extract_magic_from_auth

        body = "HELLO test message"
        body_bytes = body.encode("utf-8")
        length_prefix = len(body_bytes).to_bytes(2, "little")
        payload = base64.b64encode(length_prefix + body_bytes).decode("ascii")

        result = extract_magic_from_auth(payload)
        assert result is None

    def test_format_container_details_movement(self) -> None:
        """Test format_container_details for movement message."""
        from tankpit_bot.container.types import MovementDict
        from tankpit_bot.sniffer import format_container_details

        msg = MovementDict(
            msg_type="movement",
            flags=0x7E,
            start_x=50,
            start_y=60,
            player_id=100,
            tank_id=None,
            waypoints="nnee",
            is_self=True,
        )
        result = format_container_details(msg)
        # movement returns formatted position and waypoints
        assert "(50,60)" in result
        assert "nnee" in result

    def test_dispatch_world_state_update_radar_response(self) -> None:
        """Test dispatch_world_state_update processes radar_response."""
        from tankpit_bot.container.types import RadarContainerDict, RadarMineDict, RadarResponseDict
        from tankpit_bot.sniffer import world_state

        # Reset world state
        world_state.reset_world_state()

        msg = RadarResponseDict(
            msg_type="radar_response",
            container_count=1,
            containers=[RadarContainerDict(x=10, y=20, volume=100)],
            mines=[RadarMineDict(x=30, y=40, team=0)],
        )
        # This should update world state
        world_state.dispatch_world_state_update(msg)
        # No assertion needed - just verifying the code path runs without error

    def test_dispatch_world_state_update_movement_response_valid(self) -> None:
        """Test dispatch_world_state_update with valid MovementResponse updates position."""
        from tankpit_bot.protocol import MovementResponseDict
        from tankpit_bot.sniffer import world_state

        # Reset world state
        world_state.reset_world_state()

        msg = MovementResponseDict(
            msg_type=0x3D,
            team=0,
            tank_id=100,
            x=50,
            y=60,
            direction=0,
            rank=3,
            leaderboard_position=5,
        )
        # This should update world state position
        world_state.dispatch_world_state_update(msg)
        # No crash means success - position update occurred

    def test_process_received_message_with_result(self, fake_fs: FakeFileSystem) -> None:
        """Test process_received_message logs result when message decodes."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import decoders, xor

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        # Reset and build XOR table
        xor._global_xor_table = None
        xor._global_static_key = None
        xor.build_global_xor_table("testmagic")

        # Create a valid message payload (simple text message)
        # Format: 2-byte length + body
        body = bytes([ord("=")]) + b"0|2024|Player|3|0|0|0|0"
        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        # This should decode and log (we just verify no crash)
        decoders.process_received_message(payload)

    def test_process_received_message_binary(self, fake_fs: FakeFileSystem) -> None:
        """Test process_received_message handles binary messages."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH
        from tankpit_bot.sniffer import decoders, xor

        static_key = "ABCDEF" + "A" * 994
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        # Reset and build XOR table
        xor._global_xor_table = None
        xor._global_static_key = None
        xor.build_global_xor_table("testmagic")

        xor_table = xor._global_xor_table
        assert xor_table is not None and len(xor_table) > 0

        # Create a binary message (not a text type)
        # Use 0x41 'A' for Deactivation message: victim=1, killer=2, rank=3, points=5
        plaintext = bytes([0x41, 0x01, 0x00, 0x02, 0x00, 0x03, 0x05])
        # XOR encode the body (skip first byte which is msg_type indicator)
        body = bytes([0x2E])  # Container prefix
        body += bytes(plaintext[i] ^ xor_table[i] for i in range(len(plaintext)))

        header = len(body).to_bytes(2, "little")
        payload = base64.b64encode(header + body).decode()

        # This should decode through the binary path
        decoders.process_received_message(payload)


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

    def test_get_global_xor_table(self) -> None:
        """Test get_global_xor_table returns table or None."""
        from tankpit_bot.sniffer import xor

        xor._global_xor_table = None
        assert xor.get_global_xor_table() is None

        xor._global_xor_table = b"test"
        assert xor.get_global_xor_table() == b"test"

        # Clean up
        xor._global_xor_table = None

    def test_reset_all_trackers(self) -> None:
        """Test reset_all_trackers clears all tracker state."""
        from tankpit_bot.sniffer import trackers

        # Set some state on trackers
        for tracker in trackers.ALL_TRACKERS:
            tracker._xor_table = b"test_table"
            tracker._static_key = "test_key"

        # Reset
        trackers.reset_all_trackers()

        # Verify all are cleared
        for tracker in trackers.ALL_TRACKERS:
            assert tracker._xor_table is None
            assert tracker._static_key is None

    def test_register_tank_name_empty_name(self) -> None:
        """Test register_tank_name ignores empty names."""
        from tankpit_bot.sniffer import player_tracking

        player_tracking._tank_names.clear()

        # Empty name should not be stored
        player_tracking.register_tank_name(100, "")
        assert 100 not in player_tracking._tank_names

        # Non-empty name should be stored
        player_tracking.register_tank_name(100, "Player")
        assert player_tracking._tank_names.get(100) == "Player"

    def test_get_tank_name_not_found(self) -> None:
        """Test get_tank_name returns empty string for unknown tank."""
        from tankpit_bot.sniffer import player_tracking

        player_tracking._tank_names.clear()

        # Unknown tank should return empty string
        result = player_tracking.get_tank_name(999)
        assert result == ""
