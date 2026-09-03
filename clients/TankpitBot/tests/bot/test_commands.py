"""Tests for bot command types and encoding."""

from __future__ import annotations


class TestBotCommandTypes:
    """Tests for bot command factory functions."""

    def test_make_move_command(self) -> None:
        """Test make_move_command creates correct TypedDict."""
        from tankpit_bot.bot.types import make_move_command

        cmd = make_move_command(100, 150)
        assert cmd["cmd_type"] == "move"
        assert cmd["target_x"] == 100
        assert cmd["target_y"] == 150

    def test_make_shoot_command(self) -> None:
        """Test make_shoot_command creates correct TypedDict."""
        from tankpit_bot.bot.types import make_shoot_command

        cmd = make_shoot_command(50, 75)
        assert cmd["cmd_type"] == "shoot"
        assert cmd["target_x"] == 50
        assert cmd["target_y"] == 75

    def test_make_radar_command(self) -> None:
        """Test make_radar_command creates correct TypedDict."""
        from tankpit_bot.bot.types import make_radar_command

        cmd = make_radar_command()
        assert cmd["cmd_type"] == "radar"

    def test_make_pickup_fuel_command(self) -> None:
        """Test make_pickup_fuel_command creates correct TypedDict."""
        from tankpit_bot.bot.types import make_pickup_fuel_command

        cmd = make_pickup_fuel_command(200, 100)
        assert cmd["cmd_type"] == "pickup_fuel"
        assert cmd["target_x"] == 200
        assert cmd["target_y"] == 100

    def test_make_pickup_equipment_command(self) -> None:
        """Test make_pickup_equipment_command creates correct TypedDict."""
        from tankpit_bot.bot.types import make_pickup_equipment_command

        cmd = make_pickup_equipment_command(200, 100)
        assert cmd["cmd_type"] == "pickup_equipment"
        assert cmd["target_x"] == 200
        assert cmd["target_y"] == 100

    def test_make_teleport_command(self) -> None:
        """Test make_teleport_command creates correct TypedDict."""
        from tankpit_bot.bot.types import make_teleport_command

        cmd = make_teleport_command(128, 64)
        assert cmd["cmd_type"] == "teleport"
        assert cmd["target_x"] == 128
        assert cmd["target_y"] == 64


class TestBotCommandEncoding:
    """Tests for bot command encoding functions."""

    def test_encode_move_command(self) -> None:
        """Test encode_move_command encodes command to expected bytes."""
        from tankpit_bot.bot.commands import encode_move_command
        from tankpit_bot.bot.types import make_move_command
        from tankpit_bot.protocol.command_builders import build_move_command

        cmd = make_move_command(100, 150)
        result = encode_move_command(cmd)
        expected = build_move_command(100, 150)
        assert result == expected

    def test_encode_shoot_command(self) -> None:
        """Test encode_shoot_command encodes command to expected bytes."""
        from tankpit_bot.bot.commands import encode_shoot_command
        from tankpit_bot.bot.types import make_shoot_command
        from tankpit_bot.protocol.command_builders import build_shoot_command

        cmd = make_shoot_command(50, 75)
        result = encode_shoot_command(cmd)
        expected = build_shoot_command(50, 75)
        assert result == expected

    def test_encode_radar_command(self) -> None:
        """Test encode_radar_command encodes command to expected bytes."""
        from tankpit_bot.bot.commands import encode_radar_command
        from tankpit_bot.bot.types import make_radar_command
        from tankpit_bot.protocol.command_builders import build_query_command
        from tankpit_bot.protocol.commands import CMD_RADAR

        cmd = make_radar_command()
        result = encode_radar_command(cmd)
        expected = build_query_command(CMD_RADAR)
        assert result == expected

    def test_encode_pickup_fuel_command(self) -> None:
        """Test encode_pickup_fuel_command encodes command to expected bytes."""
        from tankpit_bot.bot.commands import encode_pickup_fuel_command
        from tankpit_bot.bot.types import make_pickup_fuel_command
        from tankpit_bot.protocol.command_builders import build_pickup_fuel_command

        cmd = make_pickup_fuel_command(200, 100)
        result = encode_pickup_fuel_command(cmd)
        expected = build_pickup_fuel_command(200, 100)
        assert result == expected

    def test_encode_pickup_equipment_command(self) -> None:
        """Test encode_pickup_equipment_command encodes command to expected bytes."""
        from tankpit_bot.bot.commands import encode_pickup_equipment_command
        from tankpit_bot.bot.types import make_pickup_equipment_command
        from tankpit_bot.protocol.command_builders import build_pickup_equipment_command

        cmd = make_pickup_equipment_command(200, 100)
        result = encode_pickup_equipment_command(cmd)
        expected = build_pickup_equipment_command(200, 100)
        assert result == expected

    def test_encode_teleport_command(self) -> None:
        """Test encode_teleport_command encodes command to expected bytes."""
        from tankpit_bot.bot.commands import encode_teleport_command
        from tankpit_bot.bot.types import make_teleport_command
        from tankpit_bot.protocol.command_builders import build_teleport_command

        cmd = make_teleport_command(128, 64)
        result = encode_teleport_command(cmd)
        expected = build_teleport_command(128, 64)
        assert result == expected
