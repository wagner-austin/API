"""Tests for tankpit_bot.commands module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.protocol.command_builders import (
    build_move_command,
    build_pickup_equipment_command,
    build_pickup_fuel_command,
    build_query_command,
    build_scope_command,
    build_shoot_command,
    build_teleport_command,
    build_toggle_equipment_command,
)
from tankpit_bot.protocol.command_frames import (
    ActionCommand,
    QueryCommand,
    decode_action_command,
    decode_query_command,
    deserialize_command,
    encode_action_command,
    encode_query_command,
    make_action_command,
    make_query_command,
    serialize_action_command,
    serialize_query_command,
)
from tankpit_bot.protocol.commands import (
    CMD_MAP_TELEPORT,
    CMD_MOVE,
    CMD_PICKUP_EQUIPMENT,
    CMD_PICKUP_FUEL,
    CMD_RADAR,
    CMD_SCOPE,
    CMD_SHOOT,
    CMD_TOGGLE_EQUIPMENT,
    COMMAND_PREFIX,
    SCOPE_NORTH,
    TYPE_COMBAT,
    TYPE_MOVEMENT,
    TYPE_QUERY,
    TYPE_UI,
)

# =============================================================================
# Constants Tests
# =============================================================================


def test_command_prefix_is_exclamation() -> None:
    """Test COMMAND_PREFIX is '!' (0x21)."""
    assert ord("!") == COMMAND_PREFIX
    assert COMMAND_PREFIX == 0x21


# =============================================================================
# QueryCommand Tests
# =============================================================================


def test_make_query_command() -> None:
    """Test creating a query command."""
    cmd = make_query_command(0x42)

    assert cmd["kind"] == "query"
    assert cmd["cmd_id"] == 0x42


def test_encode_query_command() -> None:
    """Test encoding QueryCommand to JSON."""
    cmd = QueryCommand(kind="query", cmd_id=0x03)

    result = encode_query_command(cmd)

    assert result["kind"] == "query"
    assert result["cmd_id"] == 0x03


def test_decode_query_command() -> None:
    """Test decoding QueryCommand from JSON."""
    data: JSONObject = {"kind": "query", "cmd_id": 0x01}

    result = decode_query_command(data)

    assert result["kind"] == "query"
    assert result["cmd_id"] == 0x01


def test_decode_query_command_wrong_kind_raises() -> None:
    """Test decode_query_command raises for wrong kind."""
    data: JSONObject = {"kind": "action", "cmd_id": 1}

    with pytest.raises(JSONTypeError, match="Expected kind='query'"):
        decode_query_command(data)


def test_decode_query_command_missing_kind_raises() -> None:
    """Test decode_query_command raises for missing kind."""
    data: JSONObject = {"cmd_id": 1}

    with pytest.raises(JSONTypeError, match="kind"):
        decode_query_command(data)


def test_decode_query_command_missing_cmd_id_raises() -> None:
    """Test decode_query_command raises for missing cmd_id."""
    data: JSONObject = {"kind": "query"}

    with pytest.raises(JSONTypeError, match="cmd_id"):
        decode_query_command(data)


# =============================================================================
# ActionCommand Tests
# =============================================================================


def test_make_action_command() -> None:
    """Test creating an action command."""
    cmd = make_action_command(0x10, b"\x12\x34")

    assert cmd["kind"] == "action"
    assert cmd["cmd_id"] == 0x10
    assert cmd["data"] == "1234"


def test_make_action_command_empty_data() -> None:
    """Test creating an action command with empty data."""
    cmd = make_action_command(0x12, b"")

    assert cmd["data"] == ""


def test_encode_action_command() -> None:
    """Test encoding ActionCommand to JSON."""
    cmd = ActionCommand(kind="action", cmd_id=0x11, data="abcd")

    result = encode_action_command(cmd)

    assert result["kind"] == "action"
    assert result["cmd_id"] == 0x11
    assert result["data"] == "abcd"


def test_decode_action_command() -> None:
    """Test decoding ActionCommand from JSON."""
    data: JSONObject = {"kind": "action", "cmd_id": 0x10, "data": "ff00"}

    result = decode_action_command(data)

    assert result["kind"] == "action"
    assert result["cmd_id"] == 0x10
    assert result["data"] == "ff00"


def test_decode_action_command_wrong_kind_raises() -> None:
    """Test decode_action_command raises for wrong kind."""
    data: JSONObject = {"kind": "query", "cmd_id": 1, "data": "00"}

    with pytest.raises(JSONTypeError, match="Expected kind='action'"):
        decode_action_command(data)


def test_decode_action_command_missing_data_raises() -> None:
    """Test decode_action_command raises for missing data."""
    data: JSONObject = {"kind": "action", "cmd_id": 1}

    with pytest.raises(JSONTypeError, match="data"):
        decode_action_command(data)


# =============================================================================
# Serialization Tests
# =============================================================================


def test_serialize_query_command() -> None:
    """Test serializing query command to bytes."""
    cmd = QueryCommand(kind="query", cmd_id=0x42)
    type_byte = 0x30

    result = serialize_query_command(cmd, type_byte)

    assert result == bytes([0x21, 0x30, 0x42])
    assert len(result) == 3


def test_serialize_action_command() -> None:
    """Test serializing action command to bytes."""
    cmd = ActionCommand(kind="action", cmd_id=0x10, data="abcd")
    type_byte = 0x25

    result = serialize_action_command(cmd, type_byte)

    # '!' + type + cmd + data
    assert result == bytes([0x21, 0x25, 0x10, 0xAB, 0xCD])


def test_serialize_action_command_empty_data() -> None:
    """Test serializing action command with empty data."""
    cmd = ActionCommand(kind="action", cmd_id=0x20, data="")
    type_byte = 0x30

    result = serialize_action_command(cmd, type_byte)

    assert result == bytes([0x21, 0x30, 0x20])


# =============================================================================
# Deserialization Tests
# =============================================================================


def test_deserialize_query_command() -> None:
    """Test deserializing a 3-byte command as query."""
    data = bytes([0x21, 0x30, 0x42])
    type_byte = 0x30

    result = deserialize_command(data, type_byte)

    assert result["kind"] == "query"
    assert result["cmd_id"] == 0x42


def test_deserialize_action_command() -> None:
    """Test deserializing a command with data as action."""
    data = bytes([0x21, 0x25, 0x10, 0xAB, 0xCD])
    type_byte = 0x25

    result = deserialize_command(data, type_byte)

    assert result["kind"] == "action"
    assert result["cmd_id"] == 0x10
    assert result["data"] == "abcd"


def test_deserialize_command_too_short_raises() -> None:
    """Test deserialize raises for command < 3 bytes."""
    data = bytes([0x21, 0x30])

    with pytest.raises(ValueError, match="Command too short"):
        deserialize_command(data, 0x30)


def test_deserialize_command_wrong_prefix_raises() -> None:
    """Test deserialize raises for wrong prefix byte."""
    data = bytes([0x20, 0x30, 0x42])  # 0x20 instead of '!' (0x21)

    with pytest.raises(ValueError, match="Invalid command prefix"):
        deserialize_command(data, 0x30)


def test_deserialize_command_type_mismatch_raises() -> None:
    """Test deserialize raises for type byte mismatch."""
    data = bytes([0x21, 0x30, 0x42])
    type_byte = 0x99  # Doesn't match 0x30

    with pytest.raises(ValueError, match="Type byte mismatch"):
        deserialize_command(data, type_byte)


# =============================================================================
# Round-trip Tests
# =============================================================================


def test_query_command_roundtrip_encode_decode() -> None:
    """Test QueryCommand encode then decode returns equivalent."""
    original = QueryCommand(kind="query", cmd_id=0x03)

    encoded = encode_query_command(original)
    decoded = decode_query_command(encoded)

    assert decoded["kind"] == original["kind"]
    assert decoded["cmd_id"] == original["cmd_id"]


def test_action_command_roundtrip_encode_decode() -> None:
    """Test ActionCommand encode then decode returns equivalent."""
    original = ActionCommand(kind="action", cmd_id=0x10, data="1234abcd")

    encoded = encode_action_command(original)
    decoded = decode_action_command(encoded)

    assert decoded["kind"] == original["kind"]
    assert decoded["cmd_id"] == original["cmd_id"]
    assert decoded["data"] == original["data"]


def test_query_command_roundtrip_serialize_deserialize() -> None:
    """Test serialize then deserialize returns equivalent QueryCommand."""
    original = QueryCommand(kind="query", cmd_id=0x02)
    type_byte = 0x42

    serialized = serialize_query_command(original, type_byte)
    deserialized = deserialize_command(serialized, type_byte)

    assert deserialized["kind"] == original["kind"]
    assert deserialized["cmd_id"] == original["cmd_id"]


def test_action_command_roundtrip_serialize_deserialize() -> None:
    """Test serialize then deserialize returns equivalent ActionCommand."""
    original = ActionCommand(kind="action", cmd_id=0x11, data="deadbeef")
    type_byte = 0x33

    serialized = serialize_action_command(original, type_byte)
    deserialized = deserialize_command(serialized, type_byte)

    assert deserialized["kind"] == original["kind"]
    assert deserialized["cmd_id"] == original["cmd_id"]
    assert deserialized["data"] == original["data"]


# =============================================================================
# Type Constants Tests
# =============================================================================


def test_type_query_value() -> None:
    """Test TYPE_QUERY is raw type number 2."""
    assert TYPE_QUERY == 2


def test_type_ui_value() -> None:
    """Test TYPE_UI is raw type number 3."""
    assert TYPE_UI == 3


def test_type_movement_value() -> None:
    """Test TYPE_MOVEMENT is raw type number 4."""
    assert TYPE_MOVEMENT == 4


def test_type_combat_value() -> None:
    """Test TYPE_COMBAT is raw type number 6."""
    assert TYPE_COMBAT == 6


# =============================================================================
# Wire Format Command Builder Tests
# =============================================================================


class TestBuildQueryCommand:
    """Tests for build_query_command."""

    def test_builds_correct_format(self) -> None:
        """Test query command has correct wire format."""
        result = build_query_command(CMD_RADAR)

        # Format: [len_lo, len_hi] + ! + 0x22 + cmd_id
        assert result[0] == 3  # body length
        assert result[1] == 0  # high byte of length
        assert result[2] == COMMAND_PREFIX  # '!'
        assert result[3] == TYPE_QUERY  # 0x22
        assert result[4] == CMD_RADAR  # 0x66

    def test_total_length_is_5(self) -> None:
        """Test query command is 5 bytes total."""
        result = build_query_command(0x42)
        assert len(result) == 5

    def test_body_length_is_3(self) -> None:
        """Test body length header is 3."""
        result = build_query_command(0x01)
        body_length = result[0] | (result[1] << 8)
        assert body_length == 3


class TestBuildMoveCommand:
    """Tests for build_move_command."""

    def test_builds_correct_format(self) -> None:
        """Test move command has correct wire format."""
        result = build_move_command(92, 91)

        # Format: [len_lo, len_hi] + ! + 0x24 + 0x70 + X + Y
        assert result[0] == 5  # body length
        assert result[1] == 0  # high byte
        assert result[2] == COMMAND_PREFIX  # '!'
        assert result[3] == TYPE_MOVEMENT  # 0x24
        assert result[4] == CMD_MOVE  # 0x70
        assert result[5] == 92  # X
        assert result[6] == 91  # Y

    def test_total_length_is_7(self) -> None:
        """Test move command is 7 bytes total."""
        result = build_move_command(0, 0)
        assert len(result) == 7

    def test_matches_raw_format(self) -> None:
        """Test output matches raw (pre-XOR) command format.

        Raw format: [len_lo, len_hi, '!', type=4, cmd=0x70, x, y]
        XOR encoding is applied later by the bot before sending.
        """
        result = build_move_command(92, 91)
        assert result.hex() == "05002104705c5b"

    def test_coordinates_masked_to_byte(self) -> None:
        """Test coordinates are masked to single byte."""
        result = build_move_command(256, 257)
        assert result[5] == 0  # 256 & 0xFF
        assert result[6] == 1  # 257 & 0xFF


class TestBuildPickupFuelCommand:
    """Tests for build_pickup_fuel_command."""

    def test_builds_correct_format(self) -> None:
        """Test fuel pickup command has correct wire format."""
        result = build_pickup_fuel_command(50, 60)

        assert result[2] == COMMAND_PREFIX
        assert result[3] == TYPE_MOVEMENT
        assert result[4] == CMD_PICKUP_FUEL
        assert result[5] == 50
        assert result[6] == 60

    def test_total_length_is_7(self) -> None:
        """Test fuel pickup command is 7 bytes total."""
        result = build_pickup_fuel_command(0, 0)
        assert len(result) == 7


class TestBuildPickupEquipmentCommand:
    """Tests for build_pickup_equipment_command."""

    def test_builds_correct_format(self) -> None:
        """Test equipment pickup command has correct wire format."""
        result = build_pickup_equipment_command(50, 60)

        assert result[2] == COMMAND_PREFIX
        assert result[3] == TYPE_MOVEMENT
        assert result[4] == CMD_PICKUP_EQUIPMENT
        assert result[5] == 50
        assert result[6] == 60

    def test_total_length_is_7(self) -> None:
        """Test equipment pickup command is 7 bytes total."""
        result = build_pickup_equipment_command(0, 0)
        assert len(result) == 7


class TestBuildTeleportCommand:
    """Tests for build_teleport_command."""

    def test_builds_correct_format(self) -> None:
        """Test teleport command has correct wire format."""
        result = build_teleport_command(100, 200)

        assert result[2] == COMMAND_PREFIX
        assert result[3] == TYPE_MOVEMENT
        assert result[4] == CMD_MAP_TELEPORT
        assert result[5] == 100
        assert result[6] == 200

    def test_total_length_is_7(self) -> None:
        """Test teleport command is 7 bytes total."""
        result = build_teleport_command(0, 0)
        assert len(result) == 7


class TestBuildShootCommand:
    """Tests for build_shoot_command."""

    def test_builds_correct_format_no_target(self) -> None:
        """Test shoot command with no target has correct wire format."""
        result = build_shoot_command(80, 90)

        # Format: [len_lo, len_hi] + ! + 0x26 + 0x73 + X + Y + id_lo + id_hi
        assert result[0] == 7  # body length
        assert result[1] == 0  # high byte
        assert result[2] == COMMAND_PREFIX
        assert result[3] == TYPE_COMBAT
        assert result[4] == CMD_SHOOT
        assert result[5] == 80  # X
        assert result[6] == 90  # Y
        assert result[7] == 0  # target_id low
        assert result[8] == 0  # target_id high

    def test_builds_correct_format_with_target(self) -> None:
        """Test shoot command with target ID has correct wire format."""
        result = build_shoot_command(10, 20, target_id=0x1234)

        assert result[5] == 10  # X
        assert result[6] == 20  # Y
        assert result[7] == 0x34  # target_id low byte
        assert result[8] == 0x12  # target_id high byte

    def test_total_length_is_9(self) -> None:
        """Test shoot command is 9 bytes total."""
        result = build_shoot_command(0, 0)
        assert len(result) == 9

    def test_target_id_little_endian(self) -> None:
        """Test target ID is encoded as little-endian uint16."""
        result = build_shoot_command(0, 0, target_id=0xABCD)
        assert result[7] == 0xCD  # low byte first
        assert result[8] == 0xAB  # high byte second


class TestBuildScopeCommand:
    """Tests for build_scope_command."""

    def test_builds_correct_format(self) -> None:
        """Test scope command has correct wire format."""
        result = build_scope_command(SCOPE_NORTH)

        # Format: [len_lo, len_hi] + ! + 0x23 + 0x5a + direction
        assert result[0] == 4  # body length
        assert result[1] == 0
        assert result[2] == COMMAND_PREFIX
        assert result[3] == TYPE_UI
        assert result[4] == CMD_SCOPE
        assert result[5] == SCOPE_NORTH

    def test_total_length_is_6(self) -> None:
        """Test scope command is 6 bytes total."""
        result = build_scope_command(0x00)
        assert len(result) == 6


class TestBuildToggleEquipmentCommand:
    """Tests for build_toggle_equipment_command."""

    def test_builds_correct_format_slot_1(self) -> None:
        """Test toggle equipment command for slot 1 (armor)."""
        result = build_toggle_equipment_command(1)

        assert result[2] == COMMAND_PREFIX
        assert result[3] == TYPE_UI
        assert result[4] == CMD_TOGGLE_EQUIPMENT
        assert result[5] == 0x31  # '1' ASCII

    def test_builds_correct_format_slot_5(self) -> None:
        """Test toggle equipment command for slot 5 (radar)."""
        result = build_toggle_equipment_command(5)

        assert result[5] == 0x35  # '5' ASCII

    def test_total_length_is_6(self) -> None:
        """Test toggle equipment command is 6 bytes total."""
        result = build_toggle_equipment_command(1)
        assert len(result) == 6

    def test_slot_converted_to_ascii_digit(self) -> None:
        """Test slot number is converted to ASCII digit character."""
        for slot in range(1, 6):
            result = build_toggle_equipment_command(slot)
            assert result[5] == 0x30 + slot  # '0' + slot
