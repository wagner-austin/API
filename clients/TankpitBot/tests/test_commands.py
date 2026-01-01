"""Tests for tankpit_bot.commands module."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from tankpit_bot.commands import (
    COMMAND_PREFIX,
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
