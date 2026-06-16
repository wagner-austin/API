"""Tests for tankpit_bot.sniffer decoder functions."""

from __future__ import annotations

import base64
import logging

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.sniffer.decoders import (
    decode_8byte_state,
    decode_command,
    decode_join_confirm,
    decode_message,
    decode_plus_message,
    decode_state_message,
    decode_text_message,
    process_received_message,
    set_protocol_frame_logging,
)
from tests.conftest import FakeFileSystem
from tests.sniffer.conftest import make_payload

# =============================================================================
# Basic Decode Message Tests
# =============================================================================


def test_decode_message_invalid_base64() -> None:
    """Test decode_message handles invalid base64."""
    result = decode_message("not valid base64!!!", "sent")
    assert result == "[SENT] (invalid base64)"


def test_decode_message_too_short() -> None:
    """Test decode_message handles messages shorter than 2 bytes."""
    payload = base64.b64encode(b"x").decode()
    result = decode_message(payload, "received")
    assert "[RECEIVED] (too short:" in result


def test_decode_message_auth() -> None:
    """Test decode_message decodes AUTH messages."""
    payload = make_payload(b"%AUTH !be 12345|token|auth extra")
    result = decode_message(payload, "sent")
    assert result == "[SENT] AUTH: %AUTH !be 12345|token|auth extra..."


def test_decode_message_select() -> None:
    """Test decode_message decodes SELECT messages."""
    payload = make_payload(b"*4")
    result = decode_message(payload, "sent")
    assert result == "[SENT] SELECT: room=4"


def test_decode_message_select_does_not_mutate_selected_room() -> None:
    """Sent SELECT packets do not mark the room as joined."""
    from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state

    reset_world_state()

    payload = make_payload(b"*4")
    result = decode_message(payload, "sent")

    assert result == "[SENT] SELECT: room=4"
    assert get_world_service().selected_room is None


def test_decode_message_response() -> None:
    """Test decode_message decodes RESPONSE messages."""
    payload = make_payload(b"$4|0")
    result = decode_message(payload, "received")
    assert result == "[RECEIVED] RESPONSE: $4|0"


def test_decode_message_state() -> None:
    """Test decode_message decodes STATE messages (binary with '.' prefix)."""
    # Create a 14-byte state message (subtype 0x03, not fuel-related)
    state_body = bytes.fromhex("2e033c020300005c190000ca0300")
    payload = make_payload(state_body)
    result = decode_message(payload, "received")
    # 14-byte STATE message with subtype shown
    assert "[RECEIVED] STATE: sub=0x03 len=14" in result
    assert "hex=" in result


def test_decode_message_state_short() -> None:
    """Test decode_message decodes short position messages."""
    # Short state message (4-11 bytes) - shows as POS
    short_state = bytes([0x2E, 0x01, 0x02, 0x03])  # 4 bytes
    payload = make_payload(short_state)
    result = decode_message(payload, "received")
    assert "[RECEIVED] POS: len=4 hex=2e010203" in result


def test_decode_text_message_state_without_body() -> None:
    """Test decode_text_message fallback when body is None."""
    # When body is None, should fall back to simple len display
    result = decode_text_message(".state data", 11, "RECEIVED", body=None)
    assert result == "[RECEIVED] STATE: len=11 bytes"


def test_decode_state_message_extracts_fields() -> None:
    """Test decode_state_message handles medium-length UPDATE messages."""
    # 20-byte message is ENTITY type
    body = bytes.fromhex("2e10200003000000190000e80300000000000000")  # 20 bytes
    result = decode_state_message(body, "RECV")
    assert "[RECV] ENTITY: sub=0x10 len=20" in result


def test_decode_state_message_sync() -> None:
    """Test decode_state_message handles SYNC messages (2-3 bytes)."""
    body = bytes.fromhex("2e62")  # 2 bytes
    result = decode_state_message(body, "RECV")
    assert "[RECV] SYNC: 2e62" in result


def test_decode_state_message_map_data() -> None:
    """Test decode_state_message handles MAP_DATA (>500 bytes)."""
    body = bytes([0x2E]) + bytes(600)  # 601 bytes total
    result = decode_state_message(body, "RECV")
    assert "[RECV] MAP_DATA: len=601" in result


def test_decode_state_message_hit() -> None:
    """Test decode_state_message handles HIT messages (12 bytes)."""
    body = bytes.fromhex("2e650b110f8b7bc412fd676f")  # 12 bytes
    result = decode_state_message(body, "RECV")
    assert "[RECV] HIT: 2e650b110f8b7bc412fd676f" in result


def test_decode_state_message_entity() -> None:
    """Test decode_state_message handles ENTITY messages (17-30 bytes)."""
    body = bytes([0x2E]) + bytes(19)  # 20 bytes total, subtype is 0x00
    result = decode_state_message(body, "RECV")
    assert "[RECV] ENTITY: sub=0x00 len=20" in result


def test_decode_state_message_update() -> None:
    """Test decode_state_message handles UPDATE messages (31-500 bytes)."""
    body = bytes([0x2E]) + bytes(49)  # 50 bytes total
    result = decode_state_message(body, "RECV")
    assert "[RECV] UPDATE: len=50" in result


def test_decode_message_unknown() -> None:
    """Test decode_message handles unknown message types."""
    payload = make_payload(b"some unknown message format")
    result = decode_message(payload, "received")
    assert "[RECEIVED] ???:" in result


def test_decode_message_quit() -> None:
    """Test decode_message decodes QUIT messages (dash character)."""
    payload = make_payload(b"-")
    result = decode_message(payload, "sent")
    assert result == "[SENT] QUIT: -"


def test_process_received_message_respects_protocol_frame_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Protocol frame logging can be disabled for concise bot terminal output."""
    payload = make_payload(b"$1|0")

    set_protocol_frame_logging(False)
    with caplog.at_level(logging.INFO):
        process_received_message(payload)
    assert not caplog.records

    set_protocol_frame_logging(True)
    with caplog.at_level(logging.INFO):
        process_received_message(payload)
    assert any("[RECEIVED] RESPONSE: $1|0" in record.message for record in caplog.records)


# =============================================================================
# Plus Message Tests
# =============================================================================


def test_decode_plus_message_room_list() -> None:
    """Test decode_plus_message decodes ROOM_LIST messages."""
    result = decode_plus_message(
        "+4|World (Meltdown)|42|1,1,1,0,1,0,0|3|n|field42.gif|2026",
        "RECV",
    )
    assert result == "[RECV] ROOM_LIST: room=4 name=World (Meltdown)"


def test_decode_plus_message_action() -> None:
    """Test decode_plus_message decodes ACTION messages (non-room-list format)."""
    result = decode_plus_message("+1|2|116|79|extra", "SENT")
    assert result == "[SENT] ACTION: room=1 coords=116,79"


def test_decode_plus_message_action_short() -> None:
    """Test decode_plus_message handles short ACTION messages."""
    result = decode_plus_message("+4|2", "SENT")
    assert result == "[SENT] ACTION: room=4 coords=?"


def test_decode_plus_message_action_does_not_register_room_image() -> None:
    """ACTION messages do not mutate room-image registration."""
    from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state

    reset_world_state()

    result = decode_plus_message("+1|2|118|101|manual-click", "SENT")

    assert result == "[SENT] ACTION: room=1 coords=118,101"
    assert "1" not in get_world_service().room_images


# =============================================================================
# Join Confirm Tests
# =============================================================================


def test_decode_join_confirm() -> None:
    """Test decode_join_confirm decodes JOIN_CONFIRM messages."""
    result = decode_join_confirm("=4|Sep. 25, 2012|Yuppler|4|9|10", "RECV")
    assert result == "[RECV] JOIN_CONFIRM: room=4 tank=Yuppler lieutenant"


def test_decode_join_confirm_short() -> None:
    """Test decode_join_confirm handles short messages."""
    result = decode_join_confirm("=4|date", "RECV")
    assert result == "[RECV] JOIN_CONFIRM: room=4 tank=? rank-1"


def test_decode_join_confirm_empty_room_id_does_not_select_room() -> None:
    """Empty room IDs do not mutate selected-room state."""
    from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state

    reset_world_state()

    result = decode_join_confirm("=|date", "RECV")

    assert result == "[RECV] JOIN_CONFIRM: room= tank=? rank-1"
    assert get_world_service().selected_room is None


# =============================================================================
# Command Tests
# =============================================================================


def test_decode_command_with_rest() -> None:
    """Test decode_command decodes commands with additional data (no magic)."""
    # Use actual binary bytes, not ASCII string
    body = bytes([0x21, 0x31, 0x2D, 0x43, 0xFE])  # !1-C<0xFE>
    result = decode_command(body, "SENT")
    assert result == "[SENT] CMD: ! 21312d43fe"


def test_decode_command_with_magic_xor_decryption() -> None:
    """Test decode_command decodes commands with XOR decryption when magic provided."""
    from pathlib import Path

    # Check if static key exists (required for XOR decryption)
    static_key_path = Path(__file__).parent.parent.parent / "xor_static_key.txt"
    if not static_key_path.exists():
        pytest.skip("xor_static_key.txt not found")

    # Read the static key
    static_key = static_key_path.read_text().strip()
    magic = "test_magic_key_20char"  # 20 char magic key

    # Build the XOR table manually to know what encoded bytes to send
    table = bytearray(len(static_key))
    for i in range(len(static_key)):
        table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])

    # We want to decode type=2, id=63 (enter game command)
    # Encoded bytes: type_encoded = 2 ^ table[0], id_encoded = 63 ^ table[1]
    type_encoded = 2 ^ table[0]
    id_encoded = 63 ^ table[1]
    body = bytes([0x21, type_encoded, id_encoded])

    result = decode_command(body, "SENT", magic)
    assert result == "[SENT] CMD: ! type=2 id=63"


def test_decode_command_short() -> None:
    """Test decode_command handles short command messages."""
    result = decode_command(b"!", "SENT")
    assert result == "[SENT] CMD: ! (too short: 21)"


def test_decode_command_non_ascii() -> None:
    """Test decode_command handles non-ASCII command bytes (no magic)."""
    body = bytes([0x21, 0x90, 0xAB, 0xCD])  # !<0x90><0xAB><0xCD>
    result = decode_command(body, "SENT")
    assert result == "[SENT] CMD: ! 2190abcd"


# =============================================================================
# Decode Message Routing Tests
# =============================================================================


def test_decode_message_calls_decode_plus_for_room_list() -> None:
    """Test decode_message routes to decode_plus_message for ROOM_LIST."""
    payload = make_payload(b"+3|Practice|1|0,0,0,0,0,0,0|1|p|field01.gif|2025")
    result = decode_message(payload, "received")
    assert result == "[RECEIVED] ROOM_LIST: room=3 name=Practice"


def test_decode_message_calls_decode_join_confirm() -> None:
    """Test decode_message routes to decode_join_confirm."""
    payload = make_payload(b"=4|Sep. 25, 2012|Yuppler|4|9|10|10|9")
    result = decode_message(payload, "received")
    assert result == "[RECEIVED] JOIN_CONFIRM: room=4 tank=Yuppler lieutenant"


def test_decode_message_calls_decode_command() -> None:
    """Test decode_message routes to decode_command."""
    payload = make_payload(b"!7b")
    result = decode_message(payload, "sent")
    # Without magic key, just shows raw hex
    assert result == "[SENT] CMD: ! 213762"


def test_decode_message_command_with_magic_but_no_static_key() -> None:
    """Test decode_message falls back to hex when static key file doesn't exist.

    This covers the branch where magic is provided but static key file is missing.
    """
    # Create a fake filesystem that has NO static key file
    fs = FakeFileSystem()
    _test_hooks.read_text = fs.read_text
    _test_hooks.path_exists = fs.path_exists

    payload = make_payload(b"!7b")
    # Provide magic key, but static key file doesn't exist
    result = decode_message(payload, "sent", magic="test_magic")

    # Should fall back to hex output since static key file is missing
    assert result == "[SENT] CMD: ! 213762"


# =============================================================================
# 8-Byte State Tests
# =============================================================================


class TestDecode8ByteState:
    """Tests for decode_8byte_state function."""

    def test_item_pickup_subtype(self) -> None:
        """Test 0x49 subtype returns ITEM_PICKUP."""
        body = bytes([0x2E, 0x49, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        result = decode_8byte_state(body, "RECV")
        assert "[RECV] ITEM_PICKUP:" in result

    def test_game_state_subtype(self) -> None:
        """Test 0x67 subtype returns GAME_STATE."""
        body = bytes([0x2E, 0x67, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        result = decode_8byte_state(body, "RECV")
        assert "[RECV] GAME_STATE:" in result

    def test_unknown_subtype(self) -> None:
        """Test unknown subtype returns MSG_8B."""
        body = bytes([0x2E, 0x99, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        result = decode_8byte_state(body, "RECV")
        assert "[RECV] MSG_8B: sub=0x99" in result


class TestDecodeStateMessageFuelRaw:
    """Tests for decode_state_message FUEL_RAW branch."""

    def test_fuel_raw_17_bytes_subtype_0x10(self) -> None:
        """Test 17-byte message with subtype 0x10 returns FUEL_RAW."""
        # 17 bytes with subtype 0x10
        body = bytes([0x2E, 0x10]) + bytes(13) + bytes([0xE8, 0x03])
        result = decode_state_message(body, "RECV")
        assert "[RECV] FUEL_RAW:" in result
        assert "p15=1000" in result


# =============================================================================
# Try Decode Binary Tests
# =============================================================================


class TestTryDecodeBinary:
    """Tests for try_decode_binary function."""

    def test_try_decode_binary_unknown_type(self) -> None:
        """Test try_decode_binary returns UNKNOWN for unknown message types."""
        from tankpit_bot.sniffer.decoders import try_decode_binary

        # Use message type 0x01 which is not in MSG_MIN_LENGTHS
        result = try_decode_binary(0x01, b"\x00\x01\x02\x03", b"\x01\x00\x01\x02\x03")
        assert "UNKNOWN 0x01" in result
        assert "len=5" in result

    def test_try_decode_binary_short_data(self) -> None:
        """Test try_decode_binary returns SHORT when data is too short."""
        from tankpit_bot.sniffer.decoders import try_decode_binary

        # ShootEvent (ord('S')=0x53) needs 12 bytes minimum
        result = try_decode_binary(ord("S"), b"\x00\x01\x02", b"S\x00\x01\x02")
        assert "SHORT 0x53 'S'" in result
        assert "need=12" in result
        assert "got=3" in result

    def test_try_decode_binary_valid_shoot_event(self) -> None:
        """Test try_decode_binary decodes valid ShootEvent."""
        from tankpit_bot.sniffer.decoders import try_decode_binary

        # ShootEvent needs 12 bytes: [0]=dir, [1-2]=tank_id, [3-4]=aid,
        # [5]=type, [6]=damage, [7-8]=x, [9-10]=y, [11]=flags
        data = bytes([0x01, 0x0A, 0x00, 0x0B, 0x00, 0x02, 0x10, 0x40, 0x00, 0x50, 0x00, 0x00])
        result = try_decode_binary(ord("S"), data, b"S" + data)
        assert "shoot" in result.lower() or "SHOOT" in result

    def test_try_decode_binary_non_printable_char(self) -> None:
        """Test try_decode_binary shows '?' for non-printable message types."""
        from tankpit_bot.sniffer.decoders import try_decode_binary

        # Use message type 0x00 which is not printable
        result = try_decode_binary(0x00, b"\x00\x01", b"\x00\x00\x01")
        assert "UNKNOWN 0x00 '?'" in result

    def test_try_decode_binary_long_data_preview(self) -> None:
        """Test try_decode_binary truncates long data preview with ellipsis."""
        from tankpit_bot.sniffer.decoders import try_decode_binary

        # Unknown type with more than 20 bytes of data
        long_data = bytes(range(30))
        result = try_decode_binary(0x02, long_data, b"\x02" + long_data)
        assert "..." in result


# =============================================================================
# Try Decode Received Tests
# =============================================================================


class TestTryDecodeReceived:
    """Tests for try_decode_received function."""

    def test_try_decode_received_text_message(self) -> None:
        """Test try_decode_received handles text messages."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        # '+' (0x2B) is a text message type
        body = b"+field=42\n"
        payload = base64.b64encode(len(body).to_bytes(2, "little") + body).decode()
        result = try_decode_received(payload)
        if result is None:
            raise AssertionError("Expected non-None result")
        assert "[RECEIVED]" in result

    def test_try_decode_received_binary_empty_decoded(self) -> None:
        """Test try_decode_received returns EMPTY for 1-byte binary body."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        # 1-byte binary body → xor_decode strips msg_type → empty decoded data
        body = bytes([0x47])
        payload = base64.b64encode(len(body).to_bytes(2, "little") + body).decode()
        result = try_decode_received(payload)
        if result is None:
            raise AssertionError("Expected non-None result")
        assert "EMPTY" in result
        assert "0x47" in result

    def test_try_decode_received_binary_multi_byte(self) -> None:
        """Test try_decode_received decodes multi-byte binary message."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        # 2-byte body: msg_type 0x01 (unknown) + data byte
        # xor_decode produces 1-byte decoded data, type not in MSG_MIN_LENGTHS
        body = bytes([0x01, 0xAB])
        payload = base64.b64encode(len(body).to_bytes(2, "little") + body).decode()
        result = try_decode_received(payload)
        if result is None:
            raise AssertionError("Expected non-None result")
        assert "[RECEIVED]" in result
        assert "UNKNOWN 0x01" in result

    def test_try_decode_received_short_payload(self) -> None:
        """Test try_decode_received returns None for short payloads."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        # Payload that decodes to less than 3 bytes
        payload = base64.b64encode(b"\x01\x00").decode()
        result = try_decode_received(payload)
        assert result is None

    def test_try_decode_received_invalid_base64(self) -> None:
        """Test try_decode_received returns None for invalid base64."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        result = try_decode_received("not valid base64!!!")
        assert result is None


# =============================================================================
# Decode And Log Binary Tests
# =============================================================================


class TestDecodeAndLogBinary:
    """Tests for decode_and_log_binary function."""

    def test_decode_and_log_binary_logs_result(self) -> None:
        """Test decode_and_log_binary logs the decoded result."""
        from tankpit_bot.sniffer.decoders import decode_and_log_binary

        # Use unknown type to ensure it logs UNKNOWN
        decode_and_log_binary(0x01, b"\x00\x01\x02", "RECEIVED", b"\x01\x00\x01\x02")
        # Should not raise - it just logs


# =============================================================================
# Command Movement Branch Tests
# =============================================================================


class TestDecodeCommandMovementBranches:
    """Tests for decode_command movement and shoot branches."""

    def test_decode_command_move(self, fake_fs: FakeFileSystem) -> None:
        """Test decode_command decodes MOVE command."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        magic = "testmagic123"
        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        # Build XOR table the same way decode_command does
        magic_bytes = magic.encode("utf-8")
        xor_table = bytes(
            ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
        )

        # Plaintext command: [type=4, id=112, x=100, y=50]
        plaintext = bytes([0x04, 112, 100, 50])  # After '!'
        encrypted = bytes([0x21]) + bytes(
            plaintext[i] ^ xor_table[i] for i in range(len(plaintext))
        )

        result = decode_command(encrypted, "CMD", magic)
        assert "MOVE" in result
        assert "100" in result
        assert "50" in result

    def test_decode_command_shoot(self, fake_fs: FakeFileSystem) -> None:
        """Test decode_command decodes SHOOT command."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        magic = "testmagic123"
        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        magic_bytes = magic.encode("utf-8")
        xor_table = bytes(
            ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
        )

        # Plaintext command: [type=6, id=any, x=80, y=120]
        plaintext = bytes([0x06, 0x01, 80, 120])
        encrypted = bytes([0x21]) + bytes(
            plaintext[i] ^ xor_table[i] for i in range(len(plaintext))
        )

        result = decode_command(encrypted, "CMD", magic)
        assert "SHOOT" in result
        assert "80" in result
        assert "120" in result

    def test_decode_command_pickup(self, fake_fs: FakeFileSystem) -> None:
        """Test decode_command decodes PICKUP command."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        magic = "testmagic123"
        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        magic_bytes = magic.encode("utf-8")
        xor_table = bytes(
            ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
        )

        # Plaintext command: [type=4, id=106, x=60, y=70]
        plaintext = bytes([0x04, 106, 60, 70])
        encrypted = bytes([0x21]) + bytes(
            plaintext[i] ^ xor_table[i] for i in range(len(plaintext))
        )

        result = decode_command(encrypted, "CMD", magic)
        assert "PICKUP" in result
        assert "60" in result
        assert "70" in result

    def test_decode_command_teleport(self, fake_fs: FakeFileSystem) -> None:
        """Test decode_command decodes TELEPORT command."""
        from tankpit_bot.protocol.codec import DEFAULT_STATIC_KEY_PATH

        magic = "testmagic123"
        static_key = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "A" * 974
        fake_fs.write_text(DEFAULT_STATIC_KEY_PATH, static_key)

        magic_bytes = magic.encode("utf-8")
        xor_table = bytes(
            ord(static_key[i]) ^ magic_bytes[i % len(magic_bytes)] for i in range(len(static_key))
        )

        # Plaintext command: [type=4, id=116, x=200, y=180]
        plaintext = bytes([0x04, 116, 200, 180])
        encrypted = bytes([0x21]) + bytes(
            plaintext[i] ^ xor_table[i] for i in range(len(plaintext))
        )

        result = decode_command(encrypted, "CMD", magic)
        assert "TELEPORT" in result
        assert "200" in result
        assert "180" in result


# =============================================================================
# Process Received Message Tests
# =============================================================================


class TestProcessReceivedMessage:
    """Tests for process_received_message internal paths."""

    def test_single_byte_binary_returns_early(self) -> None:
        """process_received_message handles 1-byte binary body (empty decoded data)."""
        from tankpit_bot.sniffer.decoders import process_received_message

        # Frame: 2-byte LE length (1) + 1-byte binary body (0x01)
        # xor_decode strips msg_type → empty decoded_data → early return
        frame = b"\x01\x00\x01"
        payload = base64.b64encode(frame).decode()
        process_received_message(payload)  # should not raise

    def test_unknown_binary_type_logs_fallback(self) -> None:
        """process_received_message logs fallback for unrecognized binary type."""
        from tankpit_bot.sniffer.decoders import process_received_message

        # Frame: 2-byte LE length (3) + 3-byte body with unknown type 0x01
        # msg_type 0x01 not in TEXT_MESSAGE_TYPES or MSG_MIN_LENGTHS
        body = bytes([0x01, 0xAB, 0xCD])
        frame = len(body).to_bytes(2, "little") + body
        payload = base64.b64encode(frame).decode()
        process_received_message(payload)  # should log fallback, not raise

    def test_decodable_binary_dispatches(self) -> None:
        """process_received_message decodes and dispatches a valid binary message.

        Uses a 0x3F (Sync) message which has min_len=0 and is fully decodable.
        """
        from tankpit_bot.sniffer.decoders import process_received_message

        # 0x3F body with 2 bytes → decoded_data = 1 byte via xor_decode
        # MSG_MIN_LENGTHS[0x3F] = 0, so condition passes
        # try_decode_binary_message returns SyncDict
        body = bytes([0x3F, 0x00])
        frame = len(body).to_bytes(2, "little") + body
        payload = base64.b64encode(frame).decode()
        process_received_message(payload)  # should decode, log, and dispatch

    def test_malformed_frame_length_breaks_early(self) -> None:
        """process_received_message breaks on zero-length or oversized sub-message."""
        from tankpit_bot.sniffer.decoders import process_received_message

        # Frame with msg_len=0 → triggers break at line 127
        frame = b"\x00\x00"
        payload = base64.b64encode(frame).decode()
        process_received_message(payload)  # should not raise

    def test_oversized_submessage_breaks_early(self) -> None:
        """process_received_message breaks when sub-message extends beyond frame."""
        from tankpit_bot.sniffer.decoders import process_received_message

        # Frame claims 100 bytes but only has 2 → offset + msg_len > len(data)
        frame = b"\x64\x00\x01\x02"
        payload = base64.b64encode(frame).decode()
        process_received_message(payload)  # should not raise

    def test_chat_message_decodes_and_dispatches(self) -> None:
        """process_received_message decodes 0x4D ChatMessage through full path.

        All types in MSG_MIN_LENGTHS have decoders, so decode_message
        succeeds and the message is dispatched to world state.
        """
        from tankpit_bot.sniffer.decoders import process_received_message

        # 0x4D ('M') has min_len=3, needs 4+ byte body for 3+ decoded bytes
        body = bytes([0x4D, 0x01, 0x02, 0x03])
        frame = len(body).to_bytes(2, "little") + body
        payload = base64.b64encode(frame).decode()
        process_received_message(payload)  # should decode, log, and dispatch
