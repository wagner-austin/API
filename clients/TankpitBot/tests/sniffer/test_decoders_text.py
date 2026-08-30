"""Tests for the lobby text and command decoders.

Every ``decode_*_message`` branch, including the XOR-decrypted command
path and the movement sub-branches.
"""

from __future__ import annotations

import base64

import pytest

from tankpit_bot import _test_hooks
from tankpit_bot.capture.xor import (
    XorStaticKeyUnavailableError,
    build_session_xor_table,
    reset_static_key_cache,
)
from tankpit_bot.sniffer.decoders import (
    decode_8byte_state,
    decode_command,
    decode_join_confirm,
    decode_message,
    decode_plus_message,
    decode_state_message,
    decode_text_message,
)
from tankpit_bot.sniffer.world_service import WorldService
from tests.conftest import FakeFileSystem
from tests.wire_builders import (
    frame_payload,
)


def test_decode_message_invalid_base64() -> None:
    """Test decode_message handles invalid base64."""
    result = decode_message(WorldService(), "not valid base64!!!", "sent", None)
    assert result == "[SENT] (invalid base64)"


def test_decode_message_too_short() -> None:
    """Test decode_message handles messages shorter than 2 bytes."""
    payload = base64.b64encode(b"x").decode()
    result = decode_message(WorldService(), payload, "received", None)
    assert "[RECEIVED] (too short:" in result


def test_decode_message_auth() -> None:
    """Test decode_message decodes AUTH messages."""
    payload = frame_payload(b"%AUTH !be 12345|token|auth extra")
    result = decode_message(WorldService(), payload, "sent", None)
    assert result == "[SENT] AUTH: %AUTH !be 12345|token|auth extra..."


def test_decode_message_select() -> None:
    """Test decode_message decodes SELECT messages."""
    payload = frame_payload(b"*4")
    result = decode_message(WorldService(), payload, "sent", None)
    assert result == "[SENT] SELECT: room=4"


def test_decode_message_select_does_not_mutate_selected_room() -> None:
    """Sent SELECT packets do not mark the room as joined."""

    ws = WorldService()
    payload = frame_payload(b"*4")
    result = decode_message(ws, payload, "sent", None)

    assert result == "[SENT] SELECT: room=4"
    assert ws.selected_room is None


def test_decode_message_response() -> None:
    """Test decode_message decodes RESPONSE messages."""
    payload = frame_payload(b"$4|0")
    result = decode_message(WorldService(), payload, "received", None)
    assert result == "[RECEIVED] RESPONSE: $4|0"


def test_decode_message_state() -> None:
    """Test decode_message decodes STATE messages (binary with '.' prefix)."""
    # Create a 14-byte state message (subtype 0x03, not fuel-related)
    state_body = bytes.fromhex("2e033c020300005c190000ca0300")
    payload = frame_payload(state_body)
    result = decode_message(WorldService(), payload, "received", None)
    # 14-byte STATE message with subtype shown
    assert "[RECEIVED] STATE: sub=0x03 len=14" in result
    assert "hex=" in result


def test_decode_message_state_short() -> None:
    """Test decode_message decodes short position messages."""
    # Short state message (4-11 bytes) - shows as POS
    short_state = bytes([0x2E, 0x01, 0x02, 0x03])  # 4 bytes
    payload = frame_payload(short_state)
    result = decode_message(WorldService(), payload, "received", None)
    assert "[RECEIVED] POS: len=4 hex=2e010203" in result


def test_decode_text_message_state_without_body() -> None:
    """Test decode_text_message fallback when body is None."""
    # When body is None, should fall back to simple len display
    result = decode_text_message(WorldService(), ".state data", 11, "RECEIVED", body=None)
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
    payload = frame_payload(b"some unknown message format")
    result = decode_message(WorldService(), payload, "received", None)
    assert "[RECEIVED] ???:" in result


def test_decode_message_quit() -> None:
    """Test decode_message decodes QUIT messages (dash character)."""
    payload = frame_payload(b"-")
    result = decode_message(WorldService(), payload, "sent", None)
    assert result == "[SENT] QUIT: -"


def test_decode_plus_message_room_list() -> None:
    """Test decode_plus_message decodes ROOM_LIST messages."""
    result = decode_plus_message(
        WorldService(),
        "+4|World (Meltdown)|42|1,1,1,0,1,0,0|3|n|field42.gif|2026",
        "RECV",
    )
    assert result == "[RECV] ROOM_LIST: room=4 name=World (Meltdown)"


def test_decode_plus_message_action() -> None:
    """Test decode_plus_message decodes ACTION messages (non-room-list format)."""
    result = decode_plus_message(WorldService(), "+1|2|116|79|extra", "SENT")
    assert result == "[SENT] ACTION: room=1 coords=116,79"


def test_decode_plus_message_action_short() -> None:
    """Test decode_plus_message handles short ACTION messages."""
    result = decode_plus_message(WorldService(), "+4|2", "SENT")
    assert result == "[SENT] ACTION: room=4 coords=?"


def test_decode_plus_message_action_does_not_register_room_image() -> None:
    """ACTION messages do not mutate room-image registration."""

    ws = WorldService()
    result = decode_plus_message(ws, "+1|2|118|101|manual-click", "SENT")

    assert result == "[SENT] ACTION: room=1 coords=118,101"
    assert "1" not in ws.room_images


def test_decode_join_confirm() -> None:
    """Test decode_join_confirm decodes JOIN_CONFIRM messages.

    Fields 5-8 ride the line verbatim: they are unidentified, and the
    log is the only place their values can be read against a known
    room population.
    """
    result = decode_join_confirm(WorldService(), "=4|Sep. 25, 2012|Yuppler|4|9|10", "RECV")
    assert result == "[RECV] JOIN_CONFIRM: room=4 tank=Yuppler lieutenant f5-8=9,10"


def test_decode_join_confirm_short() -> None:
    """Test decode_join_confirm handles short messages."""
    result = decode_join_confirm(WorldService(), "=4|date", "RECV")
    assert result == "[RECV] JOIN_CONFIRM: room=4 tank=? rank-1 f5-8=-"


def test_decode_join_confirm_empty_room_id_does_not_select_room() -> None:
    """Empty room IDs do not mutate selected-room state."""

    ws = WorldService()
    result = decode_join_confirm(ws, "=|date", "RECV")

    assert result == "[RECV] JOIN_CONFIRM: room= tank=? rank-1 f5-8=-"
    assert ws.selected_room is None


def test_decode_command_with_rest() -> None:
    """Test decode_command decodes commands with additional data (no magic)."""
    # Use actual binary bytes, not ASCII string
    body = bytes([0x21, 0x31, 0x2D, 0x43, 0xFE])  # !1-C<0xFE>
    result = decode_command(body, "SENT", None)
    assert result == "[SENT] CMD: ! 21312d43fe"


def test_decode_command_with_magic_xor_decryption() -> None:
    """decode_command decrypts a command against the session cipher.

    This used to read the key through its own copy of the path
    expression, skip itself when the file was absent, and build the
    table with its own copy of the math. The repo's key is present —
    every other cipher test depends on it — so the skip was a branch
    that could only ever hide a broken checkout
    ([[session-state-deglobalisation]]).
    """
    magic = "test_magic_key_20char"
    table = build_session_xor_table(magic)

    # Decode target: type=2, id=63 (enter game command).
    body = bytes([0x21, 2 ^ table[0], 63 ^ table[1]])

    result = decode_command(body, "SENT", magic)
    assert result == "[SENT] CMD: ! type=2 id=63"


def test_decode_command_short() -> None:
    """Test decode_command handles short command messages."""
    result = decode_command(b"!", "SENT", None)
    assert result == "[SENT] CMD: ! (too short: 21)"


def test_decode_command_non_ascii() -> None:
    """Test decode_command handles non-ASCII command bytes (no magic)."""
    body = bytes([0x21, 0x90, 0xAB, 0xCD])  # !<0x90><0xAB><0xCD>
    result = decode_command(body, "SENT", None)
    assert result == "[SENT] CMD: ! 2190abcd"


def test_decode_message_calls_decode_plus_for_room_list() -> None:
    """Test decode_message routes to decode_plus_message for ROOM_LIST."""
    payload = frame_payload(b"+3|Practice|1|0,0,0,0,0,0,0|1|p|field01.gif|2025")
    result = decode_message(WorldService(), payload, "received", None)
    assert result == "[RECEIVED] ROOM_LIST: room=3 name=Practice"


def test_decode_message_calls_decode_join_confirm() -> None:
    """Test decode_message routes to decode_join_confirm."""
    payload = frame_payload(b"=4|Sep. 25, 2012|Yuppler|4|9|10|10|9")
    result = decode_message(WorldService(), payload, "received", None)
    assert result == "[RECEIVED] JOIN_CONFIRM: room=4 tank=Yuppler lieutenant f5-8=9,10,10,9"


def test_decode_message_calls_decode_command() -> None:
    """Test decode_message routes to decode_command."""
    payload = frame_payload(b"!7b")
    result = decode_message(WorldService(), payload, "sent", None)
    # Without magic key, just shows raw hex
    assert result == "[SENT] CMD: ! 213762"


def test_decode_message_command_with_magic_but_no_static_key() -> None:
    """A session with magic but no static key is fatal, not a hex dump.

    This asserted a fallback: ``decode_command`` inlined its own copy of
    the key path, read, table build and XOR loop, and when the key file
    was absent it fell through to printing raw hex — a display that
    looks like a decode failure but reads like data
    ([[session-state-deglobalisation]]).
    """
    fs = FakeFileSystem()
    _test_hooks.read_text = fs.read_text
    _test_hooks.path_exists = fs.path_exists
    reset_static_key_cache()

    payload = frame_payload(b"!7b")
    with pytest.raises(XorStaticKeyUnavailableError, match="static XOR key unavailable"):
        decode_message(WorldService(), payload, "sent", magic="test_magic")


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
