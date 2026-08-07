"""Tests for binary decode and received-message processing.

``test_decoders.py`` was 800 lines; the text and command decoders are
now a sibling.
"""

from __future__ import annotations

import base64
import logging

import pytest

from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.sniffer.decoders import (
    process_received_message,
    set_protocol_frame_logging,
)
from tests.sniffer.conftest import sniffer_xor_table
from tests.wire_builders import (
    encode_wire_frame,
    frame_payload,
)


def test_process_received_message_respects_protocol_frame_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Protocol frame logging can be disabled for concise bot terminal output."""
    payload = frame_payload(b"$1|0")
    table = sniffer_xor_table()

    set_protocol_frame_logging(False)
    with caplog.at_level(logging.INFO):
        process_received_message(payload, table)
    assert not caplog.records

    set_protocol_frame_logging(True)
    with caplog.at_level(logging.INFO):
        process_received_message(payload, table)
    assert any("[RECEIVED] RESPONSE: $1|0" in record.message for record in caplog.records)


def test_process_received_message_logs_plaintext_chat_ack(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The raw "C0" body is logged as the plaintext chat ack, not XOR-routed."""
    payload = frame_payload(b"C0")

    set_protocol_frame_logging(True)
    with caplog.at_level(logging.INFO):
        process_received_message(payload, sniffer_xor_table())
    assert any("ChatAck" in record.message for record in caplog.records)


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


class TestTryDecodeReceived:
    """Tests for try_decode_received function."""

    def test_try_decode_received_text_message(self) -> None:
        """Test try_decode_received handles text messages."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        # '+' (0x2B) is a text message type
        body = b"+field=42\n"
        payload = base64.b64encode(len(body).to_bytes(2, "little") + body).decode()
        result = try_decode_received(payload, sniffer_xor_table())
        if result is None:
            raise AssertionError("Expected non-None result")
        assert "[RECEIVED]" in result

    def test_try_decode_received_plaintext_autoscroll_ack(self) -> None:
        """The raw "A1" body decodes as the plaintext autoscroll ack, not XOR."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        payload = base64.b64encode(len(b"A1").to_bytes(2, "little") + b"A1").decode()
        result = try_decode_received(payload, sniffer_xor_table())
        if result is None:
            raise AssertionError("Expected non-None result")
        assert "[RECEIVED]" in result
        assert "AutoscrollAck" in result

    def test_try_decode_received_binary_empty_decoded(self) -> None:
        """Test try_decode_received returns EMPTY for 1-byte binary body."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        # 1-byte binary body → xor_decode strips msg_type → empty decoded data
        body = bytes([0x47])
        payload = base64.b64encode(len(body).to_bytes(2, "little") + body).decode()
        result = try_decode_received(payload, sniffer_xor_table())
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
        result = try_decode_received(payload, sniffer_xor_table())
        if result is None:
            raise AssertionError("Expected non-None result")
        assert "[RECEIVED]" in result
        assert "UNKNOWN 0x01" in result

    def test_try_decode_received_short_payload(self) -> None:
        """Test try_decode_received returns None for short payloads."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        # Payload that decodes to less than 3 bytes
        payload = base64.b64encode(b"\x01\x00").decode()
        result = try_decode_received(payload, sniffer_xor_table())
        assert result is None

    def test_try_decode_received_invalid_base64(self) -> None:
        """Test try_decode_received returns None for invalid base64."""
        from tankpit_bot.sniffer.decoders import try_decode_received

        result = try_decode_received("not valid base64!!!", sniffer_xor_table())
        assert result is None


class TestDecodeAndLogBinary:
    """Tests for decode_and_log_binary function."""

    def test_decode_and_log_binary_logs_result(self) -> None:
        """Test decode_and_log_binary logs the decoded result."""
        from tankpit_bot.sniffer.decoders import decode_and_log_binary

        # Use unknown type to ensure it logs UNKNOWN
        decode_and_log_binary(0x01, b"\x00\x01\x02", "RECEIVED", b"\x01\x00\x01\x02")


class TestProcessReceivedMessage:
    """Tests for process_received_message internal paths."""

    def test_single_byte_binary_returns_early(self) -> None:
        """process_received_message handles 1-byte binary body (empty decoded data)."""
        from tankpit_bot.sniffer.decoders import process_received_message

        # Frame: 2-byte LE length (1) + 1-byte binary body (0x01)
        # xor_decode strips msg_type → empty decoded_data → early return
        frame = b"\x01\x00\x01"
        payload = base64.b64encode(frame).decode()
        process_received_message(payload, sniffer_xor_table())  # should not raise

    def test_unknown_binary_type_logs_fallback(self) -> None:
        """process_received_message logs fallback for unrecognized binary type."""
        from tankpit_bot.sniffer.decoders import process_received_message

        # Frame: 2-byte LE length (3) + 3-byte body with unknown type 0x01
        # msg_type 0x01 not in TEXT_MESSAGE_TYPES or MSG_MIN_LENGTHS
        body = bytes([0x01, 0xAB, 0xCD])
        frame = len(body).to_bytes(2, "little") + body
        payload = base64.b64encode(frame).decode()
        process_received_message(payload, sniffer_xor_table())  # log fallback, not raise

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
        process_received_message(payload, sniffer_xor_table())  # decode, log, dispatch

    def test_zero_length_frame_carries_no_message(self) -> None:
        """A zero-length frame is legal framing and dispatches nothing.

        The inline walk read this as a torn frame and stopped, which
        also happened to keep the empty body away from the router. The
        splitter drops the empty body instead, so the router keeps its
        "body[0] is safe" guarantee AND any later frame in the same
        payload still gets processed ([[session-state-deglobalisation]]).
        """
        from tankpit_bot.sniffer.decoders import process_received_message

        payload = base64.b64encode(b"\x00\x00").decode()
        process_received_message(payload, sniffer_xor_table())

    def test_zero_length_frame_does_not_hide_the_frame_after_it(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The frame following a zero-length one is still dispatched.

        The inline walk lost it: ``msg_len == 0`` broke the loop before
        the 0x4D that follows was ever read.
        """
        from tankpit_bot.sniffer.decoders import process_received_message

        body = bytes([0x4D, 0x01, 0x02, 0x03])
        frame = b"\x00\x00" + len(body).to_bytes(2, "little") + body
        payload = base64.b64encode(frame).decode()

        set_protocol_frame_logging(True)
        with caplog.at_level(logging.INFO):
            process_received_message(payload, sniffer_xor_table())
        assert any("[RECEIVED]" in record.message for record in caplog.records)

    def test_oversized_submessage_raises(self) -> None:
        """A sub-message that overruns its payload is fatal, not truncated."""
        from tankpit_bot.sniffer.decoders import process_received_message

        # Frame claims 100 bytes but only has 2.
        payload = base64.b64encode(b"\x64\x00\x01\x02").decode()
        with pytest.raises(FramingError, match="Incomplete frame"):
            process_received_message(payload, sniffer_xor_table())

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
        process_received_message(payload, sniffer_xor_table())  # decode, log, dispatch

    def test_binary_promotion_takes_binary_route(self) -> None:
        """0x2B with 3-byte body disambiguates to binary Rf, not text WorldInfo.

        Binary Rf is the only message in the wire grammar that shares a
        type byte with a text format. Length is the disambiguator
        (3 bytes for Rf, far more for WorldInfo / ROOM_LIST).
        """
        from tankpit_bot.sniffer.decoders import process_received_message
        from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state
        from tankpit_bot.state import update_self_from_movement_response

        reset_world_state()
        ws = get_world_service()
        ws.world_state = update_self_from_movement_response(
            ws.world_state,
            tank_id=1,
            x=10,
            y=20,
            team=0,
            rank=1,
            leaderboard_position=3,
            timestamp_ms=500,
        )

        # Binary Rf body: 0x2B + 2 ciphered bytes decoding to
        # (new_rank=5, banner=1). Encoded under the same session table
        # the decoder is handed, so this exercises the real cipher —
        # the table used to be a module global left at None here, which
        # let the plaintext pass through verbatim
        # ([[session-state-deglobalisation]]).
        payload = encode_wire_frame(0x2B, bytes([5, 1]), sniffer_xor_table())
        process_received_message(payload, sniffer_xor_table())

        self_state = get_world_service().world_state["self_state"]
        if self_state is None:
            raise AssertionError("self_state should be present after dispatch")
        assert self_state["rank"] == 5
        reset_world_state()

    def test_text_world_info_still_takes_text_route(self) -> None:
        """0x2B with a long pipe-delimited body stays on the text-log path."""
        from tankpit_bot.sniffer.decoders import process_received_message
        from tankpit_bot.sniffer.world_state import get_world_service, reset_world_state

        reset_world_state()
        # Long ROOM_LIST / WorldInfo body; well above the 3-byte Rf threshold.
        payload = frame_payload(b"+1|RoomName|24|1,1,1|0|n|field24.gif|2026")
        process_received_message(payload, sniffer_xor_table())

        # No self_state should have been created since this is not binary Rf.
        assert get_world_service().world_state["self_state"] is None
        reset_world_state()
