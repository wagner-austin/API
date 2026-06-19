"""Tests for capture viewport analysis."""

from __future__ import annotations

import base64

import pytest
from platform_core.json_utils import JSONTypeError

from tankpit_bot.capture import viewport_analysis as va
from tankpit_bot.capture.viewport_analysis import (
    ViewportAnalysisDict,
    ViewportShiftDict,
    analyze_capture_session,
    decode_viewport_analysis,
    encode_viewport_analysis,
    format_viewport_analysis,
)
from tankpit_bot.capture.viewport_analysis_types import (
    ViewportAnalysisStateDict,
)
from tankpit_bot.capture.xor import build_xor_table, xor_decode_body
from tankpit_bot.types.message import CapturedMessage
from tankpit_bot.types.session import CaptureSession


def _encode_received_frame(msg_type: int, decoded_data: bytes, xor_table: bytes) -> str:
    """Encode a single received message frame.

    Args:
        msg_type: Protocol message type byte.
        decoded_data: XOR-decoded message data without the type byte.
        xor_table: Session XOR table.

    Returns:
        Base64-encoded frame payload.
    """
    encoded_body = bytes([msg_type]) + xor_decode_body(decoded_data, xor_table)
    length = len(encoded_body)
    frame = bytes([length & 0xFF, length >> 8]) + encoded_body
    return base64.b64encode(frame).decode("ascii")


def _make_movement_response_payload(
    tank_id: int,
    x: int,
    y: int,
    xor_table: bytes,
) -> str:
    """Create a MovementResponse payload.

    Args:
        tank_id: Tank identifier.
        x: Absolute x position.
        y: Absolute y position.
        xor_table: Session XOR table.

    Returns:
        Base64-encoded received frame payload.
    """
    decoded_data = bytes(
        [
            1,
            tank_id & 0xFF,
            tank_id >> 8,
            x,
            y,
            0,
            0,
            1,
            0,
            0,
            5,
            0,  # carrying byte (a[11]) per JS Mg.h
        ]
    )
    return _encode_received_frame(0x3D, decoded_data, xor_table)


def _make_viewport_update_payload(
    viewport_left: int,
    viewport_top: int,
    xor_table: bytes,
) -> str:
    """Create a ViewportUpdate payload with explicit viewport origin.

    Args:
        viewport_left: Absolute viewport left coordinate.
        viewport_top: Absolute viewport top coordinate.
        xor_table: Session XOR table.

    Returns:
        Base64-encoded received frame payload.
    """
    decoded_data = bytes([viewport_left, viewport_top])
    return _encode_received_frame(0x5A, decoded_data, xor_table)


def _make_position_update_payload(
    tank_id: int,
    x: int,
    y: int,
    extra_x: int,
    extra_y: int,
    xor_table: bytes,
) -> str:
    """Create an absolute self position_update payload.

    Args:
        tank_id: Tank identifier.
        x: Absolute x position.
        y: Absolute y position.
        extra_x: First extra_data byte.
        extra_y: Second extra_data byte.
        xor_table: Session XOR table.

    Returns:
        Base64-encoded received frame payload.
    """
    decoded_data = bytes(
        [
            0x24,
            0x02,
            tank_id & 0xFF,
            tank_id >> 8,
            x,
            y,
            extra_x,
            extra_y,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    return _encode_received_frame(0x2E, decoded_data, xor_table)


def _make_sync_payload(xor_table: bytes) -> str:
    """Create a Sync payload.

    Args:
        xor_table: Session XOR table.

    Returns:
        Base64-encoded received frame payload.
    """
    return _encode_received_frame(0x3F, b"", xor_table)


def _make_unknown_payload(xor_table: bytes) -> str:
    """Create an unsupported binary payload.

    Args:
        xor_table: Session XOR table.

    Returns:
        Base64-encoded received frame payload.
    """
    return _encode_received_frame(0x2B, b"\x00\x01\x02", xor_table)


def _make_session(messages: list[CapturedMessage], magic: str) -> CaptureSession:
    """Create a typed capture session for tests.

    Args:
        messages: Captured messages.
        magic: Session magic string.

    Returns:
        CaptureSession containing the provided messages.
    """
    return CaptureSession(
        session_id="viewport-analysis-test",
        start_timestamp_ms=1000,
        end_timestamp_ms=2000,
        base_url="https://tankpit.com/play",
        messages=messages,
        magic=magic,
        game_log=[],
        tank_names={},
    )


class TestAnalyzeCaptureSession:
    """Tests for analyze_capture_session."""

    def test_matches_position_update_extra_bytes_against_inferred_viewport(self) -> None:
        """Confirms 0x5A viewport origin makes position_update bytes comparable."""
        magic = "analysis-magic"
        static_key = "A" * 64
        xor_table = build_xor_table(static_key, magic)

        messages = [
            CapturedMessage(
                timestamp_ms=1000,
                direction="received",
                payload=_make_movement_response_payload(638, 144, 137, xor_table),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1100,
                direction="received",
                payload=_make_viewport_update_payload(136, 134, xor_table),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1200,
                direction="received",
                payload=_make_position_update_payload(638, 144, 137, 8, 3, xor_table),
                ws_url="wss://test/ws",
            ),
        ]

        result = analyze_capture_session(_make_session(messages, magic), xor_table)

        assert result["self_tank_id"] == 638
        assert result["movement_response_count"] == 1
        assert result["viewport_update_count"] == 1
        assert result["position_update_count"] == 1
        assert result["thirteen_byte_0x2e_count"] == 1
        assert result["thirteen_byte_shapes"] == [
            {"first_byte": 0x24, "second_byte": 0x02, "count": 1}
        ]
        assert len(result["viewport_inferences"]) == 1
        assert result["viewport_inferences"][0]["viewport_left"] == 136
        assert result["viewport_inferences"][0]["viewport_top"] == 134
        assert result["comparable_position_count"] == 1
        assert result["extra_x_match_count"] == 1
        assert result["extra_y_match_count"] == 1
        assert result["position_evidence"][0]["expected_viewport_x"] == 8
        assert result["position_evidence"][0]["expected_viewport_y"] == 3

    def test_records_viewport_shift_when_inferred_origin_changes(self) -> None:
        """Records a viewport shift when later 0x5A origin changes."""
        magic = "shift-magic"
        static_key = "B" * 64
        xor_table = build_xor_table(static_key, magic)

        messages = [
            CapturedMessage(
                timestamp_ms=1000,
                direction="received",
                payload=_make_movement_response_payload(700, 144, 137, xor_table),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1100,
                direction="received",
                payload=_make_viewport_update_payload(136, 134, xor_table),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1200,
                direction="received",
                payload=_make_viewport_update_payload(137, 134, xor_table),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1300,
                direction="received",
                payload=_make_movement_response_payload(700, 145, 137, xor_table),
                ws_url="wss://test/ws",
            ),
        ]

        result = analyze_capture_session(_make_session(messages, magic), xor_table)

        assert result["movement_response_count"] == 2
        assert result["viewport_update_count"] == 2
        assert result["position_update_count"] == 0
        assert result["thirteen_byte_0x2e_count"] == 0
        assert result["thirteen_byte_shapes"] == []
        assert len(result["viewport_inferences"]) == 2
        assert len(result["viewport_shifts"]) == 1
        assert result["viewport_shifts"][0]["old_left"] == 136
        assert result["viewport_shifts"][0]["old_top"] == 134
        assert result["viewport_shifts"][0]["new_left"] == 137
        assert result["viewport_shifts"][0]["new_top"] == 134


class TestViewportAnalysisEncoding:
    """Tests for viewport analysis encode/decode helpers."""

    def test_round_trips_analysis_result(self) -> None:
        """Encodes and decodes a viewport analysis result without loss."""
        result: ViewportAnalysisDict = {
            "self_tank_id": 638,
            "viewport_inferences": [
                {
                    "message_index": 1,
                    "timestamp_ms": 1100,
                    "viewport_left": 136,
                    "viewport_top": 134,
                }
            ],
            "position_evidence": [
                {
                    "message_index": 3,
                    "timestamp_ms": 1300,
                    "tank_id": 638,
                    "x": 144,
                    "y": 137,
                    "extra_x": 8,
                    "extra_y": 3,
                    "viewport_left": 136,
                    "viewport_top": 134,
                    "expected_viewport_x": 8,
                    "expected_viewport_y": 3,
                    "matches_x": True,
                    "matches_y": True,
                }
            ],
            "viewport_shifts": [
                {
                    "message_index": 2,
                    "timestamp_ms": 1200,
                    "old_left": 136,
                    "old_top": 134,
                    "new_left": 137,
                    "new_top": 134,
                }
            ],
            "movement_response_count": 2,
            "viewport_update_count": 1,
            "position_update_count": 1,
            "thirteen_byte_0x2e_count": 1,
            "thirteen_byte_shapes": [
                {
                    "first_byte": 0x24,
                    "second_byte": 0x02,
                    "count": 1,
                }
            ],
            "comparable_position_count": 1,
            "extra_x_match_count": 1,
            "extra_y_match_count": 1,
        }

        encoded = encode_viewport_analysis(result)
        decoded = decode_viewport_analysis(encoded)

        assert decoded == result

    def test_round_trips_viewport_shift(self) -> None:
        """Encodes and decodes viewport shift evidence without loss."""
        shift = ViewportShiftDict(
            message_index=5,
            timestamp_ms=1450,
            old_left=136,
            old_top=134,
            new_left=137,
            new_top=134,
        )

        assert va.decode_viewport_shift(va.encode_viewport_shift(shift)) == shift

    def test_formats_analysis_report(self) -> None:
        """Formats summary counts and evidence lines for terminal output."""
        result: ViewportAnalysisDict = {
            "self_tank_id": 638,
            "viewport_inferences": [
                {
                    "message_index": 1,
                    "timestamp_ms": 1100,
                    "viewport_left": 136,
                    "viewport_top": 134,
                }
            ],
            "position_evidence": [
                {
                    "message_index": 3,
                    "timestamp_ms": 1300,
                    "tank_id": 638,
                    "x": 144,
                    "y": 137,
                    "extra_x": 8,
                    "extra_y": 3,
                    "viewport_left": 136,
                    "viewport_top": 134,
                    "expected_viewport_x": 8,
                    "expected_viewport_y": 3,
                    "matches_x": True,
                    "matches_y": True,
                }
            ],
            "viewport_shifts": [
                {
                    "message_index": 2,
                    "timestamp_ms": 1200,
                    "old_left": 136,
                    "old_top": 134,
                    "new_left": 137,
                    "new_top": 134,
                }
            ],
            "movement_response_count": 2,
            "viewport_update_count": 1,
            "position_update_count": 1,
            "thirteen_byte_0x2e_count": 1,
            "thirteen_byte_shapes": [
                {
                    "first_byte": 0x24,
                    "second_byte": 0x02,
                    "count": 1,
                }
            ],
            "comparable_position_count": 1,
            "extra_x_match_count": 1,
            "extra_y_match_count": 1,
        }

        formatted = format_viewport_analysis(result)

        assert "self_tank_id=638" in formatted
        assert "capture_status=position_update_comparable" in formatted
        assert "movement_responses=2" in formatted
        assert "viewport_updates=1" in formatted
        assert "position_updates=1" in formatted
        assert "raw_thirteen_byte_0x2e=1" in formatted
        assert "first=0x24 second=0x02 count=1" in formatted
        assert "position_extra_x_matches=1/1" in formatted
        assert "position_extra_y_matches=1/1" in formatted
        assert "viewport=(136,134)" in formatted
        assert "expected=(8,3)" in formatted
        assert "(136,134) -> (137,134)" in formatted

    def test_formats_empty_analysis_report(self) -> None:
        """Formats the empty-evidence message when nothing is comparable."""
        result: ViewportAnalysisDict = {
            "self_tank_id": None,
            "viewport_inferences": [],
            "position_evidence": [],
            "viewport_shifts": [],
            "movement_response_count": 0,
            "viewport_update_count": 0,
            "position_update_count": 0,
            "thirteen_byte_0x2e_count": 0,
            "thirteen_byte_shapes": [],
            "comparable_position_count": 0,
            "extra_x_match_count": 0,
            "extra_y_match_count": 0,
        }

        formatted = format_viewport_analysis(result)

        assert "self_tank_id=None" in formatted
        assert "capture_status=missing_movement_response" in formatted
        assert "No comparable absolute self position_update samples were found." in formatted

    def test_rejects_non_object_nested_entries(self) -> None:
        """Rejects invalid nested analysis entries with precise errors."""
        with pytest.raises(JSONTypeError, match="viewport_inferences\\[0\\]"):
            decode_viewport_analysis(
                {
                    "self_tank_id": None,
                    "viewport_inferences": [1],
                    "position_evidence": [],
                    "viewport_shifts": [],
                    "movement_response_count": 0,
                    "viewport_update_count": 0,
                    "position_update_count": 0,
                    "thirteen_byte_0x2e_count": 0,
                    "thirteen_byte_shapes": [],
                    "comparable_position_count": 0,
                    "extra_x_match_count": 0,
                    "extra_y_match_count": 0,
                }
            )

        with pytest.raises(JSONTypeError, match="position_evidence\\[0\\]"):
            decode_viewport_analysis(
                {
                    "self_tank_id": None,
                    "viewport_inferences": [],
                    "position_evidence": [1],
                    "viewport_shifts": [],
                    "movement_response_count": 0,
                    "viewport_update_count": 0,
                    "position_update_count": 0,
                    "thirteen_byte_0x2e_count": 0,
                    "thirteen_byte_shapes": [],
                    "comparable_position_count": 0,
                    "extra_x_match_count": 0,
                    "extra_y_match_count": 0,
                }
            )

        with pytest.raises(JSONTypeError, match="viewport_shifts\\[0\\]"):
            decode_viewport_analysis(
                {
                    "self_tank_id": None,
                    "viewport_inferences": [],
                    "position_evidence": [],
                    "viewport_shifts": [1],
                    "movement_response_count": 0,
                    "viewport_update_count": 0,
                    "position_update_count": 0,
                    "thirteen_byte_0x2e_count": 0,
                    "thirteen_byte_shapes": [],
                    "comparable_position_count": 0,
                    "extra_x_match_count": 0,
                    "extra_y_match_count": 0,
                }
            )

        with pytest.raises(JSONTypeError, match="thirteen_byte_shapes\\[0\\]"):
            decode_viewport_analysis(
                {
                    "self_tank_id": None,
                    "viewport_inferences": [],
                    "position_evidence": [],
                    "viewport_shifts": [],
                    "movement_response_count": 0,
                    "viewport_update_count": 0,
                    "position_update_count": 0,
                    "thirteen_byte_0x2e_count": 0,
                    "thirteen_byte_shapes": [1],
                    "comparable_position_count": 0,
                    "extra_x_match_count": 0,
                    "extra_y_match_count": 0,
                }
            )


class TestViewportAnalysisHelpers:
    """Tests for internal helper branches in viewport analysis."""

    def test_split_frame_messages_stops_on_zero_length_frame(self) -> None:
        """Stops splitting when a zero-length frame prefix appears."""
        assert va._split_frame_messages(b"\x00\x00") == []

    def test_decode_received_binary_records_skips_sent_invalid_and_sync(self) -> None:
        """Skips sent and invalid frames while decoding unmatched sync messages."""
        magic = "decode-magic"
        static_key = "C" * 64
        xor_table = build_xor_table(static_key, magic)
        messages = [
            CapturedMessage(
                timestamp_ms=1000,
                direction="sent",
                payload=_make_sync_payload(xor_table),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1100,
                direction="received",
                payload="not-base64!",
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1200,
                direction="received",
                payload=base64.b64encode(b"\x00\x00").decode("ascii"),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1250,
                direction="received",
                payload=_make_unknown_payload(xor_table),
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1300,
                direction="received",
                payload=_make_sync_payload(xor_table),
                ws_url="wss://test/ws",
            ),
        ]

        records = va._decode_received_binary_records(_make_session(messages, magic), xor_table)

        assert len(records) == 1
        assert records[0]["decoded"]["msg_type"] == 0x3F

    def test_collect_thirteen_byte_shapes_filters_and_sorts(self) -> None:
        """Collects only received 13-byte 0x2E bodies and sorts shape counts."""
        magic = "shape-magic"
        static_key = "E" * 64
        xor_table = build_xor_table(static_key, magic)

        frame_one = _encode_received_frame(0x2E, bytes.fromhex("2402" + "00" * 11), xor_table)
        frame_two = _encode_received_frame(0x2E, bytes.fromhex("2402" + "11" * 11), xor_table)
        frame_three = _encode_received_frame(0x2E, bytes.fromhex("3d01" + "22" * 11), xor_table)
        short_frame = _encode_received_frame(0x2E, b"\x24\x02", xor_table)
        non_container = _encode_received_frame(0x3F, b"", xor_table)

        messages = [
            CapturedMessage(
                timestamp_ms=1000,
                direction="sent",
                payload=frame_one,
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1100,
                direction="received",
                payload="not-base64!",
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1200,
                direction="received",
                payload=frame_one,
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1300,
                direction="received",
                payload=frame_two,
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1400,
                direction="received",
                payload=frame_three,
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1500,
                direction="received",
                payload=short_frame,
                ws_url="wss://test/ws",
            ),
            CapturedMessage(
                timestamp_ms=1600,
                direction="received",
                payload=non_container,
                ws_url="wss://test/ws",
            ),
        ]

        total_count, shapes = va._collect_thirteen_byte_shapes(
            _make_session(messages, magic),
            xor_table,
        )

        assert total_count == 3
        assert shapes == [
            {"first_byte": 0x24, "second_byte": 0x02, "count": 2},
            {"first_byte": 0x3D, "second_byte": 0x01, "count": 1},
        ]

    def test_format_capture_status_covers_remaining_outcomes(self) -> None:
        """Reports the exact missing evidence stage for each outcome."""
        missing_position: ViewportAnalysisDict = {
            "self_tank_id": 638,
            "viewport_inferences": [],
            "position_evidence": [],
            "viewport_shifts": [],
            "movement_response_count": 1,
            "viewport_update_count": 1,
            "position_update_count": 0,
            "thirteen_byte_0x2e_count": 0,
            "thirteen_byte_shapes": [],
            "comparable_position_count": 0,
            "extra_x_match_count": 0,
            "extra_y_match_count": 0,
        }
        missing_viewport: ViewportAnalysisDict = {
            "self_tank_id": 638,
            "viewport_inferences": [],
            "position_evidence": [],
            "viewport_shifts": [],
            "movement_response_count": 1,
            "viewport_update_count": 0,
            "position_update_count": 0,
            "thirteen_byte_0x2e_count": 0,
            "thirteen_byte_shapes": [],
            "comparable_position_count": 0,
            "extra_x_match_count": 0,
            "extra_y_match_count": 0,
        }
        missing_comparable: ViewportAnalysisDict = {
            "self_tank_id": 638,
            "viewport_inferences": [],
            "position_evidence": [],
            "viewport_shifts": [],
            "movement_response_count": 1,
            "viewport_update_count": 1,
            "position_update_count": 1,
            "thirteen_byte_0x2e_count": 1,
            "thirteen_byte_shapes": [{"first_byte": 0x24, "second_byte": 0x02, "count": 1}],
            "comparable_position_count": 0,
            "extra_x_match_count": 0,
            "extra_y_match_count": 0,
        }

        assert (
            va._format_capture_status(missing_position)
            == "capture_status=missing_proven_position_update"
        )
        assert (
            va._format_capture_status(missing_viewport) == "capture_status=missing_viewport_update"
        )
        assert (
            va._format_capture_status(missing_comparable)
            == "capture_status=position_update_not_comparable_yet"
        )

    def test_handle_movement_response_ignores_other_tanks(self) -> None:
        """Leaves viewport state unchanged when movement belongs to another tank."""
        state = ViewportAnalysisStateDict(
            self_tank_id=638,
            current_viewport_left=136,
            current_viewport_top=134,
        )
        updated = va._handle_movement_response(state, 999)

        assert updated == state

    def test_handle_movement_response_keeps_same_viewport_without_shift(self) -> None:
        """Learns self tank id without mutating known viewport origin."""
        state = ViewportAnalysisStateDict(
            self_tank_id=None,
            current_viewport_left=136,
            current_viewport_top=134,
        )
        updated = va._handle_movement_response(state, 638)

        assert updated["current_viewport_left"] == 136
        assert updated["current_viewport_top"] == 134
        assert updated["self_tank_id"] == 638

    def test_handle_viewport_update_requires_known_self_and_entity(self) -> None:
        """Records direct viewport origins and shift transitions."""
        state = ViewportAnalysisStateDict(
            self_tank_id=638,
            current_viewport_left=136,
            current_viewport_top=134,
        )
        viewport_inferences: list[va.ViewportInferenceDict] = []
        viewport_shifts: list[va.ViewportShiftDict] = []
        updated = va._handle_viewport_update(
            state,
            2,
            1100,
            137,
            134,
            viewport_inferences,
            viewport_shifts,
        )
        assert updated["current_viewport_left"] == 137
        assert viewport_inferences == [
            {"message_index": 2, "timestamp_ms": 1100, "viewport_left": 137, "viewport_top": 134}
        ]
        assert viewport_shifts == [
            {
                "message_index": 2,
                "timestamp_ms": 1100,
                "old_left": 136,
                "old_top": 134,
                "new_left": 137,
                "new_top": 134,
            }
        ]

    def test_handle_position_update_early_returns_and_mismatch_counts(self) -> None:
        """Handles all early-return conditions and mixed x/y match counting."""
        evidence: list[va.PositionViewportEvidenceDict] = []
        empty_state = ViewportAnalysisStateDict(
            self_tank_id=None,
            current_viewport_left=None,
            current_viewport_top=None,
        )

        no_flag = va._handle_position_update(
            empty_state,
            1,
            1000,
            0x00,
            638,
            144,
            137,
            b"\x08\x03",
            evidence,
        )
        assert no_flag == empty_state

        relative_update = va._handle_position_update(
            empty_state,
            2,
            1100,
            0x02,
            638,
            8,
            3,
            b"\x08\x03",
            evidence,
        )
        assert relative_update["self_tank_id"] == 638
        assert evidence == []

        short_extra = va._handle_position_update(
            relative_update,
            3,
            1200,
            0x02,
            638,
            144,
            137,
            b"\x08",
            evidence,
        )
        assert short_extra["self_tank_id"] == 638
        assert evidence == []

        wrong_tank = va._handle_position_update(
            ViewportAnalysisStateDict(
                self_tank_id=638,
                current_viewport_left=136,
                current_viewport_top=134,
            ),
            4,
            1300,
            0x02,
            999,
            144,
            137,
            b"\x08\x03",
            evidence,
        )
        assert wrong_tank["self_tank_id"] == 638
        assert evidence == []

        no_viewport = va._handle_position_update(
            ViewportAnalysisStateDict(
                self_tank_id=638,
                current_viewport_left=None,
                current_viewport_top=None,
            ),
            5,
            1400,
            0x02,
            638,
            144,
            137,
            b"\x08\x03",
            evidence,
        )
        assert no_viewport["self_tank_id"] == 638
        assert evidence == []

        matched = va._handle_position_update(
            ViewportAnalysisStateDict(
                self_tank_id=638,
                current_viewport_left=136,
                current_viewport_top=134,
            ),
            6,
            1500,
            0x02,
            638,
            144,
            137,
            b"\x08\x03",
            evidence,
        )
        assert matched["current_viewport_left"] == 136
        assert len(evidence) == 1

        va._handle_position_update(
            matched,
            7,
            1600,
            0x02,
            638,
            145,
            137,
            b"\x08\x02",
            evidence,
        )
        assert va._count_matches(evidence) == (1, 1)

    def test_analyze_capture_session_ignores_unmatched_sync_messages(self) -> None:
        """Ignores decoded messages that are outside the viewport-analysis cases."""
        magic = "analyze-other"
        static_key = "D" * 64
        xor_table = build_xor_table(static_key, magic)
        messages = [
            CapturedMessage(
                timestamp_ms=1000,
                direction="received",
                payload=_make_sync_payload(xor_table),
                ws_url="wss://test/ws",
            )
        ]

        result = analyze_capture_session(_make_session(messages, magic), xor_table)

        assert result["self_tank_id"] is None
        assert result["movement_response_count"] == 0
        assert result["viewport_update_count"] == 0
        assert result["position_update_count"] == 0
        assert result["thirteen_byte_0x2e_count"] == 0
        assert result["thirteen_byte_shapes"] == []
        assert result["viewport_inferences"] == []
        assert result["position_evidence"] == []
