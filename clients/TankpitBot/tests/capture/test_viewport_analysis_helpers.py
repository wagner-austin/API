"""Tests for viewport-analysis helper detail."""

from __future__ import annotations

from tankpit_bot.capture import viewport_analysis as va
from tankpit_bot.capture.viewport_analysis import (
    ViewportAnalysisDict,
    analyze_capture_session,
)
from tankpit_bot.capture.viewport_analysis_types import (
    ViewportAnalysisStateDict,
)
from tankpit_bot.protocol.codec import build_xor_table
from tankpit_bot.types.message import CapturedMessage
from tests.capture._viewport_analysis_fixtures import (
    _make_session,
    _make_sync_payload,
)


class TestViewportAnalysisHelperDetail:
    """Tests for viewport-analysis helper detail."""

    def test_format_capture_status_covers_remaining_outcomes(self) -> None:
        """Reports the exact missing evidence stage for each outcome."""
        missing_viewport: ViewportAnalysisDict = {
            "self_tank_id": 638,
            "viewport_inferences": [],
            "viewport_shifts": [],
            "movement_response_count": 1,
            "viewport_update_count": 0,
            "thirteen_byte_0x2e_count": 0,
            "thirteen_byte_shapes": [],
        }
        inferred: ViewportAnalysisDict = {
            "self_tank_id": 638,
            "viewport_inferences": [
                {
                    "message_index": 1,
                    "timestamp_ms": 1100,
                    "viewport_left": 136,
                    "viewport_top": 134,
                }
            ],
            "viewport_shifts": [],
            "movement_response_count": 1,
            "viewport_update_count": 1,
            "thirteen_byte_0x2e_count": 0,
            "thirteen_byte_shapes": [],
        }

        assert (
            va._format_capture_status(missing_viewport) == "capture_status=missing_viewport_update"
        )
        assert va._format_capture_status(inferred) == "capture_status=viewport_inferred"

    def test_handle_movement_response_ignores_other_tanks(self) -> None:
        """Leaves viewport state unchanged when self id is already learned."""
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

    def test_handle_viewport_update_records_inference_and_shift(self) -> None:
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
        assert result["thirteen_byte_0x2e_count"] == 0
        assert result["thirteen_byte_shapes"] == []
