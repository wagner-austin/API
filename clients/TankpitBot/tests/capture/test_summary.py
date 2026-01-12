"""Tests for tankpit_bot.capture.summary module."""

from __future__ import annotations

from tankpit_bot.capture.summary import build_session_summary
from tankpit_bot.types import CaptureSession, GameLogEntryWithTimestamp


class TestBuildSessionSummary:
    """Tests for build_session_summary function."""

    def test_skips_unknown_combat_event_types(self) -> None:
        """Test skips combat entries that don't match known patterns."""
        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic="test",
            game_log=[
                GameLogEntryWithTimestamp(
                    timestamp_ms=100,
                    text="Unknown combat message format",
                    category="combat",
                ),
            ],
            tank_names={},
        )

        result = build_session_summary(session)
        # Unknown event type should be skipped
        assert len(result["combat"]) == 0

    def test_extracts_killed_you_event(self) -> None:
        """Test extracts killed_you combat event."""
        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic="test",
            game_log=[
                GameLogEntryWithTimestamp(
                    timestamp_ms=100,
                    text="Enemy killed you",
                    category="combat",
                ),
            ],
            tank_names={},
        )

        result = build_session_summary(session)
        assert len(result["combat"]) == 1
        assert result["combat"][0]["event_type"] == "killed_by"
        assert result["combat"][0]["target"] == "Enemy"

    def test_handles_empty_game_log(self) -> None:
        """Test handles empty game log."""
        session = CaptureSession(
            session_id="test",
            start_timestamp_ms=0,
            end_timestamp_ms=1000,
            base_url="test",
            messages=[],
            magic="test",
            game_log=[],
            tank_names={},
        )

        result = build_session_summary(session)
        assert len(result["combat"]) == 0
        assert result["game_log"] == []
