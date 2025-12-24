"""Tests for platform_stt.merger module."""

from __future__ import annotations

from platform_stt.merger import TranscriptMerger, merge_segment_text
from platform_stt.types import AudioChunk, TranscriptSegment


class TestTranscriptMerger:
    """Tests for TranscriptMerger class."""

    def test_merge_single_chunk(self) -> None:
        """Merge segments from a single chunk."""
        merger = TranscriptMerger()
        chunk = AudioChunk(
            path="/tmp/chunk.mp3",
            start_seconds=0.0,
            duration_seconds=10.0,
            size_bytes=1024,
        )
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=0.5),
            TranscriptSegment(text="world", start=0.5, duration=0.5),
        ]
        result = merger.merge([(chunk, segments)])
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[1]["text"] == "world"

    def test_merge_multiple_chunks(self) -> None:
        """Merge segments from multiple chunks with offset adjustment."""
        merger = TranscriptMerger()
        chunk1 = AudioChunk(
            path="/tmp/chunk1.mp3",
            start_seconds=0.0,
            duration_seconds=10.0,
            size_bytes=1024,
        )
        chunk2 = AudioChunk(
            path="/tmp/chunk2.mp3",
            start_seconds=10.0,
            duration_seconds=10.0,
            size_bytes=1024,
        )
        segments1 = [TranscriptSegment(text="First", start=0.0, duration=5.0)]
        segments2 = [TranscriptSegment(text="Second", start=0.0, duration=5.0)]

        result = merger.merge([(chunk1, segments1), (chunk2, segments2)])
        assert len(result) == 2
        assert result[0]["text"] == "First"
        assert result[0]["start"] == 0.0
        assert result[1]["text"] == "Second"
        assert result[1]["start"] == 10.0  # Adjusted by chunk offset

    def test_merge_empty_chunk(self) -> None:
        """Handle chunk with no segments."""
        merger = TranscriptMerger()
        chunk = AudioChunk(
            path="/tmp/empty.mp3",
            start_seconds=0.0,
            duration_seconds=10.0,
            size_bytes=1024,
        )
        result = merger.merge([(chunk, [])])
        assert len(result) == 0

    def test_merge_sorts_by_start(self) -> None:
        """Result segments are sorted by start time."""
        merger = TranscriptMerger()
        chunk1 = AudioChunk(
            path="/tmp/chunk1.mp3",
            start_seconds=10.0,
            duration_seconds=10.0,
            size_bytes=1024,
        )
        chunk2 = AudioChunk(
            path="/tmp/chunk2.mp3",
            start_seconds=0.0,
            duration_seconds=10.0,
            size_bytes=1024,
        )
        segments1 = [TranscriptSegment(text="Second", start=0.0, duration=5.0)]
        segments2 = [TranscriptSegment(text="First", start=0.0, duration=5.0)]

        # Process out of order
        result = merger.merge([(chunk1, segments1), (chunk2, segments2)])
        assert result[0]["text"] == "First"  # 0.0 offset
        assert result[1]["text"] == "Second"  # 10.0 offset

    def test_merge_clamps_negative_start(self) -> None:
        """Clamp negative start times to 0."""
        merger = TranscriptMerger()
        chunk = AudioChunk(
            path="/tmp/chunk.mp3",
            start_seconds=0.0,
            duration_seconds=10.0,
            size_bytes=1024,
        )
        # Segment with negative start (shouldn't happen but handle gracefully)
        segments = [TranscriptSegment(text="Negative", start=-1.0, duration=1.0)]
        result = merger.merge([(chunk, segments)])
        assert result[0]["start"] == 0.0  # Clamped to 0


class TestMergeSegmentText:
    """Tests for merge_segment_text function."""

    def test_merge_segment_text_basic(self) -> None:
        """Concatenate segment texts with spaces."""
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=0.5),
            TranscriptSegment(text="world", start=0.5, duration=0.5),
        ]
        result = merge_segment_text(segments)
        assert result == "Hello world"

    def test_merge_segment_text_empty_list(self) -> None:
        """Return empty string for empty list."""
        result = merge_segment_text([])
        assert result == ""

    def test_merge_segment_text_strips_whitespace(self) -> None:
        """Strip whitespace from each segment."""
        segments = [
            TranscriptSegment(text="  Hello  ", start=0.0, duration=0.5),
            TranscriptSegment(text="  world  ", start=0.5, duration=0.5),
        ]
        result = merge_segment_text(segments)
        assert result == "Hello world"

    def test_merge_segment_text_skips_empty(self) -> None:
        """Skip empty segments."""
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=0.5),
            TranscriptSegment(text="", start=0.5, duration=0.1),
            TranscriptSegment(text="world", start=0.6, duration=0.4),
        ]
        result = merge_segment_text(segments)
        assert result == "Hello world"

    def test_merge_segment_text_whitespace_only(self) -> None:
        """Skip whitespace-only segments."""
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=0.5),
            TranscriptSegment(text="   ", start=0.5, duration=0.1),
            TranscriptSegment(text="world", start=0.6, duration=0.4),
        ]
        result = merge_segment_text(segments)
        assert result == "Hello world"
