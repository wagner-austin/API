"""Tests for platform_stt.srt module."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from platform_stt import _test_hooks
from platform_stt.srt import (
    SrtEntry,
    decode_srt_entry,
    encode_srt_entry,
    format_srt,
    format_srt_entry,
    format_timestamp,
    require_srt_entry,
    segments_to_srt_entries,
    write_srt,
)
from platform_stt.testing import FakeWriteTextFile, reset_hooks
from platform_stt.types import VerboseSegment


class TestFormatTimestamp:
    """Tests for format_timestamp function."""

    def test_zero_seconds(self) -> None:
        """Format zero seconds."""
        result = format_timestamp(0.0)
        assert result == "00:00:00,000"

    def test_milliseconds_only(self) -> None:
        """Format sub-second time."""
        result = format_timestamp(0.5)
        assert result == "00:00:00,500"

    def test_seconds_only(self) -> None:
        """Format seconds without minutes."""
        result = format_timestamp(45.0)
        assert result == "00:00:45,000"

    def test_minutes_and_seconds(self) -> None:
        """Format minutes and seconds."""
        result = format_timestamp(125.0)
        assert result == "00:02:05,000"

    def test_hours_minutes_seconds(self) -> None:
        """Format full timestamp with hours."""
        result = format_timestamp(3661.5)
        assert result == "01:01:01,500"

    def test_large_hours(self) -> None:
        """Format timestamp with many hours."""
        result = format_timestamp(36000.0)  # 10 hours
        assert result == "10:00:00,000"

    def test_precise_milliseconds(self) -> None:
        """Format timestamp with precise milliseconds."""
        result = format_timestamp(1.234)
        assert result == "00:00:01,234"

    def test_rounding_milliseconds(self) -> None:
        """Round milliseconds correctly."""
        result = format_timestamp(1.2345)
        assert result == "00:00:01,234"

    def test_rounding_up(self) -> None:
        """Round up when appropriate."""
        result = format_timestamp(1.9999)
        assert result == "00:00:02,000"

    def test_negative_raises(self) -> None:
        """Raise ValueError for negative seconds."""
        with pytest.raises(ValueError, match="non-negative"):
            format_timestamp(-1.0)


class TestSrtEntryEncode:
    """Tests for encode_srt_entry function."""

    def test_encode_basic(self) -> None:
        """Encode basic SRT entry."""
        entry = SrtEntry(
            index=1,
            start_seconds=0.0,
            end_seconds=2.5,
            text="Hello world",
        )
        result = encode_srt_entry(entry)
        assert result == {
            "index": 1,
            "start_seconds": 0.0,
            "end_seconds": 2.5,
            "text": "Hello world",
        }

    def test_encode_preserves_all_fields(self) -> None:
        """Encode preserves all field values."""
        entry = SrtEntry(
            index=42,
            start_seconds=100.5,
            end_seconds=105.75,
            text="Test subtitle",
        )
        result = encode_srt_entry(entry)
        assert result["index"] == 42
        assert result["start_seconds"] == 100.5
        assert result["end_seconds"] == 105.75
        assert result["text"] == "Test subtitle"


class TestSrtEntryDecode:
    """Tests for decode_srt_entry function."""

    def test_decode_valid(self) -> None:
        """Decode valid SRT entry."""
        obj: JSONObject = {
            "index": 1,
            "start_seconds": 0.0,
            "end_seconds": 2.5,
            "text": "Hello",
        }
        result = decode_srt_entry(obj)
        assert result["index"] == 1
        assert result["start_seconds"] == 0.0
        assert result["end_seconds"] == 2.5
        assert result["text"] == "Hello"

    def test_decode_missing_index(self) -> None:
        """Raise for missing index field."""
        obj: JSONObject = {"start_seconds": 0.0, "end_seconds": 1.0, "text": "Hello"}
        with pytest.raises(JSONTypeError):
            decode_srt_entry(obj)

    def test_decode_missing_start(self) -> None:
        """Raise for missing start_seconds field."""
        obj: JSONObject = {"index": 1, "end_seconds": 1.0, "text": "Hello"}
        with pytest.raises(JSONTypeError):
            decode_srt_entry(obj)

    def test_decode_missing_end(self) -> None:
        """Raise for missing end_seconds field."""
        obj: JSONObject = {"index": 1, "start_seconds": 0.0, "text": "Hello"}
        with pytest.raises(JSONTypeError):
            decode_srt_entry(obj)

    def test_decode_missing_text(self) -> None:
        """Raise for missing text field."""
        obj: JSONObject = {"index": 1, "start_seconds": 0.0, "end_seconds": 1.0}
        with pytest.raises(JSONTypeError):
            decode_srt_entry(obj)

    def test_decode_wrong_type_index(self) -> None:
        """Raise for wrong type index."""
        obj: JSONObject = {
            "index": "one",
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "text": "Hello",
        }
        with pytest.raises(JSONTypeError):
            decode_srt_entry(obj)

    def test_decode_wrong_type_start(self) -> None:
        """Raise for wrong type start_seconds."""
        obj: JSONObject = {
            "index": 1,
            "start_seconds": "zero",
            "end_seconds": 1.0,
            "text": "Hello",
        }
        with pytest.raises(JSONTypeError):
            decode_srt_entry(obj)

    def test_decode_index_less_than_one(self) -> None:
        """Raise for index less than 1."""
        obj: JSONObject = {
            "index": 0,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "text": "Hello",
        }
        with pytest.raises(ValueError, match="index must be >= 1"):
            decode_srt_entry(obj)

    def test_decode_negative_start(self) -> None:
        """Raise for negative start_seconds."""
        obj: JSONObject = {
            "index": 1,
            "start_seconds": -1.0,
            "end_seconds": 1.0,
            "text": "Hello",
        }
        with pytest.raises(ValueError, match="start_seconds must be >= 0"):
            decode_srt_entry(obj)

    def test_decode_negative_end(self) -> None:
        """Raise for negative end_seconds."""
        obj: JSONObject = {
            "index": 1,
            "start_seconds": 0.0,
            "end_seconds": -1.0,
            "text": "Hello",
        }
        with pytest.raises(ValueError, match="end_seconds must be >= 0"):
            decode_srt_entry(obj)

    def test_decode_end_before_start(self) -> None:
        """Raise when end_seconds < start_seconds."""
        obj: JSONObject = {
            "index": 1,
            "start_seconds": 5.0,
            "end_seconds": 3.0,
            "text": "Hello",
        }
        with pytest.raises(ValueError, match=r"end_seconds.*must be >= start_seconds"):
            decode_srt_entry(obj)


class TestRequireSrtEntry:
    """Tests for require_srt_entry function."""

    def test_require_valid(self) -> None:
        """Require valid SRT entry from JSONValue."""
        obj: JSONObject = {
            "index": 1,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "text": "Hello",
        }
        result = require_srt_entry(obj)
        assert result["index"] == 1

    def test_require_non_dict(self) -> None:
        """Raise for non-dict value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_srt_entry("not a dict")

    def test_require_list(self) -> None:
        """Raise for list value."""
        with pytest.raises(JSONTypeError, match="Expected object"):
            require_srt_entry([1, 2, 3])


class TestFormatSrtEntry:
    """Tests for format_srt_entry function."""

    def test_format_basic(self) -> None:
        """Format basic SRT entry."""
        entry = SrtEntry(
            index=1,
            start_seconds=0.0,
            end_seconds=2.5,
            text="Hello world",
        )
        result = format_srt_entry(entry)
        expected = "1\n00:00:00,000 --> 00:00:02,500\nHello world"
        assert result == expected

    def test_format_with_hours(self) -> None:
        """Format entry with hours in timestamps."""
        entry = SrtEntry(
            index=42,
            start_seconds=3661.0,
            end_seconds=3665.5,
            text="Later in video",
        )
        result = format_srt_entry(entry)
        expected = "42\n01:01:01,000 --> 01:01:05,500\nLater in video"
        assert result == expected

    def test_format_strips_text_whitespace(self) -> None:
        """Strip whitespace from text."""
        entry = SrtEntry(
            index=1,
            start_seconds=0.0,
            end_seconds=1.0,
            text="  Hello  ",
        )
        result = format_srt_entry(entry)
        assert result.endswith("Hello")

    def test_format_multiline_text(self) -> None:
        """Preserve multiline text."""
        entry = SrtEntry(
            index=1,
            start_seconds=0.0,
            end_seconds=2.0,
            text="Line one\nLine two",
        )
        result = format_srt_entry(entry)
        assert "Line one\nLine two" in result


class TestSegmentsToSrtEntries:
    """Tests for segments_to_srt_entries function."""

    def test_empty_list(self) -> None:
        """Convert empty segment list."""
        result = segments_to_srt_entries([])
        assert result == []

    def test_single_segment(self) -> None:
        """Convert single segment."""
        segments = [VerboseSegment(text="Hello", start=0.0, end=1.5)]
        result = segments_to_srt_entries(segments)
        assert len(result) == 1
        assert result[0]["index"] == 1
        assert result[0]["start_seconds"] == 0.0
        assert result[0]["end_seconds"] == 1.5
        assert result[0]["text"] == "Hello"

    def test_multiple_segments(self) -> None:
        """Convert multiple segments with correct indices."""
        segments = [
            VerboseSegment(text="First", start=0.0, end=1.0),
            VerboseSegment(text="Second", start=1.5, end=3.0),
            VerboseSegment(text="Third", start=3.5, end=5.0),
        ]
        result = segments_to_srt_entries(segments)
        assert len(result) == 3
        assert result[0]["index"] == 1
        assert result[1]["index"] == 2
        assert result[2]["index"] == 3


class TestFormatSrt:
    """Tests for format_srt function."""

    def test_empty_segments(self) -> None:
        """Format empty segment list."""
        result = format_srt([])
        assert result == ""

    def test_single_segment(self) -> None:
        """Format single segment."""
        segments = [VerboseSegment(text="Hello", start=0.0, end=1.5)]
        result = format_srt(segments)
        expected = "1\n00:00:00,000 --> 00:00:01,500\nHello"
        assert result == expected

    def test_multiple_segments(self) -> None:
        """Format multiple segments with blank line separator."""
        segments = [
            VerboseSegment(text="First", start=0.0, end=1.0),
            VerboseSegment(text="Second", start=1.5, end=3.0),
        ]
        result = format_srt(segments)
        lines = result.split("\n\n")
        assert len(lines) == 2
        assert lines[0].startswith("1\n")
        assert lines[1].startswith("2\n")

    def test_preserves_segment_order(self) -> None:
        """Preserve segment order in output."""
        segments = [
            VerboseSegment(text="A", start=0.0, end=1.0),
            VerboseSegment(text="B", start=2.0, end=3.0),
            VerboseSegment(text="C", start=4.0, end=5.0),
        ]
        result = format_srt(segments)
        assert result.index("A") < result.index("B") < result.index("C")


class TestWriteSrt:
    """Tests for write_srt function."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_write_uses_hook(self) -> None:
        """Write uses the write_text_file hook."""
        fake_writer = FakeWriteTextFile()
        _test_hooks.write_text_file = fake_writer

        content = "1\n00:00:00,000 --> 00:00:01,000\nHello"
        path = Path("/fake/output.srt")
        write_srt(content, path)

        assert len(fake_writer.writes) == 1
        assert fake_writer.writes[0][0] == path
        assert fake_writer.writes[0][1] == content

    def test_write_real_file(self, tmp_path: Path) -> None:
        """Write real file using production hook."""
        content = "1\n00:00:00,000 --> 00:00:01,000\nHello"
        output_path = tmp_path / "output.srt"

        write_srt(content, output_path)

        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == content


class TestRoundTrip:
    """Tests for encode/decode round-trip."""

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode produces identical entry."""
        original = SrtEntry(
            index=5,
            start_seconds=10.5,
            end_seconds=15.75,
            text="Round trip test",
        )
        encoded = encode_srt_entry(original)
        decoded = decode_srt_entry(encoded)
        assert decoded == original


class TestIntegration:
    """Integration tests for full SRT generation workflow."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow from segments to SRT file."""
        segments = [
            VerboseSegment(text="Welcome to the video", start=0.0, end=2.5),
            VerboseSegment(text="Today we will learn", start=3.0, end=5.0),
            VerboseSegment(text="Something new", start=5.5, end=7.0),
        ]

        srt_content = format_srt(segments)
        output_path = tmp_path / "subtitles.srt"
        write_srt(srt_content, output_path)

        # Verify file contents
        file_content = output_path.read_text(encoding="utf-8")
        assert "1\n00:00:00,000 --> 00:00:02,500\nWelcome to the video" in file_content
        assert "2\n00:00:03,000 --> 00:00:05,000\nToday we will learn" in file_content
        assert "3\n00:00:05,500 --> 00:00:07,000\nSomething new" in file_content

    def test_workflow_with_fake_writer(self) -> None:
        """Test workflow with fake writer for unit testing."""
        fake_writer = FakeWriteTextFile()
        _test_hooks.write_text_file = fake_writer

        segments = [VerboseSegment(text="Test", start=0.0, end=1.0)]
        srt_content = format_srt(segments)
        write_srt(srt_content, Path("/fake/test.srt"))

        assert len(fake_writer.writes) == 1
        written_content = fake_writer.writes[0][1]
        assert "1\n00:00:00,000 --> 00:00:01,000\nTest" in written_content
