"""Tests for VTT/SRT subtitle parser."""

from __future__ import annotations

import tempfile

import pytest

from transcript_api.types import RawTranscriptItem
from transcript_api.vtt_parser import (
    _make_segment,
    _parse_vtt_line,
    _parse_vtt_timestamp,
    _process_vtt_line,
    _should_skip_vtt_line,
    _strip_vtt_tags,
    parse_vtt_content,
    parse_vtt_file,
)


class TestParseVttTimestamp:
    """Tests for _parse_vtt_timestamp."""

    def test_parses_standard_vtt_format(self) -> None:
        result = _parse_vtt_timestamp("00:01:30.500")
        assert result == 90.5

    def test_parses_srt_format_with_comma(self) -> None:
        result = _parse_vtt_timestamp("00:01:30,500")
        assert result == 90.5

    def test_parses_short_form(self) -> None:
        result = _parse_vtt_timestamp("01:30.500")
        assert result == 90.5

    def test_raises_on_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            _parse_vtt_timestamp("invalid")


class TestParseVttLine:
    """Tests for _parse_vtt_line."""

    def test_parses_standard_timestamp_line(self) -> None:
        result = _parse_vtt_line("00:00:00.000 --> 00:00:05.000")
        assert result == (0.0, 5.0)

    def test_parses_with_positioning_info(self) -> None:
        result = _parse_vtt_line("00:00:00.000 --> 00:00:05.000 align:start")
        assert result == (0.0, 5.0)

    def test_returns_none_for_non_timestamp_line(self) -> None:
        result = _parse_vtt_line("This is not a timestamp")
        assert result is None

    def test_returns_none_for_malformed_arrow(self) -> None:
        result = _parse_vtt_line("00:00:00.000 -> 00:00:05.000")
        assert result is None

    def test_returns_none_for_multiple_arrows(self) -> None:
        result = _parse_vtt_line("00:00:00.000 --> 00:00:05.000 --> 00:00:10.000")
        assert result is None


class TestStripVttTags:
    """Tests for _strip_vtt_tags."""

    def test_strips_voice_tags(self) -> None:
        result = _strip_vtt_tags("<v Speaker>Hello world")
        assert result == "Hello world"

    def test_strips_class_tags(self) -> None:
        result = _strip_vtt_tags("<c.classname>Hello</c>")
        assert result == "Hello"

    def test_strips_formatting_tags(self) -> None:
        result = _strip_vtt_tags("<b>Bold</b> <i>Italic</i> <u>Underline</u>")
        assert result == "Bold Italic Underline"

    def test_strips_timestamp_tags(self) -> None:
        result = _strip_vtt_tags("<00:00:01.000>Hello<00:00:02.000>world")
        assert result == "Helloworld"

    def test_handles_plain_text(self) -> None:
        result = _strip_vtt_tags("Plain text")
        assert result == "Plain text"


class TestShouldSkipVttLine:
    """Tests for _should_skip_vtt_line."""

    def test_skips_webvtt_header(self) -> None:
        assert _should_skip_vtt_line("WEBVTT") is True
        assert _should_skip_vtt_line("WEBVTT - Description") is True

    def test_skips_note_comments(self) -> None:
        assert _should_skip_vtt_line("NOTE This is a comment") is True

    def test_skips_numeric_cue_ids(self) -> None:
        assert _should_skip_vtt_line("1") is True
        assert _should_skip_vtt_line("123") is True

    def test_skips_style_blocks(self) -> None:
        assert _should_skip_vtt_line("STYLE") is True
        assert _should_skip_vtt_line("STYLE::cue {}") is True

    def test_does_not_skip_regular_text(self) -> None:
        assert _should_skip_vtt_line("Hello world") is False
        assert _should_skip_vtt_line("00:00:00.000 --> 00:00:05.000") is False


class TestMakeSegment:
    """Tests for _make_segment."""

    def test_creates_segment_from_text_lines(self) -> None:
        result = _make_segment(["Hello", "world"], 0.0, 1.0)
        if result is None:
            pytest.fail("expected segment result")
        assert result["text"] == "Hello world"
        assert result["start"] == 0.0
        assert result["duration"] == 1.0

    def test_returns_none_for_empty_text(self) -> None:
        result = _make_segment([], 0.0, 1.0)
        assert result is None

    def test_returns_none_for_whitespace_only(self) -> None:
        result = _make_segment(["   ", "  "], 0.0, 1.0)
        assert result is None

    def test_strips_vtt_tags_from_text(self) -> None:
        result = _make_segment(["<b>Hello</b>", "<i>world</i>"], 0.0, 1.0)
        if result is None:
            pytest.fail("expected segment result")
        assert result["text"] == "Hello world"


class TestProcessVttLine:
    """Tests for _process_vtt_line."""

    def test_processes_timestamp_line(self) -> None:
        segments: list[RawTranscriptItem] = []
        result = _process_vtt_line(
            "00:00:00.000 --> 00:00:05.000",
            segments,
            None,
            None,
            [],
        )
        assert result == (0.0, 5.0, [])
        assert len(segments) == 0

    def test_saves_previous_segment_on_timestamp(self) -> None:
        segments: list[RawTranscriptItem] = []
        result = _process_vtt_line(
            "00:00:05.000 --> 00:00:10.000",
            segments,
            0.0,
            5.0,
            ["Hello"],
        )
        assert result == (5.0, 10.0, [])
        assert len(segments) == 1
        assert segments[0]["text"] == "Hello"

    def test_processes_empty_line(self) -> None:
        segments: list[RawTranscriptItem] = []
        result = _process_vtt_line(
            "",
            segments,
            0.0,
            5.0,
            ["Hello"],
        )
        assert result == (None, None, [])
        assert len(segments) == 1

    def test_accumulates_text_lines(self) -> None:
        segments: list[RawTranscriptItem] = []
        text_lines = ["Hello"]
        result = _process_vtt_line(
            "world",
            segments,
            0.0,
            5.0,
            text_lines,
        )
        assert result == (0.0, 5.0, text_lines)
        assert text_lines == ["Hello", "world"]

    def test_ignores_text_without_timestamp(self) -> None:
        segments: list[RawTranscriptItem] = []
        result = _process_vtt_line(
            "Hello",
            segments,
            None,
            None,
            [],
        )
        assert result == (None, None, [])


class TestParseVttContent:
    """Tests for parse_vtt_content."""

    def test_parses_basic_vtt(self) -> None:
        content = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hello world

00:00:05.000 --> 00:00:10.000
Goodbye world
"""
        result = parse_vtt_content(content)
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"
        assert result[0]["start"] == 0.0
        assert result[0]["duration"] == 5.0
        assert result[1]["text"] == "Goodbye world"

    def test_parses_srt_format(self) -> None:
        content = """1
00:00:00,000 --> 00:00:05,000
Hello world

2
00:00:05,000 --> 00:00:10,000
Goodbye world
"""
        result = parse_vtt_content(content)
        assert len(result) == 2

    def test_handles_multiline_cues(self) -> None:
        content = """WEBVTT

00:00:00.000 --> 00:00:05.000
Line one
Line two
Line three
"""
        result = parse_vtt_content(content)
        assert len(result) == 1
        assert result[0]["text"] == "Line one Line two Line three"

    def test_handles_vtt_tags(self) -> None:
        content = """WEBVTT

00:00:00.000 --> 00:00:05.000
<v Speaker><b>Hello</b> world</v>
"""
        result = parse_vtt_content(content)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"

    def test_skips_style_blocks(self) -> None:
        content = """WEBVTT

STYLE
::cue { color: white; }

00:00:00.000 --> 00:00:05.000
Hello world
"""
        result = parse_vtt_content(content)
        assert len(result) == 1

    def test_skips_note_comments(self) -> None:
        content = """WEBVTT

NOTE This is a comment

00:00:00.000 --> 00:00:05.000
Hello world
"""
        result = parse_vtt_content(content)
        assert len(result) == 1

    def test_handles_empty_content(self) -> None:
        result = parse_vtt_content("")
        assert len(result) == 0

    def test_handles_content_without_trailing_newline(self) -> None:
        content = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hello world"""
        result = parse_vtt_content(content)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"

    def test_skips_empty_cues(self) -> None:
        content = """WEBVTT

00:00:00.000 --> 00:00:05.000


00:00:05.000 --> 00:00:10.000
Hello world
"""
        result = parse_vtt_content(content)
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"


class TestParseVttFile:
    """Tests for parse_vtt_file."""

    def test_parses_vtt_file(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".vtt", delete=False, encoding="utf-8"
        ) as f:
            f.write("""WEBVTT

00:00:00.000 --> 00:00:05.000
Hello world
""")
            f.flush()
            result = parse_vtt_file(f.name)
            assert len(result) == 1
            assert result[0]["text"] == "Hello world"

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_vtt_file("/nonexistent/path/file.vtt")
