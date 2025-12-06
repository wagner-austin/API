"""VTT/SRT subtitle file parser.

Parses WebVTT and SRT subtitle formats into RawTranscriptItem segments.
Strictly typed with no Any, cast, or type: ignore.
"""

from __future__ import annotations

import re
from pathlib import Path

from .types import RawTranscriptItem


def _parse_vtt_timestamp(ts: str) -> float:
    """Parse VTT/SRT timestamp to seconds.

    Formats supported:
    - HH:MM:SS.mmm (VTT standard)
    - HH:MM:SS,mmm (SRT standard)
    - MM:SS.mmm (VTT short form)
    """
    # Normalize comma to dot for SRT format
    ts = ts.replace(",", ".")

    parts = ts.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600.0 + minutes * 60.0 + seconds
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds

    raise ValueError(f"Invalid timestamp format: {ts}")


def _parse_vtt_line(line: str) -> tuple[float, float] | None:
    """Parse a VTT/SRT timestamp line.

    Returns (start_seconds, end_seconds) or None if not a timestamp line.
    Expected format: "00:00:00.000 --> 00:00:05.000"
    """
    if " --> " not in line:
        return None

    parts = line.split(" --> ")
    if len(parts) != 2:
        return None

    start_str = parts[0].strip()
    # End timestamp may have positioning info after it (e.g., "00:05.000 align:start")
    end_str = parts[1].split()[0].strip()

    start = _parse_vtt_timestamp(start_str)
    end = _parse_vtt_timestamp(end_str)
    return (start, end)


def _strip_vtt_tags(text: str) -> str:
    """Remove VTT formatting tags like <c>, </c>, <v Speaker>, etc."""
    # Remove voice tags: <v Speaker> and </v>
    text = re.sub(r"</?v[^>]*>", "", text)
    # Remove class tags: <c>, </c>, <c.classname>
    text = re.sub(r"</?c[^>]*>", "", text)
    # Remove other common tags: <b>, </b>, <i>, </i>, <u>, </u>
    text = re.sub(r"</?[biu]>", "", text)
    # Remove timestamp tags: <00:00:00.000>
    text = re.sub(r"<\d{2}:\d{2}[:\d.]*>", "", text)
    return text.strip()


def _should_skip_vtt_line(line: str) -> bool:
    """Check if a VTT line should be skipped (header, comment, style, cue id)."""
    if line.startswith("WEBVTT") or line.startswith("NOTE"):
        return True
    if line.isdigit():
        return True
    return line.startswith("STYLE")


def _make_segment(
    text_lines: list[str],
    start: float,
    end: float,
) -> RawTranscriptItem | None:
    """Create a segment from accumulated text lines, or None if empty."""
    text = " ".join(text_lines)
    text = _strip_vtt_tags(text)
    if not text.strip():
        return None
    duration = end - start
    return RawTranscriptItem(text=text, start=start, duration=duration)


def _append_segment_if_valid(
    segments: list[RawTranscriptItem],
    text_lines: list[str],
    start: float | None,
    end: float | None,
) -> None:
    """Append a segment to segments list if start/end are valid and text is non-empty."""
    if start is not None and end is not None:
        seg = _make_segment(text_lines, start, end)
        if seg is not None:
            segments.append(seg)


def _process_vtt_line(
    line: str,
    segments: list[RawTranscriptItem],
    current_start: float | None,
    current_end: float | None,
    current_text_lines: list[str],
) -> tuple[float | None, float | None, list[str]]:
    """Process a single VTT line, returning updated state."""
    # Try to parse as timestamp line
    timestamp = _parse_vtt_line(line)
    if timestamp is not None:
        _append_segment_if_valid(segments, current_text_lines, current_start, current_end)
        return timestamp[0], timestamp[1], []

    # Empty line ends a cue
    if not line:
        _append_segment_if_valid(segments, current_text_lines, current_start, current_end)
        return None, None, []

    # Accumulate text lines
    if current_start is not None:
        current_text_lines.append(line)

    return current_start, current_end, current_text_lines


def parse_vtt_content(content: str) -> list[RawTranscriptItem]:
    """Parse VTT/SRT content string into transcript segments.

    Args:
        content: Raw VTT or SRT file content as string.

    Returns:
        List of RawTranscriptItem with text, start, and duration.
    """
    lines = content.splitlines()
    segments: list[RawTranscriptItem] = []

    current_start: float | None = None
    current_end: float | None = None
    current_text_lines: list[str] = []

    for line in lines:
        line = line.strip()

        if _should_skip_vtt_line(line):
            continue

        current_start, current_end, current_text_lines = _process_vtt_line(
            line, segments, current_start, current_end, current_text_lines
        )

    # Handle final segment without trailing newline
    _append_segment_if_valid(segments, current_text_lines, current_start, current_end)

    return segments


def parse_vtt_file(path: str) -> list[RawTranscriptItem]:
    """Parse VTT/SRT file into transcript segments.

    Args:
        path: Path to the subtitle file.

    Returns:
        List of RawTranscriptItem with text, start, and duration.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnicodeDecodeError: If the file cannot be decoded as UTF-8.
    """
    content = Path(path).read_text(encoding="utf-8")
    return parse_vtt_content(content)


__all__ = [
    "parse_vtt_content",
    "parse_vtt_file",
]
