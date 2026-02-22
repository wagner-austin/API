"""SRT subtitle file generation from Whisper transcription segments.

This module provides functions to convert VerboseResponse segments to SRT format,
a standard subtitle format supported by most video players and editors.

SRT Format:
    1
    00:00:01,000 --> 00:00:04,500
    First subtitle text

    2
    00:00:05,000 --> 00:00:08,200
    Second subtitle text

Usage:
    from pathlib import Path
    from platform_stt import OpenAISttClient, format_srt, write_srt

    client = OpenAISttClient(api_key="sk-...")
    with open("audio.mp3", "rb") as f:
        response = client.transcribe(file=f, language="en")

    srt_content = format_srt(response["segments"])
    write_srt(srt_content, Path("subtitles.srt"))

Functions:
    format_srt(segments) - Convert VerboseSegment list to SRT string
    format_srt_entry(entry) - Format single SrtEntry to string block
    format_timestamp(seconds) - Convert float seconds to "HH:MM:SS,mmm"
    write_srt(content, path) - Write SRT string to file
    segments_to_srt_entries(segments) - Convert VerboseSegment list to SrtEntry list
    encode_srt_entry(entry) - Encode SrtEntry to JSONObject
    decode_srt_entry(obj) - Decode JSONObject to SrtEntry with validation
    require_srt_entry(obj) - Validate JSONValue as SrtEntry
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from . import _test_hooks
from .types import VerboseSegment

# =============================================================================
# SRT Entry TypedDict
# =============================================================================


class SrtEntry(TypedDict):
    """A single SRT subtitle entry with index, timing, and text.

    Attributes:
        index: 1-based sequence number for the subtitle entry.
        start_seconds: Start time in seconds from video beginning.
        end_seconds: End time in seconds from video beginning.
        text: The subtitle text content.
    """

    index: int
    start_seconds: float
    end_seconds: float
    text: str


def encode_srt_entry(entry: SrtEntry) -> JSONObject:
    """Encode SrtEntry to JSON-compatible dict.

    Args:
        entry: The SRT entry to encode.

    Returns:
        JSON-compatible dictionary representation.
    """
    return {
        "index": entry["index"],
        "start_seconds": entry["start_seconds"],
        "end_seconds": entry["end_seconds"],
        "text": entry["text"],
    }


def decode_srt_entry(obj: JSONObject) -> SrtEntry:
    """Decode JSON object to SrtEntry with validation.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated SrtEntry.

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
        ValueError: If index is less than 1 or times are negative.
    """
    index = require_int(obj, "index")
    start_seconds = require_float(obj, "start_seconds")
    end_seconds = require_float(obj, "end_seconds")
    text = require_str(obj, "text")

    if index < 1:
        raise ValueError(f"SRT index must be >= 1, got {index}")
    if start_seconds < 0.0:
        raise ValueError(f"start_seconds must be >= 0, got {start_seconds}")
    if end_seconds < 0.0:
        raise ValueError(f"end_seconds must be >= 0, got {end_seconds}")
    if end_seconds < start_seconds:
        raise ValueError(f"end_seconds ({end_seconds}) must be >= start_seconds ({start_seconds})")

    return SrtEntry(
        index=index,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        text=text,
    )


def require_srt_entry(obj: JSONValue) -> SrtEntry:
    """Validate and convert JSONValue to SrtEntry.

    Args:
        obj: JSON value to validate.

    Returns:
        Validated SrtEntry.

    Raises:
        JSONTypeError: If validation fails.
        ValueError: If semantic validation fails.
    """
    if not isinstance(obj, dict):
        raise JSONTypeError(f"Expected object, got {type(obj).__name__}")
    return decode_srt_entry(obj)


# =============================================================================
# Timestamp Formatting
# =============================================================================


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds, must be non-negative.

    Returns:
        Formatted timestamp string in SRT format (e.g., "01:02:03,456").

    Raises:
        ValueError: If seconds is negative.
    """
    if seconds < 0.0:
        raise ValueError(f"Timestamp seconds must be non-negative, got {seconds}")

    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60

    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


# =============================================================================
# SRT Entry Formatting
# =============================================================================


def format_srt_entry(entry: SrtEntry) -> str:
    """Format a single SRT entry as a string block.

    Args:
        entry: The SRT entry to format.

    Returns:
        Formatted SRT entry block with index, timestamps, and text.
    """
    start_ts = format_timestamp(entry["start_seconds"])
    end_ts = format_timestamp(entry["end_seconds"])
    text = entry["text"].strip()

    return f"{entry['index']}\n{start_ts} --> {end_ts}\n{text}"


# =============================================================================
# Full SRT Generation
# =============================================================================


def segments_to_srt_entries(segments: list[VerboseSegment]) -> list[SrtEntry]:
    """Convert Whisper VerboseSegment list to SrtEntry list.

    Args:
        segments: List of VerboseSegment from Whisper transcription.

    Returns:
        List of SrtEntry with 1-based indices.
    """
    entries: list[SrtEntry] = []
    for i, segment in enumerate(segments, start=1):
        entry = SrtEntry(
            index=i,
            start_seconds=segment["start"],
            end_seconds=segment["end"],
            text=segment["text"],
        )
        entries.append(entry)
    return entries


def format_srt(segments: list[VerboseSegment]) -> str:
    """Format Whisper segments as complete SRT file content.

    Args:
        segments: List of VerboseSegment from Whisper transcription.

    Returns:
        Complete SRT file content as a string.
    """
    entries = segments_to_srt_entries(segments)
    formatted_entries: list[str] = []
    for entry in entries:
        formatted_entries.append(format_srt_entry(entry))
    return "\n\n".join(formatted_entries)


def write_srt(content: str, path: Path) -> None:
    """Write SRT content to a file.

    Args:
        content: The SRT file content to write.
        path: Destination file path.

    Raises:
        OSError: If the file cannot be written.
    """
    _test_hooks.write_text_file(path, content)


__all__ = [
    "SrtEntry",
    "decode_srt_entry",
    "encode_srt_entry",
    "format_srt",
    "format_srt_entry",
    "format_timestamp",
    "require_srt_entry",
    "segments_to_srt_entries",
    "write_srt",
]
