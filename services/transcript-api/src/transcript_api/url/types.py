"""Type definitions for parsed video URLs."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from platform_core.json_utils import JSONTypeError, JSONValue
from typing_extensions import TypedDict


class URLSource(StrEnum):
    """Video source platform identifier."""

    YOUTUBE = "youtube"
    VIMEO = "vimeo"
    DIRECT = "direct"


class YouTubeParsedURL(TypedDict):
    """Parsed YouTube URL with extracted video ID.

    Attributes:
        source: Always "youtube".
        video_id: 11-character YouTube video identifier.
        canonical_url: Normalized YouTube watch URL.
    """

    source: Literal["youtube"]
    video_id: str
    canonical_url: str


class VimeoParsedURL(TypedDict):
    """Parsed Vimeo URL with extracted video ID.

    Attributes:
        source: Always "vimeo".
        video_id: Numeric Vimeo video identifier.
        canonical_url: Normalized Vimeo URL.
    """

    source: Literal["vimeo"]
    video_id: str
    canonical_url: str


class DirectParsedURL(TypedDict):
    """Parsed direct video URL.

    Attributes:
        source: Always "direct".
        video_id: MD5 hash of the URL for identification.
        canonical_url: The original URL (unchanged).
        extension: File extension (e.g., "mp4", "webm").
    """

    source: Literal["direct"]
    video_id: str
    canonical_url: str
    extension: str


# Union type for all parsed URL variants
ParsedURL = YouTubeParsedURL | VimeoParsedURL | DirectParsedURL


def encode_parsed_url(parsed: ParsedURL) -> dict[str, str]:
    """Encode a ParsedURL to a JSON-serializable dict.

    Args:
        parsed: The parsed URL to encode.

    Returns:
        Dictionary with string values suitable for JSON serialization.
    """
    result: dict[str, str] = {
        "source": parsed["source"],
        "video_id": parsed["video_id"],
        "canonical_url": parsed["canonical_url"],
    }
    if parsed["source"] == "direct":
        direct_parsed: DirectParsedURL = parsed
        result["extension"] = direct_parsed["extension"]
    return result


def _require_str(data: dict[str, JSONValue], key: str) -> str:
    """Extract and validate a required string field.

    Args:
        data: Dictionary to extract from.
        key: Key to extract.

    Returns:
        The string value.

    Raises:
        JSONTypeError: If key is missing or value is not a string.
    """
    val = data.get(key)
    if not isinstance(val, str):
        raise JSONTypeError(f"{key} must be a string")
    return val


def decode_parsed_url(data: dict[str, JSONValue]) -> ParsedURL:
    """Decode a dict to a ParsedURL.

    Args:
        data: Dictionary with parsed URL data.

    Returns:
        Typed ParsedURL (YouTube, Vimeo, or Direct).

    Raises:
        JSONTypeError: If required fields are missing or have wrong types.
    """
    source = _require_str(data, "source")
    video_id = _require_str(data, "video_id")
    canonical_url = _require_str(data, "canonical_url")

    if source == "youtube":
        result_yt: YouTubeParsedURL = {
            "source": "youtube",
            "video_id": video_id,
            "canonical_url": canonical_url,
        }
        return result_yt

    if source == "vimeo":
        result_vim: VimeoParsedURL = {
            "source": "vimeo",
            "video_id": video_id,
            "canonical_url": canonical_url,
        }
        return result_vim

    if source == "direct":
        extension = _require_str(data, "extension")
        result_direct: DirectParsedURL = {
            "source": "direct",
            "video_id": video_id,
            "canonical_url": canonical_url,
            "extension": extension,
        }
        return result_direct

    raise JSONTypeError(f"Unknown source: {source}")


__all__ = [
    "DirectParsedURL",
    "ParsedURL",
    "URLSource",
    "VimeoParsedURL",
    "YouTubeParsedURL",
    "decode_parsed_url",
    "encode_parsed_url",
]
