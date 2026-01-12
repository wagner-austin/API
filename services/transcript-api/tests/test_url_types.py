"""Tests for url/types.py encode/decode functions."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from transcript_api.url.types import (
    DirectParsedURL,
    URLSource,
    VimeoParsedURL,
    YouTubeParsedURL,
    decode_parsed_url,
    encode_parsed_url,
)


class TestURLSource:
    """Tests for URLSource enum."""

    def test_youtube_value(self) -> None:
        assert URLSource.YOUTUBE.value == "youtube"

    def test_vimeo_value(self) -> None:
        assert URLSource.VIMEO.value == "vimeo"

    def test_direct_value(self) -> None:
        assert URLSource.DIRECT.value == "direct"


class TestEncodeYouTube:
    """Tests for encoding YouTube parsed URLs."""

    def test_encode_youtube_url(self) -> None:
        parsed: YouTubeParsedURL = {
            "source": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "canonical_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }
        encoded = encode_parsed_url(parsed)
        assert encoded == {
            "source": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "canonical_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }

    def test_encode_youtube_no_extension_key(self) -> None:
        parsed: YouTubeParsedURL = {
            "source": "youtube",
            "video_id": "abc12345678",
            "canonical_url": "https://www.youtube.com/watch?v=abc12345678",
        }
        encoded = encode_parsed_url(parsed)
        assert "extension" not in encoded


class TestEncodeVimeo:
    """Tests for encoding Vimeo parsed URLs."""

    def test_encode_vimeo_url(self) -> None:
        parsed: VimeoParsedURL = {
            "source": "vimeo",
            "video_id": "123456789",
            "canonical_url": "https://vimeo.com/123456789",
        }
        encoded = encode_parsed_url(parsed)
        assert encoded == {
            "source": "vimeo",
            "video_id": "123456789",
            "canonical_url": "https://vimeo.com/123456789",
        }

    def test_encode_vimeo_no_extension_key(self) -> None:
        parsed: VimeoParsedURL = {
            "source": "vimeo",
            "video_id": "987654321",
            "canonical_url": "https://vimeo.com/987654321",
        }
        encoded = encode_parsed_url(parsed)
        assert "extension" not in encoded


class TestEncodeDirect:
    """Tests for encoding direct parsed URLs."""

    def test_encode_direct_url(self) -> None:
        parsed: DirectParsedURL = {
            "source": "direct",
            "video_id": "abc123def456",
            "canonical_url": "https://example.com/video.mp4",
            "extension": "mp4",
        }
        encoded = encode_parsed_url(parsed)
        assert encoded == {
            "source": "direct",
            "video_id": "abc123def456",
            "canonical_url": "https://example.com/video.mp4",
            "extension": "mp4",
        }

    def test_encode_direct_includes_extension(self) -> None:
        parsed: DirectParsedURL = {
            "source": "direct",
            "video_id": "xyz789",
            "canonical_url": "https://cdn.example.org/media.webm",
            "extension": "webm",
        }
        encoded = encode_parsed_url(parsed)
        assert "extension" in encoded
        assert encoded["extension"] == "webm"


class TestDecodeYouTube:
    """Tests for decoding YouTube parsed URLs."""

    def test_decode_youtube_url(self) -> None:
        data: dict[str, JSONValue] = {
            "source": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "canonical_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }
        parsed = decode_parsed_url(data)
        assert parsed["source"] == "youtube"
        assert parsed["video_id"] == "dQw4w9WgXcQ"
        assert parsed["canonical_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


class TestDecodeVimeo:
    """Tests for decoding Vimeo parsed URLs."""

    def test_decode_vimeo_url(self) -> None:
        data: dict[str, JSONValue] = {
            "source": "vimeo",
            "video_id": "123456789",
            "canonical_url": "https://vimeo.com/123456789",
        }
        parsed = decode_parsed_url(data)
        assert parsed["source"] == "vimeo"
        assert parsed["video_id"] == "123456789"
        assert parsed["canonical_url"] == "https://vimeo.com/123456789"


class TestDecodeDirect:
    """Tests for decoding direct parsed URLs."""

    def test_decode_direct_url(self) -> None:
        data: dict[str, JSONValue] = {
            "source": "direct",
            "video_id": "abc123",
            "canonical_url": "https://example.com/video.mp4",
            "extension": "mp4",
        }
        parsed = decode_parsed_url(data)
        assert parsed["source"] == "direct"
        assert parsed["video_id"] == "abc123"
        assert parsed["canonical_url"] == "https://example.com/video.mp4"
        assert "extension" in parsed
        assert parsed["extension"] == "mp4"


class TestDecodeErrors:
    """Tests for decode error handling."""

    def test_missing_source_raises(self) -> None:
        data: dict[str, JSONValue] = {
            "video_id": "abc",
            "canonical_url": "https://example.com",
        }
        with pytest.raises(JSONTypeError, match="source must be a string"):
            decode_parsed_url(data)

    def test_missing_video_id_raises(self) -> None:
        data: dict[str, JSONValue] = {
            "source": "youtube",
            "canonical_url": "https://example.com",
        }
        with pytest.raises(JSONTypeError, match="video_id must be a string"):
            decode_parsed_url(data)

    def test_missing_canonical_url_raises(self) -> None:
        data: dict[str, JSONValue] = {
            "source": "youtube",
            "video_id": "abc123",
        }
        with pytest.raises(JSONTypeError, match="canonical_url must be a string"):
            decode_parsed_url(data)

    def test_unknown_source_raises(self) -> None:
        data: dict[str, JSONValue] = {
            "source": "tiktok",
            "video_id": "abc123",
            "canonical_url": "https://tiktok.com/video",
        }
        with pytest.raises(JSONTypeError, match="Unknown source: tiktok"):
            decode_parsed_url(data)

    def test_direct_missing_extension_raises(self) -> None:
        data: dict[str, JSONValue] = {
            "source": "direct",
            "video_id": "abc123",
            "canonical_url": "https://example.com/video.mp4",
        }
        with pytest.raises(JSONTypeError, match="extension must be a string"):
            decode_parsed_url(data)

    def test_source_not_string_raises(self) -> None:
        data: dict[str, JSONValue] = {
            "source": 123,
            "video_id": "abc",
            "canonical_url": "https://example.com",
        }
        with pytest.raises(JSONTypeError, match="source must be a string"):
            decode_parsed_url(data)

    def test_video_id_not_string_raises(self) -> None:
        data: dict[str, JSONValue] = {
            "source": "youtube",
            "video_id": 123,
            "canonical_url": "https://example.com",
        }
        with pytest.raises(JSONTypeError, match="video_id must be a string"):
            decode_parsed_url(data)


class TestRoundTrip:
    """Tests for encode/decode round-trip consistency."""

    def test_youtube_roundtrip(self) -> None:
        original: YouTubeParsedURL = {
            "source": "youtube",
            "video_id": "dQw4w9WgXcQ",
            "canonical_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        }
        encoded = encode_parsed_url(original)
        decoded = decode_parsed_url(_to_json_value_dict(encoded))
        assert decoded == original

    def test_vimeo_roundtrip(self) -> None:
        original: VimeoParsedURL = {
            "source": "vimeo",
            "video_id": "123456789",
            "canonical_url": "https://vimeo.com/123456789",
        }
        encoded = encode_parsed_url(original)
        decoded = decode_parsed_url(_to_json_value_dict(encoded))
        assert decoded == original

    def test_direct_roundtrip(self) -> None:
        original: DirectParsedURL = {
            "source": "direct",
            "video_id": "abc123",
            "canonical_url": "https://example.com/video.mp4",
            "extension": "mp4",
        }
        encoded = encode_parsed_url(original)
        decoded = decode_parsed_url(_to_json_value_dict(encoded))
        assert decoded == original


def _to_json_value_dict(d: dict[str, str]) -> dict[str, JSONValue]:
    """Convert dict[str, str] to dict[str, JSONValue] for type safety."""
    result: dict[str, JSONValue] = {}
    for k, v in d.items():
        result[k] = v
    return result
