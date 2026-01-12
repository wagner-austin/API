"""Tests for url/vimeo.py URL parsing."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, TranscriptErrorCode

from transcript_api.url.vimeo import is_vimeo_url, parse_vimeo_url


class TestIsVimeoURL:
    """Tests for is_vimeo_url detection."""

    def test_vimeo_com_standard(self) -> None:
        assert is_vimeo_url("https://vimeo.com/123456789") is True

    def test_www_vimeo_com(self) -> None:
        assert is_vimeo_url("https://www.vimeo.com/123456789") is True

    def test_player_vimeo_com(self) -> None:
        assert is_vimeo_url("https://player.vimeo.com/video/123456789") is True

    def test_without_scheme(self) -> None:
        assert is_vimeo_url("vimeo.com/123456789") is True

    def test_youtube_returns_false(self) -> None:
        assert is_vimeo_url("https://www.youtube.com/watch?v=abc") is False

    def test_direct_url_returns_false(self) -> None:
        assert is_vimeo_url("https://example.com/video.mp4") is False

    def test_empty_string_returns_false(self) -> None:
        assert is_vimeo_url("") is False

    def test_whitespace_returns_false(self) -> None:
        assert is_vimeo_url("   ") is False

    def test_random_domain_returns_false(self) -> None:
        assert is_vimeo_url("https://dailymotion.com/video/x123") is False


class TestParseVimeoURLStandardFormat:
    """Tests for parsing vimeo.com standard URLs."""

    def test_standard_vimeo_url(self) -> None:
        result = parse_vimeo_url("https://vimeo.com/123456789")
        assert result["source"] == "vimeo"
        assert result["video_id"] == "123456789"
        assert result["canonical_url"] == "https://vimeo.com/123456789"

    def test_www_vimeo_url(self) -> None:
        result = parse_vimeo_url("https://www.vimeo.com/987654321")
        assert result["source"] == "vimeo"
        assert result["video_id"] == "987654321"
        assert result["canonical_url"] == "https://vimeo.com/987654321"

    def test_vimeo_with_query_params(self) -> None:
        result = parse_vimeo_url("https://vimeo.com/123456789?h=abc123")
        assert result["video_id"] == "123456789"
        assert result["canonical_url"] == "https://vimeo.com/123456789"

    def test_vimeo_http_scheme(self) -> None:
        result = parse_vimeo_url("http://vimeo.com/123456789")
        assert result["video_id"] == "123456789"


class TestParseVimeoURLPlayerFormat:
    """Tests for parsing player.vimeo.com URLs."""

    def test_player_vimeo_url(self) -> None:
        result = parse_vimeo_url("https://player.vimeo.com/video/123456789")
        assert result["source"] == "vimeo"
        assert result["video_id"] == "123456789"
        assert result["canonical_url"] == "https://vimeo.com/123456789"

    def test_player_vimeo_with_params(self) -> None:
        result = parse_vimeo_url("https://player.vimeo.com/video/123456789?autoplay=1")
        assert result["video_id"] == "123456789"
        assert result["canonical_url"] == "https://vimeo.com/123456789"


class TestParseVimeoURLEdgeCases:
    """Tests for edge cases in Vimeo URL parsing."""

    def test_url_with_whitespace(self) -> None:
        result = parse_vimeo_url("  https://vimeo.com/123456789  ")
        assert result["video_id"] == "123456789"

    def test_without_scheme(self) -> None:
        result = parse_vimeo_url("vimeo.com/123456789")
        assert result["video_id"] == "123456789"

    def test_single_digit_video_id(self) -> None:
        result = parse_vimeo_url("https://vimeo.com/1")
        assert result["video_id"] == "1"

    def test_long_video_id(self) -> None:
        result = parse_vimeo_url("https://vimeo.com/12345678901234567890")
        assert result["video_id"] == "12345678901234567890"


class TestParseVimeoURLErrors:
    """Tests for Vimeo URL parsing error cases."""

    def test_empty_url_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_vimeo_url("")
        assert exc_info.value.code is TranscriptErrorCode.VIMEO_URL_INVALID
        assert exc_info.value.http_status == 400

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_vimeo_url("   ")
        assert exc_info.value.code is TranscriptErrorCode.VIMEO_URL_INVALID

    def test_non_vimeo_url_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_vimeo_url("https://youtube.com/watch?v=abc12345678")
        assert exc_info.value.code is TranscriptErrorCode.VIMEO_URL_INVALID
        assert exc_info.value.http_status == 400

    def test_vimeo_with_non_numeric_id_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_vimeo_url("https://vimeo.com/abc123")
        assert exc_info.value.code is TranscriptErrorCode.VIMEO_VIDEO_ID_INVALID
        assert exc_info.value.http_status == 400

    def test_vimeo_empty_path_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_vimeo_url("https://vimeo.com/")
        assert exc_info.value.code is TranscriptErrorCode.VIMEO_VIDEO_ID_INVALID

    def test_player_vimeo_wrong_path_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_vimeo_url("https://player.vimeo.com/embed/123456789")
        assert exc_info.value.code is TranscriptErrorCode.VIMEO_VIDEO_ID_INVALID

    def test_vimeo_with_letters_in_id_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_vimeo_url("https://vimeo.com/video123")
        assert exc_info.value.code is TranscriptErrorCode.VIMEO_VIDEO_ID_INVALID
