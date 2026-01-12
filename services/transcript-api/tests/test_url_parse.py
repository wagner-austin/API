"""Tests for url/parse.py unified URL parsing."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, TranscriptErrorCode

from transcript_api.url import parse_video_url


class TestParseVideoURLYouTube:
    """Tests for parsing YouTube URLs via unified parser."""

    def test_youtube_watch_url(self) -> None:
        result = parse_video_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["source"] == "youtube"
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["canonical_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_youtube_short_url(self) -> None:
        result = parse_video_url("https://youtu.be/dQw4w9WgXcQ")
        assert result["source"] == "youtube"
        assert result["video_id"] == "dQw4w9WgXcQ"

    def test_youtube_shorts_url(self) -> None:
        result = parse_video_url("https://www.youtube.com/shorts/abc12345678")
        assert result["source"] == "youtube"
        assert result["video_id"] == "abc12345678"

    def test_youtube_live_url(self) -> None:
        result = parse_video_url("https://www.youtube.com/live/xyz98765432")
        assert result["source"] == "youtube"
        assert result["video_id"] == "xyz98765432"


class TestParseVideoURLVimeo:
    """Tests for parsing Vimeo URLs via unified parser."""

    def test_vimeo_standard_url(self) -> None:
        result = parse_video_url("https://vimeo.com/123456789")
        assert result["source"] == "vimeo"
        assert result["video_id"] == "123456789"
        assert result["canonical_url"] == "https://vimeo.com/123456789"

    def test_vimeo_player_url(self) -> None:
        result = parse_video_url("https://player.vimeo.com/video/987654321")
        assert result["source"] == "vimeo"
        assert result["video_id"] == "987654321"

    def test_vimeo_www_url(self) -> None:
        result = parse_video_url("https://www.vimeo.com/123456789")
        assert result["source"] == "vimeo"


class TestParseVideoURLDirect:
    """Tests for parsing direct URLs via unified parser."""

    def test_direct_mp4_url(self) -> None:
        result = parse_video_url("https://example.com/video.mp4")
        assert result["source"] == "direct"
        assert result["canonical_url"] == "https://example.com/video.mp4"
        assert "extension" in result
        assert result["extension"] == "mp4"

    def test_direct_webm_url(self) -> None:
        result = parse_video_url("https://cdn.example.org/media.webm")
        assert result["source"] == "direct"
        assert "extension" in result
        assert result["extension"] == "webm"

    def test_direct_mp3_url(self) -> None:
        result = parse_video_url("https://example.com/audio.mp3")
        assert result["source"] == "direct"
        assert "extension" in result
        assert result["extension"] == "mp3"


class TestParseVideoURLPriority:
    """Tests for URL source detection priority."""

    def test_youtube_takes_precedence_over_mp4_extension(self) -> None:
        # A URL that looks like it could be direct (has extension-like path)
        # but is actually YouTube should be detected as YouTube
        result = parse_video_url("https://www.youtube.com/watch?v=abc12345678")
        assert result["source"] == "youtube"

    def test_vimeo_takes_precedence(self) -> None:
        result = parse_video_url("https://vimeo.com/123456789")
        assert result["source"] == "vimeo"


class TestParseVideoURLEdgeCases:
    """Tests for edge cases in unified URL parsing."""

    def test_url_with_whitespace(self) -> None:
        result = parse_video_url("  https://youtu.be/dQw4w9WgXcQ  ")
        assert result["source"] == "youtube"
        assert result["video_id"] == "dQw4w9WgXcQ"

    def test_url_without_scheme_youtube(self) -> None:
        result = parse_video_url("youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["source"] == "youtube"

    def test_url_without_scheme_vimeo(self) -> None:
        result = parse_video_url("vimeo.com/123456789")
        assert result["source"] == "vimeo"

    def test_url_without_scheme_direct(self) -> None:
        result = parse_video_url("example.com/video.mp4")
        assert result["source"] == "direct"


class TestParseVideoURLErrors:
    """Tests for unified URL parsing error cases."""

    def test_empty_url_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_video_url("")
        assert exc_info.value.code is TranscriptErrorCode.VIDEO_URL_REQUIRED
        assert exc_info.value.http_status == 400
        assert "Please provide a video URL" in exc_info.value.message

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_video_url("   ")
        assert exc_info.value.code is TranscriptErrorCode.VIDEO_URL_REQUIRED

    def test_unsupported_platform_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_video_url("https://dailymotion.com/video/x123456")
        assert exc_info.value.code is TranscriptErrorCode.VIDEO_URL_UNSUPPORTED
        assert exc_info.value.http_status == 400
        assert "YouTube, Vimeo, or a direct video file" in exc_info.value.message

    def test_random_webpage_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_video_url("https://google.com/search?q=test")
        assert exc_info.value.code is TranscriptErrorCode.VIDEO_URL_UNSUPPORTED

    def test_unsupported_file_extension_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_video_url("https://example.com/file.txt")
        assert exc_info.value.code is TranscriptErrorCode.VIDEO_URL_UNSUPPORTED

    def test_html_page_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_video_url("https://example.com/page.html")
        assert exc_info.value.code is TranscriptErrorCode.VIDEO_URL_UNSUPPORTED


class TestParseVideoURLIntegration:
    """Integration tests for complete parsing workflow."""

    def test_youtube_to_canonical(self) -> None:
        # Various YouTube URL formats should all canonicalize
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "youtube.com/watch?v=dQw4w9WgXcQ",
        ]
        for url in urls:
            result = parse_video_url(url)
            assert result["canonical_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_vimeo_to_canonical(self) -> None:
        # Various Vimeo URL formats should all canonicalize
        urls = [
            "https://vimeo.com/123456789",
            "https://www.vimeo.com/123456789",
            "https://player.vimeo.com/video/123456789",
            "vimeo.com/123456789",
        ]
        for url in urls:
            result = parse_video_url(url)
            assert result["canonical_url"] == "https://vimeo.com/123456789"

    def test_direct_preserves_original(self) -> None:
        # Direct URLs should preserve the original URL as canonical
        url = "https://cdn.example.com/path/to/video.mp4?token=abc123"
        result = parse_video_url(url)
        assert result["canonical_url"] == url
