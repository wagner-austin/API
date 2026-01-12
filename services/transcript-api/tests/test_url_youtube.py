"""Tests for url/youtube.py URL parsing."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, TranscriptErrorCode

from transcript_api.url.youtube import is_youtube_url, parse_youtube_url


class TestIsYouTubeURL:
    """Tests for is_youtube_url detection."""

    def test_youtube_com_watch(self) -> None:
        assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_www_youtube_com_watch(self) -> None:
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_m_youtube_com_watch(self) -> None:
        assert is_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_youtu_be_short(self) -> None:
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True

    def test_www_youtu_be_short(self) -> None:
        assert is_youtube_url("https://www.youtu.be/dQw4w9WgXcQ") is True

    def test_youtube_shorts(self) -> None:
        assert is_youtube_url("https://www.youtube.com/shorts/abc12345678") is True

    def test_youtube_live(self) -> None:
        assert is_youtube_url("https://www.youtube.com/live/abc12345678") is True

    def test_without_scheme(self) -> None:
        assert is_youtube_url("youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_vimeo_returns_false(self) -> None:
        assert is_youtube_url("https://vimeo.com/123456789") is False

    def test_direct_url_returns_false(self) -> None:
        assert is_youtube_url("https://example.com/video.mp4") is False

    def test_empty_string_returns_false(self) -> None:
        assert is_youtube_url("") is False

    def test_whitespace_returns_false(self) -> None:
        assert is_youtube_url("   ") is False

    def test_random_domain_returns_false(self) -> None:
        assert is_youtube_url("https://google.com/search") is False


class TestParseYouTubeURLWatchFormat:
    """Tests for parsing youtube.com/watch URLs."""

    def test_standard_watch_url(self) -> None:
        result = parse_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["source"] == "youtube"
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["canonical_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_watch_with_extra_params(self) -> None:
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s&list=PLxyz"
        result = parse_youtube_url(url)
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["canonical_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_watch_without_www(self) -> None:
        result = parse_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["canonical_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_watch_mobile(self) -> None:
        result = parse_youtube_url("https://m.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["canonical_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_watch_http_scheme(self) -> None:
        result = parse_youtube_url("http://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["video_id"] == "dQw4w9WgXcQ"


class TestParseYouTubeURLShortFormat:
    """Tests for parsing youtu.be short URLs."""

    def test_youtu_be_short_url(self) -> None:
        result = parse_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert result["source"] == "youtube"
        assert result["video_id"] == "dQw4w9WgXcQ"
        assert result["canonical_url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_youtu_be_with_timestamp(self) -> None:
        result = parse_youtube_url("https://youtu.be/dQw4w9WgXcQ?t=42")
        assert result["video_id"] == "dQw4w9WgXcQ"

    def test_www_youtu_be(self) -> None:
        result = parse_youtube_url("https://www.youtu.be/dQw4w9WgXcQ")
        assert result["video_id"] == "dQw4w9WgXcQ"


class TestParseYouTubeURLShortsFormat:
    """Tests for parsing YouTube Shorts URLs."""

    def test_shorts_url(self) -> None:
        result = parse_youtube_url("https://www.youtube.com/shorts/abc12345678")
        assert result["source"] == "youtube"
        assert result["video_id"] == "abc12345678"
        assert result["canonical_url"] == "https://www.youtube.com/watch?v=abc12345678"

    def test_shorts_without_www(self) -> None:
        result = parse_youtube_url("https://youtube.com/shorts/abc12345678")
        assert result["video_id"] == "abc12345678"


class TestParseYouTubeURLLiveFormat:
    """Tests for parsing YouTube Live URLs."""

    def test_live_url(self) -> None:
        result = parse_youtube_url("https://www.youtube.com/live/xyz98765432")
        assert result["source"] == "youtube"
        assert result["video_id"] == "xyz98765432"
        assert result["canonical_url"] == "https://www.youtube.com/watch?v=xyz98765432"


class TestParseYouTubeURLEdgeCases:
    """Tests for edge cases in YouTube URL parsing."""

    def test_url_with_whitespace(self) -> None:
        result = parse_youtube_url("  https://youtu.be/dQw4w9WgXcQ  ")
        assert result["video_id"] == "dQw4w9WgXcQ"

    def test_without_scheme(self) -> None:
        result = parse_youtube_url("youtube.com/watch?v=dQw4w9WgXcQ")
        assert result["video_id"] == "dQw4w9WgXcQ"

    def test_video_id_with_hyphens(self) -> None:
        result = parse_youtube_url("https://youtu.be/-abc123_xyz")
        assert result["video_id"] == "-abc123_xyz"

    def test_video_id_with_underscores(self) -> None:
        result = parse_youtube_url("https://youtu.be/abc_def_123")
        assert result["video_id"] == "abc_def_123"


class TestParseYouTubeURLErrors:
    """Tests for YouTube URL parsing error cases."""

    def test_youtube_channel_url_raises(self) -> None:
        # Tests the False branch when path isn't watch/shorts/live
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("https://www.youtube.com/channel/UCxyz")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID

    def test_youtu_be_empty_path_raises(self) -> None:
        # Tests the False branch when youtu.be path is empty
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("https://youtu.be/")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID

    def test_empty_url_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_URL_REQUIRED
        assert exc_info.value.http_status == 400

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("   ")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_URL_REQUIRED

    def test_non_youtube_url_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("https://vimeo.com/123456789")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_URL_UNSUPPORTED
        assert exc_info.value.http_status == 400

    def test_invalid_video_id_too_short_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("https://youtu.be/abc")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID
        assert exc_info.value.http_status == 400

    def test_invalid_video_id_too_long_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("https://youtu.be/abcdefghijklmnop")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID

    def test_watch_url_missing_v_param_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("https://www.youtube.com/watch")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID

    def test_watch_url_empty_v_param_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("https://www.youtube.com/watch?v=")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID

    def test_invalid_characters_in_video_id_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_youtube_url("https://youtu.be/abc!@#$%^&*(")
        assert exc_info.value.code is TranscriptErrorCode.YOUTUBE_VIDEO_ID_INVALID
