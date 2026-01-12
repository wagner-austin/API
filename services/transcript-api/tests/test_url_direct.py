"""Tests for url/direct.py URL parsing."""

from __future__ import annotations

import hashlib

import pytest
from platform_core.errors import AppError, TranscriptErrorCode

from transcript_api.url.direct import is_direct_url, parse_direct_url


class TestIsDirectURL:
    """Tests for is_direct_url detection."""

    def test_url_with_empty_path(self) -> None:
        # URL with just a domain, no path - exercises empty path branch
        assert is_direct_url("https://example.com") is False

    def test_mp4_video(self) -> None:
        assert is_direct_url("https://example.com/video.mp4") is True

    def test_webm_video(self) -> None:
        assert is_direct_url("https://cdn.example.org/file.webm") is True

    def test_mkv_video(self) -> None:
        assert is_direct_url("https://media.test.io/movie.mkv") is True

    def test_avi_video(self) -> None:
        assert is_direct_url("https://example.com/video.avi") is True

    def test_mov_video(self) -> None:
        assert is_direct_url("https://example.com/video.mov") is True

    def test_flv_video(self) -> None:
        assert is_direct_url("https://example.com/video.flv") is True

    def test_wmv_video(self) -> None:
        assert is_direct_url("https://example.com/video.wmv") is True

    def test_m4v_video(self) -> None:
        assert is_direct_url("https://example.com/video.m4v") is True

    def test_ogv_video(self) -> None:
        assert is_direct_url("https://example.com/video.ogv") is True

    def test_mp3_audio(self) -> None:
        assert is_direct_url("https://example.com/audio.mp3") is True

    def test_wav_audio(self) -> None:
        assert is_direct_url("https://example.com/audio.wav") is True

    def test_flac_audio(self) -> None:
        assert is_direct_url("https://example.com/audio.flac") is True

    def test_aac_audio(self) -> None:
        assert is_direct_url("https://example.com/audio.aac") is True

    def test_ogg_audio(self) -> None:
        assert is_direct_url("https://example.com/audio.ogg") is True

    def test_m4a_audio(self) -> None:
        assert is_direct_url("https://example.com/audio.m4a") is True

    def test_wma_audio(self) -> None:
        assert is_direct_url("https://example.com/audio.wma") is True

    def test_uppercase_extension(self) -> None:
        assert is_direct_url("https://example.com/video.MP4") is True

    def test_mixed_case_extension(self) -> None:
        assert is_direct_url("https://example.com/video.Mp4") is True

    def test_url_with_query_params(self) -> None:
        assert is_direct_url("https://example.com/video.mp4?token=abc123") is True

    def test_youtube_returns_false(self) -> None:
        assert is_direct_url("https://www.youtube.com/watch?v=abc") is False

    def test_vimeo_returns_false(self) -> None:
        assert is_direct_url("https://vimeo.com/123456789") is False

    def test_no_extension_returns_false(self) -> None:
        assert is_direct_url("https://example.com/video") is False

    def test_unsupported_extension_returns_false(self) -> None:
        assert is_direct_url("https://example.com/file.txt") is False

    def test_html_page_returns_false(self) -> None:
        assert is_direct_url("https://example.com/page.html") is False

    def test_empty_string_returns_false(self) -> None:
        assert is_direct_url("") is False

    def test_whitespace_returns_false(self) -> None:
        assert is_direct_url("   ") is False


class TestParseDirectURLVideoFormats:
    """Tests for parsing direct video URLs."""

    def test_mp4_url(self) -> None:
        url = "https://example.com/video.mp4"
        result = parse_direct_url(url)
        assert result["source"] == "direct"
        assert result["canonical_url"] == url
        assert result["extension"] == "mp4"
        expected_id = hashlib.md5(url.encode("utf-8")).hexdigest()
        assert result["video_id"] == expected_id

    def test_webm_url(self) -> None:
        url = "https://cdn.example.org/media.webm"
        result = parse_direct_url(url)
        assert result["source"] == "direct"
        assert result["extension"] == "webm"

    def test_mkv_url(self) -> None:
        url = "https://example.com/movie.mkv"
        result = parse_direct_url(url)
        assert result["extension"] == "mkv"


class TestParseDirectURLAudioFormats:
    """Tests for parsing direct audio URLs."""

    def test_mp3_url(self) -> None:
        url = "https://example.com/podcast.mp3"
        result = parse_direct_url(url)
        assert result["source"] == "direct"
        assert result["extension"] == "mp3"

    def test_wav_url(self) -> None:
        url = "https://example.com/recording.wav"
        result = parse_direct_url(url)
        assert result["extension"] == "wav"

    def test_flac_url(self) -> None:
        url = "https://example.com/music.flac"
        result = parse_direct_url(url)
        assert result["extension"] == "flac"

    def test_m4a_url(self) -> None:
        url = "https://example.com/audio.m4a"
        result = parse_direct_url(url)
        assert result["extension"] == "m4a"


class TestParseDirectURLEdgeCases:
    """Tests for edge cases in direct URL parsing."""

    def test_url_with_whitespace(self) -> None:
        url = "https://example.com/video.mp4"
        result = parse_direct_url(f"  {url}  ")
        assert result["canonical_url"] == url

    def test_uppercase_extension(self) -> None:
        url = "https://example.com/video.MP4"
        result = parse_direct_url(url)
        assert result["extension"] == "mp4"

    def test_mixed_case_extension(self) -> None:
        url = "https://example.com/video.Mp4"
        result = parse_direct_url(url)
        assert result["extension"] == "mp4"

    def test_url_with_query_params(self) -> None:
        url = "https://example.com/video.mp4?token=abc&expire=123"
        result = parse_direct_url(url)
        assert result["extension"] == "mp4"
        assert result["canonical_url"] == url

    def test_url_with_path_segments(self) -> None:
        url = "https://example.com/path/to/video.mp4"
        result = parse_direct_url(url)
        assert result["extension"] == "mp4"

    def test_without_scheme(self) -> None:
        url = "example.com/video.mp4"
        result = parse_direct_url(url)
        assert result["extension"] == "mp4"

    def test_deterministic_video_id(self) -> None:
        url = "https://example.com/video.mp4"
        result1 = parse_direct_url(url)
        result2 = parse_direct_url(url)
        assert result1["video_id"] == result2["video_id"]

    def test_different_urls_different_ids(self) -> None:
        url1 = "https://example.com/video1.mp4"
        url2 = "https://example.com/video2.mp4"
        result1 = parse_direct_url(url1)
        result2 = parse_direct_url(url2)
        assert result1["video_id"] != result2["video_id"]


class TestIsDirectURLDownloadPatterns:
    """Tests for is_direct_url with download URL patterns."""

    def test_canvas_files_download(self) -> None:
        url = "https://canvas.eee.uci.edu/files/33764330/download"
        assert is_direct_url(url) is True

    def test_canvas_files_download_with_params(self) -> None:
        url = "https://canvas.eee.uci.edu/files/33764330/download?download_frd=1&verifier=abc123"
        assert is_direct_url(url) is True

    def test_generic_download_endpoint(self) -> None:
        url = "https://cdn.example.com/download/12345"
        assert is_direct_url(url) is True

    def test_media_endpoint(self) -> None:
        url = "https://media.example.com/media/98765"
        assert is_direct_url(url) is True

    def test_attachments_endpoint(self) -> None:
        url = "https://api.example.com/attachments/54321"
        assert is_direct_url(url) is True

    def test_blob_storage(self) -> None:
        url = "https://storage.azure.com/container/blob/video-file"
        assert is_direct_url(url) is True

    def test_s3_objects(self) -> None:
        url = "https://bucket.s3.amazonaws.com/objects/media-file"
        assert is_direct_url(url) is True

    def test_verifier_param_makes_url_valid(self) -> None:
        url = "https://example.com/content?verifier=abc123"
        assert is_direct_url(url) is True

    def test_download_frd_param_makes_url_valid(self) -> None:
        url = "https://example.com/content?download_frd=1"
        assert is_direct_url(url) is True

    def test_token_param_makes_url_valid(self) -> None:
        url = "https://cdn.example.com/protected?token=xyz789"
        assert is_direct_url(url) is True

    def test_html_with_download_param_returns_false(self) -> None:
        url = "https://example.com/page.html?download=1"
        assert is_direct_url(url) is False

    def test_php_with_download_param_returns_false(self) -> None:
        url = "https://example.com/script.php?download=1"
        assert is_direct_url(url) is False

    def test_case_insensitive_pattern_matching(self) -> None:
        url = "https://canvas.example.edu/FILES/123/DOWNLOAD"
        assert is_direct_url(url) is True


class TestParseDirectURLDownloadPatterns:
    """Tests for parsing download pattern URLs."""

    def test_canvas_download_url(self) -> None:
        url = "https://canvas.eee.uci.edu/files/33764330/download?download_frd=1&verifier=NfOSyfJrG2gVND3KgvhQWK5YG30xF8byZSkdzbtI"
        result = parse_direct_url(url)
        assert result["source"] == "direct"
        assert result["canonical_url"] == url
        assert result["extension"] == ""  # Unknown until download
        assert len(result["video_id"]) == 32  # MD5 hash

    def test_download_url_extension_empty(self) -> None:
        url = "https://cdn.example.com/download/12345"
        result = parse_direct_url(url)
        assert result["extension"] == ""

    def test_media_endpoint_url(self) -> None:
        url = "https://media.example.com/media/98765"
        result = parse_direct_url(url)
        assert result["source"] == "direct"
        assert result["extension"] == ""

    def test_blob_storage_url(self) -> None:
        url = "https://storage.azure.com/container/blob/video-file"
        result = parse_direct_url(url)
        assert result["source"] == "direct"
        assert result["extension"] == ""

    def test_verifier_param_url(self) -> None:
        url = "https://example.com/content?verifier=abc123"
        result = parse_direct_url(url)
        assert result["source"] == "direct"
        assert result["extension"] == ""

    def test_deterministic_id_for_download_url(self) -> None:
        url = "https://canvas.edu/files/123/download?verifier=abc"
        result1 = parse_direct_url(url)
        result2 = parse_direct_url(url)
        assert result1["video_id"] == result2["video_id"]


class TestParseDirectURLErrors:
    """Tests for direct URL parsing error cases."""

    def test_empty_url_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_direct_url("")
        assert exc_info.value.code is TranscriptErrorCode.DIRECT_URL_INVALID
        assert exc_info.value.http_status == 400

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_direct_url("   ")
        assert exc_info.value.code is TranscriptErrorCode.DIRECT_URL_INVALID

    def test_no_extension_no_pattern_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_direct_url("https://example.com/video")
        assert exc_info.value.code is TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID
        assert exc_info.value.http_status == 400

    def test_unsupported_extension_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_direct_url("https://example.com/file.txt")
        assert exc_info.value.code is TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID
        assert "Unsupported file extension" in exc_info.value.message
        assert ".txt" in exc_info.value.message

    def test_html_file_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_direct_url("https://example.com/page.html")
        assert exc_info.value.code is TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID

    def test_html_download_pattern_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_direct_url("https://example.com/files/123/download.html")
        assert exc_info.value.code is TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID
        assert "webpage" in exc_info.value.message.lower()

    def test_php_with_verifier_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_direct_url("https://example.com/page.php?verifier=abc")
        assert exc_info.value.code is TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID

    def test_pdf_file_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_direct_url("https://example.com/document.pdf")
        assert exc_info.value.code is TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID

    def test_image_file_raises(self) -> None:
        with pytest.raises(AppError) as exc_info:
            parse_direct_url("https://example.com/image.jpg")
        assert exc_info.value.code is TranscriptErrorCode.DIRECT_URL_EXTENSION_INVALID
