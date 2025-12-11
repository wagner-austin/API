"""Tests for yt-dlp subtitle functionality."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue

from transcript_api import _test_hooks
from transcript_api.adapters.yt_dlp_client import (
    _find_subtitle_lang,
    _select_lang_from_dict,
)
from transcript_api.stt_provider import YtDlpCaptionProvider
from transcript_api.types import (
    SubtitleResultTD,
    TranscriptOptions,
    YtInfoTD,
)


class _StubProbeClient:
    """Stub probe client for testing YtDlpCaptionProvider."""

    def __init__(
        self,
        subtitle_result: SubtitleResultTD | None = None,
    ) -> None:
        self._subtitle_result = subtitle_result
        self.download_subtitles_calls: list[tuple[str, str | None, list[str]]] = []

    def probe(self, url: str) -> YtInfoTD:
        return {"duration": 60}

    def download_audio(self, url: str, *, cookies_path: str | None) -> str:
        return ""

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None:
        self.download_subtitles_calls.append((url, cookies_path, preferred_langs))
        return self._subtitle_result


class TestSelectLangFromDict:
    """Tests for _select_lang_from_dict."""

    def test_selects_preferred_language(self) -> None:
        subtitle_dict: dict[str, JSONValue] = {"en": True, "es": True, "fr": True}
        result = _select_lang_from_dict(subtitle_dict, ["es", "en"])
        assert result == "es"

    def test_selects_first_preferred_available(self) -> None:
        subtitle_dict: dict[str, JSONValue] = {"en": True, "fr": True}
        result = _select_lang_from_dict(subtitle_dict, ["de", "es", "en"])
        assert result == "en"

    def test_falls_back_to_first_available(self) -> None:
        subtitle_dict: dict[str, JSONValue] = {"ja": True, "ko": True}
        result = _select_lang_from_dict(subtitle_dict, ["en", "es"])
        # First key in dict
        assert result in ["ja", "ko"]

    def test_returns_none_for_empty_dict(self) -> None:
        empty_dict: dict[str, JSONValue] = {}
        result = _select_lang_from_dict(empty_dict, ["en"])
        assert result is None


class TestFindSubtitleLang:
    """Tests for _find_subtitle_lang."""

    def test_prefers_manual_subtitles(self) -> None:
        info: dict[str, JSONValue] = {
            "subtitles": {"en": True},
            "automatic_captions": {"en": True},
        }
        lang, is_auto = _find_subtitle_lang(info, ["en"])
        assert lang == "en"
        assert is_auto is False

    def test_falls_back_to_auto_captions(self) -> None:
        empty_subs: dict[str, JSONValue] = {}
        info: dict[str, JSONValue] = {
            "subtitles": empty_subs,
            "automatic_captions": {"en": True},
        }
        lang, is_auto = _find_subtitle_lang(info, ["en"])
        assert lang == "en"
        assert is_auto is True

    def test_returns_none_when_no_subtitles(self) -> None:
        empty_subs: dict[str, JSONValue] = {}
        empty_auto: dict[str, JSONValue] = {}
        info: dict[str, JSONValue] = {
            "subtitles": empty_subs,
            "automatic_captions": empty_auto,
        }
        lang, is_auto = _find_subtitle_lang(info, ["en"])
        assert lang is None
        assert is_auto is False

    def test_handles_missing_subtitle_keys(self) -> None:
        info: dict[str, JSONValue] = {}
        lang, is_auto = _find_subtitle_lang(info, ["en"])
        assert lang is None
        assert is_auto is False

    def test_handles_non_dict_subtitles(self) -> None:
        info: dict[str, JSONValue] = {
            "subtitles": "not a dict",
            "automatic_captions": None,
        }
        lang, is_auto = _find_subtitle_lang(info, ["en"])
        assert lang is None
        assert is_auto is False

    def test_empty_string_lang_in_subtitles_falls_through(self) -> None:
        """Test that empty string language key in subtitles falls through to auto captions."""
        info: dict[str, JSONValue] = {
            "subtitles": {"": True},  # Empty string key is falsy when selected
            "automatic_captions": {"en": True},
        }
        lang, is_auto = _find_subtitle_lang(info, ["fr"])  # fr not found
        # Empty string is selected but is falsy, so falls through to auto captions
        assert lang == "en"
        assert is_auto is True

    def test_empty_string_lang_in_auto_captions_returns_none(self) -> None:
        """Test that empty string language key in auto captions returns None."""
        empty_subs: dict[str, JSONValue] = {}
        info: dict[str, JSONValue] = {
            "subtitles": empty_subs,
            "automatic_captions": {"": True},  # Empty string key is falsy when selected
        }
        lang, is_auto = _find_subtitle_lang(info, ["fr"])  # fr not found
        # Empty string is selected but is falsy, so returns None
        assert lang is None
        assert is_auto is False


class TestYtDlpCaptionProvider:
    """Tests for YtDlpCaptionProvider."""

    def test_fetch_success(self, tmp_path: Path) -> None:
        # Create a VTT file
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hello world
"""
        vtt_path = tmp_path / "subs.vtt"
        vtt_path.write_text(vtt_content, encoding="utf-8")

        stub = _StubProbeClient(
            subtitle_result={"path": str(vtt_path), "lang": "en", "is_auto": False}
        )
        provider = YtDlpCaptionProvider(probe_client=stub)

        opts: TranscriptOptions = {"preferred_langs": ["en"]}
        result = provider.fetch("test_video_id", opts)

        assert len(result) == 1
        assert result[0]["text"] == "Hello world"
        assert result[0]["start"] == 0.0
        assert result[0]["duration"] == 5.0

        # Check that download_subtitles was called correctly
        assert len(stub.download_subtitles_calls) == 1
        url, _cookies, langs = stub.download_subtitles_calls[0]
        assert "test_video_id" in url
        assert langs == ["en"]

    def test_fetch_no_subtitles_raises(self) -> None:
        stub = _StubProbeClient(subtitle_result=None)
        provider = YtDlpCaptionProvider(probe_client=stub)

        opts: TranscriptOptions = {"preferred_langs": ["en"]}
        with pytest.raises(AppError) as exc_info:
            provider.fetch("test_video_id", opts)

        assert "No captions available" in str(exc_info.value)

    def test_fetch_uses_default_langs(self, tmp_path: Path) -> None:
        vtt_path = tmp_path / "subs.vtt"
        vtt_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi", encoding="utf-8")

        stub = _StubProbeClient(
            subtitle_result={"path": str(vtt_path), "lang": "en", "is_auto": False}
        )
        provider = YtDlpCaptionProvider(probe_client=stub)

        # Empty preferred_langs should use default
        opts: TranscriptOptions = {"preferred_langs": []}
        # TypedDict.get returns the default when key is present but value is empty
        result = provider.fetch("vid123", opts)

        assert len(result) == 1

    def test_fetch_with_cookies_text(self, tmp_path: Path) -> None:
        import base64

        vtt_path = tmp_path / "subs.vtt"
        vtt_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi", encoding="utf-8")

        stub = _StubProbeClient(
            subtitle_result={"path": str(vtt_path), "lang": "en", "is_auto": False}
        )

        # Create base64 encoded cookies
        cookies_content = "# Netscape HTTP Cookie File\n"
        cookies_b64 = base64.b64encode(cookies_content.encode("utf-8")).decode("ascii")

        provider = YtDlpCaptionProvider(
            probe_client=stub,
            cookies_text=cookies_b64,
        )

        opts: TranscriptOptions = {"preferred_langs": ["en"]}
        result = provider.fetch("vid123", opts)

        assert len(result) == 1
        # Cookies path should have been passed
        _, cookies_path, _ = stub.download_subtitles_calls[0]
        if cookies_path is None:
            pytest.fail("expected cookies_path to be set")

    def test_fetch_with_cookies_path(self, tmp_path: Path) -> None:
        vtt_path = tmp_path / "subs.vtt"
        vtt_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi", encoding="utf-8")

        cookies_path = tmp_path / "cookies.txt"
        cookies_path.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

        stub = _StubProbeClient(
            subtitle_result={"path": str(vtt_path), "lang": "en", "is_auto": False}
        )

        provider = YtDlpCaptionProvider(
            probe_client=stub,
            cookies_path=str(cookies_path),
        )

        opts: TranscriptOptions = {"preferred_langs": ["en"]}
        result = provider.fetch("vid123", opts)

        assert len(result) == 1
        _, used_cookies, _ = stub.download_subtitles_calls[0]
        assert used_cookies == str(cookies_path)

    def test_cleanup_removes_subtitle_file(self, tmp_path: Path) -> None:
        # Create a temp directory to simulate yt-dlp behavior
        sub_dir = tmp_path / "ytsubs_test"
        sub_dir.mkdir()
        vtt_path = sub_dir / "subs.vtt"
        vtt_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi", encoding="utf-8")

        stub = _StubProbeClient(
            subtitle_result={"path": str(vtt_path), "lang": "en", "is_auto": False}
        )
        provider = YtDlpCaptionProvider(probe_client=stub)

        opts: TranscriptOptions = {"preferred_langs": ["en"]}
        result = provider.fetch("vid123", opts)

        assert len(result) == 1
        # File should be cleaned up
        assert not vtt_path.exists()
        # Directory should be cleaned up if empty
        assert not sub_dir.exists()

    def test_handles_invalid_cookies_text(self, tmp_path: Path) -> None:
        vtt_path = tmp_path / "subs.vtt"
        vtt_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi", encoding="utf-8")

        stub = _StubProbeClient(
            subtitle_result={"path": str(vtt_path), "lang": "en", "is_auto": False}
        )

        # Invalid base64
        provider = YtDlpCaptionProvider(
            probe_client=stub,
            cookies_text="not valid base64!!!",
        )

        opts: TranscriptOptions = {"preferred_langs": ["en"]}
        result = provider.fetch("vid123", opts)

        assert len(result) == 1
        # Should fall back to no cookies
        _, cookies_path, _ = stub.download_subtitles_calls[0]
        assert cookies_path is None

    def test_cleanup_oserror_is_logged(self, tmp_path: Path) -> None:
        """Test that OSError during cleanup is caught and logged."""
        # Create a VTT file
        sub_dir = tmp_path / "ytsubs_cleanup_error"
        sub_dir.mkdir()
        vtt_path = sub_dir / "subs.vtt"
        vtt_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi", encoding="utf-8")

        stub = _StubProbeClient(
            subtitle_result={"path": str(vtt_path), "lang": "en", "is_auto": False}
        )
        provider = YtDlpCaptionProvider(probe_client=stub)

        # Make os.remove raise OSError to trigger the except branch via hook
        def _failing_remove(path: str) -> None:
            raise OSError("Permission denied")

        _test_hooks.os_remove = _failing_remove

        opts: TranscriptOptions = {"preferred_langs": ["en"]}
        result = provider.fetch("vid123", opts)

        # Should still return results even if cleanup fails
        assert len(result) == 1
        assert result[0]["text"] == "Hi"
        # File should still exist since remove failed
        assert vtt_path.exists()
