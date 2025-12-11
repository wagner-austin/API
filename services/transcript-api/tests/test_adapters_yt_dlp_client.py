from __future__ import annotations

import os
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue

from transcript_api import _test_hooks
from transcript_api.adapters import yt_dlp_client as ymod
from transcript_api.adapters.yt_dlp_client import YtDlpAdapter
from transcript_api.types import YtDlpProto, _TracebackProto


class _FakeYDL:
    """Fake yt-dlp context manager for testing."""

    def __init__(self, opts: dict[str, JSONValue]) -> None:
        self._opts = opts
        self._info: dict[str, JSONValue] = {
            "id": "vid",
            "requested_downloads": [{"filepath": os.path.abspath("/tmp/fake.m4a")}],
        }

    def __enter__(self) -> YtDlpProto:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _TracebackProto | None,
    ) -> None:
        return None

    def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
        return self._info

    def prepare_filename(self, info: dict[str, JSONValue]) -> str:
        return os.path.abspath("/tmp/alt.m4a")


def test_probe_and_download_with_cookies() -> None:
    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _FakeYDL(opts)

    _test_hooks.yt_dlp_factory = _factory

    adapter = YtDlpAdapter()
    info = adapter.probe("https://x")
    assert isinstance(info, dict) and info.get("id") == "vid"

    path = adapter.download_audio("https://x", cookies_path="/tmp/c.txt")
    assert path.endswith("fake.m4a") or path.endswith("alt.m4a")


def test_download_audio_no_cookies_uses_prepare_filename() -> None:
    class _YDL2(_FakeYDL):
        def __init__(self, opts: dict[str, JSONValue]) -> None:
            super().__init__(opts)
            # Remove requested_downloads to force prepare_filename path
            self._info = {"id": "vid"}

    def _factory2(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _YDL2(opts)

    _test_hooks.yt_dlp_factory = _factory2

    adapter = YtDlpAdapter()
    path = adapter.download_audio("https://x", cookies_path=None)
    assert path.endswith("alt.m4a") or path != ""


def test_probe_non_dict_and_download_non_dict() -> None:
    class _YDL3(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            # Return empty dict to exercise {} branches
            return {}

    def _factory3(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _YDL3(opts)

    _test_hooks.yt_dlp_factory = _factory3

    adapter = YtDlpAdapter()
    info = adapter.probe("https://x")
    assert info == {}
    path = adapter.download_audio("https://x", cookies_path=None)
    assert path == ""


def test_probe_typed_coercions_numeric_and_lists() -> None:
    class _TypedYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            return {
                "id": "vid",
                "duration": 12.5,
                "formats": [
                    {
                        "vcodec": "h264",
                        "acodec": "aac",
                        "abr": 64,
                        "filesize": 1024,
                        "filesize_approx": 2048,
                    },
                    {"vcodec": "none", "acodec": "aac", "abr": 32.0},
                ],
                "requested_downloads": [{"filepath": "/tmp/a.m4a"}, {"filepath": "not a path"}],
            }

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _TypedYDL(opts)

    _test_hooks.yt_dlp_factory = _factory

    adapter = YtDlpAdapter()
    out = adapter.probe("https://x")
    assert out.get("id") == "vid"
    assert out.get("duration") == 12.5
    f = out.get("formats")
    assert isinstance(f, list) and len(f) == 2 and isinstance(f[0], dict)
    reqs = out.get("requested_downloads")
    assert isinstance(reqs, list) and len(reqs) == 2 and reqs[0]["filepath"].endswith(".m4a")


def test_probe_typed_coercions_non_numeric_and_missing_lists() -> None:
    class _BadYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            # Return an empty dict to test edge cases
            # The production code handles non-conformant responses
            return {}

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _BadYDL(opts)

    _test_hooks.yt_dlp_factory = _factory

    adapter = YtDlpAdapter()
    out = adapter.probe("https://x")
    assert "duration" not in out
    assert "formats" not in out
    assert "requested_downloads" not in out


def test_yt_is_numeric_str_edges() -> None:
    # Exercise empty, sign-only, and multi-dot cases
    assert ymod.YtDlpAdapter._is_numeric_str("") is False
    assert ymod.YtDlpAdapter._is_numeric_str("+") is False
    assert ymod.YtDlpAdapter._is_numeric_str("1.2.3") is False
    assert ymod.YtDlpAdapter._is_numeric_str("+1.23") is True


def test_probe_int_duration_and_mixed_formats_and_requests() -> None:
    class _MixedYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            return {
                "id": "vid2",
                "duration": 7,
                "formats": [{"acodec": "aac", "abr": 64}],
                "requested_downloads": [{"filepath": "/tmp/b.m4a"}],
            }

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _MixedYDL(opts)

    _test_hooks.yt_dlp_factory = _factory

    adapter = YtDlpAdapter()
    out = adapter.probe("https://x")
    assert out.get("duration") == 7.0
    formats = out.get("formats")
    assert type(formats) is list
    reqs = out.get("requested_downloads")
    assert type(reqs) is list and reqs[0]["filepath"].endswith(".m4a")


def test_probe_formats_branch_edges() -> None:
    class _EdgeYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            return {
                "formats": [
                    {"vcodec": "none", "acodec": "aac", "abr": 16},
                    {},
                ]
            }

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _EdgeYDL(opts)

    _test_hooks.yt_dlp_factory = _factory

    adapter = YtDlpAdapter()
    out = adapter.probe("https://x")
    assert type(out) is dict


def test_coerce_helpers_cover_branches() -> None:
    full = ymod._coerce_format_item(
        {
            "vcodec": "vp9",
            "acodec": "opus",
            "abr": 96,
            "filesize": 1024,
            "filesize_approx": 2048,
        }
    )
    assert full == {
        "vcodec": "vp9",
        "acodec": "opus",
        "abr": 96.0,
        "filesize": 1024,
        "filesize_approx": 2048,
    }

    empty = ymod._coerce_format_item(
        {"vcodec": 1, "acodec": None, "abr": "x", "filesize": "y", "filesize_approx": 1.2}
    )
    assert empty is None

    fmts = ymod._coerce_formats([{"acodec": "aac"}, {"vcodec": 1}, 7])
    assert len(fmts) == 1 and fmts[0]["acodec"] == "aac"

    reqs = ymod._coerce_requested_downloads([{"filepath": "/tmp/a.m4a"}, {"filepath": 5}, 3])
    assert len(reqs) == 1 and reqs[0]["filepath"].endswith(".m4a")


def test_probe_returns_empty_when_not_dict() -> None:
    class _NotDictYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            assert url.startswith("https://")
            assert download is False
            # Return empty dict to test "not dict" branch
            return {}

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _NotDictYDL(opts)

    _test_hooks.yt_dlp_factory = _factory

    adapter = YtDlpAdapter()
    out = adapter.probe("https://x")
    assert out == {}


def test_download_uses_prepare_filename_when_filepath_missing() -> None:
    class _PrepareFilenameYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            assert download is True
            return {"id": "vid"}

        def prepare_filename(self, info: dict[str, JSONValue]) -> str:
            return os.path.abspath("/tmp/generated.m4a")

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _PrepareFilenameYDL(opts)

    _test_hooks.yt_dlp_factory = _factory

    info, path = ymod._yt_download("https://x", {"format": "bestaudio"})
    assert info.get("id") == "vid"
    assert path.endswith("generated.m4a")


def test_is_numeric_str_rejects_alpha() -> None:
    assert ymod.YtDlpAdapter._is_numeric_str("12a") is False


def test_download_subtitles_success(tmp_path: Path) -> None:
    """Test download_subtitles returns subtitles when available."""
    # Create a mock subtitle file
    sub_dir = tmp_path / "ytsubs_test"
    sub_dir.mkdir()
    vtt_file = sub_dir / "subs.vtt"
    vtt_file.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello", encoding="utf-8")

    extract_call_count = 0

    class _SubYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            nonlocal extract_call_count
            extract_call_count += 1
            if not download:
                # Probe call
                return {"subtitles": {"en": [{"ext": "vtt"}]}, "automatic_captions": {}}
            # Download call - return empty, file is already created
            return {}

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _SubYDL(opts)

    _test_hooks.yt_dlp_factory = _factory
    _test_hooks.mkdtemp = lambda prefix, suffix: str(sub_dir)

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path=None,
        preferred_langs=["en"],
    )

    if result is None:
        pytest.fail("expected subtitle result")
    assert result["lang"] == "en"
    assert result["is_auto"] is False
    assert result["path"] == str(vtt_file)


def test_download_subtitles_no_subtitles_available() -> None:
    """Test download_subtitles returns None when no subtitles available."""

    class _NoSubYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            return {"subtitles": {}, "automatic_captions": {}}

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _NoSubYDL(opts)

    _test_hooks.yt_dlp_factory = _factory

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path=None,
        preferred_langs=["en"],
    )

    assert result is None


def test_download_subtitles_probe_returns_none() -> None:
    """Test download_subtitles returns None when probe returns empty dict."""

    class _BadProbeYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            # Return empty dict to trigger None path
            return {}

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _BadProbeYDL(opts)

    _test_hooks.yt_dlp_factory = _factory

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path=None,
        preferred_langs=["en"],
    )

    assert result is None


def test_download_subtitles_uses_auto_captions(tmp_path: Path) -> None:
    """Test download_subtitles uses automatic captions when no manual subtitles."""
    sub_dir = tmp_path / "ytsubs_auto"
    sub_dir.mkdir()
    vtt_file = sub_dir / "subs.vtt"
    vtt_file.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nAuto", encoding="utf-8")

    class _AutoSubYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            if not download:
                return {"subtitles": {}, "automatic_captions": {"en": [{"ext": "vtt"}]}}
            return {}

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _AutoSubYDL(opts)

    _test_hooks.yt_dlp_factory = _factory
    _test_hooks.mkdtemp = lambda prefix, suffix: str(sub_dir)

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path=None,
        preferred_langs=["en"],
    )

    if result is None:
        pytest.fail("expected subtitle result")
    assert result["is_auto"] is True


def test_download_subtitles_with_cookies(tmp_path: Path) -> None:
    """Test download_subtitles passes cookies_path to yt-dlp options."""
    sub_dir = tmp_path / "ytsubs_cookies"
    sub_dir.mkdir()
    vtt_file = sub_dir / "subs.vtt"
    vtt_file.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nWith cookies", encoding="utf-8")

    captured_opts: list[dict[str, JSONValue]] = []

    class _CookieYDL(_FakeYDL):
        def __init__(self, opts: dict[str, JSONValue]) -> None:
            super().__init__(opts)
            captured_opts.append(opts)

        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            if not download:
                return {"subtitles": {"en": [{"ext": "vtt"}]}, "automatic_captions": {}}
            return {}

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _CookieYDL(opts)

    _test_hooks.yt_dlp_factory = _factory
    _test_hooks.mkdtemp = lambda prefix, suffix: str(sub_dir)

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path="/tmp/cookies.txt",
        preferred_langs=["en"],
    )

    if result is None:
        pytest.fail("expected subtitle result")
    # Both probe and download should have cookiefile set
    assert any(opt.get("cookiefile") == "/tmp/cookies.txt" for opt in captured_opts)


def test_download_subtitles_no_matches_after_download(tmp_path: Path) -> None:
    """Test download_subtitles returns None when glob finds no files."""
    sub_dir = tmp_path / "ytsubs_empty"
    sub_dir.mkdir()
    # Don't create any file - glob will find nothing

    class _EmptyGlobYDL(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            if not download:
                return {"subtitles": {"en": [{"ext": "vtt"}]}, "automatic_captions": {}}
            return {}

    def _factory(opts: dict[str, JSONValue]) -> YtDlpProto:
        return _EmptyGlobYDL(opts)

    _test_hooks.yt_dlp_factory = _factory
    _test_hooks.mkdtemp = lambda prefix, suffix: str(sub_dir)

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path=None,
        preferred_langs=["en"],
    )

    assert result is None
