from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
from platform_core.json_utils import JSONValue

import transcript_api.adapters.yt_dlp_client as ymod
from transcript_api.adapters.yt_dlp_client import YtDlpAdapter
from transcript_api.types import YtInfoTD


class _YtDlpModule(ModuleType):
    YoutubeDL: ymod._YtDlpFactoryProto


class _FakeYDL:
    def __init__(self, opts: dict[str, JSONValue]) -> None:
        self._opts = opts
        self._info: dict[str, JSONValue] = {
            "id": "vid",
            "requested_downloads": [{"filepath": os.path.abspath("/tmp/fake.m4a")}],
        }

    def __enter__(self) -> ymod._YtDlpProto:
        return self

    def __exit__(self, *args: str | int | None) -> None:
        return None

    def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
        return self._info

    def prepare_filename(self, info: dict[str, JSONValue]) -> str:
        return os.path.abspath("/tmp/alt.m4a")


def test_probe_and_download_with_cookies(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _YtDlpModule("yt_dlp")

    def _factory(opts: dict[str, JSONValue]) -> ymod._YtDlpContextProto:
        return _FakeYDL(opts)

    mod.YoutubeDL = _factory
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)

    adapter = YtDlpAdapter()
    info = adapter.probe("https://x")
    assert isinstance(info, dict) and info.get("id") == "vid"

    path = adapter.download_audio("https://x", cookies_path="/tmp/c.txt")
    assert path.endswith("fake.m4a") or path.endswith("alt.m4a")


def test_download_audio_no_cookies_uses_prepare_filename(monkeypatch: pytest.MonkeyPatch) -> None:
    class _YDL2(_FakeYDL):
        def __init__(self, opts: dict[str, JSONValue]) -> None:
            super().__init__(opts)
            # Remove requested_downloads to force prepare_filename path
            self._info = {"id": "vid"}

    mod2 = _YtDlpModule("yt_dlp")

    def _factory2(opts: dict[str, JSONValue]) -> ymod._YtDlpContextProto:
        return _YDL2(opts)

    mod2.YoutubeDL = _factory2
    monkeypatch.setitem(sys.modules, "yt_dlp", mod2)

    adapter = YtDlpAdapter()
    path = adapter.download_audio("https://x", cookies_path=None)
    assert path.endswith("alt.m4a") or path != ""


def test_probe_non_dict_and_download_non_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    class _YDL3(_FakeYDL):
        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            # Return non-dict to exercise {}/"" branches
            return {}

    mod3 = _YtDlpModule("yt_dlp")

    def _factory3(opts: dict[str, JSONValue]) -> ymod._YtDlpContextProto:
        return _YDL3(opts)

    mod3.YoutubeDL = _factory3
    monkeypatch.setitem(sys.modules, "yt_dlp", mod3)

    adapter = YtDlpAdapter()
    info = adapter.probe("https://x")
    assert info == {}
    path = adapter.download_audio("https://x", cookies_path=None)
    assert path == ""


def test_probe_typed_coercions_numeric_and_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    def _probe(_: str) -> YtInfoTD:
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

    monkeypatch.setattr(ymod, "_yt_probe", _probe, raising=True)
    adapter = YtDlpAdapter()
    out = adapter.probe("https://x")
    assert out.get("id") == "vid"
    assert out.get("duration") == 12.5
    f = out.get("formats")
    assert isinstance(f, list) and len(f) == 2 and isinstance(f[0], dict)
    reqs = out.get("requested_downloads")
    assert isinstance(reqs, list) and len(reqs) == 2 and reqs[0]["filepath"].endswith(".m4a")


def test_probe_typed_coercions_non_numeric_and_missing_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This test now needs to exercise _coerce_yt_info with invalid data
    # Mock at extract_info level instead of _yt_probe to test coercion
    def _bad_extract(_: str, download: bool) -> dict[str, JSONValue]:
        result: dict[str, JSONValue] = {
            "duration": "not-a-number",
            "formats": "oops",
            "requested_downloads": "oops",
        }
        return result

    def _make_factory() -> ymod._YtDlpFactoryProto:
        class _FakeDL:
            def __init__(self, _opts: dict[str, JSONValue]) -> None:
                pass

            def __enter__(self) -> _FakeDL:
                return self

            def __exit__(self, *_args: str | int | None) -> None:
                pass

            def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
                return _bad_extract(url, download)

            def prepare_filename(self, info: dict[str, JSONValue]) -> str:
                return ""

        def _factory(opts: dict[str, JSONValue]) -> ymod._YtDlpContextProto:
            return _FakeDL(opts)

        factory: ymod._YtDlpFactoryProto = _factory
        return factory

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", _make_factory, raising=True)
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


def test_probe_int_duration_and_mixed_formats_and_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Test coercion with mixed types - mock at factory level
    def _mixed_extract(_: str, download: bool) -> dict[str, JSONValue]:
        acodec_dict: dict[str, JSONValue] = {"acodec": "aac", "abr": 64}
        filepath_dict: dict[str, JSONValue] = {"filepath": "/tmp/b.m4a"}
        result: dict[str, JSONValue] = {
            "id": "vid2",
            "duration": 7,
            "formats": [123, acodec_dict],
            "requested_downloads": [123, filepath_dict],
        }
        return result

    def _make_factory() -> ymod._YtDlpFactoryProto:
        class _FakeDL:
            def __init__(self, _opts: dict[str, JSONValue]) -> None:
                pass

            def __enter__(self) -> _FakeDL:
                return self

            def __exit__(self, *_args: str | int | None) -> None:
                pass

            def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
                return _mixed_extract(url, download)

            def prepare_filename(self, info: dict[str, JSONValue]) -> str:
                return ""

        def _factory(opts: dict[str, JSONValue]) -> ymod._YtDlpContextProto:
            return _FakeDL(opts)

        factory: ymod._YtDlpFactoryProto = _factory
        return factory

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", _make_factory, raising=True)
    adapter = YtDlpAdapter()
    out = adapter.probe("https://x")
    assert out.get("duration") == 7.0
    formats = out.get("formats")
    assert type(formats) is list
    reqs = out.get("requested_downloads")
    assert type(reqs) is list and reqs[0]["filepath"].endswith(".m4a")


def test_probe_formats_branch_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    def _probe(_: str) -> dict[str, list[dict[str, str | int]]]:
        return {
            "formats": [
                {"vcodec": "none", "acodec": 1, "abr": "16"},
                {},
            ]
        }

    monkeypatch.setattr(ymod, "_yt_probe", _probe, raising=True)
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


def test_probe_returns_empty_when_not_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Ctx:
        def __init__(self, _opts: dict[str, JSONValue]) -> None:
            pass

        def __enter__(self) -> _Ctx:
            return self

        def __exit__(self, *_args: str | int | None) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> str:
            assert url.startswith("https://")
            assert download is False
            return "not-a-dict"

        def prepare_filename(self, info: dict[str, JSONValue]) -> str:
            return ""

    def _factory(opts: dict[str, JSONValue]) -> _Ctx:
        return _Ctx(opts)

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", lambda: _factory, raising=True)
    adapter = YtDlpAdapter()
    out = adapter.probe("https://x")
    assert out == {}


def test_download_uses_prepare_filename_when_filepath_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_coerce(raw: dict[str, JSONValue]) -> YtInfoTD:
        assert raw == {"id": "vid"}
        return {"requested_downloads": [{}], "id": "vid"}

    class _Ctx:
        def __init__(self, _opts: dict[str, JSONValue]) -> None:
            pass

        def __enter__(self) -> _Ctx:
            return self

        def __exit__(self, *_args: str | int | None) -> None:
            return None

        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            assert download is True
            return {"id": "vid"}

        def prepare_filename(self, info: dict[str, JSONValue]) -> str:
            return os.path.abspath("/tmp/generated.m4a")

    def _factory(opts: dict[str, JSONValue]) -> _Ctx:
        return _Ctx(opts)

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", lambda: _factory, raising=True)
    monkeypatch.setattr(ymod, "_coerce_yt_info", _fake_coerce, raising=True)

    info, path = ymod._yt_download("https://x", {"format": "bestaudio"})
    assert info.get("id") == "vid"
    assert path.endswith("generated.m4a")


def test_is_numeric_str_rejects_alpha() -> None:
    assert ymod.YtDlpAdapter._is_numeric_str("12a") is False


def test_download_subtitles_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test download_subtitles returns subtitles when available."""
    # Create a mock subtitle file
    sub_dir = tmp_path / "ytsubs_test"
    sub_dir.mkdir()
    vtt_file = sub_dir / "subs.vtt"
    vtt_file.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello", encoding="utf-8")

    extract_call_count = 0

    class _SubCtx:
        def __init__(self, _opts: dict[str, JSONValue]) -> None:
            self._opts = _opts

        def __enter__(self) -> _SubCtx:
            return self

        def __exit__(self, *_args: str | int | None) -> None:
            pass

        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            nonlocal extract_call_count
            extract_call_count += 1
            if not download:
                # Probe call
                return {"subtitles": {"en": [{"ext": "vtt"}]}, "automatic_captions": {}}
            # Download call - return empty, file is already created
            return {}

        def prepare_filename(self, info: dict[str, JSONValue]) -> str:
            return ""

    def _factory(opts: dict[str, JSONValue]) -> _SubCtx:
        return _SubCtx(opts)

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", lambda: _factory, raising=True)

    # Mock tempfile.mkdtemp to return our controlled directory
    import tempfile

    def _mock_mkdtemp(prefix: str) -> str:
        return str(sub_dir)

    monkeypatch.setattr(tempfile, "mkdtemp", _mock_mkdtemp)

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


def test_download_subtitles_no_subtitles_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test download_subtitles returns None when no subtitles available."""

    class _NoSubCtx:
        def __init__(self, _opts: dict[str, JSONValue]) -> None:
            pass

        def __enter__(self) -> _NoSubCtx:
            return self

        def __exit__(self, *_args: str | int | None) -> None:
            pass

        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            return {"subtitles": {}, "automatic_captions": {}}

        def prepare_filename(self, info: dict[str, JSONValue]) -> str:
            return ""

    def _factory(opts: dict[str, JSONValue]) -> _NoSubCtx:
        return _NoSubCtx(opts)

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", lambda: _factory, raising=True)

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path=None,
        preferred_langs=["en"],
    )

    assert result is None


def test_download_subtitles_probe_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test download_subtitles returns None when probe returns non-dict."""

    class _BadProbeCtx:
        def __init__(self, _opts: dict[str, JSONValue]) -> None:
            pass

        def __enter__(self) -> _BadProbeCtx:
            return self

        def __exit__(self, *_args: str | int | None) -> None:
            pass

        def extract_info(self, url: str, download: bool) -> str:
            # Return non-dict to trigger None path
            return "not-a-dict"

        def prepare_filename(self, info: dict[str, JSONValue]) -> str:
            return ""

    def _factory(opts: dict[str, JSONValue]) -> _BadProbeCtx:
        return _BadProbeCtx(opts)

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", lambda: _factory, raising=True)

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path=None,
        preferred_langs=["en"],
    )

    assert result is None


def test_download_subtitles_uses_auto_captions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test download_subtitles uses automatic captions when no manual subtitles."""
    sub_dir = tmp_path / "ytsubs_auto"
    sub_dir.mkdir()
    vtt_file = sub_dir / "subs.vtt"
    vtt_file.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nAuto", encoding="utf-8")

    class _AutoSubCtx:
        def __init__(self, _opts: dict[str, JSONValue]) -> None:
            self._opts = _opts

        def __enter__(self) -> _AutoSubCtx:
            return self

        def __exit__(self, *_args: str | int | None) -> None:
            pass

        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            if not download:
                return {"subtitles": {}, "automatic_captions": {"en": [{"ext": "vtt"}]}}
            return {}

        def prepare_filename(self, info: dict[str, JSONValue]) -> str:
            return ""

    def _factory(opts: dict[str, JSONValue]) -> _AutoSubCtx:
        return _AutoSubCtx(opts)

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", lambda: _factory, raising=True)

    import tempfile

    def _mock_mkdtemp(prefix: str) -> str:
        return str(sub_dir)

    monkeypatch.setattr(tempfile, "mkdtemp", _mock_mkdtemp)

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path=None,
        preferred_langs=["en"],
    )

    if result is None:
        pytest.fail("expected subtitle result")
    assert result["is_auto"] is True


def test_download_subtitles_with_cookies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test download_subtitles passes cookies_path to yt-dlp options."""
    sub_dir = tmp_path / "ytsubs_cookies"
    sub_dir.mkdir()
    vtt_file = sub_dir / "subs.vtt"
    vtt_file.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nWith cookies", encoding="utf-8")

    captured_opts: list[dict[str, JSONValue]] = []

    class _CookieCtx:
        def __init__(self, opts: dict[str, JSONValue]) -> None:
            captured_opts.append(opts)

        def __enter__(self) -> _CookieCtx:
            return self

        def __exit__(self, *_args: str | int | None) -> None:
            pass

        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            if not download:
                return {"subtitles": {"en": [{"ext": "vtt"}]}, "automatic_captions": {}}
            return {}

        def prepare_filename(self, info: dict[str, JSONValue]) -> str:
            return ""

    def _factory(opts: dict[str, JSONValue]) -> _CookieCtx:
        return _CookieCtx(opts)

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", lambda: _factory, raising=True)

    import tempfile

    def _mock_mkdtemp(prefix: str) -> str:
        return str(sub_dir)

    monkeypatch.setattr(tempfile, "mkdtemp", _mock_mkdtemp)

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


def test_download_subtitles_no_matches_after_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test download_subtitles returns None when glob finds no files."""
    sub_dir = tmp_path / "ytsubs_empty"
    sub_dir.mkdir()
    # Don't create any file - glob will find nothing

    class _EmptyGlobCtx:
        def __init__(self, _opts: dict[str, JSONValue]) -> None:
            pass

        def __enter__(self) -> _EmptyGlobCtx:
            return self

        def __exit__(self, *_args: str | int | None) -> None:
            pass

        def extract_info(self, url: str, download: bool) -> dict[str, JSONValue]:
            if not download:
                return {"subtitles": {"en": [{"ext": "vtt"}]}, "automatic_captions": {}}
            return {}

        def prepare_filename(self, info: dict[str, JSONValue]) -> str:
            return ""

    def _factory(opts: dict[str, JSONValue]) -> _EmptyGlobCtx:
        return _EmptyGlobCtx(opts)

    monkeypatch.setattr(ymod, "_create_yt_dlp_factory", lambda: _factory, raising=True)

    import tempfile

    def _mock_mkdtemp(prefix: str) -> str:
        return str(sub_dir)

    monkeypatch.setattr(tempfile, "mkdtemp", _mock_mkdtemp)

    adapter = YtDlpAdapter()
    result = adapter.download_subtitles(
        "https://youtube.com/watch?v=test",
        cookies_path=None,
        preferred_langs=["en"],
    )

    assert result is None
