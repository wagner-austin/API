"""Tests for scripts.download_fields module."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from scripts._test_hooks import HttpGetResponseProtocol
from scripts.download_fields import download_field_gifs, main

from scripts import _test_hooks


class _FakeResponse:
    """Typed fake HTTP response for download tests."""

    def __init__(self, status_code: int, content: bytes) -> None:
        self._status_code = status_code
        self._content = content

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def content(self) -> bytes:
        return self._content


_OK_RESPONSE: HttpGetResponseProtocol = _FakeResponse(200, b"GIF89a_fake_data")
_HTML_RESPONSE: HttpGetResponseProtocol = _FakeResponse(200, b"<!DOCTYPE html>")
_NOT_FOUND_RESPONSE: HttpGetResponseProtocol = _FakeResponse(404, b"")


@pytest.fixture(autouse=True)
def _isolate_hooks() -> Generator[None, None, None]:
    """Save and restore hooks around each test."""
    orig_path_exists = _test_hooks.path_exists
    orig_http_get = _test_hooks.http_get
    orig_setup = _test_hooks.setup_rich_logging
    yield
    _test_hooks.path_exists = orig_path_exists
    _test_hooks.http_get = orig_http_get
    _test_hooks.setup_rich_logging = orig_setup


def test_download_skips_existing(tmp_path: Path) -> None:
    """Already-downloaded GIFs are skipped."""
    _test_hooks.path_exists = lambda p: True
    paths = download_field_gifs(output_dir=tmp_path)
    assert len(paths) == 50


def test_download_handles_http_errors(tmp_path: Path) -> None:
    """Non-200 responses are skipped without error."""

    def _fake_get(url: str) -> HttpGetResponseProtocol:
        return _NOT_FOUND_RESPONSE

    _test_hooks.path_exists = lambda p: False
    _test_hooks.http_get = _fake_get
    paths = download_field_gifs(output_dir=tmp_path)
    assert len(paths) == 0


def test_download_skips_non_gif_response(tmp_path: Path) -> None:
    """200 responses with non-GIF content are skipped."""

    def _fake_get(url: str) -> HttpGetResponseProtocol:
        return _HTML_RESPONSE

    _test_hooks.path_exists = lambda p: False
    _test_hooks.http_get = _fake_get
    paths = download_field_gifs(output_dir=tmp_path)
    assert len(paths) == 0


def test_download_saves_valid_response(tmp_path: Path) -> None:
    """Valid responses are saved to disk, fetched from the client's real path.

    The URL is pinned because the former ``/play/fieldXX.gif`` path now
    serves the SPA's HTML page (discovered 2026-07-19); the live client
    JS builds ``/images/maps/field`` + id, and the downloader must
    match it or every fetch gets skipped by the GIF-magic guard.
    """
    requested: list[str] = []

    def _fake_get(url: str) -> HttpGetResponseProtocol:
        requested.append(url)
        return _OK_RESPONSE

    _test_hooks.path_exists = lambda p: False
    _test_hooks.http_get = _fake_get
    paths = download_field_gifs(output_dir=tmp_path)
    assert len(paths) == 50
    assert (tmp_path / "field01_r.gif").read_bytes() == b"GIF89a_fake_data"
    assert requested[0] == "https://tankpit.com/images/maps/field01.gif"


def test_main_runs_without_error(tmp_path: Path) -> None:
    """main() returns 0 on success."""
    _test_hooks.path_exists = lambda p: True
    called: list[str] = []

    def _fake_setup(level: _test_hooks.LogLevel) -> None:
        called.append(level)

    _test_hooks.setup_rich_logging = _fake_setup
    result = main()
    assert result == 0
    assert called == ["INFO"]


def test_main_module_entry() -> None:
    """Running as __main__ invokes main() and exits 0."""
    import runpy
    import sys
    import warnings

    _test_hooks.path_exists = lambda p: True
    called: list[str] = []

    def _fake_setup(level: _test_hooks.LogLevel) -> None:
        called.append(level)

    _test_hooks.setup_rich_logging = _fake_setup
    sys.modules.pop("scripts.download_fields", None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("scripts.download_fields", run_name="__main__")
    assert exc_info.value.code == 0
