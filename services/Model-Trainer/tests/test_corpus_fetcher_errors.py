from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from platform_core.data_bank_client import AuthorizationError, NotFoundError

from model_trainer.core import _test_hooks
from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher


class _ErrorMockServer:
    """Mock HTTP server that returns error responses."""

    def __init__(self, error_code: int, error_text: str) -> None:
        self._error_code = error_code
        self._error_text = error_text

    def handle(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(self._error_code, text=self._error_text)


class _HeadThenDownloadErrorServer:
    """Mock server: HEAD succeeds, download fails with error."""

    def __init__(self, size: int, etag: str, download_error_code: int) -> None:
        self._size = size
        self._etag = etag
        self._download_error_code = download_error_code

    def handle(self, request: httpx.Request) -> httpx.Response:
        hdrs: dict[str, str] = {k.lower(): v for k, v in request.headers.items()}
        if hdrs.get("x-api-key") != "k":
            return httpx.Response(401, text="unauthorized")

        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={
                    "Content-Length": str(self._size),
                    "ETag": self._etag,
                    "Content-Type": "text/plain",
                },
            )
        if request.method == "GET":
            return httpx.Response(self._download_error_code, text="download error")
        return httpx.Response(500, text="unhandled")


class _SizeMismatchServer:
    """Mock server that returns wrong size data."""

    def __init__(self, reported_size: int, actual_data: bytes) -> None:
        self._reported_size = reported_size
        self._actual_data = actual_data
        self._etag = sha256(actual_data).hexdigest()

    def handle(self, request: httpx.Request) -> httpx.Response:
        hdrs: dict[str, str] = {k.lower(): v for k, v in request.headers.items()}
        if hdrs.get("x-api-key") != "k":
            return httpx.Response(401, text="unauthorized")

        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={
                    "Content-Length": str(self._reported_size),
                    "ETag": self._etag,
                    "Content-Type": "text/plain",
                },
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                content=self._actual_data,
                headers={
                    "Content-Length": str(len(self._actual_data)),
                    "ETag": self._etag,
                },
            )
        return httpx.Response(500, text="unhandled")


def test_fetcher_head_404_raises(tmp_path: Path) -> None:
    server = _ErrorMockServer(404, "not found")
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://db", "k", tmp_path)
    with pytest.raises(NotFoundError):
        _ = f.fetch("deadbeef")


def test_fetcher_get_401_raises(tmp_path: Path) -> None:
    # HEAD succeeds but download fails with 401
    server = _HeadThenDownloadErrorServer(size=10, etag="abcd", download_error_code=401)
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://db", "k", tmp_path)
    with pytest.raises(AuthorizationError):
        _ = f.fetch("deadbeef")


def test_fetcher_size_mismatch(tmp_path: Path) -> None:
    # Report size=10 but return only 5 bytes
    server = _SizeMismatchServer(reported_size=10, actual_data=b"abcde")
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://db", "k", tmp_path)
    with pytest.raises(RuntimeError):
        _ = f.fetch("deadbeef")


def test_fetcher_resume_size_mismatch(tmp_path: Path) -> None:
    # Report size=6 but return only 3 bytes total (partial file has 2)
    # The server returns only 1 additional byte making total 3, but expected 6
    actual_data = b"abc"  # 3 bytes
    server = _SizeMismatchServer(reported_size=6, actual_data=actual_data)
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://db", "k", tmp_path)

    # Seed partial temp file to trigger Range branch
    fid = "resume"
    cache_path = tmp_path / f"{fid}.txt"
    tmp_file = cache_path.with_suffix(".tmp")
    tmp_file.write_bytes(b"")  # empty partial file

    # Size mismatch: HEAD says 6 bytes but GET returns 3 bytes
    with pytest.raises(RuntimeError, match="Size mismatch"):
        _ = f.fetch(fid)


def test_fetcher_zero_size_cached(tmp_path: Path) -> None:
    """Test that zero-size file works when already cached."""
    # Zero-size files are an edge case - the DataBankClient skips download
    # when size=0, so this test verifies the cache-hit path works.
    f = CorpusFetcher("http://db", "k", tmp_path)

    # Pre-create a cached zero-size file
    fid = "zero"
    cache_path = tmp_path / f"{fid}.txt"
    cache_path.write_bytes(b"")

    # Should return cached file without making network request
    def _fail_factory(*, timeout_seconds: float = 30.0) -> httpx.Client:
        raise AssertionError("httpx.Client should not be created on cache hit")

    _test_hooks.httpx_client_factory = _fail_factory

    out = f.fetch(fid)
    assert out.exists() and out.stat().st_size == 0
