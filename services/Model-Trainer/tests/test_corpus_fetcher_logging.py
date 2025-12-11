from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import TypeVar

import httpx
import pytest
from platform_core.logging import stdlib_logging

from model_trainer.core import _test_hooks
from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher

_T = TypeVar("_T")


def _get_log_extra(
    record: stdlib_logging.LogRecord, key: str, expected_type: type[_T]
) -> _T | None:
    """Type-safe accessor for log record extra fields."""
    val: str | int | float | bool | None = getattr(record, key, None)
    if val is None:
        return None
    if not isinstance(val, expected_type):
        return None
    return val


def _has_log_extra(record: stdlib_logging.LogRecord, key: str) -> bool:
    """Check if a log record has a specific extra field."""
    if not hasattr(record, key):
        return False
    val: str | int | float | bool | None = getattr(record, key, None)
    return val is not None


class _SuccessServer:
    """Mock server that returns successful responses."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._etag = sha256(data).hexdigest()

    def handle(self, request: httpx.Request) -> httpx.Response:
        hdrs: dict[str, str] = {k.lower(): v for k, v in request.headers.items()}
        if hdrs.get("x-api-key") != "k":
            return httpx.Response(401, text="unauthorized")

        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={
                    "Content-Length": str(len(self._data)),
                    "ETag": self._etag,
                    "Content-Type": "text/plain",
                },
            )
        if request.method == "GET":
            rng_header = hdrs.get("range")
            if rng_header and rng_header.startswith("bytes="):
                start = int(rng_header.split("=")[1].split("-")[0])
                part = self._data[start:]
                return httpx.Response(
                    206,
                    content=part,
                    headers={
                        "Content-Length": str(len(part)),
                        "Content-Range": f"bytes {start}-{len(self._data) - 1}/{len(self._data)}",
                        "ETag": self._etag,
                    },
                )
            return httpx.Response(
                200,
                content=self._data,
                headers={
                    "Content-Length": str(len(self._data)),
                    "ETag": self._etag,
                },
            )
        return httpx.Response(500, text="unhandled")


class _SizeMismatchServer:
    """Mock server that returns wrong size in HEAD vs actual data."""

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


def test_fetcher_logs_cache_hit(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that cache hits are logged."""
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)
    fid = "cached_file"
    cache_path = cache / f"{fid}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("cached", encoding="utf-8")

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch(fid)

    assert any("Corpus cache hit" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Corpus cache hit" in r.message)
    assert _get_log_extra(record, "file_id", str) == fid
    assert _get_log_extra(record, "path", str) == str(cache_path)


def test_fetcher_logs_fetch_start(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that fetch start is logged with structured fields."""
    payload = b"test"
    cache = tmp_path / "cache"

    server = _SuccessServer(payload)
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://test", "k", cache)

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch("test_file")

    assert any(
        "Starting corpus fetch from data bank" in record.message for record in caplog.records
    )
    record = next(r for r in caplog.records if "Starting corpus fetch" in r.message)
    assert _get_log_extra(record, "file_id", str) == "test_file"
    assert _get_log_extra(record, "api_url", str) == "http://test"


def test_fetcher_logs_head_request(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that HEAD request is logged."""
    payload = b"test"
    cache = tmp_path / "cache"

    server = _SuccessServer(payload)
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://test", "k", cache)

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch("test_file")

    assert any("Sending HEAD request to data bank" in r.message for r in caplog.records)
    assert any("HEAD request successful" in r.message for r in caplog.records)


def test_fetcher_logs_download_start(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that download start is logged with size."""
    payload = b"test"
    cache = tmp_path / "cache"

    server = _SuccessServer(payload)
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://test", "k", cache)

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch("test_file")

    assert any("Starting corpus download" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Starting corpus download" in r.message)
    assert _get_log_extra(record, "expected_size", int) == len(payload)


def test_fetcher_logs_completion(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that successful completion is logged with elapsed time."""
    payload = b"test"
    cache = tmp_path / "cache"

    server = _SuccessServer(payload)
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://test", "k", cache)

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch("test_file")

    assert any("Corpus fetch completed successfully" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Corpus fetch completed successfully" in r.message)
    assert _get_log_extra(record, "file_id", str) == "test_file"
    assert _get_log_extra(record, "size", int) == len(payload)
    assert _has_log_extra(record, "elapsed_seconds")


def test_fetcher_logs_resume(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that resume is logged with offset."""
    payload = b"test data"
    cache = tmp_path / "cache"

    server = _SuccessServer(payload)
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://test", "k", cache)

    fid = "resume_file"
    cache_path = cache / f"{fid}.txt"
    tmp = cache_path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(payload[:4])

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch(fid)

    assert any("Resuming partial download" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Resuming partial download" in r.message)
    assert _get_log_extra(record, "resume_from", int) == 4


def test_fetcher_logs_size_mismatch_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verify that size mismatch is logged as error."""
    payload = b"small"
    cache = tmp_path / "cache"

    # Report size=100 but return only 5 bytes
    server = _SizeMismatchServer(reported_size=100, actual_data=payload)
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: httpx.Client(
        transport=httpx.MockTransport(server.handle)
    )

    f = CorpusFetcher("http://test", "k", cache)

    stdlib_logging.getLogger().setLevel(stdlib_logging.ERROR)
    with pytest.raises(RuntimeError):
        _ = f.fetch("bad_file")

    assert any("Downloaded file size mismatch" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Downloaded file size mismatch" in r.message)
    assert _get_log_extra(record, "expected_size", int) == 100
    assert _get_log_extra(record, "actual_size", int) == len(payload)
