from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import pytest
from platform_core.data_bank_client import HeadInfo
from platform_core.logging import stdlib_logging

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


def test_fetcher_logs_fetch_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that fetch start is logged with structured fields."""
    payload = b"test"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C:
        def __init__(
            self: _C, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as out:
                out.write(payload)
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C)

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch("test_file")

    assert any(
        "Starting corpus fetch from data bank" in record.message for record in caplog.records
    )
    record = next(r for r in caplog.records if "Starting corpus fetch" in r.message)
    assert _get_log_extra(record, "file_id", str) == "test_file"
    assert _get_log_extra(record, "api_url", str) == "http://test"


def test_fetcher_logs_head_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that HEAD request is logged."""
    payload = b"test"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C2:
        def __init__(
            self: _C2, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C2, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C2,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            with dest.open("wb") as out:
                out.write(payload)
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C2)

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch("test_file")

    assert any("Sending HEAD request to data bank" in r.message for r in caplog.records)
    assert any("HEAD request successful" in r.message for r in caplog.records)


def test_fetcher_logs_download_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that download start is logged with size."""
    payload = b"test"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C3:
        def __init__(
            self: _C3, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C3, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C3,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            with dest.open("wb") as out:
                out.write(payload)
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C3)

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch("test_file")

    assert any("Starting corpus download" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Starting corpus download" in r.message)
    assert _get_log_extra(record, "expected_size", int) == len(payload)


def test_fetcher_logs_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that successful completion is logged with elapsed time."""
    payload = b"test"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C:
        def __init__(
            self: _C, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            with dest.open("wb") as out:
                out.write(payload)
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C)

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch("test_file")

    assert any("Corpus fetch completed successfully" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Corpus fetch completed successfully" in r.message)
    assert _get_log_extra(record, "file_id", str) == "test_file"
    assert _get_log_extra(record, "size", int) == len(payload)
    assert _has_log_extra(record, "elapsed_seconds")


def test_fetcher_logs_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that resume is logged with offset."""
    payload = b"test data"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    fid = "resume_file"
    cache_path = cache / f"{fid}.txt"
    tmp = cache_path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(payload[:4])

    class _C:
        def __init__(
            self: _C, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            start = dest.stat().st_size if dest.exists() else 0
            with dest.open("ab" if start > 0 else "wb") as out:
                out.write(payload[start:])
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C)

    stdlib_logging.getLogger().setLevel(stdlib_logging.INFO)
    _ = f.fetch(fid)

    assert any("Resuming partial download" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Resuming partial download" in r.message)
    assert _get_log_extra(record, "resume_from", int) == 4


def test_fetcher_logs_size_mismatch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify that size mismatch is logged as error."""
    payload = b"small"
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C:
        def __init__(
            self: _C, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return HeadInfo(size=100, etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            with dest.open("wb") as out:
                out.write(payload)
            return HeadInfo(size=100, etag="abcd", content_type="text/plain")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C)

    stdlib_logging.getLogger().setLevel(stdlib_logging.ERROR)
    with pytest.raises(RuntimeError):
        _ = f.fetch("bad_file")

    assert any("Downloaded file size mismatch" in record.message for record in caplog.records)
    record = next(r for r in caplog.records if "Downloaded file size mismatch" in r.message)
    assert _get_log_extra(record, "expected_size", int) == 100
    assert _get_log_extra(record, "actual_size", int) == len(payload)
