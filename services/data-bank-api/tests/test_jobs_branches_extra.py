from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO, Protocol, TypedDict

import pytest
from platform_core.data_bank_client import DataBankClientError
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.job_events import (
    decode_job_event,
    is_completed,
    is_failed,
    is_progress,
    is_started,
)
from platform_core.job_types import job_key
from platform_workers.testing import FakeRedis

from data_bank_api import _test_hooks
from data_bank_api.api.config import Settings
from data_bank_api.api.jobs import (
    JobParams,
    LocalCorpusService,
    LoggerLike,
    process_corpus_impl,
)


class _LogExtraDict(TypedDict, total=False):
    job_id: str


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url="http://db",
        data_bank_api_key="K",
    )


def _noop_ensure(
    *,
    source: str,
    language: str,
    data_dir: str,
    max_sentences: int,
    transliterate: bool,
    confidence_threshold: float,
) -> None:
    """No-op corpus ensure function for testing."""
    return


class _FakeCorpusService:
    """Fake corpus service that yields test lines."""

    def __init__(self, data_dir: str) -> None:
        pass

    def stream(self, spec: JobParams) -> Generator[str, None, None]:
        yield from ["one", "two"]


class _FakeCorpusServiceSingle:
    """Fake corpus service that yields a single test line."""

    def __init__(self, data_dir: str) -> None:
        pass

    def stream(self, spec: JobParams) -> Generator[str, None, None]:
        yield from ["x"]


def _make_fake_corpus_factory(
    service_class: type[_FakeCorpusService] | type[_FakeCorpusServiceSingle],
) -> _test_hooks.LocalCorpusServiceFactoryProtocol:
    """Create a factory function for the given corpus service class."""

    def _factory(data_dir: str) -> _test_hooks.LocalCorpusServiceProtocol:
        return service_class(data_dir)

    return _factory


class _DataBankClientProtocol(Protocol):
    """Protocol for mock DataBankClient classes used in tests."""

    def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 60.0) -> None: ...

    def upload(
        self,
        file_id: str,
        stream: BinaryIO,
        *,
        content_type: str = "application/octet-stream",
        request_id: str | None = None,
    ) -> FileUploadResponse: ...


def _make_mock_client_factory(
    client_class: type[_DataBankClientProtocol],
) -> _test_hooks.DataBankUploaderFactoryProtocol:
    """Create a factory function for the given client class."""

    def _factory(
        api_url: str, api_key: str, *, timeout_seconds: float
    ) -> _test_hooks.DataBankUploaderProtocol:
        return client_class(api_url, api_key, timeout_seconds=timeout_seconds)

    return _factory


class _Log(LoggerLike):
    """Simple logger stub for testing."""

    def info(self, msg: str, *, extra: _LogExtraDict | None = None) -> None:
        return None

    def error(self, msg: str, *, extra: _LogExtraDict | None = None) -> None:
        return None


def test_local_corpus_service_defaults(tmp_path: Path) -> None:
    # Ensure default implementation is covered: __init__ + empty iterator
    svc = LocalCorpusService(str(tmp_path))
    empty_params: JobParams = {
        "source": "",
        "language": "",
        "max_sentences": 0,
        "transliterate": False,
        "confidence_threshold": 0.0,
    }
    assert list(svc.stream(empty_params)) == []


def test_local_corpus_service_streams_file_content(tmp_path: Path) -> None:
    """Test that LocalCorpusService reads and yields lines from corpus file."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "oscar_kk.txt"
    corpus_file.write_text("hello\nworld\ntest\n", encoding="utf-8")

    svc = LocalCorpusService(str(tmp_path))
    params: JobParams = {
        "source": "oscar",
        "language": "kk",
        "max_sentences": 0,  # unlimited
        "transliterate": False,
        "confidence_threshold": 0.0,
    }
    result = list(svc.stream(params))
    assert result == ["hello", "world", "test"]


def test_local_corpus_service_respects_max_sentences(tmp_path: Path) -> None:
    """Test that LocalCorpusService respects max_sentences limit."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "oscar_en.txt"
    corpus_file.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

    svc = LocalCorpusService(str(tmp_path))
    params: JobParams = {
        "source": "oscar",
        "language": "en",
        "max_sentences": 2,
        "transliterate": False,
        "confidence_threshold": 0.0,
    }
    result = list(svc.stream(params))
    assert result == ["line1", "line2"]


def test_local_corpus_service_skips_empty_lines(tmp_path: Path) -> None:
    """Test that LocalCorpusService skips empty lines."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "oscar_ru.txt"
    corpus_file.write_text("line1\n\nline2\n\n\nline3\n", encoding="utf-8")

    svc = LocalCorpusService(str(tmp_path))
    params: JobParams = {
        "source": "oscar",
        "language": "ru",
        "max_sentences": 0,
        "transliterate": False,
        "confidence_threshold": 0.0,
    }
    result = list(svc.stream(params))
    assert result == ["line1", "line2", "line3"]


def test_jobs_upload_handles_client_error(tmp_path: Path) -> None:
    class _ErrorClient:
        def __init__(
            self,
            base_url: str,
            api_key: str,
            *,
            timeout_seconds: float = 60.0,
        ) -> None:
            pass

        def upload(
            self,
            file_id: str,
            stream: BinaryIO,
            *,
            content_type: str = "application/octet-stream",
            request_id: str | None = None,
        ) -> FileUploadResponse:
            raise DataBankClientError("upload failed")

    # Configure hooks
    _test_hooks.ensure_corpus_file = _noop_ensure
    _test_hooks.local_corpus_service_factory = _make_fake_corpus_factory(_FakeCorpusService)
    _test_hooks.data_bank_client_factory = _make_mock_client_factory(_ErrorClient)

    # Execute
    r = FakeRedis()
    s = _settings(tmp_path)

    with pytest.raises(DataBankClientError, match="upload failed"):
        process_corpus_impl(
            "job-xx",
            params={
                "source": "oscar",
                "language": "kk",
                "max_sentences": 2,
                "transliterate": False,
                "confidence_threshold": 0.9,
            },
            redis=r,
            settings=s,
            logger=_Log(),
        )
    assert r.hgetall(job_key("databank", "job-xx")) == {}
    r.assert_only_called({"publish", "hgetall"})


def test_process_corpus_upload_server_error_raises(tmp_path: Path) -> None:
    class _ServerErrorClient:
        def __init__(
            self,
            base_url: str,
            api_key: str,
            *,
            timeout_seconds: float = 60.0,
        ) -> None:
            pass

        def upload(
            self,
            file_id: str,
            stream: BinaryIO,
            *,
            content_type: str = "application/octet-stream",
            request_id: str | None = None,
        ) -> FileUploadResponse:
            raise DataBankClientError("HTTP 500: Internal Server Error")

    # Configure hooks
    _test_hooks.ensure_corpus_file = _noop_ensure
    _test_hooks.local_corpus_service_factory = _make_fake_corpus_factory(_FakeCorpusServiceSingle)
    _test_hooks.data_bank_client_factory = _make_mock_client_factory(_ServerErrorClient)

    r = FakeRedis()
    s: Settings = {
        "data_dir": str(tmp_path),
        "environment": "test",
        "data_bank_api_url": "http://db",
        "data_bank_api_key": "K",
    }

    with pytest.raises(DataBankClientError, match="HTTP 500"):
        process_corpus_impl(
            "jid",
            params={
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": False,
                "confidence_threshold": 0.9,
            },
            redis=r,
            settings=s,
            logger=_Log(),
        )
    events = [decode_job_event(record.payload) for record in r.published]
    assert any(is_started(ev) for ev in events)
    assert any(is_failed(ev) for ev in events)
    assert any(is_progress(ev) for ev in events)
    assert not any(is_completed(ev) for ev in events)
    r.assert_only_called({"publish"})
