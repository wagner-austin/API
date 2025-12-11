from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO, Final, Protocol, TypedDict

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
from data_bank_api.api.jobs import JobParams, LoggerLike, process_corpus_impl


class _LogExtraDict(TypedDict, total=False):
    job_id: str


class _FakeLogger(LoggerLike):
    """Minimal logger stub compatible with LoggerLike protocol."""

    def __init__(self) -> None:
        self.records: list[dict[str, str | _LogExtraDict]] = []

    def info(self, msg: str, *, extra: _LogExtraDict | None = None) -> None:
        self.records.append({"level": "info", "msg": msg, "extra": extra or {}})

    def error(self, msg: str, *, extra: _LogExtraDict | None = None) -> None:
        self.records.append({"level": "error", "msg": msg, "extra": extra or {}})


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


class _MockDataBankClient:
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
        return {
            "file_id": "deadbeef",
            "size": 3,
            "sha256": "a" * 64,
            "content_type": "text/plain",
            "created_at": "2025-01-01T00:00:00Z",
        }


def _settings(tmp_path: Path, *, url: str = "", key: str = "") -> Settings:
    return Settings(
        data_dir=str(tmp_path),
        environment="test",
        data_bank_api_url=url,
        data_bank_api_key=key,
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
        yield from ["a", "b", "c"]


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


def _make_mock_client_factory(
    client_class: type[_DataBankClientProtocol],
) -> _test_hooks.DataBankUploaderFactoryProtocol:
    """Create a factory function for the given client class."""

    def _factory(
        api_url: str, api_key: str, *, timeout_seconds: float
    ) -> _test_hooks.DataBankUploaderProtocol:
        return client_class(api_url, api_key, timeout_seconds=timeout_seconds)

    return _factory


def test_jobs_uploads_to_data_bank_and_sets_file_id(tmp_path: Path) -> None:
    # Arrange a tiny corpus stream and environment
    job_id: Final[str] = "job-123"

    # Configure hooks
    _test_hooks.ensure_corpus_file = _noop_ensure
    _test_hooks.local_corpus_service_factory = _make_fake_corpus_factory(_FakeCorpusService)
    _test_hooks.data_bank_client_factory = _make_mock_client_factory(_MockDataBankClient)

    # Redis in-memory
    r = FakeRedis()
    s = _settings(tmp_path, url="http://db", key="K")
    log = _FakeLogger()

    # Act
    out = process_corpus_impl(
        job_id,
        params={
            "source": "oscar",
            "language": "kk",
            "max_sentences": 3,
            "transliterate": False,
            "confidence_threshold": 0.9,
        },
        redis=r,
        settings=s,
        logger=log,
    )

    # file_id persisted in redis
    data = r.hgetall(job_key("databank", job_id))
    assert data.get("file_id") == "deadbeef"
    events = [decode_job_event(record.payload) for record in r.published]
    assert any(is_started(ev) for ev in events)
    assert any(is_progress(ev) for ev in events)
    assert any(is_completed(ev) for ev in events)
    assert not any(is_failed(ev) for ev in events)
    # function result reflects completion
    assert out["status"] == "completed"
    r.assert_only_called({"publish", "hset", "expire", "hgetall"})


def test_jobs_upload_handles_missing_file_id_raises(tmp_path: Path) -> None:
    # Arrange minimal environment and noop corpus

    class _BadClient:
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
            raise DataBankClientError("upload response missing file_id")

    # Configure hooks
    _test_hooks.ensure_corpus_file = _noop_ensure
    _test_hooks.local_corpus_service_factory = _make_fake_corpus_factory(_FakeCorpusServiceSingle)
    _test_hooks.data_bank_client_factory = _make_mock_client_factory(_BadClient)

    r = FakeRedis()
    s = _settings(tmp_path, url="http://db", key="K")
    with pytest.raises(DataBankClientError, match="missing file_id"):
        process_corpus_impl(
            "job-2",
            params={
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": False,
                "confidence_threshold": 0.9,
            },
            redis=r,
            settings=s,
            logger=_FakeLogger(),
        )
    events = [decode_job_event(record.payload) for record in r.published]
    assert any(is_started(ev) for ev in events)
    assert any(is_progress(ev) for ev in events)
    assert any(is_failed(ev) for ev in events)
    assert not any(is_completed(ev) for ev in events)
    r.assert_only_called({"publish"})


def test_jobs_upload_config_missing_raises(tmp_path: Path) -> None:
    # Arrange minimal environment and noop corpus
    # Configure hooks
    _test_hooks.ensure_corpus_file = _noop_ensure
    _test_hooks.local_corpus_service_factory = _make_fake_corpus_factory(_FakeCorpusServiceSingle)

    r = FakeRedis()
    # Empty config triggers error before client is created
    s = _settings(tmp_path, url="", key="")
    with pytest.raises(DataBankClientError, match="configuration missing"):
        process_corpus_impl(
            "job-cfg",
            params={
                "source": "oscar",
                "language": "kk",
                "max_sentences": 1,
                "transliterate": False,
                "confidence_threshold": 0.9,
            },
            redis=r,
            settings=s,
            logger=_FakeLogger(),
        )
    events = [decode_job_event(record.payload) for record in r.published]
    assert any(is_started(ev) for ev in events)
    assert any(is_failed(ev) for ev in events)
    assert not any(is_completed(ev) for ev in events)
    assert not any(is_progress(ev) for ev in events)
    r.assert_only_called({"publish"})
